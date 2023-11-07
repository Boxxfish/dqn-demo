mod cartpole;
mod dqn;
mod env;
mod model;
mod replay_buffer;

use crate::{dqn::train_dqn, replay_buffer::ReplayBuffer};
use anyhow::Result;
use candle_core::{DType, Device, Module, Shape, Tensor, D};
use candle_nn as nn;
use env::{GridEnv, GRID_SIZE, NUM_CHANNELS};
use indicatif::{ProgressIterator, ProgressStyle};
use model::QNet;
use nn::{AdamW, VarBuilder, VarMap};
use rand::Rng;

// Hyperparameters
const TRAIN_STEPS: usize = 20;
const ITERATIONS: usize = 20000;
const TRAIN_ITERS: usize = 1; // Number of passes over the samples collected.
const TRAIN_BATCH_SIZE: usize = 64; // Minibatch size while training models.
const DISCOUNT: f64 = 0.99; // Discount factor applied to rewards.
const Q_EPSILON: f32 = 0.8; // Epsilon for epsilon greedy strategy. This gets annealed over time.
const EVAL_STEPS: usize = 8; // Number of eval runs to average over.
const MAX_EVAL_STEPS: usize = 300; // Max number of steps to take during each eval run.
const Q_LR: f64 = 0.0001; // Learning rate of the q net.
const WARMUP_STEPS: usize = 500; // For the first n number of steps, we will only sample randomly.
const BUFFER_SIZE: usize = 10000; // Number of elements that can be stored in the buffer.
const TARGET_UPDATE: usize = 500; // Number of iterations before updating Q target.
const START_PRIORITY: f32 = 0.4; // Priority to start with, for priority sampling.

fn process_obs(state: Vec<Vec<Vec<bool>>>) -> Result<Tensor> {
    Ok(Tensor::from_vec(
        state
            .iter()
            .flatten()
            .flatten()
            .map(|&b| if b { 1. } else { 0. })
            .collect::<Vec<f32>>(),
        &[NUM_CHANNELS, GRID_SIZE, GRID_SIZE],
        &Device::Cpu,
    )?
    .unsqueeze(0)?)
}

fn mask_to_tensor(mask: &[bool]) -> candle_core::Result<Tensor> {
    Tensor::new(
        mask.iter().map(|&b| b as u8 as f32).collect::<Vec<_>>(),
        &Device::Cpu,
    )?.unsqueeze(0)
}

fn main() -> Result<()> {
    let device = Device::Cpu;

    let mut train_env = GridEnv::new();
    let mut test_env = GridEnv::new();

    // Initialize Q network
    let obs_channels = NUM_CHANNELS;
    let act_space = 4;
    let mut vm = VarMap::new();
    let vs = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
    // let data = std::fs::read("temp/q_net_grid.safetensors")?;
    // let vs = VarBuilder::from_buffered_safetensors(data, DType::F32, &Device::Cpu)?;
    let q_net = QNet::new(vs, obs_channels, act_space)?;
    let mut target_vm = VarMap::new();
    let target_vs = VarBuilder::from_varmap(&target_vm, DType::F32, &device);
    let q_net_target = QNet::new(target_vs, obs_channels, act_space)?;
    let mut q_opt = AdamW::new_lr(vm.all_vars(), Q_LR)?;

    // A replay buffer stores experience collected over all sampling runs
    let mut buffer = ReplayBuffer::new(
        Shape::from_dims(&[obs_channels, GRID_SIZE, GRID_SIZE]),
        BUFFER_SIZE,
    );

    let (obs_, mask_) = train_env.reset();
    let mut obs = process_obs(obs_)?;
    let mut mask = mask_to_tensor(&mask_)?;
    let mut rng = rand::thread_rng();
    for step in (0..ITERATIONS).progress_with_style(ProgressStyle::with_template(
        "[{eta_precise}] {wide_bar} {pos:>7}/{len:7}",
    )?) {
        let percent_done = step as f32 / ITERATIONS as f32;

        // Collect experience
        for _ in 0..TRAIN_STEPS {
            let action = if rng.gen::<f32>() < Q_EPSILON * f32::max(1.0 - percent_done, 0.05)
                || step < WARMUP_STEPS
            {
                rng.gen_range(0..act_space) as u32
            } else {
                let q_vals = (q_net.forward(&obs)?.detach()?
                    * (1. - &mask)? + (&mask * -f64::INFINITY)?)?;
                q_vals.argmax(1)?.squeeze(0)?.to_scalar::<u32>()?
            };
            // train_env.render();
            let (obs_, reward, done, trunc, next_mask) = train_env.step(action);
            let next_obs = process_obs(obs_)?;
            let next_mask = mask_to_tensor(&next_mask)?;
            buffer.insert_step(
                obs,
                next_obs.clone(),
                Tensor::new(&[action], &Device::Cpu)?,
                &[reward],
                &[done],
                next_mask.clone(),
            );
            obs = next_obs;
            mask = next_mask;
            if done || trunc {
                let (obs_, mask_) = train_env.reset();
                obs = process_obs(obs_)?;
                mask = mask_to_tensor(&mask_)?;
            }
        }

        // Train
        let priority = (START_PRIORITY + percent_done * (1. - START_PRIORITY)) as f64;
        if buffer.filled {
            let total_q_loss = train_dqn(
                &q_net,
                &q_net_target,
                &mut q_opt,
                &mut vm,
                &mut buffer,
                &device,
                TRAIN_ITERS,
                TRAIN_BATCH_SIZE,
                DISCOUNT,
                priority,
            )?;

            // Evaluate the network's performance after this training iteration.
            if step % 100 == 0 {
                // with torch.no_grad(){
                let mut reward_total = 0.;
                let (obs_, mut eval_masks) = test_env.reset();
                let mut eval_obs = process_obs(obs_)?;
                for _ in 0..EVAL_STEPS {
                    for _ in 0..MAX_EVAL_STEPS {
                        let masks_tensor = Tensor::new(
                            eval_masks.iter().map(|&b| b as u8 as f32).collect::<Vec<_>>(),
                            &Device::Cpu,
                        )?;
                        let q_vals = (q_net.forward(&eval_obs)?.detach()?.squeeze(0)?
                            * (1. - &masks_tensor)? + (&masks_tensor * -f64::INFINITY)?)?;
                        let action = q_vals.argmax(0)?.to_scalar()?;
                        // pred_reward_total += (
                        //     q_net(eval_obs.unsqueeze(0)).squeeze().max(0).values.item()
                        // );
                        let (obs_, reward, eval_done, eval_trunc, next_eval_masks) =
                            test_env.step(action);
                        eval_obs = process_obs(obs_)?;
                        eval_masks = next_eval_masks;
                        reward_total += reward;
                        if eval_done || eval_trunc {
                            let (obs_, masks_) = test_env.reset();
                            eval_obs = process_obs(obs_)?;
                            eval_masks = masks_;
                            break;
                        }
                    }
                }
                println!(
                    "Eval reward: {}, Total Q Loss: {total_q_loss}",
                    reward_total / EVAL_STEPS as f32
                );
            }
            // }

            // Update Q target
            if (step + 1) % TARGET_UPDATE == 0 {
                for (name, v) in vm.data().lock().unwrap().iter() {
                    target_vm.set_one(name, v.as_tensor().clone())?;
                }
            }

            // Save network
            if (step + 1) % 10 == 0 {
                vm.save("temp/q_net_grid.safetensors")?;
            }
        }
    }
    Ok(())
}
