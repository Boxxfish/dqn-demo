use anyhow::Result;
use candle_core::{backprop::GradStore, Device, IndexOp, Module};
use candle_nn::{Optimizer, VarBuilder, VarMap};

use crate::replay_buffer::ReplayBuffer;

/// Performs the DQN training loop.
#[allow(clippy::too_many_arguments)]
pub fn train_dqn<M: Module, O: Optimizer>(
    q_net: &M,
    q_net_target: &M,
    q_opt: &mut O,
    vm: &mut VarMap,
    buffer: &mut ReplayBuffer,
    device: &Device,
    train_iters: usize,
    train_batch_size: usize,
    discount: f64,
) -> Result<f32> {
    let mut total_q_loss = 0.;
    for v in vm.all_vars() {
        v.to_device(device)?;
    }

    for _ in 0..train_iters {
        let (prev_states, states, actions, rewards, dones) = buffer.sample(train_batch_size)?;

        // Move batch to device if applicable
        let prev_states = prev_states.to_device(device)?;
        let states = states.to_device(device)?;
        let actions = actions.to_device(device)?;
        let rewards = rewards.to_device(device)?;
        let dones = dones.to_device(device)?;

        // Train q network
        // q_opt.zero_grad();
        // with torch.no_grad() {
        let next_actions = q_net.forward(&states)?.argmax(1)?.squeeze(0)?;
        let q_target = (rewards
            + discount
                * (q_net_target
                    .forward(&states)?
                    .detach()?
                    .gather(&next_actions.unsqueeze(1)?, 1)?
                    * (1. - &dones.unsqueeze(1)?))?
                    .squeeze(1)?)?;
        // };
        let diff = (q_net
            .forward(&prev_states)?
            .gather(&actions.unsqueeze(1)?, 1)?
            .squeeze(1)?
            - q_target)?;
        let q_loss = (&diff * &diff)?.mean(0)?;
        q_opt.backward_step(&q_loss)?;
        total_q_loss += q_loss.to_scalar::<f32>()?;
    }

    if device.is_cpu() {
        for v in vm.all_vars() {
            v.to_device(&Device::Cpu)?;
        }
    }
    Ok(total_q_loss)
}
