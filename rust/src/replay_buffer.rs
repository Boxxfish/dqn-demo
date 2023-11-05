use anyhow::{Error, Result};
use candle_core::{DType, Device, IndexOp, Shape, Tensor};
use rand::Rng;

/// A replay buffer for use with off policy algorithms.
/// Stores transitions and generates mini batches.
pub struct ReplayBuffer {
    pub capacity: usize,
    pub next: usize,
    pub states: Vec<Tensor>,
    pub next_states: Vec<Tensor>,
    pub actions: Vec<Tensor>,
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
    pub filled: bool,
}

impl ReplayBuffer {
    pub fn new(state_shape: Shape, capacity: usize) -> Self {
        let s = move || -> Result<_, candle_core::Error> {
            // let k = DType::F32;
            // let state_shape = [&[capacity], state_shape.dims()].concat();
            // let d = candle_core::Device::Cpu;
            let states = Vec::new();
            let next_states = Vec::new();
            let actions = Vec::new();
            let rewards = Vec::new();
            // Technically this is the "terminated" flag
            let dones = Vec::new();
            let filled = false;
            let next = 0;
            Ok(Self {
                capacity,
                next,
                states,
                next_states,
                actions,
                rewards,
                dones,
                filled,
            })
        }()
        .unwrap();
        s
    }

    /// Inserts a transition from each environment into the buffer. Make sure
    /// more data than steps aren't inserted.
    pub fn insert_step(
        &mut self,
        states: Tensor,
        next_states: Tensor,
        actions: Tensor,
        rewards: &[f32],
        dones: &[bool],
    ) {
        move || -> Result<_> {
            let batch_size = dones.len();
            let d = candle_core::Device::Cpu;
            let indices = ((self.next..(self.next + batch_size))
                .map(|i| (i % self.capacity) as usize)
                .collect::<Vec<_>>());
            for (val_i, &i) in indices.iter().enumerate() {
                if self.filled {
                    self.states[i] = states.i(val_i)?;
                    self.next_states[i] = next_states.i(val_i)?;
                    self.actions[i] = actions.i(val_i)?;
                    self.rewards[i] = rewards[val_i];
                    self.dones[i] = dones[val_i];
                }
                else {
                    self.states.push(states.i(val_i)?);
                    self.next_states.push(next_states.i(val_i)?);
                    self.actions.push(actions.i(val_i)?);
                    self.rewards.push(rewards[val_i]);
                    self.dones.push(dones[val_i]);
                }
            }
            self.next = (self.next + batch_size) % self.capacity;
            if self.next == 0 {
                self.filled = true;
            }
            Ok(())
        }()
        .unwrap()
    }

    /// Generates minibatches of experience.
    pub fn sample(
        &self,
        batch_size: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor), Error> {
        let mut rng = rand::thread_rng();
        let indices = (0..batch_size)
            .map(|_| rng.gen_range(0..self.capacity) as usize)
            .collect::<Vec<_>>();
        let mut rand_states_vec = Vec::new();
        let mut rand_next_states_vec = Vec::new();
        let mut rand_actions_vec = Vec::new();
        let mut rand_rewards_vec = Vec::new();
        let mut rand_dones_vec = Vec::new();
        for i in indices {
            rand_states_vec.push(&self.states[i]);
            rand_next_states_vec.push(&self.next_states[i]);
            rand_actions_vec.push(&self.actions[i]);
            rand_rewards_vec.push(self.rewards[i]);
            rand_dones_vec.push(if self.dones[i] { 1. } else { 0. });
        }
        Ok((
            Tensor::stack(&rand_states_vec, 0)?,
            Tensor::stack(&rand_next_states_vec, 0)?,
            Tensor::stack(&rand_actions_vec, 0)?,
            Tensor::new(rand_rewards_vec, &Device::Cpu)?,
            Tensor::new(rand_dones_vec, &Device::Cpu)?,
        ))
    }
}
