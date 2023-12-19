use anyhow::{Error, Result};
use candle_core::{DType, Device, IndexOp, Shape, Tensor};
use rand::{
    seq::{index::sample_weighted, IteratorRandom, SliceRandom},
    Rng,
};

pub type Samples = (
    Vec<usize>,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
);

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
    pub masks: Vec<Tensor>,
    pub priorities: Vec<f32>,
    pub filled: bool,
    pub max_priority: f32,
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
            let masks = Vec::new();
            let priorities = Vec::new();
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
                max_priority: 0.1,
                priorities,
                masks,
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
        masks: Tensor,
    ) {
        move || -> Result<_> {
            let batch_size = dones.len();
            let indices = (self.next..(self.next + batch_size))
                .map(|i| i % self.capacity)
                .collect::<Vec<_>>();
            for (val_i, &i) in indices.iter().enumerate() {
                if self.filled {
                    self.states[i] = states.i(val_i)?;
                    self.next_states[i] = next_states.i(val_i)?;
                    self.actions[i] = actions.i(val_i)?;
                    self.rewards[i] = rewards[val_i];
                    self.dones[i] = dones[val_i];
                    self.masks[i] = masks.i(val_i)?;
                    self.priorities[i] = self.max_priority;
                } else {
                    self.states.push(states.i(val_i)?);
                    self.next_states.push(next_states.i(val_i)?);
                    self.actions.push(actions.i(val_i)?);
                    self.rewards.push(rewards[val_i]);
                    self.dones.push(dones[val_i]);
                    self.masks.push(masks.i(val_i)?);
                    self.priorities.push(self.max_priority);
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
    pub fn sample(&self, batch_size: usize) -> Result<Samples, Error> {
        let mut rng = rand::thread_rng();
        let sum_priorities: f32 = self.priorities.iter().sum();
        let probs: Vec<_> = self
            .priorities
            .iter()
            .map(|p| 1.
                // *p / sum_priorities
            )
            .collect();
        let indices: Vec<_> = sample_weighted(&mut rng, self.capacity, |i| probs[i], batch_size)
            .unwrap()
            .into_vec();
        let mut rand_states_vec = Vec::new();
        let mut rand_next_states_vec = Vec::new();
        let mut rand_actions_vec = Vec::new();
        let mut rand_rewards_vec = Vec::new();
        let mut rand_dones_vec = Vec::new();
        let mut rand_masks_vec = Vec::new();
        for &i in &indices {
            rand_states_vec.push(&self.states[i]);
            rand_next_states_vec.push(&self.next_states[i]);
            rand_actions_vec.push(&self.actions[i]);
            rand_rewards_vec.push(self.rewards[i]);
            rand_dones_vec.push(if self.dones[i] { 1_f32 } else { 0. });
            rand_masks_vec.push(&self.masks[i]);
        }
        let probs = Tensor::new(probs, &Device::Cpu)?.gather(
            &Tensor::new(
                indices
                    .iter()
                    .copied()
                    .map(|i| i as u32)
                    .collect::<Vec<_>>(),
                &Device::Cpu,
            )?,
            0,
        )?;
        Ok((
            indices,
            probs,
            Tensor::stack(&rand_states_vec, 0)?,
            Tensor::stack(&rand_next_states_vec, 0)?,
            Tensor::stack(&rand_actions_vec, 0)?,
            Tensor::new(rand_rewards_vec, &Device::Cpu)?,
            Tensor::new(rand_dones_vec, &Device::Cpu)?,
            Tensor::stack(&rand_masks_vec, 0)?,
        ))
    }

    /// Updates transition TD errors.
    pub fn update_errors(&mut self, indices: &[usize], errors: &[f32]) {
        for (&i, &error) in indices.iter().zip(errors) {
            let priority = error.abs() + 0.00001;
            self.priorities[i] = priority;
            self.max_priority = self.max_priority.max(priority);
        }
    }
}
