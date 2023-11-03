use anyhow::{Error, Result};
use candle_core::{DType, IndexOp, Shape, Tensor};
use rand::Rng;

/// A replay buffer for use with off policy algorithms.
/// Stores transitions and generates mini batches.
struct ReplayBuffer {
    pub capacity: usize,
    pub next: usize,
    pub states: Tensor,
    pub next_states: Tensor,
    pub actions: Tensor,
    pub rewards: Tensor,
    pub dones: Tensor,
    pub filled: bool,
}

impl ReplayBuffer {
    fn new(state_shape: Shape, action_masks_shape: Shape, capacity: usize) -> Self {
        let s = move || -> Result<_, candle_core::Error> {
            let k = DType::F32;
            let state_shape = [&[capacity], state_shape.dims()].concat();
            let d = candle_core::Device::Cpu;
            let states = Tensor::zeros(state_shape.clone(), k, &d)?;
            let next_states = Tensor::zeros(state_shape, k, &d)?;
            let actions = Tensor::zeros(&[capacity], DType::I64, &d)?;
            let rewards = Tensor::zeros(&[capacity], k, &d)?;
            // Technically this is the "terminated" flag
            let dones = Tensor::zeros(&[capacity], k, &d)?;
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
    fn insert_step(
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
            let indices = Tensor::new(
                (self.next..(self.next + batch_size))
                    .map(|i| (i % self.capacity) as u32)
                    .collect::<Vec<_>>(),
                &d,
            )?;
            *self.states.i(&indices).as_mut().unwrap() = states;
            *self.next_states.i(&indices).as_mut().unwrap() = next_states;
            *self.actions.i(&indices).as_mut().unwrap() = actions;
            *self.rewards.i(&indices).as_mut().unwrap() = Tensor::new(rewards, &d)?;
            *self.dones.i(&indices).as_mut().unwrap() = Tensor::new(
                dones
                    .iter()
                    .map(|x| if *x { 1. } else { 0. })
                    .collect::<Vec<f32>>()
                    .as_slice(),
                &d,
            )?;
            self.next = (self.next + batch_size) % self.capacity;
            if self.next == 0 {
                self.filled = true;
            }
            Ok(())
        }()
        .unwrap()
    }

    /// Generates minibatches of experience.
    fn sample(&self, batch_size: usize) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor), Error> {
        let mut rng = rand::thread_rng();
        let indices = Tensor::new(
            (0..batch_size)
                .map(|_| rng.gen_range(0..self.capacity) as u32)
                .collect::<Vec<_>>(),
            &candle_core::Device::Cpu,
        )?;
        let rand_states = self.states.index_select(&indices, 0)?;
        let rand_next_states = self.next_states.index_select(&indices, 0)?;
        let rand_actions = self.actions.index_select(&indices, 0)?;
        let rand_rewards = self.rewards.index_select(&indices, 0)?;
        let rand_dones = self.dones.index_select(&indices, 0)?;
        Ok((
            rand_states,
            rand_next_states,
            rand_actions,
            rand_rewards,
            rand_dones,
        ))
    }
}
