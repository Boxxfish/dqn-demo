mod dqn;
mod env;
mod model;
mod replay_buffer;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use model::QNet;
use safetensors::SafeTensors;
use wasm_bindgen::prelude::*;

use crate::env::{GRID_SIZE, NUM_CHANNELS};

#[wasm_bindgen]
pub struct DQN {
    net: QNet,
}

#[wasm_bindgen]
impl DQN {
    pub fn load(data: &[u8]) -> Self {
        let vs =
            VarBuilder::from_buffered_safetensors(data.to_vec(), DType::F32, &Device::Cpu).unwrap();
        let net = QNet::new(vs, NUM_CHANNELS, 4).unwrap();
        Self { net }
    }

    pub fn eval_state(&self, state: &[u8]) -> Vec<f32> {
        move || -> Result<_> {
            let state = Tensor::new(state, &Device::Cpu)?
                .reshape(&[NUM_CHANNELS, GRID_SIZE, GRID_SIZE])?
                .to_dtype(DType::F32)?
                .unsqueeze(0)?;
            let q_vals = self.net.forward(&state)?.squeeze(0)?;
            q_vals.to_vec1()
        }()
        .unwrap()
    }
}
