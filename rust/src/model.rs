use anyhow::Result;
use candle_core::{Module, Tensor, D};
use nn::VarBuilder;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::env::GRID_SIZE;
use candle_nn as nn;

pub struct QNet {
    net: nn::sequential::Sequential,
    advantage: nn::sequential::Sequential,
    value: nn::sequential::Sequential,
    action_count: usize,
}

impl QNet {
    pub fn new(vs: VarBuilder, in_channels: usize, action_count: usize) -> Result<Self> {
        let net = nn::seq()
            .add(nn::conv2d(
                in_channels,
                8,
                3,
                nn::Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                },
                vs.pp("conv1"),
            )?)
            .add(nn::Activation::Relu)
            .add(nn::conv2d(
                8,
                32,
                3,
                nn::Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                },
                vs.pp("conv2"),
            )?)
            .add(nn::Activation::Relu);
        let advantage = nn::seq()
            .add(nn::linear(32, 64, vs.pp("a_ln1"))?)
            .add(nn::Activation::Relu)
            .add(nn::linear(64, action_count, vs.pp("a_ln2"))?);
        let value = nn::seq()
            .add(nn::linear(32, 64, vs.pp("v_ln1"))?)
            .add(nn::Activation::Relu)
            .add(nn::linear(64, 1, vs.pp("v_ln2"))?);
        Ok(Self {
            net,
            advantage,
            value,
            action_count,
        })
    }
}

impl Module for QNet {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self
            .net
            .forward(xs)?
            .max_pool2d(GRID_SIZE)?
            .squeeze(D::Minus1)?
            .squeeze(D::Minus1)?;
        let advantage = self.advantage.forward(&xs)?;
        let value = self.value.forward(&xs)?;
        &value.repeat(&[1, self.action_count])? + &advantage
            - &advantage.mean_keepdim(1)?.repeat(&[1, self.action_count])?
    }
}
