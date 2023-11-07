use anyhow::Result;
use candle_core::{Module, Tensor, D};
use nn::{
    var_builder::{Backend, VarBuilderArgs},
    VarBuilder,
};

use crate::env::GRID_SIZE;
use candle_nn as nn;

/// A skip connection.
struct Skip {
    module: nn::Sequential,
}

impl Module for Skip {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xformed = self.module.forward(xs)?;
        xs + xformed
    }
}

/// Creates a skip connection.
fn skip(features: usize, vs: VarBuilder) -> candle_core::Result<Skip> {
    let module = nn::seq()
        .add(nn::Activation::Relu)
        .add(nn::conv2d(
            features,
            features,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vs.pp("l1"),
        )?)
        .add(nn::Activation::Relu)
        .add(nn::conv2d(
            features,
            features,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vs.pp("l2"),
        )?);
    Ok(Skip { module })
}

pub struct QNet {
    net: nn::sequential::Sequential,
    rep_net: nn::sequential::Sequential,
    advantage: nn::sequential::Sequential,
    value: nn::sequential::Sequential,
    action_count: usize,
    out_net: nn::Conv2d,
}

impl QNet {
    pub fn new(vs: VarBuilder, in_channels: usize, action_count: usize) -> Result<Self> {
        let conv_conf = nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_features = 32;
        let net = nn::seq().add(nn::conv2d(
            in_channels,
            conv_features,
            3,
            conv_conf,
            vs.pp("conv1"),
        )?);
        let rep_net = nn::seq()
            .add(skip(conv_features, vs.pp("skip1"))?);
        let out_net = nn::conv2d(conv_features, 32, 3, Default::default(), vs.pp("conv_out"))?;
        let advantage = nn::seq()
            .add(nn::linear(32, 32, vs.pp("a_ln1"))?)
            .add(nn::Activation::Relu)
            .add(nn::linear(32, action_count, vs.pp("a_ln2"))?);
        let value = nn::seq()
            .add(nn::linear(32, 32, vs.pp("v_ln1"))?)
            .add(nn::Activation::Relu)
            .add(nn::linear(32, 1, vs.pp("v_ln2"))?);
        Ok(Self {
            net,
            rep_net,
            advantage,
            value,
            action_count,
            out_net,
        })
    }
}

impl Module for QNet {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut xs = self.net.forward(xs)?;
        for _ in 0..1 {
            xs = self.rep_net.forward(&xs)?;
        }
        let xs = self
            .out_net
            .forward(&xs)?
            .max_pool2d(GRID_SIZE - 2)?
            .squeeze(D::Minus1)?
            .squeeze(D::Minus1)?;
        let advantage = self.advantage.forward(&xs)?;
        let value = self.value.forward(&xs)?;
        &value.repeat(&[1, self.action_count])? + &advantage
            - &advantage.mean_keepdim(1)?.repeat(&[1, self.action_count])?
    }
}
