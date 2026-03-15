use crate::activation::Activation;
use crate::backend::Backend;
use ndarray::Dimension;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Tanh;

impl<B: Backend> Activation<B> for Tanh {
    fn activate<D: Dimension>(&self, x: &B::Tensor<D>) -> B::Tensor<D> {
        B::mapv(x, |v| v.tanh())
    }
    fn derivative<D: Dimension>(&self, x: &B::Tensor<D>) -> B::Tensor<D> {
        B::mapv(x, |v| {
            let t = v.tanh();
            1.0 - t * t
        })
    }
}
