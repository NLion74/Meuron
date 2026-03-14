use crate::activation::Activation;
use crate::backend::Backend;
use ndarray::Dimension;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct ReLU;

impl<B: Backend> Activation<B> for ReLU {
    fn activate<D: Dimension>(&self, x: &B::Tensor<D>) -> B::Tensor<D> {
        B::mapv(x, |v| v.max(0.0))
    }
    fn derivative<D: Dimension>(&self, x: &B::Tensor<D>) -> B::Tensor<D> {
        B::mapv(x, |v| if v > 0.0 { 1.0 } else { 0.0 })
    }
}
