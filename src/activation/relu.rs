use crate::activation::Activation;
use crate::backend::Backend;
use crate::backend::unary_ops;
use ndarray::Dimension;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct ReLU;

impl<B: Backend> Activation<B> for ReLU {
    fn activate<D: Dimension>(&self, x: &B::Tensor<D>) -> B::Tensor<D> {
        B::unary(x, unary_ops::RELU)
    }
    fn derivative<D: Dimension>(&self, x: &B::Tensor<D>) -> B::Tensor<D> {
        B::unary(x, unary_ops::RELU_DERIV)
    }
}
