use crate::activation::Activation;
use crate::backend::Backend;
use crate::backend::unary_ops;
use ndarray::Dimension;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Tanh;

impl<B: Backend> Activation<B> for Tanh {
    fn activate<D: Dimension>(&self, x: &B::Tensor<D>) -> B::Tensor<D> {
        B::unary(x, unary_ops::TANH)
    }
    fn derivative<D: Dimension>(&self, x: &B::Tensor<D>) -> B::Tensor<D> {
        B::unary(x, unary_ops::TANH_DERIV)
    }
}
