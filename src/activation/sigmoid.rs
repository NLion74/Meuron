use crate::activation::Activation;
use crate::backend::Backend;
use crate::backend::unary_ops;
use ndarray::Dimension;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Sigmoid;

impl<B: Backend> Activation<B> for Sigmoid {
    fn activate<D: Dimension>(&self, x: &B::Tensor<D>) -> B::Tensor<D> {
        B::unary(x, unary_ops::SIGMOID)
    }

    fn derivative<D: Dimension>(&self, x: &B::Tensor<D>) -> B::Tensor<D> {
        let s = B::unary(x, unary_ops::SIGMOID);
        B::mul(&s, &B::scalar_sub(1.0, &s))
    }
}
