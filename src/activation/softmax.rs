use ndarray::{Dimension};
use serde::{Deserialize, Serialize};
use crate::activation::Activation;
use crate::backend::Backend;

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Softmax;

impl<B: Backend> Activation<B> for Softmax {
    fn activate<D: Dimension>(&self, x: &B::Tensor<D>) -> B::Tensor<D> {
        B::softmax(x)
    }
    fn derivative<D: Dimension>(&self, x: &B::Tensor<D>) -> B::Tensor<D> {
        let s = B::softmax(x);
        B::mul(&s, &B::scalar_sub(1.0, &s))
    }
    fn vjp<D: Dimension>(&self, z: &B::Tensor<D>, grad: &B::Tensor<D>) -> B::Tensor<D> {
        B::softmax_vjp(z, grad)
    }
}