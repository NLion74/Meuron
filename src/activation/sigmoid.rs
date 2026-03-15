use crate::activation::Activation;
use crate::backend::Backend;
use ndarray::Dimension;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Sigmoid;

impl<B: Backend> Activation<B> for Sigmoid {
    fn activate<D: Dimension>(&self, x: &B::Tensor<D>) -> B::Tensor<D> {
        B::mapv(x, |v| 1.0 / (1.0 + (-v).exp()))
    }

    fn derivative<D: Dimension>(&self, x: &B::Tensor<D>) -> B::Tensor<D> {
        let s = B::mapv(x, |v| 1.0 / (1.0 + (-v).exp()));
        B::mul(&s, &B::scalar_sub(1.0, &s))
    }
}
