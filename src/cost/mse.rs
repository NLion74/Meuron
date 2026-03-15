use crate::backend::Backend;
use crate::cost::Cost;
use ndarray::Dimension;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct MSE;

impl<B: Backend> Cost<B> for MSE {
    fn loss<D: Dimension>(&self, predicted: &B::Tensor<D>, target: &B::Tensor<D>) -> f32 {
        let diff = B::sub(predicted, target);
        B::mean(&B::mul(&diff, &diff)).unwrap_or(0.0)
    }
    fn gradient<D: Dimension>(
        &self,
        predicted: &B::Tensor<D>,
        target: &B::Tensor<D>,
    ) -> B::Tensor<D> {
        B::sub(predicted, target)
    }
}
