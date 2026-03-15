use ndarray::Dimension;
use serde::{Deserialize, Serialize};
use crate::backend::Backend;
use crate::cost::Cost;

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct CrossEntropy;

impl<B: Backend> Cost<B> for CrossEntropy {
    fn loss<D: Dimension>(&self, predicted: &B::Tensor<D>, target: &B::Tensor<D>) -> f32 {
        let eps = 1e-15_f32;
        let clipped = B::mapv(predicted, |v| v.clamp(eps, 1.0 - eps));
        -B::mean(&B::mul(target, &B::mapv(&clipped, |v| v.ln()))).unwrap_or(0.0)
    }

    fn gradient<D: Dimension>(
        &self,
        predicted: &B::Tensor<D>,
        target: &B::Tensor<D>,
    ) -> B::Tensor<D> {
        B::sub(predicted, target)
    }
}
