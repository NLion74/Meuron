use crate::backend::Backend;
use crate::backend::unary_ops;
use crate::cost::Cost;
use ndarray::Dimension;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct CrossEntropy;

impl<B: Backend> Cost<B> for CrossEntropy {
    fn loss<D: Dimension>(&self, predicted: &B::Tensor<D>, target: &B::Tensor<D>) -> f32 {
        let eps = 1e-15_f32;
        let clipped = B::clamp(predicted, eps, 1.0 - eps);
        let ln_clip = B::unary(&clipped, unary_ops::LN);
        -B::mean(&B::mul(target, &ln_clip)).unwrap_or(0.0)
    }

    fn gradient<D: Dimension>(
        &self,
        predicted: &B::Tensor<D>,
        target: &B::Tensor<D>,
    ) -> B::Tensor<D> {
        let eps = 1e-15_f32;
        let clipped = B::clamp(predicted, eps, 1.0 - eps);
        B::div(&B::scale(target, -1.0), &clipped)
    }
}
