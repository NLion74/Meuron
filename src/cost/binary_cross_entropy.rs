use crate::backend::Backend;
use crate::backend::unary_ops;
use crate::cost::Cost;
use ndarray::Dimension;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct BinaryCrossEntropy;

impl<B: Backend> Cost<B> for BinaryCrossEntropy {
    fn loss<D: Dimension>(&self, predicted: &B::Tensor<D>, target: &B::Tensor<D>) -> f32 {
        let eps = 1e-15_f32;
        let c = B::clamp(predicted, eps, 1.0 - eps);
        let ln_c = B::unary(&c, unary_ops::LN);
        let ln_1mc = B::unary(&B::scalar_sub(1.0, &c), unary_ops::LN);
        let loss = B::add(
            &B::mul(target, &ln_c),
            &B::mul(&B::scalar_sub(1.0, target), &ln_1mc),
        );
        -B::mean(&loss).unwrap_or(0.0)
    }

    fn gradient<D: Dimension>(
        &self,
        predicted: &B::Tensor<D>,
        target: &B::Tensor<D>,
    ) -> B::Tensor<D> {
        let eps = 1e-15_f32;
        let c = B::clamp(predicted, eps, 1.0 - eps);
        let c1mc = B::mul(&c, &B::scalar_sub(1.0, &c));
        B::div(&B::sub(&c, target), &c1mc)
    }
}
