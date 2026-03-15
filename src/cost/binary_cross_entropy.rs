use crate::backend::Backend;
use crate::cost::Cost;
use ndarray::Dimension;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct BinaryCrossEntropy;

impl<B: Backend> Cost<B> for BinaryCrossEntropy {
    fn loss<D: Dimension>(&self, predicted: &B::Tensor<D>, target: &B::Tensor<D>) -> f32 {
        let eps = 1e-15_f32;
        let c = B::mapv(predicted, |v| v.clamp(eps, 1.0 - eps));
        let loss = B::add(
            &B::mul(target, &B::mapv(&c, |v| v.ln())),
            &B::mul(
                &B::scalar_sub(1.0, target),
                &B::mapv(&c, |v| (1.0 - v).ln()),
            ),
        );
        -B::mean(&loss).unwrap_or(0.0)
    }

    fn gradient<D: Dimension>(
        &self,
        predicted: &B::Tensor<D>,
        target: &B::Tensor<D>,
    ) -> B::Tensor<D> {
        let eps = 1e-15_f32;
        let c = B::mapv(predicted, |v| v.clamp(eps, 1.0 - eps));
        B::div(&B::sub(&c, target), &B::mul(&c, &B::scalar_sub(1.0, &c)))
    }
}
