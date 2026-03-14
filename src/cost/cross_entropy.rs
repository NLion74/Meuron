use ndarray::{ArrayBase, Dimension, OwnedRepr};
use serde::{Deserialize, Serialize};
use crate::cost::Cost;

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct CrossEntropy;

impl Cost for CrossEntropy {
    fn loss<D: Dimension>(
        &self,
        predicted: &ArrayBase<OwnedRepr<f32>, D>,
        target: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> f32 {
        let epsilon = 1e-15_f32;
        let clipped = predicted.mapv(|v| v.clamp(epsilon, 1.0 - epsilon));
        -(target * &clipped.mapv(|v| v.ln())).mean().unwrap()
    }

    fn gradient<D: Dimension>(
        &self,
        predicted: &ArrayBase<OwnedRepr<f32>, D>,
        target: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> ArrayBase<OwnedRepr<f32>, D> {
        predicted - target
    }
}
