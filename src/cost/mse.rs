use ndarray::{ArrayBase, Dimension, OwnedRepr};
use serde::{Deserialize, Serialize};
use crate::cost::Cost;

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct MSE;

impl Cost for MSE {
    fn loss<D: Dimension>(
        &self,
        predicted: &ArrayBase<OwnedRepr<f32>, D>,
        target: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> f32 {
        let diff = predicted - target;
        (&diff * &diff).mean().unwrap()
    }

    fn gradient<D: Dimension>(
        &self,
        predicted: &ArrayBase<OwnedRepr<f32>, D>,
        target: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> ArrayBase<OwnedRepr<f32>, D> {
        predicted - target
    }
}
