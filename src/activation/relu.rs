use ndarray::{ArrayBase, Dimension, OwnedRepr};
use serde::{Deserialize, Serialize};
use crate::activation::Activation;

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct ReLU;

impl Activation for ReLU {
    fn activate<D: Dimension>(
        &self,
        x: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> ArrayBase<OwnedRepr<f32>, D> {
        x.mapv(|v| v.max(0.0))
    }

    fn derivative<D: Dimension>(
        &self,
        x: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> ArrayBase<OwnedRepr<f32>, D> {
        x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
    }
}
