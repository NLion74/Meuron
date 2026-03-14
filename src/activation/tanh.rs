use ndarray::{ArrayBase, Dimension, OwnedRepr};
use serde::{Deserialize, Serialize};
use crate::activation::Activation;

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Tanh;

impl Activation for Tanh {
    fn activate<D: Dimension>(
        &self,
        x: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> ArrayBase<OwnedRepr<f32>, D> {
        x.mapv(|v| v.tanh())
    }

    fn derivative<D: Dimension>(
        &self,
        x: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> ArrayBase<OwnedRepr<f32>, D> {
        x.mapv(|v| {
            let t = v.tanh();
            1.0 - t * t
        })
    }
}
