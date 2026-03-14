use ndarray::{ArrayBase, Dimension, OwnedRepr};
use serde::{Deserialize, Serialize};
use crate::activation::Activation;

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Sigmoid;

impl Activation for Sigmoid {
    fn activate<D: Dimension>(
        &self,
        x: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> ArrayBase<OwnedRepr<f32>, D> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn derivative<D: Dimension>(
        &self,
        x: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> ArrayBase<OwnedRepr<f32>, D> {
        let s = self.activate(x);
        &s * &(1.0 - &s)
    }
}
