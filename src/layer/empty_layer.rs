use ndarray::{ArrayBase, Dimension, OwnedRepr};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use crate::layer::Layer;

#[derive(Serialize, Deserialize)]
pub struct EmptyLayer<D> {
    _phantom: PhantomData<D>,
}

impl<D> EmptyLayer<D> {
    pub fn new() -> Self {
        EmptyLayer { _phantom: PhantomData }
    }
}

impl<D: Dimension> Layer for EmptyLayer<D> {
    type Input = D;
    type Output = D;

    fn forward(&mut self, input: &ArrayBase<OwnedRepr<f32>, D>) -> ArrayBase<OwnedRepr<f32>, D> {
        input.clone()
    }

    fn backward(&mut self, grad_output: &ArrayBase<OwnedRepr<f32>, D>) -> ArrayBase<OwnedRepr<f32>, D> {
        grad_output.clone()
    }
}
