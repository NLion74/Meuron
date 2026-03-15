use crate::backend::{Backend, DefaultBackend};
use crate::layer::Layer;
use ndarray::Dimension;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

#[derive(Serialize, Deserialize)]
pub struct EmptyLayer<D, B: Backend = DefaultBackend> {
    _phantom: PhantomData<(D, B)>,
}

impl<D, B: Backend> Default for EmptyLayer<D, B> {
    fn default() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<D, B: Backend> EmptyLayer<D, B> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<D: Dimension, B: Backend> Layer<B> for EmptyLayer<D, B> {
    type Input = D;
    type Output = D;

    fn forward(&mut self, input: &B::Tensor<D>) -> B::Tensor<D> {
        input.clone()
    }

    fn backward(&mut self, grad_output: &B::Tensor<D>) -> B::Tensor<D> {
        grad_output.clone()
    }
}
