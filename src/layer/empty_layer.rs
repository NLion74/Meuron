use ndarray::Dimension;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use crate::backend::{Backend, NdarrayBackend};
use crate::layer::Layer;

#[derive(Serialize, Deserialize)]
pub struct EmptyLayer<D, B: Backend = NdarrayBackend> {
    _phantom: PhantomData<(D, B)>,
}

impl<D, B: Backend> EmptyLayer<D, B> {
    pub fn new() -> Self {
        EmptyLayer { _phantom: PhantomData }
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
