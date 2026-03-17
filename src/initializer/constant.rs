use crate::backend::Backend;
use crate::initializer::Initializer;
use ndarray::Dimension;

#[derive(Clone, Copy, Debug)]
pub struct Constant {
    pub value: f32,
}

impl Constant {
    pub fn new(value: f32) -> Self {
        Self { value }
    }
}

impl<B: Backend> Initializer<B> for Constant {
    fn init<D: Dimension>(&self, shape: D) -> B::Tensor<D> {
        B::scale(&B::zeros(shape), self.value)
    }
}
