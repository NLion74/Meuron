use crate::backend::Backend;
use crate::initializer::Initializer;
use ndarray::Dimension;

#[derive(Clone, Copy, Debug, Default)]
pub struct Zeros;

impl<B: Backend> Initializer<B> for Zeros {
    fn init<D: Dimension>(&self, shape: D) -> B::Tensor<D> {
        B::zeros(shape)
    }
}
