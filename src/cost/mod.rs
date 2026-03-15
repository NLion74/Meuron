pub mod binary_cross_entropy;
pub mod cross_entropy;
pub mod mse;
use crate::backend::Backend;
pub use binary_cross_entropy::BinaryCrossEntropy;
pub use cross_entropy::CrossEntropy;
pub use mse::MSE;
use ndarray::Dimension;

pub trait Cost<B: Backend> {
    fn loss<D: Dimension>(&self, predicted: &B::Tensor<D>, target: &B::Tensor<D>) -> f32;
    fn gradient<D: Dimension>(
        &self,
        predicted: &B::Tensor<D>,
        target: &B::Tensor<D>,
    ) -> B::Tensor<D>;
}
