pub mod mse; pub mod cross_entropy; pub mod binary_cross_entropy;
pub use mse::MSE; pub use cross_entropy::CrossEntropy;
pub use binary_cross_entropy::BinaryCrossEntropy;
use crate::backend::Backend;
use ndarray::Dimension;

pub trait Cost<B: Backend> {
    fn loss<D: Dimension>(&self, predicted: &B::Tensor<D>, target: &B::Tensor<D>) -> f32;
    fn gradient<D: Dimension>(&self, predicted: &B::Tensor<D>, target: &B::Tensor<D>) -> B::Tensor<D>;
}
