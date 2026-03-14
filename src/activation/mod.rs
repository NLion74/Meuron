pub mod relu;
pub mod sigmoid;
pub mod softmax;
pub mod tanh;

pub use relu::ReLU;
pub use sigmoid::Sigmoid;
pub use softmax::Softmax;
pub use tanh::Tanh;

use ndarray::{ArrayBase, Dimension, OwnedRepr};

pub trait Activation: Clone {
    fn activate<D: Dimension>(
        &self,
        x: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> ArrayBase<OwnedRepr<f32>, D>;

    fn derivative<D: Dimension>(
        &self,
        x: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> ArrayBase<OwnedRepr<f32>, D>;

    fn vjp<D: Dimension>(
        &self,
        z: &ArrayBase<OwnedRepr<f32>, D>,
        grad_output: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> ArrayBase<OwnedRepr<f32>, D> {
        grad_output * &self.derivative(z)
    }
}
