pub mod relu;
pub mod sigmoid;
pub mod softmax;
pub mod tanh;

pub use relu::ReLU;
pub use sigmoid::Sigmoid;
pub use softmax::Softmax;
pub use tanh::Tanh;

use crate::backend::Backend;
use ndarray::Dimension;

pub trait Activation<B: Backend>: Clone {
    fn activate<D: Dimension>(&self, x: &B::Tensor<D>) -> B::Tensor<D>;
    fn derivative<D: Dimension>(&self, x: &B::Tensor<D>) -> B::Tensor<D>;

    fn vjp<D: Dimension>(&self, z: &B::Tensor<D>, grad: &B::Tensor<D>) -> B::Tensor<D> {
        B::mul(grad, &self.derivative(z))
    }
}
