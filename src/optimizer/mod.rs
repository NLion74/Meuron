pub mod sgd;
pub use sgd::SGD;

use crate::backend::Backend;
use ndarray::Dimension;

pub trait Optimizer<B: Backend> {
    fn update_param<D: Dimension>(&mut self, param: &mut B::Tensor<D>, grad: &B::Tensor<D>);
}

// Allow usage of f32 scalars directly as learning rates, by treating them as a simple SGD optimizer.
impl<B: Backend, O: Optimizer<B>> Optimizer<B> for &mut O {
    fn update_param<D: Dimension>(&mut self, param: &mut B::Tensor<D>, grad: &B::Tensor<D>) {
        (**self).update_param(param, grad);
    }
}
