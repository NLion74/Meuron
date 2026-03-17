pub mod sgd;
pub mod sgd_momentum;

pub use sgd::SGD;
pub use sgd_momentum::SGDMomentum;

use crate::backend::Backend;
use ndarray::Dimension;

pub trait Optimizer<B: Backend> {
    fn update_param<D: Dimension + 'static>(
        &mut self,
        param: &mut B::Tensor<D>,
        grad: &B::Tensor<D>,
    ) where
        B::Tensor<D>: 'static;
}

impl<B: Backend, O: Optimizer<B>> Optimizer<B> for &mut O {
    fn update_param<D: Dimension + 'static>(
        &mut self,
        param: &mut B::Tensor<D>,
        grad: &B::Tensor<D>,
    ) where
        B::Tensor<D>: 'static,
    {
        (**self).update_param(param, grad);
    }
}
