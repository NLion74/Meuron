pub mod sgd;

pub use sgd::SGD;

use ndarray::{ArrayBase, Dimension, OwnedRepr};

pub trait Optimizer {
    fn update_param<D: Dimension>(
        &mut self,
        param: &mut ArrayBase<OwnedRepr<f32>, D>,
        grad: &ArrayBase<OwnedRepr<f32>, D>,
    );
}

// Allow usage of f32 scalars directly as learning rates, by treating them as a simple SGD optimizer.
impl<O: Optimizer> Optimizer for &mut O {
    fn update_param<D: Dimension>(
        &mut self,
        param: &mut ArrayBase<OwnedRepr<f32>, D>,
        grad: &ArrayBase<OwnedRepr<f32>, D>,
    ) {
        (**self).update_param(param, grad);
    }
}