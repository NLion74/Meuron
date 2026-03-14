use ndarray::{ArrayBase, Dimension, OwnedRepr};
use crate::optimizer::Optimizer;

pub struct SGD {
    pub learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        SGD { learning_rate }
    }
}

impl Optimizer for SGD {
    fn update_param<D: Dimension>(
        &mut self,
        param: &mut ArrayBase<OwnedRepr<f32>, D>,
        grad: &ArrayBase<OwnedRepr<f32>, D>,
    ) {
        *param -= &(self.learning_rate * grad);
    }
}