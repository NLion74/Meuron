use crate::backend::Backend;
use crate::optimizer::Optimizer;
use ndarray::Dimension;

pub struct SGD {
    pub learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        SGD { learning_rate }
    }
}

impl<B: Backend> Optimizer<B> for SGD {
    fn update_param<D: Dimension>(&mut self, param: &mut B::Tensor<D>, grad: &B::Tensor<D>) {
        let updated = B::sub(param, &B::scale(grad, self.learning_rate));
        B::assign(param, updated);
    }
}
