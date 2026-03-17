use crate::backend::Backend;
use crate::optimizer::Optimizer;
use ndarray::Dimension;
use std::any::Any;
use std::collections::HashMap;

pub struct SGDMomentum {
    pub learning_rate: f32,
    pub momentum: f32,
    velocities: HashMap<usize, Box<dyn Any>>,
}

impl SGDMomentum {
    pub fn new(learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            velocities: HashMap::new(),
        }
    }
}

impl<B: Backend> Optimizer<B> for SGDMomentum {
    fn update_param<D: Dimension + 'static>(
        &mut self,
        param: &mut B::Tensor<D>,
        grad: &B::Tensor<D>,
    ) where
        B::Tensor<D>: 'static,
    {
        let key = param as *mut _ as usize;

        let velocity = self
            .velocities
            .entry(key)
            .or_insert_with(|| Box::new(B::scale(param, 0.0)));

        let v = velocity.downcast_mut::<B::Tensor<D>>().unwrap();

        // v = momentum * v + grad
        let new_v = B::add(&B::scale(v, self.momentum), grad);
        B::assign(v, new_v.clone());

        // param -= lr * v
        let updated = B::sub(param, &B::scale(&new_v, self.learning_rate));
        B::assign(param, updated);
    }
}
