pub mod activation;
pub mod backend;
pub mod cost;
pub mod layer;
pub mod metric;
pub mod optimizer;
pub mod initializer;
pub mod serialization;
pub mod train;

pub use activation::{ReLU, Sigmoid, Softmax, Tanh};
pub use backend::DefaultBackend;
pub use cost::{BinaryCrossEntropy, CrossEntropy, MSE};
pub use layer::DenseLayer;
pub use metric::classification::accuracy;
pub use optimizer::{SGD, SGDMomentum};
pub use train::{PrintCallback, TrainCallback, TrainOptions};
pub use initializer::{Constant, HeNormal, Initializer, XavierUniform, Zeros};

use crate::backend::Backend;
use crate::layer::Layer;
use ndarray::{Array, Dimension};
use std::marker::PhantomData;

pub struct NeuralNetwork<L, C, B: Backend = DefaultBackend>
where
    L: Layer<B>,
{
    pub layers: L,
    pub cost: C,
    pub(crate) _backend: PhantomData<B>,
}

impl<L, C, B> NeuralNetwork<L, C, B>
where
    B: Backend,
    L: Layer<B>,
{
    pub fn new(layers: L, cost: C) -> Self {
        NeuralNetwork {
            layers,
            cost,
            _backend: PhantomData,
        }
    }

    pub fn predict<D>(&mut self, input: Array<f32, D>) -> Array<f32, L::Output>
    where
        D: Dimension,
        Array<f32, D>: Into<B::Tensor<L::Input>>,
    {
        B::to_array(&self.layers.forward(&input.into()))
    }

    #[inline]
    pub fn forward<I>(&mut self, input: I) -> B::Tensor<L::Output>
    where
        I: Into<B::Tensor<L::Input>>,
    {
        self.layers.forward(&input.into())
    }

    #[inline]
    pub fn backward<G>(&mut self, grad_output: G) -> B::Tensor<L::Input>
    where
        G: Into<B::Tensor<L::Output>>,
    {
        self.layers.backward(&grad_output.into())
    }
}

#[macro_export]
macro_rules! NetworkType {
    ($l:ty) => { $l };
    ($l1:ty, $l2:ty) => { $crate::layer::Sequential<$l1, $l2> };
    ($l1:ty, $l2:ty, $($rest:ty),+) => {
        $crate::layer::Sequential<$l1, $crate::NetworkType!($l2, $($rest),+)>
    };
}
