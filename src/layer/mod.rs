pub mod dense_layer;
pub mod empty_layer;

pub use dense_layer::DenseLayer;
pub use empty_layer::EmptyLayer;

use crate::backend::Backend;
use crate::optimizer::Optimizer;
use ndarray::Dimension;
use serde::{Deserialize, Serialize};

pub trait Layer<B: Backend> {
    type Input: Dimension;
    type Output: Dimension;

    fn forward(&mut self, input: &B::Tensor<Self::Input>) -> B::Tensor<Self::Output>;
    fn backward(&mut self, grad_output: &B::Tensor<Self::Output>) -> B::Tensor<Self::Input>;
    fn update<O: Optimizer<B>>(&mut self, _optimizer: &mut O) {}
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "L1: Serialize, L2: Serialize",
    deserialize = "L1: Deserialize<'de>, L2: Deserialize<'de>"
))]
pub struct Sequential<L1, L2> {
    pub layer1: L1,
    pub layer2: L2,
}

impl<L1, L2, B, D1, D2, D3> Layer<B> for Sequential<L1, L2>
where
    B: Backend,
    L1: Layer<B, Input = D1, Output = D2>,
    L2: Layer<B, Input = D2, Output = D3>,
    D1: Dimension,
    D2: Dimension,
    D3: Dimension,
{
    type Input = D1;
    type Output = D3;

    fn forward(&mut self, input: &B::Tensor<D1>) -> B::Tensor<D3> {
        let out = self.layer1.forward(input);
        self.layer2.forward(&out)
    }

    fn backward(&mut self, grad_output: &B::Tensor<D3>) -> B::Tensor<D1> {
        let grad = self.layer2.backward(grad_output);
        self.layer1.backward(&grad)
    }

    fn update<O: Optimizer<B>>(&mut self, optimizer: &mut O) {
        self.layer1.update(optimizer);
        self.layer2.update(optimizer);
    }
}

pub fn seq<L1, L2>(layer1: L1, layer2: L2) -> Sequential<L1, L2> {
    Sequential { layer1, layer2 }
}

#[macro_export]
macro_rules! Layers {
    ($layer:expr) => { $layer };
    ($layer1:expr, $layer2:expr) => { $crate::layer::seq($layer1, $layer2) };
    ($layer1:expr, $layer2:expr, $($rest:expr),+) => {
        $crate::layer::seq($layer1, $crate::Layers!($layer2, $($rest),+))
    };
}
