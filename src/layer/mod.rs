pub mod dense_layer;
pub mod empty_layer;

pub use dense_layer::DenseLayer;
pub use empty_layer::EmptyLayer;

use crate::optimizer::Optimizer;
use ndarray::{ArrayBase, Dimension, OwnedRepr};
use serde::{Deserialize, Serialize};

pub trait Layer {
    type Input: Dimension;
    type Output: Dimension;

    fn forward(
        &mut self,
        input: &ArrayBase<OwnedRepr<f32>, Self::Input>,
    ) -> ArrayBase<OwnedRepr<f32>, Self::Output>;

    fn backward(
        &mut self,
        grad_output: &ArrayBase<OwnedRepr<f32>, Self::Output>,
    ) -> ArrayBase<OwnedRepr<f32>, Self::Input>;

    fn update<O: Optimizer>(&mut self, _optimizer: &mut O) {}
}

#[derive(Serialize, Deserialize)]
pub struct Sequential<L1, L2> {
    pub layer1: L1,
    pub layer2: L2,
}

impl<L1, L2, D1, D2, D3> Layer for Sequential<L1, L2>
where
    L1: Layer<Input = D1, Output = D2>,
    L2: Layer<Input = D2, Output = D3>,
    D1: Dimension,
    D2: Dimension,
    D3: Dimension,
{
    type Input = D1;
    type Output = D3;

    fn forward(
        &mut self,
        input: &ArrayBase<OwnedRepr<f32>, D1>,
    ) -> ArrayBase<OwnedRepr<f32>, D3> {
        let out = self.layer1.forward(input);
        self.layer2.forward(&out)
    }

    fn backward(
        &mut self,
        grad_output: &ArrayBase<OwnedRepr<f32>, D3>,
    ) -> ArrayBase<OwnedRepr<f32>, D1> {
        let grad = self.layer2.backward(grad_output);
        self.layer1.backward(&grad)
    }

    fn update<O: Optimizer>(&mut self, optimizer: &mut O) {
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
    ($layer1:expr, $layer2:expr) => {
        $crate::layer::seq($layer1, $layer2)
    };
    ($layer1:expr, $layer2:expr, $($rest:expr),+) => {
        $crate::layer::seq($layer1, $crate::Layers!($layer2, $($rest),+))
    };
}
