use crate::NeuralNetwork;
use crate::backend::Backend;
use crate::cost::Cost;
use crate::layer::Layer;
use ndarray::{Array, Axis, Dimension};
use serde::{Deserialize, Serialize};

pub trait Forward<B: Backend> {
    type Input: Dimension;
    type Output: Dimension;
    fn run(&mut self, x: &B::Tensor<Self::Input>) -> B::Tensor<Self::Output>;
}

impl<L, C, B> Forward<B> for NeuralNetwork<L, C, B>
where
    B: Backend,
    L: Layer<B> + Serialize + for<'de> Deserialize<'de>,
    C: Cost<B>,
{
    type Input = L::Input;
    type Output = L::Output;

    fn run(&mut self, x: &B::Tensor<L::Input>) -> B::Tensor<L::Output> {
        self.forward(x)
    }
}

pub fn accuracy<B, N, I, T>(network: &mut N, test_data: I, test_labels: T) -> f32
where
    B: Backend,
    N: Forward<B>,
    I: Into<B::Tensor<N::Input>>,
    T: Into<Array<f32, N::Output>>,
{
    let test_data = test_data.into();
    let test_labels = test_labels.into();

    let predictions = network.run(&test_data);
    let pred_arr = B::to_array(&predictions).into_dyn();
    let label_arr = test_labels.into_dyn();

    let n = pred_arr.shape()[0];
    assert_eq!(n, label_arr.shape()[0], "shape mismatch");

    let argmax = |row: ndarray::ArrayView1<f32>| -> f32 {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as f32)
            .unwrap_or(0.0)
    };

    let last = Axis(pred_arr.ndim() - 1);
    let pred_classes = pred_arr.map_axis(last, argmax);
    let label_classes = label_arr.map_axis(last, argmax);

    pred_classes
        .iter()
        .zip(label_classes.iter())
        .filter(|(p, l)| (*p - *l).abs() < 0.5)
        .count() as f32
        / n as f32
}
