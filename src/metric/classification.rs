use crate::NeuralNetwork;
use crate::backend::DefaultBackend;
use crate::cost::Cost;
use crate::layer::Layer;
use ndarray::{ArrayBase, Axis, Ix2, OwnedRepr};
use serde::{Deserialize, Serialize};

pub fn accuracy<L, C>(
    network: &mut NeuralNetwork<L, C, DefaultBackend>,
    test_data: &ArrayBase<OwnedRepr<f32>, Ix2>,
    test_labels: &ArrayBase<OwnedRepr<f32>, Ix2>,
) -> f32
where
    L: Layer<DefaultBackend, Input = Ix2, Output = Ix2> + Serialize + for<'de> Deserialize<'de>,
    C: Cost<DefaultBackend>,
{
    let predictions = network.forward(test_data);

    assert_eq!(predictions.shape(), test_labels.shape(), "shape mismatch");

    let argmax = |row: ndarray::ArrayView1<f32>| {
        row.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    };

    let pred_classes = predictions.map_axis(Axis(1), argmax);
    let true_classes = test_labels.map_axis(Axis(1), argmax);

    let correct = pred_classes
        .iter()
        .zip(true_classes.iter())
        .filter(|&(p, t)| p == t)
        .count();

    correct as f32 / predictions.len_of(Axis(0)) as f32
}
