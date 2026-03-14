use ndarray::{ArrayBase, Axis, Ix2, OwnedRepr};
use crate::cost::Cost;
use crate::layer::Layer;
use crate::NeuralNetwork;

pub fn accuracy<L, C>(
    network: &mut NeuralNetwork<L, C>,
    test_data: &ArrayBase<OwnedRepr<f32>, Ix2>,
    test_labels: &ArrayBase<OwnedRepr<f32>, Ix2>,
) -> f32
where
    L: Layer<Input = Ix2, Output = Ix2> + serde::Serialize + for<'de> serde::Deserialize<'de>,
    C: Cost,
{
    let predictions = network.forward(test_data);

    assert_eq!(
        predictions.shape(),
        test_labels.shape(),
        "Predictions and labels must have the same shape"
    );

    let predicted_classes = predictions.map_axis(Axis(1), |row| {
        row.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap()
    });

    let target_classes = test_labels.map_axis(Axis(1), |row| {
        row.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap()
    });

    let correct = predicted_classes
        .iter()
        .zip(target_classes.iter())
        .filter(|&(pred, target)| pred == target)
        .count();

    correct as f32 / predictions.len_of(Axis(0)) as f32
}
