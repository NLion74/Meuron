pub mod layer;
pub mod activation;
pub mod cost;

use crate::layer::Layer;
use crate::cost::Cost;
use ndarray::{ArrayBase, OwnedRepr, Dimension, RemoveAxis, Axis};

pub struct NeuralNetwork<L, C>
where
    L: Layer,
{
    layers: Vec<L>,
    cost: C,
}

impl<L, C, D> NeuralNetwork<L, C>
where
    L: Layer<Input = D, Output = D>,
    C: Cost<D>,
    D: Dimension + RemoveAxis,
{
    pub fn new(layers: Vec<L>, cost: C) -> Self {
        NeuralNetwork { layers, cost }
    }

    pub fn forward(&mut self, input: &ArrayBase<OwnedRepr<f32>, D>) -> ArrayBase<OwnedRepr<f32>, D> {
        let mut output = input.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        output
    }

    pub fn backward(&mut self, grad_output: &ArrayBase<OwnedRepr<f32>, D>, learning_rate: f32) {
        let mut grad_input = grad_output.clone();
        for layer in self.layers.iter_mut().rev() {
            grad_input = layer.backward(&grad_input, learning_rate);
        }
    }

    pub fn train(
        &mut self,
        inputs: &ArrayBase<OwnedRepr<f32>, D>,
        targets: &ArrayBase<OwnedRepr<f32>, D>,
        learning_rate: f32,
        epochs: usize,
        _batch_size: usize,
    ) where
        D: RemoveAxis,
    {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let num_samples = inputs.len_of(Axis(0));

            for i in 0..num_samples {
                let input = inputs.select(Axis(0), &[i]);
                let target = targets.select(Axis(0), &[i]);

                let output = self.forward(&input);
                total_loss += self.cost.loss(&output, &target);

                let grad_output = self.cost.gradient(&output, &target);
                self.backward(&grad_output, learning_rate);
            }

            println!("Epoch {}: Loss = {}", epoch + 1, total_loss / num_samples as f32);
        }
    }

    pub fn accuracy(
        &mut self,
        test_data: &ArrayBase<OwnedRepr<f32>, D>,
        test_labels: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> f32
    where
        D: RemoveAxis,
    {
        let predictions = self.forward(test_data);

        assert_eq!(
            predictions.shape(),
            test_labels.shape(),
            "Predictions and labels must have the same shape."
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

        let correct_predictions = predicted_classes
            .iter()
            .zip(target_classes.iter())
            .filter(|&(pred, target)| pred == target)
            .count();

        correct_predictions as f32 / predictions.len_of(Axis(0)) as f32
    }
}
