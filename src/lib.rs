pub mod activation;
pub mod cost;
pub mod layer;

use crate::cost::Cost;
use crate::layer::Layer;
use ndarray::{ArrayBase, Axis, Dimension, OwnedRepr, RemoveAxis};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

pub struct NeuralNetwork<L, C>
where
    L: Layer,
{
    pub layers: Vec<L>,
    pub cost: C,
}

impl<L, C, D> NeuralNetwork<L, C>
where
    L: Layer<Input = D, Output = D> + Serialize + for<'de> Deserialize<'de>,
    C: Cost<D>,
    D: Dimension + RemoveAxis,
{
    pub fn new(layers: Vec<L>, cost: C) -> Self {
        NeuralNetwork { layers, cost }
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let encoded = postcard::to_allocvec(&self.layers)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        let mut file = File::create(path)?;
        file.write_all(&encoded)?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P, cost: C) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let layers: Vec<L> = postcard::from_bytes(&buffer)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        Ok(NeuralNetwork { layers, cost })
    }

    pub fn forward(
        &mut self,
        input: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> ArrayBase<OwnedRepr<f32>, D> {
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
        batch_size: usize,
    ) where
        D: RemoveAxis,
    {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let num_samples = inputs.len_of(Axis(0));

        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut batch_count = 0;

            let mut indices: Vec<usize> = (0..num_samples).collect();
            indices.shuffle(&mut thread_rng());

            for batch_start in (0..num_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(num_samples);
                let batch_indices: Vec<usize> = indices[batch_start..batch_end].to_vec();

                let batch_input = inputs.select(Axis(0), &batch_indices);
                let batch_target = targets.select(Axis(0), &batch_indices);

                let output = self.forward(&batch_input);
                total_loss += self.cost.loss(&output, &batch_target);

                let grad_output = self.cost.gradient(&output, &batch_target);
                self.backward(&grad_output, learning_rate);

                batch_count += 1;
            }

            let avg_loss = total_loss / batch_count as f32;
            println!("Epoch {}/{}: Loss = {:.6}", epoch + 1, epochs, avg_loss);
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
