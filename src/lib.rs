pub mod activation;
pub mod cost;
pub mod layer;
pub mod metric;
pub mod optimizer;

use crate::cost::Cost;
use crate::layer::Layer;
use crate::optimizer::Optimizer;
use ndarray::{ArrayBase, Axis, OwnedRepr, RemoveAxis};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

pub struct NeuralNetwork<L, C> where L: Layer {
    pub layers: L,
    pub cost: C,
}

impl<L, C> NeuralNetwork<L, C>
where
    L: Layer + Serialize + for<'de> Deserialize<'de>,
    C: Cost,
{
    pub fn new(layers: L, cost: C) -> Self {
        NeuralNetwork { layers, cost }
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let encoded = postcard::to_allocvec(&self.layers)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let mut file = File::create(path)?;
        file.write_all(&encoded)?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P, cost: C) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let layers: L = postcard::from_bytes(&buffer)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        Ok(NeuralNetwork { layers, cost })
    }

    pub fn forward(
        &mut self,
        input: &ArrayBase<OwnedRepr<f32>, L::Input>,
    ) -> ArrayBase<OwnedRepr<f32>, L::Output> {
        self.layers.forward(input)
    }

    pub fn backward(
        &mut self,
        grad_output: &ArrayBase<OwnedRepr<f32>, L::Output>,
    ) -> ArrayBase<OwnedRepr<f32>, L::Input> {
        self.layers.backward(grad_output)
    }

    pub fn train<O: Optimizer>(
        &mut self,
        inputs: &ArrayBase<OwnedRepr<f32>, L::Input>,
        targets: &ArrayBase<OwnedRepr<f32>, L::Output>,
        mut optimizer: O,
        epochs: usize,
        batch_size: usize,
    ) where
        L::Input: RemoveAxis,
        L::Output: RemoveAxis,
    {
        use rand::rng;
        use rand::seq::SliceRandom;

        let num_samples = inputs.len_of(Axis(0));
        assert_eq!(num_samples, targets.len_of(Axis(0)), "inputs and targets batch size mismatch");

        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut batch_count = 0;

            let mut indices: Vec<usize> = (0..num_samples).collect();
            indices.shuffle(&mut rng());

            for batch_start in (0..num_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(num_samples);
                let batch_indices = &indices[batch_start..batch_end];

                let batch_input = inputs.select(Axis(0), batch_indices);
                let batch_target = targets.select(Axis(0), batch_indices);

                let output = self.forward(&batch_input);
                total_loss += self.cost.loss(&output, &batch_target);

                let grad = self.cost.gradient(&output, &batch_target);
                self.backward(&grad);
                self.layers.update(&mut optimizer);
                batch_count += 1;
            }

            println!("Epoch {}/{}: Loss = {:.6}", epoch + 1, epochs, total_loss / batch_count as f32);
        }
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