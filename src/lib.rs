pub mod activation;
pub mod backend;
pub mod cost;
pub mod layer;
pub mod metric;
pub mod optimizer;

pub use backend::NdarrayBackend;

use crate::backend::Backend;
use crate::cost::Cost;
use crate::layer::Layer;
use crate::optimizer::Optimizer;
use ndarray::RemoveAxis;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::path::Path;

pub struct NeuralNetwork<L, C, B: Backend = NdarrayBackend>
where
    L: Layer<B>,
{
    pub layers: L,
    pub cost: C,
    _backend: PhantomData<B>,
}

impl<L, C, B> NeuralNetwork<L, C, B>
where
    B: Backend,
    L: Layer<B> + Serialize + for<'de> Deserialize<'de>,
    C: Cost<B>,
{
    pub fn new(layers: L, cost: C) -> Self {
        NeuralNetwork { layers, cost, _backend: PhantomData }
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
        Ok(NeuralNetwork { layers, cost, _backend: PhantomData })
    }

    pub fn forward(&mut self, input: &B::Tensor<L::Input>) -> B::Tensor<L::Output> {
        self.layers.forward(input)
    }

    pub fn backward(&mut self, grad_output: &B::Tensor<L::Output>) -> B::Tensor<L::Input> {
        self.layers.backward(grad_output)
    }

    pub fn train<O: Optimizer<B>>(
        &mut self,
        inputs: &B::Tensor<L::Input>,
        targets: &B::Tensor<L::Output>,
        mut optimizer: O,
        epochs: usize,
        batch_size: usize,
    ) where
        L::Input: RemoveAxis,
        L::Output: RemoveAxis,
    {
        use rand::rng;
        use rand::seq::SliceRandom;

        let num_samples = B::len_of(inputs, 0);
        assert_eq!(num_samples, B::len_of(targets, 0), "batch size mismatch");

        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut batch_count = 0;

            let mut indices: Vec<usize> = (0..num_samples).collect();
            indices.shuffle(&mut rng());

            for batch_start in (0..num_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(num_samples);
                let batch_indices = &indices[batch_start..batch_end];

                let batch_input = B::select(inputs, 0, batch_indices);
                let batch_target = B::select(targets, 0, batch_indices);

                let output = self.forward(&batch_input);
                total_loss += self.cost.loss(&output, &batch_target);

                let grad = self.cost.gradient(&output, &batch_target);
                self.backward(&grad);
                self.layers.update(&mut optimizer);
                batch_count += 1;
            }

            println!(
                "Epoch {}/{}: Loss = {:.6}",
                epoch + 1, epochs,
                total_loss / batch_count as f32
            );
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