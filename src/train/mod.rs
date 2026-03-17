use crate::NeuralNetwork;
use crate::backend::Backend;
use crate::cost::Cost;
use crate::layer::Layer;
use crate::optimizer::Optimizer;
use ndarray::RemoveAxis;
use serde::{Deserialize, Serialize};

pub trait TrainCallback {
    fn on_epoch(&mut self, epoch: usize, total: usize, loss: f32, val_loss: Option<f32>) -> bool;
}

pub struct PrintCallback;

impl TrainCallback for PrintCallback {
    fn on_epoch(&mut self, epoch: usize, total: usize, loss: f32, val_loss: Option<f32>) -> bool {
        print!("Epoch {epoch}/{total}: loss = {loss:.6}");
        if let Some(v) = val_loss {
            print!(" val_loss = {v:.6}");
        }
        println!();
        true
    }
}

impl<F> TrainCallback for F
where
    F: FnMut(usize, usize, f32, Option<f32>) -> bool,
{
    fn on_epoch(&mut self, epoch: usize, total: usize, loss: f32, val_loss: Option<f32>) -> bool {
        self(epoch, total, loss, val_loss)
    }
}

pub struct TrainOptions {
    pub epochs: usize,
    pub batch_size: usize,
    pub validation_split: f32,
    pub callback: Option<Box<dyn TrainCallback>>,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            epochs: 10,
            batch_size: 32,
            validation_split: 0.0,
            callback: None,
        }
    }
}

impl TrainOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn validation_split(mut self, validation_split: f32) -> Self {
        self.validation_split = validation_split;
        self
    }

    pub fn callback<C>(mut self, callback: C) -> Self
    where
        C: TrainCallback + 'static,
    {
        self.callback = Some(Box::new(callback));
        self
    }
}

impl<L, C, B> NeuralNetwork<L, C, B>
where
    B: Backend,
    L: Layer<B> + Serialize + for<'de> Deserialize<'de>,
    C: Cost<B>,
{
    pub fn train<O, I, T>(
        &mut self,
        inputs: I,
        targets: T,
        mut optimizer: O,
        options: TrainOptions,
    ) -> Vec<f32>
    where
        O: Optimizer<B>,
        I: Into<B::Tensor<L::Input>>,
        T: Into<B::Tensor<L::Output>>,
        L::Input: RemoveAxis,
        L::Output: RemoveAxis,
    {
        use ndarray::{Axis, Slice};
        use rand::rng;
        use rand::seq::SliceRandom;

        let TrainOptions {
            epochs,
            batch_size,
            validation_split,
            callback,
        } = options;

        let mut callback: Box<dyn TrainCallback> =
            callback.unwrap_or_else(|| Box::new(PrintCallback));

        let all_inputs = B::to_array(&inputs.into());
        let all_targets = B::to_array(&targets.into());

        let total = all_inputs.shape()[0];
        let val_n = (total as f32 * validation_split) as usize;
        let train_n = total - val_n;

        let train_inputs = all_inputs
            .slice_axis(Axis(0), Slice::from(..train_n))
            .to_owned();
        let train_targets = all_targets
            .slice_axis(Axis(0), Slice::from(..train_n))
            .to_owned();

        let val_data = (val_n > 0).then(|| {
            (
                all_inputs
                    .slice_axis(Axis(0), Slice::from(train_n..))
                    .to_owned(),
                all_targets
                    .slice_axis(Axis(0), Slice::from(train_n..))
                    .to_owned(),
            )
        });

        let mut losses = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            let mut indices: Vec<usize> = (0..train_n).collect();
            indices.shuffle(&mut rng());

            let mut total_loss = 0.0_f32;
            let mut batch_count = 0;

            for batch_start in (0..train_n).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(train_n);
                let batch_indices = &indices[batch_start..batch_end];

                let batch_input = B::from_array(train_inputs.select(Axis(0), batch_indices));
                let batch_target = B::from_array(train_targets.select(Axis(0), batch_indices));

                let output = self.forward(batch_input);
                total_loss += self.cost.loss(&output, &batch_target);
                let grad = self.cost.gradient(&output, &batch_target);
                self.backward(grad);
                self.layers.update(&mut optimizer);

                B::flush();
                batch_count += 1;
            }

            let loss = total_loss / batch_count as f32;
            losses.push(loss);

            let val_loss = val_data.as_ref().map(|(vi, vt)| {
                let out = self.forward(B::from_array(vi.clone()));
                self.cost.loss(&out, &B::from_array(vt.clone()))
            });

            if !callback.on_epoch(epoch + 1, epochs, loss, val_loss) {
                break;
            }
        }

        losses
    }
}
