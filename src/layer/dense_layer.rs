use ndarray::{Array1, Array2, Axis, Ix2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use serde::{Deserialize, Serialize};
use crate::activation::Activation;
use crate::layer::Layer;
use crate::optimizer::Optimizer;

#[derive(Serialize, Deserialize)]
pub struct DenseLayer<A: Activation + Serialize> {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    pub activation: A,
    #[serde(skip)]
    last_input: Option<Array2<f32>>,
    #[serde(skip)]
    last_z: Option<Array2<f32>>,
    #[serde(skip)]
    grad_weights: Option<Array2<f32>>,
    #[serde(skip)]
    grad_biases: Option<Array1<f32>>,
}

impl<A: Activation + Serialize> DenseLayer<A> {
    pub fn new(input_size: usize, output_size: usize, activation: A) -> Self {
        let scale = (2.0 / input_size as f32).sqrt();
        DenseLayer {
            weights: Array2::random(
                (input_size, output_size),
                Uniform::new(-scale, scale).unwrap(),
            ),
            biases: Array1::zeros(output_size),
            activation,
            last_input: None,
            last_z: None,
            grad_weights: None,
            grad_biases: None,
        }
    }
}

impl<A: Activation + Serialize> Layer for DenseLayer<A> {
    type Input = Ix2;
    type Output = Ix2;

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.last_input = Some(input.clone());
        let z = input.dot(&self.weights) + &self.biases;
        self.last_z = Some(z.clone());
        self.activation.activate(&z)
    }

    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        let last_z = self.last_z.as_ref().expect("forward must be called before backward");
        let last_input = self.last_input.as_ref().expect("forward must be called before backward");

        let grad_z = self.activation.vjp(last_z, grad_output);

        self.grad_weights = Some(last_input.t().dot(&grad_z));
        self.grad_biases = Some(grad_z.sum_axis(Axis(0)));

        grad_z.dot(&self.weights.t())
    }

    fn update<O: Optimizer>(&mut self, optimizer: &mut O) {
        if let (Some(gw), Some(gb)) = (self.grad_weights.take(), self.grad_biases.take()) {
            optimizer.update_param(&mut self.weights, &gw);
            optimizer.update_param(&mut self.biases, &gb);
        }
    }
}
