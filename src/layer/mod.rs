use ndarray::{ArrayBase, OwnedRepr, Dimension};

pub trait Layer {
    type Input: Dimension;
    type Output: Dimension;

    fn forward(&mut self, input: &ArrayBase<OwnedRepr<f32>, Self::Input>) -> ArrayBase<OwnedRepr<f32>, Self::Output>;
    fn backward(&mut self, grad_output: &ArrayBase<OwnedRepr<f32>, Self::Output>, learning_rate: f32) -> ArrayBase<OwnedRepr<f32>, Self::Input>;
}

// Example implementation for DenseLayer
use ndarray::{Array2, Array1, Ix2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use crate::activation::Activation;

pub struct DenseLayer<A: Activation> {
    weights: Array2<f32>,
    biases: Array1<f32>,
    activation: A,
    last_input: Option<Array2<f32>>,
    last_z: Option<Array2<f32>>,
}

impl<A: Activation> DenseLayer<A> {
    pub fn new(input_size: usize, output_size: usize, activation: A) -> Self {
        let scale = (2.0 / input_size as f32).sqrt();
        DenseLayer {
            weights: Array2::random((input_size, output_size), Uniform::new(-scale, scale).unwrap()),
            biases: Array1::zeros(output_size),
            activation,
            last_input: None,
            last_z: None,
        }
    }
}

impl<A: Activation> Layer for DenseLayer<A> {
    type Input = Ix2;
    type Output = Ix2;

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.last_input = Some(input.clone());
        let z = input.dot(&self.weights) + &self.biases;
        self.last_z = Some(z.clone());
        self.activation.activate(&z)
    }

    fn backward(&mut self, grad_output: &Array2<f32>, learning_rate: f32) -> Array2<f32> {
        let last_z = self.last_z.as_ref().expect("forward must be called first");
        let last_input = self.last_input.as_ref().expect("forward must be called first");

        let grad_z = grad_output * &self.activation.derivative(last_z);

        let grad_weights = last_input.t().dot(&grad_z);
        let grad_biases = grad_z.sum_axis(ndarray::Axis(0));
        let grad_input = grad_z.dot(&self.weights.t());

        self.weights = &self.weights - &(learning_rate * &grad_weights);
        self.biases = &self.biases - &(learning_rate * &grad_biases);

        grad_input
    }
}
