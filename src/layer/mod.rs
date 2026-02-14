use ndarray::{ArrayBase, Dimension, OwnedRepr};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use crate::activation::Activation;
use ndarray::{Array1, Array2, Axis, Ix2, RemoveAxis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

pub trait Layer {
    type Input: Dimension;
    type Output: Dimension;

    fn forward(
        &mut self,
        input: &ArrayBase<OwnedRepr<f32>, Self::Input>,
    ) -> ArrayBase<OwnedRepr<f32>, Self::Output>;
    fn backward(
        &mut self,
        grad_output: &ArrayBase<OwnedRepr<f32>, Self::Output>,
        learning_rate: f32,
    ) -> ArrayBase<OwnedRepr<f32>, Self::Input>;
}

#[derive(Serialize, Deserialize)]
pub struct EmptyLayer<D> {
    _phantom: PhantomData<D>,
}

impl<D> EmptyLayer<D> {
    pub fn new() -> Self {
        EmptyLayer {
            _phantom: PhantomData,
        }
    }
}

impl<D: Dimension + RemoveAxis> Layer for EmptyLayer<D> {
    type Input = D;
    type Output = D;

    fn forward(&mut self, input: &ArrayBase<OwnedRepr<f32>, D>) -> ArrayBase<OwnedRepr<f32>, D> {
        input.clone()
    }

    fn backward(&mut self, grad_output: &ArrayBase<OwnedRepr<f32>, D>, _learning_rate: f32) -> ArrayBase<OwnedRepr<f32>, D> {
        grad_output.clone()
    }
}


#[derive(Serialize, Deserialize)]
pub struct Sequential<L1, L2> {
    pub layer1: L1,
    pub layer2: L2,
}

impl<L1, L2, D> Layer for Sequential<L1, L2>
where
    L1: Layer<Input = D, Output = D>,
    L2: Layer<Input = D, Output = D>,
    D: Dimension + RemoveAxis,
{
    type Input = D;
    type Output = D;

    fn forward(&mut self, input: &ArrayBase<OwnedRepr<f32>, D>) -> ArrayBase<OwnedRepr<f32>, D> {
        let out1 = self.layer1.forward(input);
        self.layer2.forward(&out1)
    }

    fn backward(&mut self, grad_output: &ArrayBase<OwnedRepr<f32>, D>, learning_rate: f32) -> ArrayBase<OwnedRepr<f32>, D> {
        let grad2 = self.layer2.backward(grad_output, learning_rate);
        self.layer1.backward(&grad2, learning_rate)
    }
}

pub fn seq<L1, L2>(layer1: L1, layer2: L2) -> Sequential<L1, L2> {
    Sequential { layer1, layer2 }
}

#[macro_export]
macro_rules! Layers {
    ($layer:expr) => {
        $layer
    };
    ($layer1:expr, $layer2:expr) => {
        $crate::layer::seq($layer1, $layer2)
    };
    ($layer1:expr, $layer2:expr, $($rest:expr),+) => {
        $crate::layer::seq($layer1, Layers!($layer2, $($rest),+))
    };
}

#[derive(Serialize, Deserialize)]
pub struct DenseLayer<A: Activation + Serialize> {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    pub activation: A,
    #[serde(skip)]
    last_input: Option<Array2<f32>>,
    #[serde(skip)]
    last_z: Option<Array2<f32>>,
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

    fn backward(&mut self, grad_output: &Array2<f32>, learning_rate: f32) -> Array2<f32> {
        let last_z = self.last_z.as_ref().expect("forward must be called first");
        let last_input = self
            .last_input
            .as_ref()
            .expect("forward must be called first");

        let grad_z = grad_output * &self.activation.derivative(last_z);

        let grad_weights = last_input.t().dot(&grad_z);
        let grad_biases = grad_z.sum_axis(Axis(0));
        let grad_input = grad_z.dot(&self.weights.t());

        self.weights = &self.weights - &(learning_rate * &grad_weights);
        self.biases = &self.biases - &(learning_rate * &grad_biases);

        grad_input
    }
}
