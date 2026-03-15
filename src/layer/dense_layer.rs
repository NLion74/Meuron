use ndarray::{Ix1, Ix2};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use crate::activation::Activation;
use crate::backend::{Backend, DefaultBackend};
use crate::layer::Layer;
use crate::optimizer::Optimizer;

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "A: Serialize, B::Tensor<Ix2>: Serialize, B::Tensor<Ix1>: Serialize",
    deserialize = "A: Deserialize<'de>, B::Tensor<Ix2>: Deserialize<'de>, B::Tensor<Ix1>: Deserialize<'de>"
))]
pub struct DenseLayer<A, B: Backend = DefaultBackend>
where
    A: Activation<B>,
{
    pub weights: B::Tensor<Ix2>,
    pub biases: B::Tensor<Ix1>,
    pub activation: A,
    #[serde(skip)] last_input: Option<B::Tensor<Ix2>>,
    #[serde(skip)] last_z: Option<B::Tensor<Ix2>>,
    #[serde(skip)] grad_weights: Option<B::Tensor<Ix2>>,
    #[serde(skip)] grad_biases: Option<B::Tensor<Ix1>>,
    #[serde(skip)] _backend: PhantomData<B>,
}

impl<A: Activation<B>, B: Backend> DenseLayer<A, B> {
    pub fn new(input_size: usize, output_size: usize, activation: A) -> Self {
        let scale = (2.0 / input_size as f32).sqrt();
        DenseLayer {
            weights: B::random_uniform(Ix2(input_size, output_size), -scale, scale),
            biases: B::zeros(Ix1(output_size)),
            activation,
            last_input: None,
            last_z: None,
            grad_weights: None,
            grad_biases: None,
            _backend: PhantomData,
        }
    }
}

impl<A: Activation<B>, B: Backend> Layer<B> for DenseLayer<A, B> {
    type Input = Ix2;
    type Output = Ix2;

    fn forward(&mut self, input: &B::Tensor<Ix2>) -> B::Tensor<Ix2> {
        self.last_input = Some(input.clone());
        let z = B::broadcast_add(&B::matmul(input, &self.weights), &self.biases);
        self.last_z = Some(z.clone());
        self.activation.activate(&z)
    }

    fn backward(&mut self, grad_output: &B::Tensor<Ix2>) -> B::Tensor<Ix2> {
        let last_z = self.last_z.as_ref().expect("forward before backward");
        let last_input = self.last_input.as_ref().expect("forward before backward");

        let grad_z = self.activation.vjp(last_z, grad_output);

        self.grad_weights = Some(B::matmul(&B::transpose(last_input, 0, 1), &grad_z));
        self.grad_biases = Some(B::sum_axis(&grad_z, 0));

        B::matmul(&grad_z, &B::transpose(&self.weights, 0, 1))
    }

    fn update<O: Optimizer<B>>(&mut self, optimizer: &mut O) {
        if let (Some(gw), Some(gb)) = (self.grad_weights.take(), self.grad_biases.take()) {
            optimizer.update_param(&mut self.weights, &gw);
            optimizer.update_param(&mut self.biases, &gb);
        }
    }
}
