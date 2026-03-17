use crate::activation::Activation;
use crate::backend::{Backend, DefaultBackend};
use crate::initializer::Initializer;
use crate::layer::Layer;
use crate::optimizer::Optimizer;
use ndarray::{Ix1, Ix2};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

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
    #[serde(skip)]
    last_input: Option<B::Tensor<Ix2>>,
    #[serde(skip)]
    last_z: Option<B::Tensor<Ix2>>,
    #[serde(skip)]
    grad_weights: Option<B::Tensor<Ix2>>,
    #[serde(skip)]
    grad_biases: Option<B::Tensor<Ix1>>,
    #[serde(skip)]
    _backend: PhantomData<B>,
}

impl<A: Activation<B>, B: Backend> DenseLayer<A, B> {
    pub fn new<IW, IB>(
        input_size: usize,
        output_size: usize,
        activation: A,
        weight_init: IW,
        bias_init: IB,
    ) -> Self
    where
        IW: Initializer<B>,
        IB: Initializer<B>,
    {
        DenseLayer {
            weights: weight_init.init(Ix2(input_size, output_size)),
            biases: bias_init.init(Ix1(output_size)),
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

        let batch_size = B::len_of(last_input, 0) as f32;
        let inv_batch = 1.0 / batch_size.max(1.0);

        self.grad_weights = Some(B::scale(
            &B::matmul(&B::transpose(last_input, 0, 1), &grad_z),
            inv_batch,
        ));
        self.grad_biases = Some(B::scale(&B::sum_axis(&grad_z, 0), inv_batch));

        B::matmul(&grad_z, &B::transpose(&self.weights, 0, 1))
    }

    fn update<O: Optimizer<B>>(&mut self, optimizer: &mut O) {
        if let (Some(gw), Some(gb)) = (self.grad_weights.take(), self.grad_biases.take()) {
            optimizer.update_param(&mut self.weights, &gw);
            optimizer.update_param(&mut self.biases, &gb);
        }
    }
}
