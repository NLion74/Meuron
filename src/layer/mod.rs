use ndarray::{Array2, Axis, ArrayBase, OwnedRepr, Dimension, Ix2};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use crate::activation::Activation;

pub trait Layer {
    type Input: Dimension;
    type Output: Dimension;
    
    fn forward(&mut self, input: &ArrayBase<OwnedRepr<f32>, Self::Input>) -> ArrayBase<OwnedRepr<f32>, Self::Output>;
    fn backward(&mut self, gradient: &ArrayBase<OwnedRepr<f32>, Self::Output>, learning_rate: f32) -> ArrayBase<OwnedRepr<f32>, Self::Input>;
}

pub struct DenseLayer<A: Activation> {
    pub weights: ArrayBase<OwnedRepr<f32>, Ix2>,
    pub biases: ArrayBase<OwnedRepr<f32>, Ix2>,
    pub grad_weights: ArrayBase<OwnedRepr<f32>, Ix2>,
    pub grad_biases: ArrayBase<OwnedRepr<f32>, Ix2>,
    pub input: Option<Array2<f32>>,
    pub output: Option<Array2<f32>>,
    pub activation: A,
}

impl<A: Activation> DenseLayer<A> {
    pub fn new(input_size: usize, output_size: usize, activation: A) -> Self {
        let scale = (6.0 / (input_size + output_size) as f32).sqrt();

        let uniform = Uniform::new(-scale, scale);

        let mut rng = thread_rng();
        
        let weights = Array2::from_shape_fn((input_size, output_size), |_| uniform.sample(&mut rng));
        let biases = Array2::<f32>::zeros((1, output_size));
        let grad_weights = Array2::<f32>::zeros((input_size, output_size));
        let grad_biases = Array2::<f32>::zeros((1, output_size));

        DenseLayer {
            weights,
            biases,
            grad_weights,
            grad_biases,
            input: None,
            output: None,
            activation,
        }
    }
}

impl<A: Activation> Layer for DenseLayer<A> {
    type Input = Ix2;
    type Output = Ix2;

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        assert_eq!(input.shape()[1], self.weights.shape()[0], "Input width must match weight height.");

        self.input = Some(input.clone());

        let z = input.dot(&self.weights) + &self.biases;
        let activated_output = self.activation.activate(&z);

        self.output = Some(activated_output.clone());

        activated_output
    }

    fn backward(&mut self, grad_output: &Array2<f32>, learning_rate: f32) -> Array2<f32> {
        let input = self.input.as_ref().expect("Forward pass must be called before backward pass");
        let output = self.output.as_ref().expect("Forward pass must be called before backward pass");

        let grad_activation = self.activation.derivative(output);
        let grad_output_combined = grad_output * &grad_activation;

        self.grad_weights += &input.t().dot(&grad_output_combined);
        self.grad_biases += &grad_output_combined.sum_axis(Axis(0)).insert_axis(Axis(0));

        assert_eq!(self.weights.shape(), self.grad_weights.shape(), "Weights and grad_weights must have the same shape.");
        assert_eq!(self.biases.shape(), self.grad_biases.shape(), "Biases and grad_biases must have the same shape.");

        // let max_norm = 5.0;

        // let norm = self.grad_weights.mapv(|x| x.powi(2)).sum().sqrt();

        // // Clip if the norm exceeds the max norm
        // if norm > max_norm {
        //     self.grad_weights *= max_norm / norm; // Scale down the gradients
        // }

        // // Do the same for biases if needed
        // let bias_norm = self.grad_biases.mapv(|x| x.powi(2)).sum().sqrt();
        // if bias_norm > max_norm {
        //     self.grad_biases *= max_norm / bias_norm; // Scale down the biases
        // }

        self.weights -= &(learning_rate * &self.grad_weights);
        self.biases -= &(learning_rate * &self.grad_biases);

        self.grad_weights.fill(0.0);
        self.grad_biases.fill(0.0);

        grad_activation.dot(&self.weights.t())
    }
}

// pub trait Layer<I: Dimension, O: Dimension> {


//     fn forward(&mut self, input: &ArrayBase<OwnedRepr<f32>, I>) -> ArrayBase<OwnedRepr<f32>, O>;
//     fn backward(&mut self, input: &ArrayBase<OwnedRepr<f32>, I>, gradient: &ArrayBase<OwnedRepr<f32>, O>, learning_rate: f32) -> ArrayBase<OwnedRepr<f32>, I>; // For backward propagation
// }

// pub struct DenseLayer<A: Activation> {
//     pub weights: ArrayBase<OwnedRepr<f32>, Ix2>,
//     pub biases: ArrayBase<OwnedRepr<f32>, Ix2>,
//     pub grad_weights: ArrayBase<OwnedRepr<f32>, Ix2>,
//     pub grad_biases: ArrayBase<OwnedRepr<f32>, Ix2>,
//     pub activation: A,
// }

// impl<A: Activation> DenseLayer<A> {
//     pub fn new(input_size: usize, output_size: usize, activation: A) -> Self {
//         let scale = (6.0 / (input_size + output_size) as f32).sqrt();

//         // Define a uniform distribution for random values
//         let uniform = Uniform::new(-scale, scale);

//         let mut rng = thread_rng();
        
//         let weights = Array2::from_shape_fn((input_size, output_size), |_| uniform.sample(&mut rng));
//         let biases = Array2::<f32>::zeros((1, output_size));
//         let grad_weights = Array2::<f32>::zeros((input_size, output_size));
//         let grad_biases = Array2::<f32>::zeros((1, output_size));

//         DenseLayer {
//             weights,
//             biases,
//             grad_weights,
//             grad_biases,
//             activation,
//         }
//     }
// }

// impl<A: Activation> Layer<Ix2, Ix2> for DenseLayer<A> {
//     fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
//         assert_eq!(input.shape()[1], self.weights.shape()[0], "Input width must match weight height.");
//         let z = input.dot(&self.weights) + &self.biases;

//         self.activation.activate(&z)
//     }

//     fn backward(&mut self, input: &Array2<f32>, grad_output: &Array2<f32>, learning_rate: f32) -> Array2<f32> {
//         let grad_activation = self.activation.derivative(grad_output);

//         self.grad_weights = input.t().dot(&grad_activation);
//         self.grad_biases = grad_activation.sum_axis(Axis(0)).insert_axis(Axis(0));

//         let max_norm = 5.0; // Set a max norm for gradient clipping

//         // Compute the norm of the weights gradient
//         let norm = self.grad_weights.mapv(|x| x.powi(2)).sum().sqrt();

//         // Clip if the norm exceeds the max norm
//         if norm > max_norm {
//             self.grad_weights *= max_norm / norm; // Scale down the gradients
//         }

//         // Do the same for biases if needed
//         let bias_norm = self.grad_biases.mapv(|x| x.powi(2)).sum().sqrt();
//         if bias_norm > max_norm {
//             self.grad_biases *= max_norm / bias_norm; // Scale down the biases
//         }

//         self.weights -= &(learning_rate * &self.grad_weights);
//         self.biases -= &(learning_rate * &self.grad_biases);
        
//         self.grad_weights.fill(0.0);
//         self.grad_biases.fill(0.0);

//         grad_activation.dot(&self.weights.t())
//     }
// }

// pub trait Layer<Input, Output> {
//     fn forward(&mut self, input: &Input) -> Output;
//     fn backward(&mut self, input: &Input, gradient: &Output, learning_rate: f32) -> Input; // For backward propagation
// }

// pub struct DenseLayer<A: Activation> {
//     pub weights: Array2<f32>,
//     pub biases: Array2<f32>,
//     pub grad_weights: Array2<f32>,
//     pub grad_biases: Array2<f32>,
//     pub activation: A,
// }

// impl<A: Activation> DenseLayer<A> {
//     pub fn new(input_size: usize, output_size: usize, activation: A) -> Self {
//         let scale = (6.0 / (input_size + output_size) as f32).sqrt();

//         // Define a uniform distribution for random values
//         let uniform = Uniform::new(-scale, scale);

//         let mut rng = thread_rng();
        
//         let weights = Array2::from_shape_fn((input_size, output_size), |_| uniform.sample(&mut rng));
//         let biases = Array2::<f32>::zeros((1, output_size));
//         let grad_weights = Array2::<f32>::zeros((input_size, output_size));
//         let grad_biases = Array2::<f32>::zeros((1, output_size));

//         DenseLayer {
//             weights,
//             biases,
//             grad_weights,
//             grad_biases,
//             activation,
//         }
//     }
// }

// impl<A: Activation> Layer<Array2<f32>, Array2<f32>> for DenseLayer<A> {
//     fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
//         assert_eq!(input.shape()[1], self.weights.shape()[0], "Input width must match weight height.");
//         let z = input.dot(&self.weights) + &self.biases;

//         self.activation.activate(&z)
//     }

//     fn backward(&mut self, input: &Array2<f32>, grad_output: &Array2<f32>, learning_rate: f32) -> Array2<f32> {
//         let grad_activation = self.activation.derivative(grad_output);

//         self.grad_weights = input.t().dot(&grad_activation);
//         self.grad_biases = grad_activation.sum_axis(Axis(0)).insert_axis(Axis(0));

//         let max_norm = 5.0; // Set a max norm for gradient clipping

//         // Compute the norm of the weights gradient
//         let norm = self.grad_weights.mapv(|x| x.powi(2)).sum().sqrt();

//         // Clip if the norm exceeds the max norm
//         if norm > max_norm {
//             self.grad_weights *= max_norm / norm; // Scale down the gradients
//         }

//         // Do the same for biases if needed
//         let bias_norm = self.grad_biases.mapv(|x| x.powi(2)).sum().sqrt();
//         if bias_norm > max_norm {
//             self.grad_biases *= max_norm / bias_norm; // Scale down the biases
//         }

//         self.weights -= &(learning_rate * &self.grad_weights);
//         self.biases -= &(learning_rate * &self.grad_biases);
        
//         self.grad_weights.fill(0.0);
//         self.grad_biases.fill(0.0);

//         grad_activation.dot(&self.weights.t())
//     }
// }