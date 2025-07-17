use ndarray::{ Array2, Axis };

pub trait Activation {
    fn activate(&self, input: &Array2<f32>) -> Array2<f32>;
    fn derivative(&self, output: &Array2<f32>) -> Array2<f32>;
}

pub struct Sigmoid;
pub struct ReLU;
pub struct Softmax;

impl Activation for Sigmoid {
    fn activate(&self, input: &Array2<f32>) -> Array2<f32> {
        let clipped_input = input.mapv(|x| x.clamp(-20.0, 20.0));
        clipped_input.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    fn derivative(&self, input: &Array2<f32>) -> Array2<f32> {
        input * &(1.0 - input)
    }
}

impl Activation for ReLU {
    fn activate(&self, input: &Array2<f32>) -> Array2<f32> {
        input.mapv(|x| x.max(0.0))
    }

    fn derivative(&self, output: &Array2<f32>) -> Array2<f32> {
        output.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }
}

impl Activation for Softmax {
    fn activate(&self, input: &Array2<f32>) -> Array2<f32> {
        let max_val = input.fold(f32::NEG_INFINITY, |a, &b| a.max(b)); // Numerical stability
        let exp_input = (input - max_val).mapv(|x| x.exp()); // Apply exp
        let sum_exp = exp_input.sum_axis(Axis(1)).insert_axis(Axis(1)); // Sum along the rows
        exp_input / sum_exp // Normalize to get probabilities
    }

    fn derivative(&self, output: &Array2<f32>) -> Array2<f32> {
        let (batch_size, num_classes) = (output.shape()[0], output.shape()[1]);
        
        // Initialize the gradient with shape (batch_size, num_classes)
        let mut grad = Array2::<f32>::zeros((batch_size, num_classes));

        // Compute the gradient for each instance in the batch
        for i in 0..batch_size {
            for j in 0..num_classes {
                let s_j = output[[i, j]]; // Probability for class j in batch instance i
                for k in 0..num_classes {
                    let s_k = output[[i, k]]; // Probability for class k in batch instance i
                    if j == k {
                        grad[[i, j]] += s_j * (1.0 - s_k); // Diagonal contribution
                    } else {
                        grad[[i, j]] -= s_j * s_k; // Off-diagonal contribution
                    }
                }
            }
        }

        // Return the gradient vector with shape (batch_size, num_classes)
        grad

    }
}
