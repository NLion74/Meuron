#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, Array2};
    use Meuron::layer::{Layer, DenseLayer};
    use Meuron::activation::{Activation, Sigmoid};
    use Meuron::cost::{Cost, MSE};

    #[test]
    fn test_dense_layer_forward() {
        let input_size = 3;
        let output_size = 2;
        let dense_layer = DenseLayer::new(input_size, output_size, Sigmoid);
        
        let input = arr2(&[[1.0], [2.0], [3.0]]); // Shape (3, 1)
        let output = dense_layer.forward(&input); // Perform forward pass
        
        // You can manually compute the expected output based on initial weights
        // For the sake of the test, let's assume you know the expected output
        // Here we will use placeholder expected values
        let expected_output = arr2(&[[0.0], [0.0]]); // Replace with actual expected output
        assert!(output.abs_diff_eq(&expected_output, 1e-5));
    }

    #[test]
    fn test_dense_layer_backward() {
        let input_size = 3;
        let output_size = 2;
        let mut dense_layer = DenseLayer::new(input_size, output_size, Sigmoid);
        
        let input = arr2(&[[1.0], [2.0], [3.0]]);
        let output = dense_layer.forward(&input);
        
        // Simulating gradients coming from the next layer
        let grad_output = arr2(&[[0.5], [0.5]]); // Placeholder gradients
        let grad_input = dense_layer.backward(&input, &grad_output); // Backward pass
        
        // Check the shape of the gradient input
        assert_eq!(grad_input.shape(), &[input_size, 1]);

        // Verify that gradients are being computed correctly
        // You can manually compute the expected gradient based on your layer's implementation
        let expected_grad_input = arr2(&[[0.0], [0.0], [0.0]]); // Replace with actual expected gradients
        assert!(grad_input.abs_diff_eq(&expected_grad_input, 1e-5));
    }

    #[test]
    fn test_dense_layer_weight_update() {
        let input_size = 3;
        let output_size = 2;
        let mut dense_layer = DenseLayer::new(input_size, output_size, Sigmoid);
        
        let input = arr2(&[[1.0], [2.0], [3.0]]);
        let output = dense_layer.forward(&input);

        // Simulating a loss gradient
        let grad_output = arr2(&[[0.5], [0.5]]); // Placeholder gradients
        dense_layer.backward(&input, &grad_output); // Perform backward pass
        
        // Save original weights for comparison
        let original_weights = dense_layer.weights.clone();

        // Update weights
        let learning_rate = 0.1;
        dense_layer.update(learning_rate);

        // Check if weights have changed
        assert!(!dense_layer.weights.abs_diff_eq(&original_weights, 1e-5)); // Ensure weights are updated
    }
}

