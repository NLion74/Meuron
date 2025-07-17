pub mod layer;
pub mod activation;
pub mod cost;

use crate::layer::{Layer};
use crate::cost::{Cost};
use ndarray::{ArrayBase, Data, Dimension, RawDataClone, OwnedRepr, RemoveAxis, Axis, s};

pub struct NeuralNetwork<'a, L, T>
where
    L: Layer + 'a,
{
    layers: Vec<L>,
    cost: &'a dyn Cost<T>,
}

impl<'a, L, T> NeuralNetwork<'a, L, T>
where
    L: Layer<Input = T, Output = T> + 'a,
    T: Clone + Dimension + RemoveAxis,
{
    pub fn new(layers: Vec<L>, cost: &'a dyn Cost<T>) -> Self {
        NeuralNetwork { layers, cost }
    }

    pub fn forward(&mut self, input: &ArrayBase<OwnedRepr<f32>, L::Input>) -> ArrayBase<OwnedRepr<f32>, L::Output> {
        let mut output = input.clone();

        for layer in &mut self.layers {
            output = layer.forward(&output);
        }

        output
    }

    pub fn backward(
        &mut self,
        grad_output: &ArrayBase<OwnedRepr<f32>, L::Output>,
        learning_rate: f32,
    ) {
        let mut grad_input = grad_output.clone();

        for layer in self.layers.iter_mut().rev() {
            grad_input = layer.backward(&grad_input, learning_rate);
        }
    }

    pub fn accuracy(
        &mut self,
        test_data: &ArrayBase<OwnedRepr<f32>, L::Input>,
        test_labels: &ArrayBase<OwnedRepr<f32>, L::Output>,
    ) -> f32 {
        // Forward pass on the test dataset to get predictions
        let predictions = self.forward(test_data);

        // Ensure that predictions and labels have the same shape
        assert_eq!(
            predictions.shape(),
            test_labels.shape(),
            "Predictions and labels must have the same shape."
        );

        // Get the index of the maximum value in each row for predictions (predicted class)
        let predicted_classes = predictions.map_axis(Axis(1), |row| {
            row.iter().enumerate().fold(0, |max_idx, (i, &val)| {
                if val > row[max_idx] {
                    i
                } else {
                    max_idx
                }
            })
        });

        // Get the index of the correct class from the test labels (assuming one-hot encoding)
        let target_classes = test_labels.map_axis(Axis(1), |row| {
            row.iter().enumerate().fold(0, |max_idx, (i, &val)| {
                if val > row[max_idx] {
                    i
                } else {
                    max_idx
                }
            })
        });

        // Count the number of correct predictions
        let correct_predictions = predicted_classes.iter()
            .zip(target_classes.iter())
            .filter(|&(pred, target)| pred == target)
            .count();

        // Calculate accuracy as the percentage of correct predictions
        correct_predictions as f32 / predictions.len_of(Axis(0)) as f32
    }

    pub fn train(
        &mut self,
        inputs: &ArrayBase<OwnedRepr<f32>, L::Input>,
        targets: &ArrayBase<OwnedRepr<f32>, L::Output>,
        learning_rate: f32,
        epochs: usize,
        batch_size: usize,
    ) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for i in 0..inputs.shape()[0] {
                let input = inputs.index_axis(Axis(0), i.clone()).to_owned().insert_axis(Axis(0));
                let target = targets.index_axis(Axis(0), i.clone()).to_owned().insert_axis(Axis(0));
    
                let out = self.forward(&input);
    
                total_loss += self.cost.loss(&out, &target);
    
                let grad_output = self.cost.gradient(&out, &target);
    
                self.backward(&grad_output, learning_rate);
            }

            println!("Epoch: {}", epoch);
            println!("Loss: {}", total_loss / inputs.len_of(Axis(0)) as f32);
        };
    }
}

// pub struct NeuralNetwork<'a, I, O>
// where
//     I: Dimension, // Input dimension
//     O: Dimension, // Output dimension
// {
//     layers: Vec<&'a mut dyn Layer<I, O>>, // Vector of layers
//     loss_function: &'a dyn Cost,          // Reference to the cost function
//     _marker: PhantomData<(I, O)>,         // PhantomData to handle generics
// }

// impl<'a, I, O> NeuralNetwork<'a, I, O>
// where
//     I: Dimension, // Input dimension
//     O: Dimension, // Output dimension
// {
//     // Constructor for creating a new neural network
//     pub fn new(layers: Vec<&'a mut dyn Layer<I, O>>, loss_function: &'a dyn Cost) -> Self {
//         NeuralNetwork {
//             layers,
//             loss_function,
//             _marker: PhantomData,
//         }
//     }

//     pub fn forward(&mut self, input: &ArrayBase<OwnedRepr<f32>, I>) -> ArrayBase<OwnedRepr<f32>, O> {
//         let mut output = input.clone();
//         for layer in self.layers.iter_mut() {
//             output = layer.forward(&output); // forward through each layer
//         }
//         output
//     }

//     // Backward pass through all the layers
//     pub fn backward(
//         &mut self,
//         input: &ArrayBase<OwnedRepr<f32>, I>,
//         grad_output: &ArrayBase<OwnedRepr<f32>, O>,
//         learning_rate: f32,
//     ) -> ArrayBase<OwnedRepr<f32>, I> {
//         let mut grad_input = grad_output.clone(); // Clone gradient output to avoid modification
//         for layer in self.layers.iter_mut().rev() {
//             grad_input = layer.backward(input, &grad_input, learning_rate); // Backprop through each layer
//         }
//         grad_input
//     }

//     pub fn train(
//         &mut self,
//         input: &ArrayBase<OwnedRepr<f32>, I>,
//         target: &ArrayBase<OwnedRepr<f32>, O>,
//         learning_rate: f32,
//     ) -> f32 {
//         // Forward pass to get the predicted output
//         let predicted = self.forward(input);

//         // Compute the loss
//         let loss = self.loss_function.compute_loss(&predicted, target);

//         // Compute the gradient of the loss
//         let grad_loss = self.loss_function.compute_loss_gradient(&predicted, target);

//         // Backpropagation
//         self.backward(input, &grad_loss, learning_rate);

//         loss // Return the loss value for reporting
//     }
// }


// pub struct NeuralNetwork<'a, T, D>
// where
//     T: Data<Elem = f32> + RawDataClone,
//     D: Dimension,
// {
//     layers: Vec<&'a mut dyn Layer<T, D>>,
//     loss_function: &'a dyn Cost,
//     _marker: PhantomData<(T, D)>,
// }

// impl<'a, T, D> NeuralNetwork<'a, T, D>
// where
//     T: Data<Elem = f32> + RawDataClone,
//     D: Dimension,
// {
//     pub fn new(layers: Vec<&'a mut dyn Layer<T, D>>, loss_function: &'a dyn Cost) -> Self {
//         NeuralNetwork {
//             layers,
//             loss_function,
//             _marker: PhantomData,
//         }
//     }

//     pub fn forward(&mut self, input: &ArrayBase<T, D>) -> ArrayBase<T, D> {
//         let mut output = input.clone();
//         for layer in &mut self.layers {
//             output = layer.forward(&output);
//         }
//         output
//     }

// pub struct NeuralNetwork<'a, C, T, D>
// where
//     C: Cost,
//     T: Data<Elem = f32> + RawDataClone,
//     D: Dimension,
// {
//     layers: Vec<&'a mut dyn Layer<T, D>>,
//     loss_function: C,
//     _marker: PhantomData<(T, D)>,
// }

// impl<'a, C, T, D> NeuralNetwork<'a, C, T, D>
// where
//     C: Cost,
//     T: Data<Elem = f32> + RawDataClone,
//     D: Dimension,
// {
//     pub fn new(layers: Vec<&'a mut dyn Layer<T, D>>, loss_function: C) -> Self {
//         NeuralNetwork {
//             layers,
//             loss_function,
//             _marker: PhantomData,
//         }
//     }

//     pub fn forward(&mut self, mut input: &ArrayBase<T, D>) -> ArrayBase<T, D> {
//         for layer in &self.layers {
//             input = layer.forward(input)
//         }

//         self.output = Some(input.clone());
//         input
//     }

    // // Training the network
    // pub fn train(&mut self, input: &ArrayBase<T, D>, target: &ArrayBase<T, D>, learning_rate: f32) {
    //     let prediction = self.forward(input);
    //     let loss = self.loss_function.loss(&prediction, target);
    //     let mut grad = self.loss_function.gradient(&prediction, target);

    //     for layer in self.layers.iter_mut().rev() {
    //         grad = layer.backward(input, &grad, learning_rate);
    //     }

    //     println!("Loss: {}", loss);
    // }
// }

// pub struct NeuralNetwork<L, C, I, O, D>
// where
//     L: Layer<ArrayBase<I, D>, ArrayBase<O, D>>,  // Layer works with input I and output O
//     C: Cost,                    // Cost works with output O
//     I: Data<Elem = f32>,                         // Input data type
//     O: Data<Elem = f32>,                         // Output data type
//     D: Dimension,                                // Dimension of the data
// {
//     layers: Vec<Layer<ArrayBase<I, D>, ArrayBase<O, D>>>,
//     loss_function: C,
// }

// impl<L, C, I, O, D> NeuralNetwork<L, C, I, O, D>
// where
//     L: Layer<ArrayBase<I, D>, ArrayBase<O, D>>,
//     C: Cost,
//     I: Data<Elem = f32>,
//     O: Data<Elem = f32>,
//     D: Dimension,
// {
//     pub fn new(layers: Vec<L>, loss_function: C) -> Self {
//         NeuralNetwork {
//             layers,
//             loss_function,
//         }
//     }

//     // Forward pass with flexible input/output types
//     pub fn forward(&mut self, input: &ArrayBase<I, D>) -> ArrayBase<O, D> {
//         let mut output = input.clone();
//         for layer in &mut self.layers {
//             output = layer.forward(&output);
//         }
//         output
//     }

//     // Training the network
//     pub fn train(&mut self, input: &ArrayBase<I, D>, target: &ArrayBase<O, D>, learning_rate: f32) {
//         let prediction = self.forward(input);
//         let loss = self.loss_function.loss(&prediction, target);
//         let mut grad = self.loss_function.gradient(&prediction, target);

//         for layer in self.layers.iter_mut().rev() {
//             grad = layer.backward(input, &grad, learning_rate);
//         }

//         println!("Loss: {}", loss);
//     }
// }
