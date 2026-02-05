# Meuron

A modular rust written library for training simple Neuronal Networks

## Features

- Modular layer system
- Multiple activation functions (ReLU, Sigmoid, Softmax)
- Multiple cost functions (MSE, CrossEntropy, BinaryCrossEntropy)
- Easy to extend with custom layers and activations

## Usage

```rust
use meuron::{NeuralNetwork, layer::DenseLayer, activation::Sigmoid, cost::MSE};

let layer1 = DenseLayer::new(784, 128, Sigmoid);
let layer2 = DenseLayer::new(128, 10, Sigmoid);

let mut nn = NeuralNetwork::new(
    vec![Box::new(layer1), Box::new(layer2)],
    Box::new(MSE),
);

nn.train(&train_data, &train_labels, 0.01, 10);
```

## Example

Run the MNIST example:

```
cargo run --example mnist
```

## Commands to run:

```
# Build the library
cargo build

# Run MNIST example
cargo run --example mnist
```
