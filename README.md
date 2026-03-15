# Meuron

**Meuron** is a modular neural network library written in rust for training simple neural networks.

> Built mainly for personal learning and experimentation, focused on clean, extensible architecture and implementing neural network concepts from scratch.

## Features

- Modular layer system
- Multiple activation functions (ReLU, Sigmoid, Softmax)
- Multiple cost functions (MSE, CrossEntropy, BinaryCrossEntropy)
- Optimizer Support
- Easy to extend with custom layers and activations

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
meuron = { version = "0.2", features = ["cpu"] }
```

## Basic Example

```rust
use meuron::{NeuralNetwork, layer::DenseLayer, activation::ReLU, activation::Softmax, cost::MSE, Layers};
use ndarray::Array2;

fn main() {
    // Create a simple 2-layer network
    let layer1 = DenseLayer::new(784, 128, ReLU);
    let layer2 = DenseLayer::new(128, 10, Softmax);

    let mut nn = NeuralNetwork::new(
        Layers![layer1, layer2],
        MSE,
    );

    // Train the network
    nn.train(&train_data, &train_labels, 0.01, 10, 32);

    // Save the model
    nn.save("model.bin").unwrap();

    // Load later
    let loaded_nn = NeuralNetwork::load("model.bin", MSE).unwrap();
}
```

### Available Components

#### Activations

- ReLU
- Sigmoid
- Softmax
- Tanh

#### Cost Functions

- MSE
- CrossEntropy
- BinaryCrossEntropy

# Optimizers

- SGD

#### Layers

- DenseLayer

## Examples

See the examples/ directory:

```
cargo run --example mnist-mlp-cpu --release
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
