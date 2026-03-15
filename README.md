# Meuron

**Meuron** is a modular neural network library written in rust for training simple neural networks.

> Built mainly for personal learning and experimentation, focused on clean, extensible architecture and implementing neural network concepts from scratch.

## Features

- Modular layer system
- CPU and GPU backends (via ndarray/wgpu)
- Multiple activation functions (ReLU, Sigmoid, Tanh, Softmax)
- Multiple cost functions (MSE, CrossEntropy, BinaryCrossEntropy)
- Optimizer support (SGD)
- Model serialization and deserialization
- Easy to extend with custom layers and activations

## Quick Start

Add to your `Cargo.toml`:

```toml
# CPU backend
meuron = { version = "0.2", features = ["cpu"] }

# GPU backend (requires WebGPU support)
meuron = { version = "0.2", features = ["gpu"] }
```

Only one backend can be active at a time.

## Basic Example

```rust
use meuron::{
    NeuralNetwork, NetworkType, Layers,
    layer::DenseLayer,
    activation::{ReLU, Softmax},
    cost::CrossEntropy,
    optimizer::SGD,
};
use ndarray::Array2;

fn main() {
    let layer1 = DenseLayer::new(784, 128, ReLU);
    let layer2 = DenseLayer::new(128, 64,  ReLU);
    let layer3 = DenseLayer::new(64,  10,  Softmax);

    type Net = NeuralNetwork<
        NetworkType!(
            DenseLayer<ReLU>,
            DenseLayer<ReLU>,
            DenseLayer<Softmax>
        ),
        CrossEntropy,
    >;

    let mut nn: Net = NeuralNetwork::new(
        Layers![layer1, layer2, layer3],
        CrossEntropy,
    );

    // Train the network
    nn.train(train_data, train_labels, SGD::new(0.01), 10, 32);

    // Save the model
    nn.save("model.bin").unwrap();

    // Load later
    let loaded: Net = NeuralNetwork::load("model.bin", CrossEntropy).unwrap();
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

#### Optimizers

- SGD

#### Layers

- DenseLayer

#### Macros

- `Layers![l1, l2, ...]` - compose layers into a Sequential Chain
- `NetworkType!(L1, L2, ...)` - produce a matching type of Sequential Chain of annotation

## Examples

See the examples/ directory:

```
cargo run --example mnist-mlp-cpu --release
```

```
cargo run --example mnist-mlp-gpu --release --no-default-features --features gpu
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
