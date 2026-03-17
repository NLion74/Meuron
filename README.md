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
meuron = { version = "0.3.1", features = ["cpu"] }

# GPU backend (requires WebGPU support)
meuron = { version = "0.3.1", features = ["gpu"] }
```

Only one backend can be active at a time.

## Basic Example

```rust
use meuron::{
    NeuralNetwork, NetworkType, Layers,
    DenseLayer, ReLU, Softmax, CrossEntropy, SGD,
};
use meuron::train::TrainOptions;
use ndarray::Array2;

type Net = NeuralNetwork<
    NetworkType![DenseLayer<ReLU>, DenseLayer<ReLU>, DenseLayer<Softmax>],
    CrossEntropy,
>;

fn main() {
    let mut nn: Net = NeuralNetwork::new(
        Layers![
            DenseLayer::new(784, 128, ReLU),
            DenseLayer::new(128, 64,  ReLU),
            DenseLayer::new(64,  10,  Softmax),
        ],
        CrossEntropy,
    );

    nn.train(
        train_data,
        train_labels,
        SGD::new(0.01),
        TrainOptions::new()
            .epochs(25)
            .batch_size(256)
            .validation_split(0.1),
    );

    nn.save("model.bin").unwrap();
    let loaded: Net = NeuralNetwork::load("model.bin", CrossEntropy).unwrap();
}
```

A custom callback receives the epoch, total, training loss, and optional validation loss:

```
TrainOptions::new()
    .epochs(25)
    .callback(|epoch, total, loss, val_loss| {
        println!("{epoch}/{total}  loss={loss:.4}");
        true // return false to stop early
    })
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
cargo run --example mnist-mlp-gpu --release --features gpu
```

For a more advanced UI based example see:

```
cargo run --example mnist-draw --release --features gpu
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
