use flate2::read::GzDecoder;
use meuron::activation::{ReLU, Softmax};
use meuron::cost::CrossEntropy;
use meuron::initializer::{HeNormal, XavierUniform, Zeros};
use meuron::layer::DenseLayer;
use meuron::metric::classification::accuracy;
use meuron::optimizer::SGDMomentum;
use meuron::train::TrainOptions;
use meuron::{Layers, NetworkType, NeuralNetwork};
use ndarray::Array2;
use std::fs::{self, File};
use std::io::{self, BufWriter, Read};
use std::path::{Path, PathBuf};

type MnistNetwork = NeuralNetwork<
    NetworkType![DenseLayer<ReLU>, DenseLayer<ReLU>, DenseLayer<Softmax>],
    CrossEntropy,
>;

const MIRROR: &str = "https://systemds.apache.org/assets/datasets/mnist";

const FILES: &[&str] = &[
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
];

fn ensure_mnist(dir: &Path) -> io::Result<()> {
    fs::create_dir_all(dir)?;

    for &gz_name in FILES {
        let dest = dir.join(gz_name.strip_suffix(".gz").unwrap());

        if dest.exists() {
            continue;
        }

        let url = format!("{}/{}", MIRROR, gz_name);
        println!("Downloading {}...", gz_name);

        let response = ureq::get(&url)
            .call()
            .map_err(|e| io::Error::other(e.to_string()))?;

        let mut body = response.into_body();
        let mut gz = GzDecoder::new(body.as_reader());
        let mut out = BufWriter::new(File::create(&dest)?);
        io::copy(&mut gz, &mut out)?;

        println!("  → saved to {}", dest.display());
    }

    Ok(())
}

fn read_u32(file: &mut File) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn load_mnist(dir: &Path, prefix: &str) -> io::Result<(Array2<f32>, Array2<f32>)> {
    ensure_mnist(dir)?;

    let mut img_f = File::open(dir.join(format!("{}-images-idx3-ubyte", prefix)))?;
    let mut lbl_f = File::open(dir.join(format!("{}-labels-idx1-ubyte", prefix)))?;

    let _ = read_u32(&mut img_f)?;
    let n = read_u32(&mut img_f)? as usize;
    let rows = read_u32(&mut img_f)? as usize;
    let cols = read_u32(&mut img_f)? as usize;

    let _ = read_u32(&mut lbl_f)?;
    let n_labels = read_u32(&mut lbl_f)? as usize;
    assert_eq!(n, n_labels);

    let mut raw_images = vec![0u8; n * rows * cols];
    img_f.read_exact(&mut raw_images)?;

    let mut raw_labels = vec![0u8; n];
    lbl_f.read_exact(&mut raw_labels)?;

    let images = Array2::from_shape_vec(
        (n, rows * cols),
        raw_images.into_iter().map(|x| x as f32 / 255.0).collect(),
    )
    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let labels = Array2::from_shape_vec(
        (n, 10),
        raw_labels
            .into_iter()
            .flat_map(|l| {
                let mut oh = [0.0f32; 10];
                oh[l as usize] = 1.0;
                oh
            })
            .collect(),
    )
    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    Ok((images, labels))
}

fn main() {
    let data_dir = PathBuf::from("./examples/mnist/data");
    let model_path = "./examples/mnist/mnist_model_gpu.bin";

    let mut nn: MnistNetwork = if PathBuf::from(model_path).exists() {
        println!("Loading existing model...");
        NeuralNetwork::load(model_path, CrossEntropy).expect("Failed to load model")
    } else {
        println!("Creating new model...");
        NeuralNetwork::new(
            Layers![
                DenseLayer::new(28 * 28, 500, ReLU, HeNormal, Zeros),
                DenseLayer::new(500, 128, ReLU, HeNormal, Zeros),
                DenseLayer::new(128, 10, Softmax, HeNormal, XavierUniform)
            ],
            CrossEntropy,
        )
    };

    let (images, labels) = load_mnist(&data_dir, "train").expect("Failed to load training data");
    println!("Loaded {} training images", images.shape()[0]);

    println!("\nTraining...");
    nn.train(
        images,
        labels,
        SGDMomentum::new(0.01, 0.9),
        TrainOptions::new()
            .epochs(25)
            .batch_size(1024)
            .validation_split(0.1),
    );

    println!("\nSaving model to {}...", model_path);
    nn.save(model_path).expect("Failed to save model");

    let (test_images, test_labels) =
        load_mnist(&data_dir, "t10k").expect("Failed to load test data");

    let acc = accuracy(&mut nn, test_images, test_labels);
    println!("\nTest accuracy: {:.2}%", acc * 100.0);
}
