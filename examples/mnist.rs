use meuron::{NeuralNetwork, Layers};
use meuron::activation::{ReLU, Softmax};
use meuron::cost::CrossEntropy;
use meuron::layer::{DenseLayer, Sequential};  // ← import Sequential for annotation
use meuron::metric::classification::accuracy;
use ndarray::Array2;
use std::fs::File;
use std::io::{self, Read};
use std::path::PathBuf;

type MnistNetwork = NeuralNetwork<
    Sequential<DenseLayer<ReLU>, DenseLayer<Softmax>>,
    CrossEntropy,
>;

fn read_u32_from_file(file: &mut File) -> Result<u32, io::Error> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn load_mnist_data(
    images_path: PathBuf,
    labels_path: PathBuf,
) -> Result<(Array2<f32>, Array2<f32>), io::Error> {
    let mut image_file = File::open(images_path)?;
    let mut label_file = File::open(labels_path)?;

    let _magic_images = read_u32_from_file(&mut image_file)?;
    let num_images = read_u32_from_file(&mut image_file)?;
    let num_rows = read_u32_from_file(&mut image_file)?;
    let num_cols = read_u32_from_file(&mut image_file)?;

    let _magic_labels = read_u32_from_file(&mut label_file)?;
    let num_labels = read_u32_from_file(&mut label_file)?;

    assert_eq!(num_images, num_labels);

    let mut image_data = vec![0u8; (num_images * num_rows * num_cols) as usize];
    image_file.read_exact(&mut image_data)?;

    let images = Array2::from_shape_vec(
        (num_images as usize, (num_rows * num_cols) as usize),
        image_data.into_iter().map(|x| x as f32 / 255.0).collect(),
    )
    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let mut label_data = vec![0u8; num_labels as usize];
    label_file.read_exact(&mut label_data)?;

    let labels = Array2::from_shape_vec(
        (num_labels as usize, 10),
        label_data
            .into_iter()
            .flat_map(|label| {
                let mut one_hot = vec![0.0f32; 10];
                one_hot[label as usize] = 1.0;
                one_hot
            })
            .collect(),
    )
    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    Ok((images, labels))
}

fn main() {
    let model_path = "mnist_model.bin";

    let mut nn: MnistNetwork = if PathBuf::from(model_path).exists() {
        println!("Loading existing model...");
        NeuralNetwork::load(model_path, CrossEntropy).expect("Failed to load model")
    } else {
        println!("Creating new model...");
        let dense_layer_1 = DenseLayer::new(28 * 28, 128, ReLU);
        let dense_layer_2 = DenseLayer::new(128, 10, Softmax);
        NeuralNetwork::new(Layers![dense_layer_1, dense_layer_2], CrossEntropy)
    };

    let (images, labels) = match load_mnist_data(
        PathBuf::from("./train-images.idx3-ubyte"),
        PathBuf::from("./train-labels.idx1-ubyte"),
    ) {
        Ok(data) => data,
        Err(e) => { eprintln!("Error loading MNIST training data: {}", e); return; }
    };

    println!("Loaded {} training images", images.shape()[0]);
    println!("\nTraining with batch size 32...");
    nn.train(&images, &labels, 0.01, 10, 32);

    println!("\nSaving model to {}...", model_path);
    nn.save(model_path).expect("Failed to save model");

    let (test_images, test_labels) = match load_mnist_data(
        PathBuf::from("./t10k-images.idx3-ubyte"),
        PathBuf::from("./t10k-labels.idx1-ubyte"),
    ) {
        Ok(data) => data,
        Err(e) => { eprintln!("Error loading MNIST test data: {}", e); return; }
    };

    let acc = accuracy(&mut nn, &test_images, &test_labels);
    println!("\nTest accuracy: {:.2}%", acc * 100.0);
}
