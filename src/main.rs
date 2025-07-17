use ndarray::{Array2, Axis, s};
use std::io::{self, Write, Read};
use std::path::PathBuf;
use std::fs::File;
use meuron::layer::{Layer, DenseLayer};
use meuron::activation::{ReLU, Sigmoid, Softmax};
use meuron::cost::{Cost, MSE, CrossEntropy, BinaryCrossEntropy};
use meuron::NeuralNetwork;

fn read_u32_from_file(file: &mut File) -> Result<u32, io::Error> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn load_mnist_data(images_path: PathBuf, labels_path: PathBuf) -> Result<(Array2<f32>, Array2<f32>), io::Error> {
    let mut image_file = File::open(images_path).expect("Failed to open file");
    let mut label_file = File::open(labels_path).expect("Failed to open file");
    
    // Read header information
    let _magic_images = read_u32_from_file(&mut image_file).expect("Failed to read header information");
    let num_images = read_u32_from_file(&mut image_file).expect("Failed to read header information");
    let num_rows = read_u32_from_file(&mut image_file).expect("Failed to read header information");
    let num_cols = read_u32_from_file(&mut image_file).expect("Failed to read header information");

    let _magic_labels = read_u32_from_file(&mut label_file).expect("Failed to read header information");
    let num_labels = read_u32_from_file(&mut label_file).expect("Failed to read header information");

    assert_eq!(num_images, num_labels, "Number of images and labels do not match");

    let mut image_data = vec![0u8; (num_images * num_rows * num_cols) as usize];
    image_file.read_exact(&mut image_data)?;

    let images = Array2::from_shape_vec(
        (num_images as usize, (num_rows * num_cols) as usize),
        image_data.into_iter().map(|x| x as f32 / 255.0).collect()
    ).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    // Read label data
    let mut label_data = vec![0u8; num_labels as usize];
    label_file.read_exact(&mut label_data)?;

    let labels = Array2::from_shape_vec(
        (num_labels as usize, 10),
        label_data.into_iter().map(|label| {
            let mut one_hot = vec![0.0; 10];
            one_hot[label as usize] = 1.0;
            one_hot
        }).flatten().collect()
    ).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    Ok((images, labels))
}

fn main() {
    let train_images_path = PathBuf::from("./train-images.idx3-ubyte");
    let train_labels_path = PathBuf::from("./train-labels.idx1-ubyte");

    let (images, labels) = match load_mnist_data(train_images_path, train_labels_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error loading MNIST data: {}", e);
            return;
        }
    };

    println!("Loaded MNIST data: {:?}", images.shape());
    println!("shape: {:?}", labels.shape());

    let output_size = 10;       // 10 classes for digits 0-9
    let input_size = 28 * 28;   // 28x28 pixel images

    let dense_layer_1 = DenseLayer::new(input_size, 128, Sigmoid);
    let dense_layer_2 = DenseLayer::new(128, output_size, Sigmoid);

    let cost = MSE;

    let mut nn = NeuralNetwork::new(
        vec![dense_layer_1, dense_layer_2],
        &cost,
    );

    // Train the network
    let learning_rate = 0.01;
    let num_epochs = 10;
    let batch_size = 1;

    nn.train(&images, &labels, learning_rate, num_epochs, batch_size);

    let test_images_path = PathBuf::from("./t10k-images.idx3-ubyte");
    let test_labels_path = PathBuf::from("./t10k-labels.idx1-ubyte");

    let (test_images, test_labels) = match load_mnist_data(test_images_path, test_labels_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error loading MNIST data: {}", e);
            return;
        }
    };

    let accuarcy = nn.accuracy(&test_images, &test_labels);

    println!("Test accuracy: {}", accuarcy);
}