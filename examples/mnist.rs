use meuron::NeuralNetwork;
use meuron::activation::Sigmoid;
use meuron::cost::MSE;
use meuron::layer::DenseLayer;
use ndarray::Array2;
use std::fs::File;
use std::io::{self, Read};
use std::path::PathBuf;

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
                let mut one_hot = vec![0.0; 10];
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

    let mut nn = if PathBuf::from(model_path).exists() {
        println!("Loading existing model...");
        NeuralNetwork::load(model_path, MSE).expect("Failed to load model")
    } else {
        println!("Creating new model...");
        let output_size = 10;
        let input_size = 28 * 28;

        let dense_layer_1 = DenseLayer::new(input_size, 128, Sigmoid);
        let dense_layer_2 = DenseLayer::new(128, output_size, Sigmoid);

        NeuralNetwork::new(vec![dense_layer_1, dense_layer_2], MSE)
    };

    let train_images_path = PathBuf::from("./train-images.idx3-ubyte");
    let train_labels_path = PathBuf::from("./train-labels.idx1-ubyte");

    let (images, labels) = match load_mnist_data(train_images_path, train_labels_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error loading MNIST data: {}", e);
            return;
        }
    };

    println!("Loaded {} training images", images.shape()[0]);

    let learning_rate = 0.01;
    let num_epochs = 10;
    let batch_size = 32;

    println!("\nTraining with batch size {}...", batch_size);
    nn.train(&images, &labels, learning_rate, num_epochs, batch_size);

    println!("\nSaving model to {}...", model_path);
    nn.save(model_path).expect("Failed to save model");

    let test_images_path = PathBuf::from("./t10k-images.idx3-ubyte");
    let test_labels_path = PathBuf::from("./t10k-labels.idx1-ubyte");

    let (test_images, test_labels) = match load_mnist_data(test_images_path, test_labels_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error loading test data: {}", e);
            return;
        }
    };

    let accuracy = nn.accuracy(&test_images, &test_labels);
    println!("\nTest accuracy: {:.2}%", accuracy * 100.0);
}
