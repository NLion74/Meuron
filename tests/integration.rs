#![cfg(feature = "gpu")]

mod common;

use common::assert_close;
use meuron::activation::{ReLU, Softmax};
use meuron::backend::{Backend, CPUBackend, GPUBackend, unary_ops};
use meuron::cost::{Cost, CrossEntropy};
use meuron::layer::{DenseLayer, Layer};
use meuron::optimizer::SGD;
use meuron::{Layers, NetworkType, NeuralNetwork};
use ndarray::{arr1, arr2};

#[test]
fn clone_after_compute_is_correct() {
    let a = arr2(&[[1.0_f32, 2.0], [3.0, 4.0]]);
    let b = arr2(&[[10.0_f32, 20.0], [30.0, 40.0]]);
    let ga = GPUBackend::from_array(a.clone());
    let gb = GPUBackend::from_array(b.clone());

    let sum = GPUBackend::add(&ga, &gb);
    let sum_clone = sum.clone();
    let cpu_sum = CPUBackend::add(&a, &b);

    assert_close("clone_orig", &cpu_sum, &GPUBackend::to_array(&sum), 1e-6);
    assert_close(
        "clone_clone",
        &cpu_sum,
        &GPUBackend::to_array(&sum_clone),
        1e-6,
    );
}

#[test]
fn multi_step_clone_chain_is_correct() {
    let x = arr2(&[[1.0_f32, -1.0, 2.0], [0.5, 0.0, -0.5]]);
    let w = arr2(&[[0.1_f32, 0.2], [-0.3, 0.4], [0.5, -0.6]]);
    let b = arr1(&[0.1_f32, -0.1]);

    let cpu_z = CPUBackend::broadcast_add(&CPUBackend::matmul(&x, &w), &b);
    let cpu_act = CPUBackend::unary(&cpu_z, unary_ops::RELU);

    let gx = GPUBackend::from_array(x);
    let gw = GPUBackend::from_array(w);
    let gb = GPUBackend::from_array(b);
    let gpu_z = GPUBackend::broadcast_add(&GPUBackend::matmul(&gx, &gw), &gb);
    let gpu_z_cl = gpu_z.clone(); // last_z stored by forward()
    let gpu_act = GPUBackend::unary(&gpu_z, unary_ops::RELU);

    assert_close("z_clone", &cpu_z, &GPUBackend::to_array(&gpu_z_cl), 1e-5);
    assert_close("act", &cpu_act, &GPUBackend::to_array(&gpu_act), 1e-5);
}

#[test]
fn dense_forward_parity() {
    use meuron::activation::ReLU;
    use meuron::layer::{DenseLayer, Layer};

    let w = arr2(&[
        [0.5_f32, -0.3, 0.2],
        [-0.1, 0.4, 0.6],
        [0.7, -0.5, 0.1],
        [0.2, 0.3, -0.4],
    ]);
    let b = arr1(&[0.1_f32, -0.2, 0.3]);
    let x = arr2(&[[1.0_f32, 0.5, -0.3, 0.2], [0.0, 1.0, 0.0, -1.0]]);

    let mut cpu_l = DenseLayer::<ReLU, CPUBackend>::new(4, 3, ReLU);
    CPUBackend::assign(&mut cpu_l.weights, CPUBackend::from_array(w.clone()));
    CPUBackend::assign(&mut cpu_l.biases, CPUBackend::from_array(b.clone()));

    let mut gpu_l = DenseLayer::<ReLU, GPUBackend>::new(4, 3, ReLU);
    GPUBackend::assign(&mut gpu_l.weights, GPUBackend::from_array(w));
    GPUBackend::assign(&mut gpu_l.biases, GPUBackend::from_array(b));

    assert_close(
        "dense_forward",
        &cpu_l.forward(&CPUBackend::from_array(x.clone())),
        &GPUBackend::to_array(&gpu_l.forward(&GPUBackend::from_array(x))),
        1e-4,
    );
}

#[test]
fn dense_backward_parity() {
    use meuron::activation::ReLU;
    use meuron::layer::{DenseLayer, Layer};
    use meuron::optimizer::SGD;

    let w = arr2(&[[0.5_f32, -0.3], [-0.1, 0.4], [0.7, -0.5]]);
    let b = arr1(&[0.1_f32, -0.2]);
    let x = arr2(&[[1.0_f32, 0.5, -0.3], [0.0, 1.0, 0.0]]);
    let g = arr2(&[[0.3_f32, -0.1], [-0.2, 0.4]]);

    let mut cpu_l = DenseLayer::<ReLU, CPUBackend>::new(3, 2, ReLU);
    CPUBackend::assign(&mut cpu_l.weights, CPUBackend::from_array(w.clone()));
    CPUBackend::assign(&mut cpu_l.biases, CPUBackend::from_array(b.clone()));

    let mut gpu_l = DenseLayer::<ReLU, GPUBackend>::new(3, 2, ReLU);
    GPUBackend::assign(&mut gpu_l.weights, GPUBackend::from_array(w));
    GPUBackend::assign(&mut gpu_l.biases, GPUBackend::from_array(b));

    cpu_l.forward(&CPUBackend::from_array(x.clone()));
    gpu_l.forward(&GPUBackend::from_array(x));

    let cpu_dx = cpu_l.backward(&CPUBackend::from_array(g.clone()));
    let gpu_dx = GPUBackend::to_array(&gpu_l.backward(&GPUBackend::from_array(g)));
    assert_close("backward_dx", &cpu_dx, &gpu_dx, 1e-4);

    cpu_l.update(&mut SGD::new(0.1));
    gpu_l.update(&mut SGD::new(0.1));
    assert_close(
        "weights_after_update",
        &CPUBackend::to_array(&cpu_l.weights),
        &GPUBackend::to_array(&gpu_l.weights),
        1e-4,
    );
    assert_close(
        "biases_after_update",
        &CPUBackend::to_array(&cpu_l.biases),
        &GPUBackend::to_array(&gpu_l.biases),
        1e-4,
    );
}

#[test]
fn sgd_update_parity() {
    use meuron::optimizer::{Optimizer, SGD};

    let w = arr2(&[[1.0_f32, 2.0], [3.0, 4.0]]);
    let g = arr2(&[[0.1_f32, -0.2], [0.3, -0.4]]);
    let mut cpu_w = CPUBackend::from_array(w.clone());
    let mut gpu_w = GPUBackend::from_array(w);
    let cpu_g = CPUBackend::from_array(g.clone());
    let gpu_g = GPUBackend::from_array(g);

    <SGD as Optimizer<CPUBackend>>::update_param(&mut SGD::new(0.1), &mut cpu_w, &cpu_g);
    <SGD as Optimizer<GPUBackend>>::update_param(&mut SGD::new(0.1), &mut gpu_w, &gpu_g);

    assert_close(
        "sgd_update",
        &CPUBackend::to_array(&cpu_w),
        &GPUBackend::to_array(&gpu_w),
        1e-6,
    );
}

type TwoLayerCpu = NeuralNetwork<
    NetworkType![DenseLayer<ReLU, CPUBackend>, DenseLayer<Softmax, CPUBackend>],
    CrossEntropy,
    CPUBackend,
>;
type TwoLayerGpu = NeuralNetwork<
    NetworkType![DenseLayer<ReLU, GPUBackend>, DenseLayer<Softmax, GPUBackend>],
    CrossEntropy,
    GPUBackend,
>;

fn cpu_gpu_pair() -> (TwoLayerCpu, TwoLayerGpu) {
    let cpu = NeuralNetwork::new(
        Layers![
            DenseLayer::<ReLU, CPUBackend>::new(4, 5, ReLU),
            DenseLayer::<Softmax, CPUBackend>::new(5, 3, Softmax)
        ],
        CrossEntropy,
    );
    let mut gpu = NeuralNetwork::new(
        Layers![
            DenseLayer::<ReLU, GPUBackend>::new(4, 5, ReLU),
            DenseLayer::<Softmax, GPUBackend>::new(5, 3, Softmax)
        ],
        CrossEntropy,
    );

    GPUBackend::assign(
        &mut gpu.layers.layer1.weights,
        GPUBackend::from_array(CPUBackend::to_array(&cpu.layers.layer1.weights)),
    );
    GPUBackend::assign(
        &mut gpu.layers.layer1.biases,
        GPUBackend::from_array(CPUBackend::to_array(&cpu.layers.layer1.biases)),
    );
    GPUBackend::assign(
        &mut gpu.layers.layer2.weights,
        GPUBackend::from_array(CPUBackend::to_array(&cpu.layers.layer2.weights)),
    );
    GPUBackend::assign(
        &mut gpu.layers.layer2.biases,
        GPUBackend::from_array(CPUBackend::to_array(&cpu.layers.layer2.biases)),
    );

    (cpu, gpu)
}

fn tiny_inputs() -> ndarray::Array2<f32> {
    arr2(&[
        [1.0_f32, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
    ])
}

fn tiny_targets() -> ndarray::Array2<f32> {
    arr2(&[
        [1.0_f32, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
}

#[test]
fn two_layer_forward_backward_update_parity() {
    let (mut cpu, mut gpu) = cpu_gpu_pair();
    let x = tiny_inputs();
    let y = tiny_targets();

    let cpu_out = cpu.forward(&CPUBackend::from_array(x.clone()));
    let gpu_out_t = gpu.forward(&GPUBackend::from_array(x.clone()));
    let gpu_out = GPUBackend::to_array(&gpu_out_t);
    assert_close(
        "two_layer_forward",
        &CPUBackend::to_array(&cpu_out),
        &gpu_out,
        1e-4,
    );

    let cpu_grad = <CrossEntropy as Cost<CPUBackend>>::gradient(
        &cpu.cost,
        &cpu_out,
        &CPUBackend::from_array(y.clone()),
    );
    let gpu_grad = <CrossEntropy as Cost<GPUBackend>>::gradient(
        &gpu.cost,
        &gpu_out_t,
        &GPUBackend::from_array(y.clone()),
    );
    let cpu_dx = cpu.backward(&cpu_grad);
    let gpu_dx = gpu.backward(&gpu_grad);
    assert_close(
        "two_layer_backward_dx",
        &CPUBackend::to_array(&cpu_dx),
        &GPUBackend::to_array(&gpu_dx),
        1e-4,
    );

    cpu.layers.update(&mut SGD::new(0.05));
    gpu.layers.update(&mut SGD::new(0.05));

    assert_close(
        "two_layer_w1_after_update",
        &CPUBackend::to_array(&cpu.layers.layer1.weights),
        &GPUBackend::to_array(&gpu.layers.layer1.weights),
        1e-4,
    );
    assert_close(
        "two_layer_b1_after_update",
        &CPUBackend::to_array(&cpu.layers.layer1.biases),
        &GPUBackend::to_array(&gpu.layers.layer1.biases),
        1e-4,
    );
    assert_close(
        "two_layer_w2_after_update",
        &CPUBackend::to_array(&cpu.layers.layer2.weights),
        &GPUBackend::to_array(&gpu.layers.layer2.weights),
        1e-4,
    );
    assert_close(
        "two_layer_b2_after_update",
        &CPUBackend::to_array(&cpu.layers.layer2.biases),
        &GPUBackend::to_array(&gpu.layers.layer2.biases),
        1e-4,
    );
}

#[test]
fn two_layer_multi_step_loss_trajectory_parity() {
    let (mut cpu, mut gpu) = cpu_gpu_pair();
    let x = tiny_inputs();
    let y = tiny_targets();

    let x_cpu = CPUBackend::from_array(x.clone());
    let y_cpu = CPUBackend::from_array(y.clone());
    let x_gpu = GPUBackend::from_array(x.clone());
    let y_gpu = GPUBackend::from_array(y.clone());

    let mut cpu_losses = Vec::new();
    let mut gpu_losses = Vec::new();

    for _ in 0..5 {
        let cpu_out = cpu.forward(&x_cpu);
        let gpu_out = gpu.forward(&x_gpu);

        cpu_losses.push(<CrossEntropy as Cost<CPUBackend>>::loss(
            &cpu.cost, &cpu_out, &y_cpu,
        ));
        gpu_losses.push(<CrossEntropy as Cost<GPUBackend>>::loss(
            &gpu.cost, &gpu_out, &y_gpu,
        ));

        let cpu_grad = <CrossEntropy as Cost<CPUBackend>>::gradient(&cpu.cost, &cpu_out, &y_cpu);
        let gpu_grad = <CrossEntropy as Cost<GPUBackend>>::gradient(&gpu.cost, &gpu_out, &y_gpu);

        cpu.backward(&cpu_grad);
        gpu.backward(&gpu_grad);
        cpu.layers.update(&mut SGD::new(0.05));
        gpu.layers.update(&mut SGD::new(0.05));
        GPUBackend::flush();
    }

    for i in 0..cpu_losses.len() {
        let d = (cpu_losses[i] - gpu_losses[i]).abs();
        assert!(
            d <= 2e-4,
            "loss trajectory mismatch at step {i}: cpu={}, gpu={}, diff={d}",
            cpu_losses[i],
            gpu_losses[i]
        );
    }
}
