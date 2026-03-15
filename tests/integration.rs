#![cfg(feature = "gpu")]

mod common;

use common::assert_close;
use meuron::backend::{unary_ops, Backend, CPUBackend, GPUBackend};
use ndarray::{arr1, arr2};

#[test]
fn clone_after_compute_is_correct() {
    let a  = arr2(&[[1.0_f32, 2.0], [3.0, 4.0]]);
    let b  = arr2(&[[10.0_f32, 20.0], [30.0, 40.0]]);
    let ga = GPUBackend::from_array(a.clone());
    let gb = GPUBackend::from_array(b.clone());

    let sum       = GPUBackend::add(&ga, &gb);
    let sum_clone = sum.clone();
    let cpu_sum   = CPUBackend::add(&a, &b);

    assert_close("clone_orig",  &cpu_sum, &GPUBackend::to_array(&sum),       1e-6);
    assert_close("clone_clone", &cpu_sum, &GPUBackend::to_array(&sum_clone), 1e-6);
}

#[test]
fn multi_step_clone_chain_is_correct() {
    let x = arr2(&[[1.0_f32, -1.0, 2.0], [0.5, 0.0, -0.5]]);
    let w = arr2(&[[0.1_f32, 0.2], [-0.3, 0.4], [0.5, -0.6]]);
    let b = arr1(&[0.1_f32, -0.1]);

    let cpu_z   = CPUBackend::broadcast_add(&CPUBackend::matmul(&x, &w), &b);
    let cpu_act = CPUBackend::unary(&cpu_z, unary_ops::RELU);

    let gx  = GPUBackend::from_array(x);
    let gw  = GPUBackend::from_array(w);
    let gb  = GPUBackend::from_array(b);
    let gpu_z    = GPUBackend::broadcast_add(&GPUBackend::matmul(&gx, &gw), &gb);
    let gpu_z_cl = gpu_z.clone();   // last_z stored by forward()
    let gpu_act  = GPUBackend::unary(&gpu_z, unary_ops::RELU);

    assert_close("z_clone", &cpu_z,   &GPUBackend::to_array(&gpu_z_cl), 1e-5);
    assert_close("act",     &cpu_act, &GPUBackend::to_array(&gpu_act),  1e-5);
}

#[test]
fn dense_forward_parity() {
    use meuron::activation::ReLU;
    use meuron::layer::{DenseLayer, Layer};

    let w = arr2(&[[0.5_f32, -0.3, 0.2], [-0.1, 0.4, 0.6], [0.7, -0.5, 0.1], [0.2, 0.3, -0.4]]);
    let b = arr1(&[0.1_f32, -0.2, 0.3]);
    let x = arr2(&[[1.0_f32, 0.5, -0.3, 0.2], [0.0, 1.0, 0.0, -1.0]]);

    let mut cpu_l = DenseLayer::<ReLU, CPUBackend>::new(4, 3, ReLU);
    CPUBackend::assign(&mut cpu_l.weights, CPUBackend::from_array(w.clone()));
    CPUBackend::assign(&mut cpu_l.biases,  CPUBackend::from_array(b.clone()));

    let mut gpu_l = DenseLayer::<ReLU, GPUBackend>::new(4, 3, ReLU);
    GPUBackend::assign(&mut gpu_l.weights, GPUBackend::from_array(w));
    GPUBackend::assign(&mut gpu_l.biases,  GPUBackend::from_array(b));

    assert_close("dense_forward",
        &cpu_l.forward(&CPUBackend::from_array(x.clone())),
        &GPUBackend::to_array(&gpu_l.forward(&GPUBackend::from_array(x))),
        1e-4);
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
    CPUBackend::assign(&mut cpu_l.biases,  CPUBackend::from_array(b.clone()));

    let mut gpu_l = DenseLayer::<ReLU, GPUBackend>::new(3, 2, ReLU);
    GPUBackend::assign(&mut gpu_l.weights, GPUBackend::from_array(w));
    GPUBackend::assign(&mut gpu_l.biases,  GPUBackend::from_array(b));

    cpu_l.forward(&CPUBackend::from_array(x.clone()));
    gpu_l.forward(&GPUBackend::from_array(x));

    let cpu_dx = cpu_l.backward(&CPUBackend::from_array(g.clone()));
    let gpu_dx = GPUBackend::to_array(&gpu_l.backward(&GPUBackend::from_array(g)));
    assert_close("backward_dx", &cpu_dx, &gpu_dx, 1e-4);

    cpu_l.update(&mut SGD::new(0.1));
    gpu_l.update(&mut SGD::new(0.1));
    assert_close("weights_after_update", &CPUBackend::to_array(&cpu_l.weights), &GPUBackend::to_array(&gpu_l.weights), 1e-4);
    assert_close("biases_after_update",  &CPUBackend::to_array(&cpu_l.biases),  &GPUBackend::to_array(&gpu_l.biases),  1e-4);
}

#[test]
fn sgd_update_parity() {
    use meuron::optimizer::{Optimizer, SGD};

    let w  = arr2(&[[1.0_f32, 2.0], [3.0, 4.0]]);
    let g  = arr2(&[[0.1_f32, -0.2], [0.3, -0.4]]);
    let mut cpu_w = CPUBackend::from_array(w.clone());
    let mut gpu_w = GPUBackend::from_array(w);
    let cpu_g = CPUBackend::from_array(g.clone());
    let gpu_g = GPUBackend::from_array(g);

    <SGD as Optimizer<CPUBackend>>::update_param(&mut SGD::new(0.1), &mut cpu_w, &cpu_g);
    <SGD as Optimizer<GPUBackend>>::update_param(&mut SGD::new(0.1), &mut gpu_w, &gpu_g);

    assert_close("sgd_update",
        &CPUBackend::to_array(&cpu_w),
        &GPUBackend::to_array(&gpu_w),
        1e-6);
}
