#![cfg(feature = "gpu")]

mod common;

use common::{assert_all_finite, assert_close};
use meuron::backend::{unary_ops, Backend, CPUBackend, GPUBackend};
use ndarray::{arr1, arr2, Array, Array2};

fn sample_2d() -> Array2<f32> {
    arr2(&[[0.5, -1.0, 2.0], [3.0, 0.0, -0.2]])
}

fn range_3d(shape: (usize, usize, usize)) -> Array<f32, ndarray::Ix3> {
    let n = shape.0 * shape.1 * shape.2;
    Array::from_shape_vec(shape, (0..n).map(|v| v as f32 / n as f32 - 0.5).collect()).unwrap()
}

#[test]
fn unary_parity() {
    let x  = sample_2d();
    let gx = GPUBackend::from_array(x.clone());
    for &(name, op, eps) in &[
        ("tanh",          unary_ops::TANH,          1e-5_f32),
        ("sigmoid",       unary_ops::SIGMOID,       1e-5),
        ("relu",          unary_ops::RELU,          1e-6),
        ("tanh_deriv",    unary_ops::TANH_DERIV,    1e-5),
        ("sigmoid_deriv", unary_ops::SIGMOID_DERIV, 1e-5),
        ("relu_deriv",    unary_ops::RELU_DERIV,    1e-6),
        ("exp",           unary_ops::EXP,           1e-5),
        ("abs",           unary_ops::ABS,           1e-6),
        ("neg",           unary_ops::NEG,           1e-6),
    ] {
        let cpu = CPUBackend::unary(&x, op);
        let gpu = GPUBackend::to_array(&GPUBackend::unary(&gx, op));
        assert_all_finite(name, &gpu);
        assert_close(name, &cpu, &gpu, eps);
    }
}

#[test]
fn scalar_parity() {
    let x  = sample_2d();
    let gx = GPUBackend::from_array(x.clone());
    macro_rules! check {
        ($name:expr, $cpu:expr, $gpu:expr, $eps:expr) => {
            assert_close($name, &$cpu, &GPUBackend::to_array(&$gpu), $eps);
        };
    }
    check!("scale",      CPUBackend::scale(&x, 2.5),          GPUBackend::scale(&gx, 2.5),          1e-6);
    check!("scale_neg",  CPUBackend::scale(&x, -0.5),         GPUBackend::scale(&gx, -0.5),         1e-6);
    check!("scale_zero", CPUBackend::scale(&x, 0.0),          GPUBackend::scale(&gx, 0.0),          1e-6);
    check!("scalar_sub", CPUBackend::scalar_sub(1.25, &x),    GPUBackend::scalar_sub(1.25, &gx),    1e-6);
    check!("scalar_max", CPUBackend::scalar_max(&x, 0.25),    GPUBackend::scalar_max(&gx, 0.25),    1e-6);
    check!("scalar_min", CPUBackend::scalar_min(&x, 0.25),    GPUBackend::scalar_min(&gx, 0.25),    1e-6);
    check!("clamp",      CPUBackend::clamp(&x, -0.3, 1.0),    GPUBackend::clamp(&gx, -0.3, 1.0),    1e-6);
}

#[test]
fn binop_parity() {
    let a  = arr2(&[[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let b  = arr2(&[[0.5_f32, 1.0, 2.0], [8.0, 2.0, 3.0]]);
    let ga = GPUBackend::from_array(a.clone());
    let gb = GPUBackend::from_array(b.clone());
    assert_close("add", &CPUBackend::add(&a, &b), &GPUBackend::to_array(&GPUBackend::add(&ga, &gb)), 1e-6);
    assert_close("sub", &CPUBackend::sub(&a, &b), &GPUBackend::to_array(&GPUBackend::sub(&ga, &gb)), 1e-6);
    assert_close("mul", &CPUBackend::mul(&a, &b), &GPUBackend::to_array(&GPUBackend::mul(&ga, &gb)), 1e-6);
    assert_close("div", &CPUBackend::div(&a, &b), &GPUBackend::to_array(&GPUBackend::div(&ga, &gb)), 1e-6);
}

#[test]
fn matmul_2d_parity() {
    let a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let b = arr2(&[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);
    assert_close("matmul_2d",
        &CPUBackend::matmul(&a, &b),
        &GPUBackend::to_array(&GPUBackend::matmul(&GPUBackend::from_array(a), &GPUBackend::from_array(b))),
        1e-4);
}

#[test]
fn matmul_identity_parity() {
    let a = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let b = arr2(&[[3.0_f32, 7.0], [-1.0, 2.0]]);
    assert_close("matmul_identity",
        &CPUBackend::matmul(&a, &b),
        &GPUBackend::to_array(&GPUBackend::matmul(&GPUBackend::from_array(a), &GPUBackend::from_array(b))),
        1e-6);
}

#[test]
fn matmul_3d_parity() {
    let a = range_3d((2, 3, 4));
    let b = Array::from_shape_vec((4, 5), (0..20).map(|v| v as f32 / 7.0 - 0.5).collect()).unwrap();
    assert_close("matmul_3d",
        &CPUBackend::matmul(&a, &b),
        &GPUBackend::to_array(&GPUBackend::matmul(&GPUBackend::from_array(a), &GPUBackend::from_array(b))),
        1e-4);
}

#[test]
fn matmul_4d_parity() {
    let a = Array::from_shape_vec((2, 2, 3, 4), (0..48).map(|v| v as f32 / 9.0 - 2.0).collect()).unwrap();
    let b = Array::from_shape_vec((4, 5), (0..20).map(|v| v as f32 / 11.0 - 0.3).collect()).unwrap();
    assert_close("matmul_4d",
        &CPUBackend::matmul(&a, &b),
        &GPUBackend::to_array(&GPUBackend::matmul(&GPUBackend::from_array(a), &GPUBackend::from_array(b))),
        1e-4);
}

#[test]
fn transpose_2d_parity() {
    let x  = sample_2d();
    let gx = GPUBackend::from_array(x.clone());
    assert_close("transpose_2d",
        &CPUBackend::transpose(&x, 0, 1),
        &GPUBackend::to_array(&GPUBackend::transpose(&gx, 0, 1)),
        1e-6);
}

#[test]
fn transpose_3d_parity() {
    let x  = range_3d((2, 3, 4));
    let gx = GPUBackend::from_array(x.clone());
    for (a1, a2) in [(0, 1), (0, 2), (1, 2)] {
        let name = format!("transpose_3d_{a1}_{a2}");
        assert_close(&name,
            &CPUBackend::transpose(&x, a1, a2),
            &GPUBackend::to_array(&GPUBackend::transpose(&gx, a1, a2)),
            1e-6);
    }
}

#[test]
fn broadcast_add_parity() {
    let a = Array::from_shape_vec((3, 4), (0..12).map(|v| v as f32).collect()).unwrap();
    let b = arr1(&[1.0_f32, -2.0, 3.0, -4.0]);
    assert_close("broadcast_add",
        &CPUBackend::broadcast_add(&a, &b),
        &GPUBackend::to_array(&GPUBackend::broadcast_add(&GPUBackend::from_array(a), &GPUBackend::from_array(b))),
        1e-6);
}

#[test]
fn broadcast_add_negative_bias_parity() {
    let a = arr2(&[[1.0_f32, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
    let b = arr1(&[-5.0_f32, 10.0, -3.0]);
    assert_close("broadcast_add_neg",
        &CPUBackend::broadcast_add(&a, &b),
        &GPUBackend::to_array(&GPUBackend::broadcast_add(&GPUBackend::from_array(a), &GPUBackend::from_array(b))),
        1e-6);
}

#[test]
fn softmax_parity() {
    let x  = sample_2d();
    let gx = GPUBackend::from_array(x.clone());
    let cpu = CPUBackend::softmax(&x);
    let gpu = GPUBackend::to_array(&GPUBackend::softmax(&gx));
    assert_all_finite("softmax", &gpu);
    assert_close("softmax", &cpu, &gpu, 1e-5);
    for row in gpu.rows() {
        let s: f32 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "row sum = {s}");
    }
}

#[test]
fn softmax_large_values_parity() {
    let x  = arr2(&[[100.0_f32, 101.0, 102.0], [-100.0, -101.0, -102.0]]);
    let gx = GPUBackend::from_array(x.clone());
    let gpu = GPUBackend::to_array(&GPUBackend::softmax(&gx));
    assert_all_finite("softmax_large", &gpu);
    assert_close("softmax_large", &CPUBackend::softmax(&x), &gpu, 1e-5);
}

#[test]
fn softmax_vjp_parity() {
    let z    = arr2(&[[1.0_f32, 2.0, 3.0], [0.5, -1.0, 2.0]]);
    let grad = arr2(&[[0.1_f32, -0.2, 0.3], [-0.1, 0.5, -0.2]]);
    let cpu  = CPUBackend::softmax_vjp(&z, &grad);
    let gpu  = GPUBackend::to_array(&GPUBackend::softmax_vjp(
        &GPUBackend::from_array(z), &GPUBackend::from_array(grad)));
    assert_all_finite("softmax_vjp", &gpu);
    assert_close("softmax_vjp", &cpu, &gpu, 1e-5);
}

#[test]
fn sum_axis_parity() {
    let x  = range_3d((2, 3, 4));
    let gx = GPUBackend::from_array(x.clone());
    for axis in 0..3 {
        assert_close(&format!("sum_axis_{axis}"),
            &CPUBackend::sum_axis(&x, axis),
            &GPUBackend::to_array(&GPUBackend::sum_axis(&gx, axis)),
            1e-5);
    }
}

#[test]
fn mean_parity() {
    let x  = arr1(&[1.0_f32, 2.0, 3.0, 4.0, -2.0]);
    let gx = GPUBackend::from_array(x.clone());
    let cpu = CPUBackend::mean(&x).unwrap();
    let gpu = GPUBackend::mean(&gx).unwrap();
    assert!((cpu - gpu).abs() < 1e-6, "mean: cpu={cpu} gpu={gpu}");
}

#[test]
fn select_parity() {
    let x  = Array::from_shape_vec((5, 3), (0..15).map(|v| v as f32).collect()).unwrap();
    let gx = GPUBackend::from_array(x.clone());
    let idx = [4usize, 1, 3, 0];
    assert_close("select",
        &CPUBackend::select(&x, 0, &idx),
        &GPUBackend::to_array(&GPUBackend::select(&gx, 0, &idx)),
        1e-6);
}
