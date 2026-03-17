pub mod cpu;
pub use cpu::CPUBackend;

#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "gpu")]
pub use gpu::GPUBackend;

#[cfg(not(feature = "gpu"))]
pub type DefaultBackend = CPUBackend;

#[cfg(feature = "gpu")]
pub type DefaultBackend = GPUBackend;

use ndarray::{Dimension, RemoveAxis};

pub trait Backend: Clone + 'static {
    type Tensor<D: Dimension>: Clone;

    fn zeros<D: Dimension>(shape: D) -> Self::Tensor<D>;
    fn random_uniform<D: Dimension>(shape: D, low: f32, high: f32) -> Self::Tensor<D>;
    fn from_array<D: Dimension>(array: ndarray::Array<f32, D>) -> Self::Tensor<D>;
    fn to_array<D: Dimension>(tensor: &Self::Tensor<D>) -> ndarray::Array<f32, D>;

    fn unary<D: Dimension>(tensor: &Self::Tensor<D>, op: u32) -> Self::Tensor<D>;

    fn add<D: Dimension>(a: &Self::Tensor<D>, b: &Self::Tensor<D>) -> Self::Tensor<D>;
    fn sub<D: Dimension>(a: &Self::Tensor<D>, b: &Self::Tensor<D>) -> Self::Tensor<D>;
    fn mul<D: Dimension>(a: &Self::Tensor<D>, b: &Self::Tensor<D>) -> Self::Tensor<D>;
    fn div<D: Dimension>(a: &Self::Tensor<D>, b: &Self::Tensor<D>) -> Self::Tensor<D>;
    fn scale<D: Dimension>(tensor: &Self::Tensor<D>, scalar: f32) -> Self::Tensor<D>;
    fn scalar_sub<D: Dimension>(scalar: f32, tensor: &Self::Tensor<D>) -> Self::Tensor<D>;
    fn scalar_max<D: Dimension>(tensor: &Self::Tensor<D>, s: f32) -> Self::Tensor<D>;
    fn scalar_min<D: Dimension>(tensor: &Self::Tensor<D>, s: f32) -> Self::Tensor<D>;
    fn clamp<D: Dimension>(tensor: &Self::Tensor<D>, low: f32, high: f32) -> Self::Tensor<D> {
        Self::scalar_min(&Self::scalar_max(tensor, low), high)
    }

    fn mean<D: Dimension>(tensor: &Self::Tensor<D>) -> Option<f32>;
    fn sum_axis<D: Dimension + RemoveAxis>(
        tensor: &Self::Tensor<D>,
        axis: usize,
    ) -> Self::Tensor<D::Smaller>;

    fn matmul<D1: Dimension, D2: Dimension>(
        a: &Self::Tensor<D1>,
        b: &Self::Tensor<D2>,
    ) -> Self::Tensor<D1>;
    fn transpose<D: Dimension>(
        tensor: &Self::Tensor<D>,
        axis1: usize,
        axis2: usize,
    ) -> Self::Tensor<D>;
    fn broadcast_add<D1: Dimension, D2: Dimension>(
        a: &Self::Tensor<D1>,
        b: &Self::Tensor<D2>,
    ) -> Self::Tensor<D1>;

    fn softmax<D: Dimension>(tensor: &Self::Tensor<D>) -> Self::Tensor<D>;
    fn softmax_vjp<D: Dimension>(
        z: &Self::Tensor<D>,
        grad_output: &Self::Tensor<D>,
    ) -> Self::Tensor<D>;

    fn assign<D: Dimension>(dst: &mut Self::Tensor<D>, src: Self::Tensor<D>);
    fn shape<D: Dimension>(tensor: &Self::Tensor<D>) -> Vec<usize>;
    fn len_of<D: Dimension>(tensor: &Self::Tensor<D>, axis: usize) -> usize;
    fn select<D: Dimension + RemoveAxis>(
        tensor: &Self::Tensor<D>,
        axis: usize,
        indices: &[usize],
    ) -> Self::Tensor<D>;

    fn flush() {}
}

pub mod unary_ops {
    pub const TANH: u32 = 0;
    pub const SIGMOID: u32 = 1;
    pub const RELU: u32 = 2;
    pub const TANH_DERIV: u32 = 3;
    pub const SIGMOID_DERIV: u32 = 4;
    pub const RELU_DERIV: u32 = 5;
    pub const EXP: u32 = 6;
    pub const LN: u32 = 7;
    pub const ABS: u32 = 8;
    pub const NEG: u32 = 9;
    pub const SQRT: u32 = 10;
}
