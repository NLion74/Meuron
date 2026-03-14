pub mod ndarray_backend;
pub use ndarray_backend::NdarrayBackend;

use ndarray::{Dimension, RemoveAxis};

pub trait Backend: Clone + 'static {
    type Tensor<D: Dimension>: Clone;

    fn zeros<D: Dimension>(shape: D) -> Self::Tensor<D>;
    fn random_uniform<D: Dimension>(shape: D, low: f32, high: f32) -> Self::Tensor<D>;
    fn from_array<D: Dimension>(array: ndarray::Array<f32, D>) -> Self::Tensor<D>;
    fn to_array<D: Dimension>(tensor: &Self::Tensor<D>) -> ndarray::Array<f32, D>;

    fn mapv<D: Dimension>(tensor: &Self::Tensor<D>, f: impl Fn(f32) -> f32) -> Self::Tensor<D>;
    fn add<D: Dimension>(a: &Self::Tensor<D>, b: &Self::Tensor<D>) -> Self::Tensor<D>;
    fn sub<D: Dimension>(a: &Self::Tensor<D>, b: &Self::Tensor<D>) -> Self::Tensor<D>;
    fn mul<D: Dimension>(a: &Self::Tensor<D>, b: &Self::Tensor<D>) -> Self::Tensor<D>;
    fn div<D: Dimension>(a: &Self::Tensor<D>, b: &Self::Tensor<D>) -> Self::Tensor<D>;
    fn scale<D: Dimension>(tensor: &Self::Tensor<D>, scalar: f32) -> Self::Tensor<D>;
    fn scalar_sub<D: Dimension>(scalar: f32, tensor: &Self::Tensor<D>) -> Self::Tensor<D>;

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
}
