use crate::backend::Backend;
use ndarray::{Array, ArrayBase, Axis, Dimension, Ix2, OwnedRepr, RemoveAxis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Normal, Uniform};

#[derive(Clone)]
pub struct CPUBackend;

impl Backend for CPUBackend {
    type Tensor<D: Dimension> = ArrayBase<OwnedRepr<f32>, D>;

    fn zeros<D: Dimension>(shape: D) -> Self::Tensor<D> {
        Array::zeros(shape)
    }

    fn random_uniform<D: Dimension>(shape: D, low: f32, high: f32) -> Self::Tensor<D> {
        Array::random(shape, Uniform::new(low, high).unwrap())
    }

    fn random_normal<D: Dimension>(shape: D, mean: f32, std: f32) -> Self::Tensor<D> {
        Array::random(shape, Normal::new(mean, std).unwrap())
    }

    fn from_array<D: Dimension>(array: ndarray::Array<f32, D>) -> Self::Tensor<D> {
        array
    }

    fn to_array<D: Dimension>(tensor: &Self::Tensor<D>) -> ndarray::Array<f32, D> {
        tensor.clone()
    }

    fn unary<D: Dimension>(tensor: &Self::Tensor<D>, op: u32) -> Self::Tensor<D> {
        tensor.mapv(|x| match op {
            0 => x.tanh(),
            1 => 1.0 / (1.0 + (-x).exp()),
            2 => x.max(0.0),
            3 => {
                let t = x.tanh();
                1.0 - t * t
            }
            4 => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            5 => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            6 => x.exp(),
            7 => x.ln(),
            8 => x.abs(),
            9 => -x,
            10 => x.sqrt(),
            _ => panic!("unknown unary op {op}"),
        })
    }

    fn add<D: Dimension>(a: &Self::Tensor<D>, b: &Self::Tensor<D>) -> Self::Tensor<D> {
        a + b
    }
    fn sub<D: Dimension>(a: &Self::Tensor<D>, b: &Self::Tensor<D>) -> Self::Tensor<D> {
        a - b
    }
    fn mul<D: Dimension>(a: &Self::Tensor<D>, b: &Self::Tensor<D>) -> Self::Tensor<D> {
        a * b
    }
    fn div<D: Dimension>(a: &Self::Tensor<D>, b: &Self::Tensor<D>) -> Self::Tensor<D> {
        a / b
    }

    fn scale<D: Dimension>(tensor: &Self::Tensor<D>, scalar: f32) -> Self::Tensor<D> {
        tensor * scalar
    }

    fn scalar_sub<D: Dimension>(scalar: f32, tensor: &Self::Tensor<D>) -> Self::Tensor<D> {
        tensor.mapv(|v| scalar - v)
    }
    fn scalar_max<D: Dimension>(tensor: &Array<f32, D>, s: f32) -> Array<f32, D> {
        tensor.mapv(|x| x.max(s))
    }
    fn scalar_min<D: Dimension>(tensor: &Array<f32, D>, s: f32) -> Array<f32, D> {
        tensor.mapv(|x| x.min(s))
    }

    fn mean<D: Dimension>(tensor: &Self::Tensor<D>) -> Option<f32> {
        tensor.mean()
    }

    fn sum_axis<D: Dimension + RemoveAxis>(
        tensor: &Self::Tensor<D>,
        axis: usize,
    ) -> Self::Tensor<D::Smaller> {
        tensor.sum_axis(Axis(axis))
    }

    fn matmul<D1: Dimension, D2: Dimension>(
        a: &Self::Tensor<D1>,
        b: &Self::Tensor<D2>,
    ) -> Self::Tensor<D1> {
        let a_dyn = a.view().into_dyn();
        let b_dyn = b.view().into_dyn();
        let out = match (a_dyn.ndim(), b_dyn.ndim()) {
            (2, 2) => {
                let a2 = a_dyn.view().into_dimensionality::<Ix2>().unwrap();
                let b2 = b_dyn.view().into_dimensionality::<Ix2>().unwrap();
                a2.dot(&b2).into_dyn()
            }
            (3, 2) => {
                let (batch, m) = (a_dyn.shape()[0], a_dyn.shape()[1]);
                let n = b_dyn.shape()[1];
                let b2 = b_dyn.view().into_dimensionality::<Ix2>().unwrap();
                let mut result = Array::zeros((batch, m, n)).into_dyn();
                for i in 0..batch {
                    let ai = a_dyn
                        .index_axis(Axis(0), i)
                        .into_dimensionality::<Ix2>()
                        .unwrap();
                    result.index_axis_mut(Axis(0), i).assign(&ai.dot(&b2));
                }
                result
            }
            (3, 3) => {
                let (batch, m) = (a_dyn.shape()[0], a_dyn.shape()[1]);
                let n = b_dyn.shape()[2];
                let mut result = Array::zeros((batch, m, n)).into_dyn();
                for i in 0..batch {
                    let ai = a_dyn
                        .index_axis(Axis(0), i)
                        .into_dimensionality::<Ix2>()
                        .unwrap();
                    let bi = b_dyn
                        .index_axis(Axis(0), i)
                        .into_dimensionality::<Ix2>()
                        .unwrap();
                    result.index_axis_mut(Axis(0), i).assign(&ai.dot(&bi));
                }
                result
            }
            (4, 2) => {
                let (b1, b2, m, _k) = (
                    a_dyn.shape()[0],
                    a_dyn.shape()[1],
                    a_dyn.shape()[2],
                    a_dyn.shape()[3],
                );
                let n = b_dyn.shape()[1];
                let bw = b_dyn.view().into_dimensionality::<Ix2>().unwrap();
                let mut out = Array::zeros((b1, b2, m, n)).into_dyn();
                for i in 0..b1 {
                    let ai = a_dyn.index_axis(Axis(0), i);
                    let mut oi = out.index_axis_mut(Axis(0), i);
                    for j in 0..b2 {
                        let aij = ai
                            .index_axis(Axis(0), j)
                            .into_dimensionality::<Ix2>()
                            .unwrap();
                        oi.index_axis_mut(Axis(0), j).assign(&aij.dot(&bw));
                    }
                }
                out
            }
            (4, 4) => {
                let (b1, b2, m) = (a_dyn.shape()[0], a_dyn.shape()[1], a_dyn.shape()[2]);
                let n = b_dyn.shape()[3];
                let mut out = Array::zeros((b1, b2, m, n)).into_dyn();
                for i in 0..b1 {
                    let ai = a_dyn.index_axis(Axis(0), i);
                    let bi = b_dyn.index_axis(Axis(0), i);
                    let mut oi = out.index_axis_mut(Axis(0), i);
                    for j in 0..b2 {
                        let aij = ai
                            .index_axis(Axis(0), j)
                            .into_dimensionality::<Ix2>()
                            .unwrap();
                        let bij = bi
                            .index_axis(Axis(0), j)
                            .into_dimensionality::<Ix2>()
                            .unwrap();
                        oi.index_axis_mut(Axis(0), j).assign(&aij.dot(&bij));
                    }
                }
                out
            }
            _ => panic!(
                "matmul: unsupported shapes {:?} x {:?}",
                a_dyn.shape(),
                b_dyn.shape()
            ),
        };
        out.into_dimensionality::<D1>()
            .expect("matmul output rank must match left operand")
            .to_owned()
    }

    fn transpose<D: Dimension>(
        tensor: &Self::Tensor<D>,
        axis1: usize,
        axis2: usize,
    ) -> Self::Tensor<D> {
        let mut t = tensor.clone().into_dyn();
        t.swap_axes(axis1, axis2);
        t.as_standard_layout()
            .into_dimensionality::<D>()
            .expect("transpose must preserve rank")
            .to_owned()
    }

    fn broadcast_add<D1: Dimension, D2: Dimension>(
        a: &Self::Tensor<D1>,
        b: &Self::Tensor<D2>,
    ) -> Self::Tensor<D1> {
        let a_dyn = a.view().into_dyn().to_owned();
        let b_dyn = b.view().into_dyn().to_owned();
        (a_dyn + b_dyn)
            .into_dimensionality::<D1>()
            .expect("broadcast_add output rank must match left operand")
    }

    fn softmax<D: Dimension>(tensor: &Self::Tensor<D>) -> Self::Tensor<D> {
        let shape = tensor.shape().to_vec();
        let last_dim = shape[shape.len() - 1];
        let batch = shape[..shape.len() - 1].iter().product::<usize>().max(1);
        let x_c = tensor.as_standard_layout();
        let raw = x_c.as_slice().unwrap();
        let mut out = vec![0.0f32; raw.len()];
        for b in 0..batch {
            let s = b * last_dim;
            let row = &raw[s..s + last_dim];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&v| (v - max).exp()).sum();
            for (i, &v) in row.iter().enumerate() {
                out[s + i] = (v - max).exp() / sum;
            }
        }
        Array::from_shape_vec(tensor.raw_dim(), out).unwrap()
    }

    fn softmax_vjp<D: Dimension>(
        z: &Self::Tensor<D>,
        grad_output: &Self::Tensor<D>,
    ) -> Self::Tensor<D> {
        let shape = z.shape().to_vec();
        let last_dim = shape[shape.len() - 1];
        let batch = shape[..shape.len() - 1].iter().product::<usize>().max(1);
        let s = Self::softmax(z);
        let s_c = s.as_standard_layout();
        let s_raw = s_c.as_slice().unwrap();
        let g_c = grad_output.as_standard_layout();
        let g_raw = g_c.as_slice().unwrap();
        let mut out = vec![0.0f32; s_raw.len()];
        for b in 0..batch {
            let start = b * last_dim;
            let sr = &s_raw[start..start + last_dim];
            let gr = &g_raw[start..start + last_dim];
            let dot: f32 = sr.iter().zip(gr).map(|(&si, &gi)| si * gi).sum();
            for i in 0..last_dim {
                out[start + i] = sr[i] * (gr[i] - dot);
            }
        }
        Array::from_shape_vec(z.raw_dim(), out).unwrap()
    }

    fn assign<D: Dimension>(dst: &mut Self::Tensor<D>, src: Self::Tensor<D>) {
        *dst = src;
    }

    fn shape<D: Dimension>(tensor: &Self::Tensor<D>) -> Vec<usize> {
        tensor.shape().to_vec()
    }

    fn len_of<D: Dimension>(tensor: &Self::Tensor<D>, axis: usize) -> usize {
        tensor.len_of(Axis(axis))
    }

    fn select<D: Dimension + RemoveAxis>(
        tensor: &Self::Tensor<D>,
        axis: usize,
        indices: &[usize],
    ) -> Self::Tensor<D> {
        tensor.select(Axis(axis), indices)
    }
}
