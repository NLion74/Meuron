use ndarray::{ArrayBase, Dimension, OwnedRepr};
use serde::{Deserialize, Serialize};
use crate::activation::Activation;

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Softmax;

impl Softmax {
    fn apply_rowwise<D: Dimension, F>(
        x: &ArrayBase<OwnedRepr<f32>, D>,
        f: F,
    ) -> ArrayBase<OwnedRepr<f32>, D>
    where
        F: Fn(&[f32], &mut [f32]),
    {
        let shape = x.shape().to_vec();
        let ndim = shape.len();
        assert!(ndim >= 1, "Softmax requires at least a 1D tensor");

        let last_dim = shape[ndim - 1];
        let batch: usize = shape[..ndim - 1].iter().product::<usize>().max(1);

        let x_c = x.as_standard_layout();
        let raw = x_c.as_slice().expect("standard layout slice failed");
        let mut output = vec![0.0f32; raw.len()];

        for b in 0..batch {
            let start = b * last_dim;
            f(&raw[start..start + last_dim], &mut output[start..start + last_dim]);
        }

        ndarray::Array::from_shape_vec(x.raw_dim(), output)
            .expect("shape reconstruction failed")
    }
}

impl Activation for Softmax {
    fn activate<D: Dimension>(
        &self,
        x: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> ArrayBase<OwnedRepr<f32>, D> {
        Self::apply_rowwise(x, |row, out| {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&v| (v - max).exp()).sum();
            for (i, &v) in row.iter().enumerate() {
                out[i] = (v - max).exp() / sum;
            }
        })
    }

    // incomplete derivative, only correct for loss functions like cross-entropy
    fn derivative<D: Dimension>(
        &self,
        x: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> ArrayBase<OwnedRepr<f32>, D> {
        let s = self.activate(x);
        &s * &(1.0 - &s)
    }

    // Full correct Vector-Jacobian Product for softmax, needed for non-cross-entropy losses
    fn vjp<D: Dimension>(
        &self,
        z: &ArrayBase<OwnedRepr<f32>, D>,
        grad_output: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> ArrayBase<OwnedRepr<f32>, D> {
        let shape = z.shape().to_vec();
        let ndim = shape.len();
        let last_dim = shape[ndim - 1];
        let batch: usize = shape[..ndim - 1].iter().product::<usize>().max(1);

        let s = self.activate(z);
        let s_c = s.as_standard_layout();
        let s_raw = s_c.as_slice().expect("standard layout slice failed");

        let g_c = grad_output.as_standard_layout();
        let g_raw = g_c.as_slice().expect("standard layout slice failed");

        let mut out = vec![0.0f32; s_raw.len()];

        for b in 0..batch {
            let start = b * last_dim;
            let s_row = &s_raw[start..start + last_dim];
            let g_row = &g_raw[start..start + last_dim];

            let dot: f32 = s_row.iter().zip(g_row.iter()).map(|(&si, &gi)| si * gi).sum();

            for i in 0..last_dim {
                out[start + i] = s_row[i] * (g_row[i] - dot);
            }
        }

        ndarray::Array::from_shape_vec(z.raw_dim(), out)
            .expect("vjp shape reconstruction failed")
    }
}
