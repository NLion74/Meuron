use ndarray::{Axis, Dimension, RemoveAxis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Normal, Uniform};
use std::sync::Arc;
use wgpu::util::DeviceExt;

use super::context::GpuContext;
use super::ops::{
    binop, dispatch_1d, dispatch_3d, scalar_op, storage_ro_buf, unary_op, uniform_buf,
};
use super::params::*;
use super::tensor::GpuTensor;
use crate::backend::Backend;

static GPU_CTX: std::sync::OnceLock<Arc<GpuContext>> = std::sync::OnceLock::new();

impl GpuContext {
    pub fn global() -> Arc<Self> {
        GPU_CTX
            .get_or_init(|| Arc::new(pollster::block_on(GpuContext::init())))
            .clone()
    }
}

#[derive(Clone)]
pub struct GPUBackend;

impl Backend for GPUBackend {
    type Tensor<D: Dimension> = GpuTensor<D>;

    fn zeros<D: Dimension>(shape: D) -> GpuTensor<D> {
        let ctx = GpuContext::global();
        let size = shape.size();
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &vec![0u8; size * 4],
                usage: TENSOR_USAGE,
            });
        GpuTensor {
            buffer: Arc::new(buffer),
            shape,
            size,
            ctx,
        }
    }

    fn random_uniform<D: Dimension>(shape: D, low: f32, high: f32) -> GpuTensor<D> {
        let arr = ndarray::Array::random(shape, Uniform::new(low, high).unwrap());
        GpuTensor::upload(arr, GpuContext::global())
    }

    fn random_normal<D: Dimension>(shape: D, mean: f32, std: f32) -> GpuTensor<D> {
        let arr = ndarray::Array::random(shape, Normal::new(mean, std).unwrap());
        GpuTensor::upload(arr, GpuContext::global())
    }

    fn from_array<D: Dimension>(array: ndarray::Array<f32, D>) -> GpuTensor<D> {
        GpuTensor::upload(array, GpuContext::global())
    }

    fn to_array<D: Dimension>(tensor: &GpuTensor<D>) -> ndarray::Array<f32, D> {
        tensor.download()
    }

    fn add<D: Dimension>(a: &GpuTensor<D>, b: &GpuTensor<D>) -> GpuTensor<D> {
        binop(a, b, 0)
    }
    fn sub<D: Dimension>(a: &GpuTensor<D>, b: &GpuTensor<D>) -> GpuTensor<D> {
        binop(a, b, 1)
    }
    fn mul<D: Dimension>(a: &GpuTensor<D>, b: &GpuTensor<D>) -> GpuTensor<D> {
        binop(a, b, 2)
    }
    fn div<D: Dimension>(a: &GpuTensor<D>, b: &GpuTensor<D>) -> GpuTensor<D> {
        binop(a, b, 3)
    }

    fn scale<D: Dimension>(tensor: &GpuTensor<D>, scalar: f32) -> GpuTensor<D> {
        scalar_op(tensor, 0, scalar)
    }
    fn scalar_sub<D: Dimension>(scalar: f32, tensor: &GpuTensor<D>) -> GpuTensor<D> {
        scalar_op(tensor, 1, scalar)
    }
    fn scalar_max<D: Dimension>(tensor: &GpuTensor<D>, s: f32) -> GpuTensor<D> {
        scalar_op(tensor, 2, s)
    }
    fn scalar_min<D: Dimension>(tensor: &GpuTensor<D>, s: f32) -> GpuTensor<D> {
        scalar_op(tensor, 3, s)
    }

    fn unary<D: Dimension>(tensor: &GpuTensor<D>, op: u32) -> GpuTensor<D> {
        unary_op(tensor, op)
    }

    /* CPU Fallbakcks START*/
    fn mean<D: Dimension>(tensor: &GpuTensor<D>) -> Option<f32> {
        if tensor.size == 0 {
            return None;
        }
        let arr = tensor.download();
        Some(arr.sum() / tensor.size as f32)
    }
    fn sum_axis<D: Dimension + RemoveAxis>(
        tensor: &GpuTensor<D>,
        axis: usize,
    ) -> GpuTensor<D::Smaller> {
        let arr = tensor.download();
        let result = arr.sum_axis(Axis(axis));
        GpuTensor::upload(result, tensor.ctx.clone())
    }
    fn select<D: Dimension + RemoveAxis>(
        tensor: &GpuTensor<D>,
        axis: usize,
        indices: &[usize],
    ) -> GpuTensor<D> {
        let arr = tensor.download();
        let owned: Vec<_> = indices
            .iter()
            .map(|&i| arr.index_axis(Axis(axis), i).to_owned())
            .collect();
        let views: Vec<_> = owned.iter().map(|s| s.view()).collect();
        let out = ndarray::stack(Axis(axis), &views)
            .unwrap()
            .into_dimensionality::<D>()
            .unwrap();
        GpuTensor::upload(out, tensor.ctx.clone())
    }
    /* CPU Fallbakcks END*/

    fn matmul<D1: Dimension, D2: Dimension>(a: &GpuTensor<D1>, b: &GpuTensor<D2>) -> GpuTensor<D1> {
        let ctx = a.ctx.clone();
        let a_s = a.shape.slice();
        let b_s = b.shape.slice();
        let a_ndim = a_s.len();
        let b_ndim = b_s.len();

        assert!(
            (2..=4).contains(&a_ndim) && (b_ndim == 2 || b_ndim == a_ndim),
            "matmul: unsupported rank combination {:?} × {:?}",
            a_s,
            b_s
        );

        let m = a_s[a_ndim - 2] as u32;
        let k = a_s[a_ndim - 1] as u32;
        let n = b_s[b_ndim - 1] as u32;
        let total_batch = a_s[..a_ndim - 2]
            .iter()
            .map(|&d| d as u32)
            .product::<u32>()
            .max(1);
        let b_shared = if b_ndim == 2 { 1u32 } else { 0u32 };
        let out_size = (total_batch * m * n) as usize;

        let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (out_size * 4) as u64,
            usage: TENSOR_USAGE,
            mapped_at_creation: false,
        });
        let ub = uniform_buf(
            &ctx,
            &MatmulDims {
                batch: total_batch,
                m,
                k,
                n,
                b_shared,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            },
        );
        dispatch_3d(
            &ctx,
            &ctx.pipelines.matmul,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ub.as_entire_binding(),
                },
            ],
            n,
            m,
            total_batch,
        );

        let mut out_shape = a.shape.clone();
        out_shape[a_ndim - 1] = n as usize;
        GpuTensor {
            buffer: Arc::new(out_buf),
            shape: out_shape,
            size: out_size,
            ctx,
        }
    }

    fn transpose<D: Dimension>(tensor: &GpuTensor<D>, axis1: usize, axis2: usize) -> GpuTensor<D> {
        let ctx = tensor.ctx.clone();
        let ndim = tensor.shape.ndim();
        assert!(
            ndim <= 6,
            "GPU transpose supports up to 6 dimensions (got {})",
            ndim
        );
        assert!(axis1 < ndim && axis2 < ndim, "axis out of bounds");

        let in_shape = tensor.shape.slice();
        let row_major = |shape: &[usize]| -> [u32; 6] {
            let mut s = [0u32; 6];
            let mut acc = 1u32;
            for i in (0..shape.len()).rev() {
                s[i] = acc;
                acc *= shape[i] as u32;
            }
            s
        };

        let mut out_shape_vec = in_shape.to_vec();
        out_shape_vec.swap(axis1, axis2);
        let mut out_shape_arr = [1u32; 6];
        for (i, &d) in out_shape_vec.iter().enumerate() {
            out_shape_arr[i] = d as u32;
        }

        let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: tensor.buffer.size(),
            usage: TENSOR_USAGE,
            mapped_at_creation: false,
        });
        let pb = storage_ro_buf(
            &ctx,
            &TransposeDims {
                ndim: ndim as u32,
                total: tensor.size as u32,
                axis1: axis1 as u32,
                axis2: axis2 as u32,
                out_shape: out_shape_arr,
                out_stride: row_major(&out_shape_vec),
                in_stride: row_major(in_shape),
                pad: [0u32; 2],
            },
        );
        dispatch_1d(
            &ctx,
            &ctx.pipelines.transpose,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tensor.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pb.as_entire_binding(),
                },
            ],
            tensor.size as u32,
        );

        let mut out_shape = tensor.shape.clone();
        for (i, &d) in out_shape_vec.iter().enumerate() {
            out_shape[i] = d;
        }
        GpuTensor {
            buffer: Arc::new(out_buf),
            shape: out_shape,
            size: tensor.size,
            ctx,
        }
    }

    fn broadcast_add<D1: Dimension, D2: Dimension>(
        a: &GpuTensor<D1>,
        b: &GpuTensor<D2>,
    ) -> GpuTensor<D1> {
        let ctx = a.ctx.clone();
        let total = a.size as u32;
        let last_dim = b.size as u32;
        let out_buf = a.alloc_like();
        let ub = uniform_buf(
            &ctx,
            &BroadcastDims {
                total,
                last_dim,
                pad0: 0,
                pad1: 0,
            },
        );
        dispatch_1d(
            &ctx,
            &ctx.pipelines.broadcast_add,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ub.as_entire_binding(),
                },
            ],
            total,
        );
        GpuTensor {
            buffer: Arc::new(out_buf),
            shape: a.shape.clone(),
            size: a.size,
            ctx,
        }
    }

    fn softmax<D: Dimension>(tensor: &GpuTensor<D>) -> GpuTensor<D> {
        let ctx = tensor.ctx.clone();
        let shape = tensor.shape.slice();
        let last_dim = *shape.last().unwrap() as u32;
        let batch = tensor.size as u32 / last_dim;
        let out_buf = tensor.alloc_like();
        let ub = uniform_buf(
            &ctx,
            &BatchDims {
                batch,
                last_dim,
                pad0: 0,
                pad1: 0,
            },
        );
        dispatch_1d(
            &ctx,
            &ctx.pipelines.softmax,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tensor.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ub.as_entire_binding(),
                },
            ],
            batch,
        );
        GpuTensor {
            buffer: Arc::new(out_buf),
            shape: tensor.shape.clone(),
            size: tensor.size,
            ctx,
        }
    }

    fn softmax_vjp<D: Dimension>(z: &GpuTensor<D>, grad: &GpuTensor<D>) -> GpuTensor<D> {
        let ctx = z.ctx.clone();
        let shape = z.shape.slice();
        let last_dim = *shape.last().unwrap() as u32;
        let batch = z.size as u32 / last_dim;
        let out_buf = z.alloc_like();
        let ub = uniform_buf(
            &ctx,
            &BatchDims {
                batch,
                last_dim,
                pad0: 0,
                pad1: 0,
            },
        );
        dispatch_1d(
            &ctx,
            &ctx.pipelines.softmax_vjp,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: z.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ub.as_entire_binding(),
                },
            ],
            batch,
        );
        GpuTensor {
            buffer: Arc::new(out_buf),
            shape: z.shape.clone(),
            size: z.size,
            ctx,
        }
    }

    fn assign<D: Dimension>(dst: &mut GpuTensor<D>, src: GpuTensor<D>) {
        *dst = src;
    }

    fn shape<D: Dimension>(tensor: &GpuTensor<D>) -> Vec<usize> {
        tensor.shape.slice().to_vec()
    }

    fn len_of<D: Dimension>(tensor: &GpuTensor<D>, axis: usize) -> usize {
        tensor.shape.slice()[axis]
    }

    fn flush() {
        GpuContext::global().flush();
    }
}
