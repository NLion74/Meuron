use bytemuck::{Pod, Zeroable};
use wgpu;

pub const TENSOR_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BinopParams {
    pub size: u32,
    pub op: u32,
    pub pad0: u32,
    pub pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ScalarParams {
    pub size: u32,
    pub op: u32,
    pub scalar: f32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct UnaryParams {
    pub size: u32,
    pub op: u32,
    pub pad0: u32,
    pub pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MatmulDims {
    pub batch: u32,
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub b_shared: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BatchDims {
    pub batch: u32,
    pub last_dim: u32,
    pub pad0: u32,
    pub pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BroadcastDims {
    pub total: u32,
    pub last_dim: u32,
    pub pad0: u32,
    pub pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SumAxisParams {
    pub outer: u32,
    pub axis_len: u32,
    pub inner: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ReduceSumParams {
    pub size: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GatherParams {
    pub outer: u32,
    pub axis_len: u32,
    pub inner: u32,
    pub n_sel: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TransposeDims {
    pub ndim: u32,
    pub total: u32,
    pub axis1: u32,
    pub axis2: u32,
    pub out_shape: [u32; 6],
    pub out_stride: [u32; 6],
    pub in_stride: [u32; 6],
    pub pad: [u32; 2],
}
