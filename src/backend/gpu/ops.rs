use bytemuck::{Pod, bytes_of};
use ndarray::Dimension;
use std::sync::Arc;
use wgpu::util::DeviceExt;

use super::context::GpuContext;
use super::params::{BinopParams, ScalarParams, UnaryParams};
use super::tensor::GpuTensor;

pub fn uniform_buf<T: Pod>(ctx: &Arc<GpuContext>, data: &T) -> wgpu::Buffer {
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytes_of(data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
}

pub fn storage_ro_buf<T: Pod>(ctx: &Arc<GpuContext>, data: &T) -> wgpu::Buffer {
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytes_of(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
}

fn make_bind_group(
    ctx: &Arc<GpuContext>,
    pipeline: &wgpu::ComputePipeline,
    entries: &[wgpu::BindGroupEntry],
) -> wgpu::BindGroup {
    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries,
    })
}

pub fn dispatch_1d(
    ctx: &Arc<GpuContext>,
    pipeline: &wgpu::ComputePipeline,
    entries: &[wgpu::BindGroupEntry],
    count: u32,
) {
    let bg = make_bind_group(ctx, pipeline, entries);
    ctx.with_encoder(|enc| {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(count.div_ceil(256), 1, 1);
    });
}

pub fn dispatch_3d(
    ctx: &Arc<GpuContext>,
    pipeline: &wgpu::ComputePipeline,
    entries: &[wgpu::BindGroupEntry],
    x: u32,
    y: u32,
    z: u32,
) {
    let bg = make_bind_group(ctx, pipeline, entries);
    ctx.with_encoder(|enc| {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(x.div_ceil(8), y.div_ceil(8), z);
    });
}

pub fn binop<D: Dimension>(a: &GpuTensor<D>, b: &GpuTensor<D>, op: u32) -> GpuTensor<D> {
    let ctx = a.ctx.clone();
    let size = a.size as u32;
    let out_buf = a.alloc_like();
    let ub = uniform_buf(
        &ctx,
        &BinopParams {
            size,
            op,
            pad0: 0,
            pad1: 0,
        },
    );
    dispatch_1d(
        &ctx,
        &ctx.pipelines.binop,
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
        size,
    );
    GpuTensor {
        buffer: Arc::new(out_buf),
        shape: a.shape.clone(),
        size: a.size,
        ctx,
    }
}

pub fn scalar_op<D: Dimension>(tensor: &GpuTensor<D>, op: u32, scalar: f32) -> GpuTensor<D> {
    let ctx = tensor.ctx.clone();
    let size = tensor.size as u32;
    let out_buf = tensor.alloc_like();
    let ub = uniform_buf(
        &ctx,
        &ScalarParams {
            size,
            op,
            scalar,
            pad: 0,
        },
    );
    dispatch_1d(
        &ctx,
        &ctx.pipelines.scalar,
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
        size,
    );
    GpuTensor {
        buffer: Arc::new(out_buf),
        shape: tensor.shape.clone(),
        size: tensor.size,
        ctx,
    }
}

pub fn unary_op<D: Dimension>(tensor: &GpuTensor<D>, op: u32) -> GpuTensor<D> {
    let ctx = tensor.ctx.clone();
    let size = tensor.size as u32;
    let out_buf = tensor.alloc_like();
    let ub = uniform_buf(
        &ctx,
        &UnaryParams {
            size,
            op,
            pad0: 0,
            pad1: 0,
        },
    );
    dispatch_1d(
        &ctx,
        &ctx.pipelines.unary,
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
        size,
    );
    GpuTensor {
        buffer: Arc::new(out_buf),
        shape: tensor.shape.clone(),
        size: tensor.size,
        ctx,
    }
}
