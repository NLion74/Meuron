use super::context::GpuContext;
use super::params::TENSOR_USAGE;
use bytemuck::cast_slice;
use ndarray::Dimension;
use std::sync::Arc;
use wgpu;
use wgpu::util::DeviceExt;

pub struct GpuTensor<D: Dimension> {
    pub buffer: Arc<wgpu::Buffer>,
    pub shape: D,
    pub size: usize,
    pub(super) ctx: Arc<GpuContext>,
}

impl<D: Dimension> GpuTensor<D> {
    pub fn upload(arr: ndarray::Array<f32, D>, ctx: Arc<GpuContext>) -> Self {
        let contig = arr.as_standard_layout();
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: cast_slice(contig.as_slice().unwrap()),
                usage: TENSOR_USAGE,
            });
        let shape = contig.raw_dim();
        let size = contig.len();
        GpuTensor {
            buffer: Arc::new(buffer),
            shape,
            size,
            ctx,
        }
    }

    pub fn download(&self) -> ndarray::Array<f32, D> {
        self.ctx.flush();

        let byte_size = (self.size * 4) as u64;
        let staging = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label:              None,
            size:               byte_size,
            usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, byte_size);
        let si = self.ctx.queue.submit([enc.finish()]);

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.ctx.device.poll(wgpu::PollType::Wait {
            submission_index: Some(si),
            timeout:          None,
        });

        let data = cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        ndarray::Array::from_shape_vec(self.shape.clone(), data).unwrap()
    }


    pub fn alloc_like(&self) -> wgpu::Buffer {
        self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (self.size * 4) as u64,
            usage: TENSOR_USAGE,
            mapped_at_creation: false,
        })
    }

    pub fn download_raw(ctx: &GpuContext, buf: &wgpu::Buffer, count: usize) -> Vec<f32> {
        let byte_size = (count * 4) as u64;
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: byte_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, byte_size);
        let si = ctx.queue.submit([enc.finish()]);

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = ctx.device.poll(wgpu::PollType::Wait {
            submission_index: Some(si),
            timeout:          None,
        });

        let data = cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        data
    }
}

impl<D: Dimension> Clone for GpuTensor<D> {
    fn clone(&self) -> Self {
        self.ctx.flush();

        let new_buf = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: self.buffer.size(),
            usage: TENSOR_USAGE,
            mapped_at_creation: false,
        });
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.buffer, 0, &new_buf, 0, self.buffer.size());
        self.ctx.queue.submit([enc.finish()]);
        GpuTensor {
            buffer: Arc::new(new_buf),
            shape: self.shape.clone(),
            size: self.size,
            ctx: self.ctx.clone(),
        }
    }
}

impl<D: Dimension + serde::Serialize> serde::Serialize for GpuTensor<D> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.download().serialize(s)
    }
}
impl<'de, D: Dimension + serde::Deserialize<'de>> serde::Deserialize<'de> for GpuTensor<D> {
    fn deserialize<De: serde::Deserializer<'de>>(d: De) -> Result<Self, De::Error> {
        let arr = ndarray::Array::<f32, D>::deserialize(d)?;
        Ok(GpuTensor::upload(arr, super::context::GpuContext::global()))
    }
}

impl<D: Dimension> From<ndarray::Array<f32, D>> for GpuTensor<D> {
    fn from(arr: ndarray::Array<f32, D>) -> Self {
        GpuTensor::upload(arr, GpuContext::global())
    }
}