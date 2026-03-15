mod shaders;

use std::sync::{Arc, OnceLock};
use bytemuck::{cast_slice, Pod, Zeroable};
use ndarray::{Array, Axis, Dimension, Ix2, RemoveAxis, IntoDimension};
use ndarray_rand::RandomExt;
use wgpu::util::DeviceExt;
use wgpu::PollType;
use crate::backend::Backend;

static GPU_CTX: OnceLock<Arc<GpuContext>> = OnceLock::new();

pub struct GpuContext {
    pub device:    wgpu::Device,
    pub queue:     wgpu::Queue,
    pub pipelines: GpuPipelines,
}

impl GpuContext {
    pub fn global() -> Arc<Self> {
        GPU_CTX.get_or_init(|| Arc::new(pollster::block_on(Self::init()))).clone()
    }

    async fn init() -> Self {
        let instance = wgpu::Instance::default();
        let adapter  = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     None,
                force_fallback_adapter: false,
            })
            .await
            .expect("No GPU adapter found - install drivers or use features = [\"cpu\"]");

        let info = adapter.get_info();
        println!("meuron GPU backend: {} ({:?})", info.name, info.backend);

        let desc = wgpu::DeviceDescriptor {
            label: Some("meuron"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        };
        let (device, queue) = adapter.request_device(&desc).await.unwrap();

        let pipelines = GpuPipelines::new(&device);
        GpuContext { device, queue, pipelines }
    }
}

pub struct GpuPipelines {
    pub binop:         wgpu::ComputePipeline,
    pub scalar:        wgpu::ComputePipeline,
    pub matmul:        wgpu::ComputePipeline,
    pub softmax:       wgpu::ComputePipeline,
    pub softmax_vjp:   wgpu::ComputePipeline,
    pub broadcast_add: wgpu::ComputePipeline,
    pub transpose:  wgpu::ComputePipeline,
}

impl GpuPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let compile = |src: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label:  None,
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        GpuPipelines {
            binop:         compile(shaders::BINOP),
            scalar:        compile(shaders::SCALAR),
            matmul:        compile(shaders::MATMUL),
            softmax:       compile(shaders::SOFTMAX),
            softmax_vjp:   compile(shaders::SOFTMAX_VJP),
            broadcast_add: compile(shaders::BROADCAST_ADD),
            transpose:  compile(shaders::TRANSPOSE),
        }
    }
}

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
struct BinopParams   { size: u32, op: u32, pad0: u32, pad1: u32 }

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
struct ScalarParams  { size: u32, op: u32, scalar: f32, pad: u32 }

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
struct MatmulDims    { m: u32, k: u32, n: u32, pad: u32 }

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
struct BatchDims     { batch: u32, last_dim: u32, pad0: u32, pad1: u32 }

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
struct BroadcastDims { total: u32, last_dim: u32, pad0: u32, pad1: u32 }

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
struct TransposeDims { rows: u32, cols: u32, pad0: u32, pad1: u32 }

const TENSOR_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

pub struct GpuTensor<D: Dimension> {
    pub buffer: Arc<wgpu::Buffer>,
    pub shape:  D,
    pub size:   usize,
    ctx:        Arc<GpuContext>,
}

impl<D: Dimension> GpuTensor<D> {
    fn upload(arr: ndarray::Array<f32, D>, ctx: Arc<GpuContext>) -> Self {
        let contig = arr.as_standard_layout();
        let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    None,
            contents: cast_slice(contig.as_slice().unwrap()),
            usage:    TENSOR_USAGE,
        });
        let shape = contig.raw_dim();
        let size  = contig.len();
        GpuTensor { buffer: Arc::new(buffer), shape, size, ctx }
    }

    pub fn download(&self) -> ndarray::Array<f32, D> {
        let byte_size = (self.size * 4) as u64;
        let staging   = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label:              None,
            size:               byte_size,
            usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, byte_size);
        self.ctx.queue.submit([enc.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());

        loop {
            #[allow(unused_must_use)]
            {
                let _ = self.ctx.device.poll(wgpu::PollType::Wait { submission_index: wgpu::SubmissionIndex::default(), timeout: None });
            }

            if let Ok(result) = rx.try_recv() {
                result.unwrap();
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        let data: Vec<f32> = cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
        ndarray::Array::from_shape_vec(self.shape.clone(), data).unwrap()
    }

    fn alloc_like(&self) -> wgpu::Buffer {
        self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label:              None,
            size:               (self.size * 4) as u64,
            usage:              TENSOR_USAGE,
            mapped_at_creation: false,
        })
    }
}

impl<D: Dimension> Clone for GpuTensor<D> {
    fn clone(&self) -> Self {
        let new_buf = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label:              None,
            size:               self.buffer.size(),
            usage:              TENSOR_USAGE,
            mapped_at_creation: false,
        });
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.buffer, 0, &new_buf, 0, self.buffer.size());
        self.ctx.queue.submit([enc.finish()]);
        GpuTensor { buffer: Arc::new(new_buf), shape: self.shape.clone(), size: self.size, ctx: self.ctx.clone() }
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
        Ok(GpuTensor::upload(arr, GpuContext::global()))
    }
}

fn uniform_buf<T: Pod>(ctx: &GpuContext, data: &T) -> wgpu::Buffer {
    ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    None,
        contents: bytemuck::bytes_of(data),
        usage:    wgpu::BufferUsages::UNIFORM,
    })
}

fn dispatch_1d(
    ctx:      &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    entries:  &[wgpu::BindGroupEntry<'_>],
    size:     u32,
) {
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   None,
        layout:  &pipeline.get_bind_group_layout(0),
        entries,
    });
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((size + 255) / 256, 1, 1);
    }
    ctx.queue.submit([enc.finish()]);
}

fn dispatch_2d(
    ctx:      &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    entries:  &[wgpu::BindGroupEntry<'_>],
    x:        u32,
    y:        u32,
) {
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   None,
        layout:  &pipeline.get_bind_group_layout(0),
        entries,
    });
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((x + 7) / 8, (y + 7) / 8, 1);
    }
    ctx.queue.submit([enc.finish()]);
}

fn binop<D: Dimension>(a: &GpuTensor<D>, b: &GpuTensor<D>, op: u32) -> GpuTensor<D> {
    let ctx     = a.ctx.clone();
    let size    = a.size as u32;
    let out_buf = a.alloc_like();
    let ub      = uniform_buf(&ctx, &BinopParams { size, op, pad0: 0, pad1: 0 });
    dispatch_1d(&ctx, &ctx.pipelines.binop, &[
        wgpu::BindGroupEntry { binding: 0, resource: a.buffer.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: b.buffer.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: ub.as_entire_binding() },
    ], size);
    GpuTensor { buffer: Arc::new(out_buf), shape: a.shape.clone(), size: a.size, ctx }
}

fn scalar_op<D: Dimension>(tensor: &GpuTensor<D>, op: u32, scalar: f32) -> GpuTensor<D> {
    let ctx     = tensor.ctx.clone();
    let size    = tensor.size as u32;
    let out_buf = tensor.alloc_like();
    let ub      = uniform_buf(&ctx, &ScalarParams { size, op, scalar, pad: 0 });
    dispatch_1d(&ctx, &ctx.pipelines.scalar, &[
        wgpu::BindGroupEntry { binding: 0, resource: tensor.buffer.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: ub.as_entire_binding() },
    ], size);
    GpuTensor { buffer: Arc::new(out_buf), shape: tensor.shape.clone(), size: tensor.size, ctx }
}

#[derive(Clone)]
pub struct WgpuBackend;

impl Backend for WgpuBackend {
    type Tensor<D: Dimension> = GpuTensor<D>;

    fn zeros<D: Dimension>(shape: D) -> GpuTensor<D> {
        let ctx    = GpuContext::global();
        let size   = shape.size();
        let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    None,
            contents: &vec![0u8; size * 4],
            usage:    TENSOR_USAGE,
        });
        GpuTensor { buffer: Arc::new(buffer), shape, size, ctx }
    }

    fn random_uniform<D: Dimension>(shape: D, low: f32, high: f32) -> GpuTensor<D> {
        let arr = ndarray::Array::random(
            shape,
            ndarray_rand::rand_distr::Uniform::new(low, high).unwrap(),
        );
        GpuTensor::upload(arr, GpuContext::global())
    }

    fn from_array<D: Dimension>(array: ndarray::Array<f32, D>) -> GpuTensor<D> {
        GpuTensor::upload(array, GpuContext::global())
    }

    fn to_array<D: Dimension>(tensor: &GpuTensor<D>) -> ndarray::Array<f32, D> {
        tensor.download()
    }

    fn add<D: Dimension>(a: &GpuTensor<D>, b: &GpuTensor<D>) -> GpuTensor<D> { binop(a, b, 0) }
    fn sub<D: Dimension>(a: &GpuTensor<D>, b: &GpuTensor<D>) -> GpuTensor<D> { binop(a, b, 1) }
    fn mul<D: Dimension>(a: &GpuTensor<D>, b: &GpuTensor<D>) -> GpuTensor<D> { binop(a, b, 2) }
    fn div<D: Dimension>(a: &GpuTensor<D>, b: &GpuTensor<D>) -> GpuTensor<D> { binop(a, b, 3) }

    fn scale<D: Dimension>(tensor: &GpuTensor<D>, scalar: f32) -> GpuTensor<D> {
        scalar_op(tensor, 0, scalar)
    }
    fn scalar_sub<D: Dimension>(scalar: f32, tensor: &GpuTensor<D>) -> GpuTensor<D> {
        scalar_op(tensor, 1, scalar)
    }

    fn mapv<D: Dimension>(tensor: &GpuTensor<D>, f: impl Fn(f32) -> f32) -> GpuTensor<D> {
        let arr = tensor.download().mapv(f);
        GpuTensor::upload(arr, tensor.ctx.clone())
    }

    fn mean<D: Dimension>(tensor: &GpuTensor<D>) -> Option<f32> {
        tensor.download().mean()
    }

    fn sum_axis<D: Dimension + RemoveAxis>(tensor: &GpuTensor<D>, axis: usize) -> GpuTensor<D::Smaller> {
        let result = tensor.download().sum_axis(Axis(axis));
        GpuTensor::upload(result, tensor.ctx.clone())
    }

    fn matmul<D1: Dimension, D2: Dimension>(a: &GpuTensor<D1>, b: &GpuTensor<D2>) -> GpuTensor<D1> {
        let ctx     = a.ctx.clone();
        let a_shape = a.shape.slice();
        let b_shape = b.shape.slice();

        if a_shape.len() == 2 && b_shape.len() == 2 {
            let (m, k, n) = (a_shape[0] as u32, a_shape[1] as u32, b_shape[1] as u32);
            let out_buf   = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label:              None,
                size:               (m * n * 4) as u64,
                usage:              TENSOR_USAGE,
                mapped_at_creation: false,
            });
            let ub = uniform_buf(&ctx, &MatmulDims { m, k, n, pad: 0 });
            dispatch_2d(&ctx, &ctx.pipelines.matmul, &[
                wgpu::BindGroupEntry { binding: 0, resource: a.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: ub.as_entire_binding() },
            ], m, n);
            let out_shape = ndarray::Ix2(m as usize, n as usize)
                .into_dimension::<D1>()
                .expect("matmul 2D output must match D1");
            GpuTensor { buffer: Arc::new(out_buf), shape: out_shape, size: (m * n) as usize, ctx }
        } else {
            let a_arr = a.download();
            let b_arr = b.download();
            let out   = crate::backend::CPUBackend::matmul(&a_arr, &b_arr);
            GpuTensor::upload(out, ctx)
        }
    }

    fn transpose<D: Dimension>(tensor: &GpuTensor<D>, axis1: usize, axis2: usize) -> GpuTensor<D> {
        let ctx   = tensor.ctx.clone();
        let shape = tensor.shape.slice();

        if shape.len() == 2 && axis1 == 0 && axis2 == 1 {
            let (rows, cols) = (shape[0] as u32, shape[1] as u32);
            let out_buf      = tensor.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label:              None,
                size:               tensor.buffer.size(),
                usage:              TENSOR_USAGE,
                mapped_at_creation: false,
            });
            let ub = uniform_buf(&ctx, &TransposeDims { rows, cols, pad0: 0, pad1: 0 });
            dispatch_2d(&ctx, &ctx.pipelines.transpose, &[
                wgpu::BindGroupEntry { binding: 0, resource: tensor.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ub.as_entire_binding() },
            ], rows, cols);
            let mut new_shape = shape.to_vec();
            new_shape.swap(axis1, axis2);
            let out_shape = ndarray::IxDyn(&new_shape)
                .into_dimension::<D>()
                .expect("transpose must preserve rank");
            GpuTensor { buffer: Arc::new(out_buf), shape: out_shape, size: tensor.size, ctx }
        } else {
            let mut arr = tensor.download().into_dyn();
            arr.swap_axes(axis1, axis2);
            let result = arr.as_standard_layout().into_dimension::<D>().unwrap().to_owned();
            GpuTensor::upload(result, ctx)
        }
    }

    fn broadcast_add<D1: Dimension, D2: Dimension>(a: &GpuTensor<D1>, b: &GpuTensor<D2>) -> GpuTensor<D1> {
        let ctx      = a.ctx.clone();
        let total    = a.size as u32;
        let last_dim = b.size as u32;
        let out_buf  = a.alloc_like();
        let ub       = uniform_buf(&ctx, &BroadcastDims { total, last_dim, pad0: 0, pad1: 0 });
        dispatch_1d(&ctx, &ctx.pipelines.broadcast_add, &[
            wgpu::BindGroupEntry { binding: 0, resource: a.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: ub.as_entire_binding() },
        ], total);
        GpuTensor { buffer: Arc::new(out_buf), shape: a.shape.clone(), size: a.size, ctx }
    }

    fn softmax<D: Dimension>(tensor: &GpuTensor<D>) -> GpuTensor<D> {
        let ctx      = tensor.ctx.clone();
        let shape    = tensor.shape.slice();
        let last_dim = *shape.last().unwrap() as u32;
        let batch    = tensor.size as u32 / last_dim;
        let out_buf  = tensor.alloc_like();
        let ub       = uniform_buf(&ctx, &BatchDims { batch, last_dim, pad0: 0, pad1: 0 });
        dispatch_1d(&ctx, &ctx.pipelines.softmax, &[
            wgpu::BindGroupEntry { binding: 0, resource: tensor.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: ub.as_entire_binding() },
        ], batch);
        GpuTensor { buffer: Arc::new(out_buf), shape: tensor.shape.clone(), size: tensor.size, ctx }
    }

    fn softmax_vjp<D: Dimension>(z: &GpuTensor<D>, grad: &GpuTensor<D>) -> GpuTensor<D> {
        let ctx      = z.ctx.clone();
        let shape    = z.shape.slice();
        let last_dim = *shape.last().unwrap() as u32;
        let batch    = z.size as u32 / last_dim;
        let out_buf  = z.alloc_like();
        let ub       = uniform_buf(&ctx, &BatchDims { batch, last_dim, pad0: 0, pad1: 0 });
        dispatch_1d(&ctx, &ctx.pipelines.softmax_vjp, &[
            wgpu::BindGroupEntry { binding: 0, resource: z.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: grad.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: ub.as_entire_binding() },
        ], batch);
        GpuTensor { buffer: Arc::new(out_buf), shape: z.shape.clone(), size: z.size, ctx }
    }

    fn assign<D: Dimension>(dst: &mut GpuTensor<D>, src: GpuTensor<D>) { *dst = src; }

    fn shape<D: Dimension>(tensor: &GpuTensor<D>) -> Vec<usize> {
        tensor.shape.slice().to_vec()
    }

    fn len_of<D: Dimension>(tensor: &GpuTensor<D>, axis: usize) -> usize {
        tensor.shape.slice()[axis]
    }

    fn select<D: Dimension + RemoveAxis>(tensor: &GpuTensor<D>, axis: usize, indices: &[usize]) -> GpuTensor<D> {
        let result = tensor.download().select(Axis(axis), indices);
        GpuTensor::upload(result, tensor.ctx.clone())
    }
}
