use super::shaders;
use std::sync::Mutex;
use wgpu;

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipelines: GpuPipelines,
    pub encoder: Mutex<Option<wgpu::CommandEncoder>>,
}

impl GpuContext {
    pub async fn init() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("No GPU adapter found");

        let info = adapter.get_info();
        println!("meuron GPU backend: {} ({:?})", info.name, info.backend);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("meuron"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .unwrap();

        let pipelines = GpuPipelines::new(&device);
        GpuContext {
            device,
            queue,
            pipelines,
            encoder: Mutex::new(None),
        }
    }

    pub fn with_encoder<F: FnOnce(&mut wgpu::CommandEncoder)>(&self, f: F) {
        let mut guard = self.encoder.lock().unwrap();
        let enc =
            guard.get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
        f(enc);
    }

    pub fn flush(&self) {
        let mut guard = self.encoder.lock().unwrap();
        if let Some(enc) = guard.take() {
            let cmd: wgpu::CommandBuffer = enc.finish();
            self.queue.submit([cmd]);
        }
    }
}

pub struct GpuPipelines {
    pub binop: wgpu::ComputePipeline,
    pub scalar: wgpu::ComputePipeline,
    pub unary: wgpu::ComputePipeline,
    pub matmul: wgpu::ComputePipeline,
    pub softmax: wgpu::ComputePipeline,
    pub softmax_vjp: wgpu::ComputePipeline,
    pub broadcast_add: wgpu::ComputePipeline,
    pub transpose: wgpu::ComputePipeline,
}

impl GpuPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let compile = |src: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
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
            binop: compile(shaders::BINOP),
            scalar: compile(shaders::SCALAR),
            unary: compile(shaders::UNARY),
            matmul: compile(shaders::MATMUL),
            softmax: compile(shaders::SOFTMAX),
            softmax_vjp: compile(shaders::SOFTMAX_VJP),
            broadcast_add: compile(shaders::BROADCAST_ADD),
            transpose: compile(shaders::TRANSPOSE),
        }
    }
}
