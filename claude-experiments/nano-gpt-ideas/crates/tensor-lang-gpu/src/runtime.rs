//! GPU runtime using `wgpu`. Creates device, compiles shaders, dispatches
//! compute passes, and reads back results.

use std::collections::HashMap;

use tensor_lang_graph::TensorRuntime;

use crate::plan::{BufId, GpuPlan, GpuStep};
use crate::wgsl::Shader;

/// GPU runtime backed by wgpu.
pub struct GpuRuntime {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl GpuRuntime {
    /// Create a new runtime, requesting a GPU device.
    pub fn new() -> Self {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("no suitable GPU adapter found");

        let mut limits = adapter.limits();
        // Request maximum storage buffer size the adapter supports
        limits.max_storage_buffer_binding_size = limits.max_storage_buffer_binding_size.max(256 * 1024 * 1024);
        limits.max_buffer_size = limits.max_buffer_size.max(256 * 1024 * 1024);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("tensor-lang-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                ..Default::default()
            })
            .await
            .expect("failed to create GPU device");

        GpuRuntime { device, queue }
    }

    /// Run a plan with concrete inputs. Returns the flat f32 output.
    pub fn run(&self, plan: &GpuPlan, inputs: &[&[f32]], output_size: usize) -> Vec<f32> {
        self.run_with_dim_params(plan, &[], inputs, output_size)
    }

    /// Run a plan with symbolic dim params and inputs.
    pub fn run_with_dim_params(
        &self,
        plan: &GpuPlan,
        dim_param_values: &[u32],
        inputs: &[&[f32]],
        output_size: usize,
    ) -> Vec<f32> {
        // Build dim param lookup
        let dim_map: HashMap<String, usize> = plan
            .dim_params
            .iter()
            .zip(dim_param_values.iter())
            .map(|(name, &val)| (name.clone(), val as usize))
            .collect();

        // Compile all shaders into pipelines
        let pipelines: Vec<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> = plan
            .shaders
            .iter()
            .map(|shader| self.compile_shader(shader))
            .collect();

        // Create dims uniform buffer if needed
        let dims_buf = if !plan.dim_params.is_empty() {
            let data: Vec<u32> = plan.dim_params.iter().map(|name| {
                *dim_map.get(name).unwrap_or_else(|| panic!("missing dim param: {name}")) as u32
            }).collect();
            // Pad to 16-byte alignment (uniform buffers require it)
            let mut padded = data.clone();
            while (padded.len() * 4) % 16 != 0 {
                padded.push(0);
            }
            let buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dims"),
                contents: bytemuck::cast_slice(&padded),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            Some(buf)
        } else {
            None
        };

        // Track GPU buffers by buf_id
        let mut gpu_bufs: HashMap<BufId, wgpu::Buffer> = HashMap::new();

        // Track which buf_id is an input (for data upload)
        let mut input_idx = 0usize;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tensor-lang-gpu"),
        });

        // Pre-create all bind groups (need to outlive the compute pass)
        struct DispatchInfo {
            pipeline_idx: usize,
            bind_group_idx: usize,
            wg_x: u32,
            wg_y: u32,
            wg_z: u32,
        }
        let mut dispatches: Vec<DispatchInfo> = Vec::new();
        let mut bind_groups: Vec<wgpu::BindGroup> = Vec::new();

        for step in &plan.steps {
            match step {
                GpuStep::AllocBuffer { buf, size } => {
                    let n_elements = size.eval(&dim_map);
                    let byte_size = (n_elements * 4) as u64;
                    let byte_size = byte_size.max(4);

                    let is_input = plan.inputs.iter().any(|(id, _)| *id == *buf);

                    if is_input {
                        let data = inputs[input_idx];
                        input_idx += 1;
                        let gpu_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(&format!("buf_{buf}")),
                            contents: bytemuck::cast_slice(data),
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                        });
                        gpu_bufs.insert(*buf, gpu_buf);
                    } else {
                        let usage = wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST;
                        let gpu_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some(&format!("buf_{buf}")),
                            size: byte_size,
                            usage,
                            mapped_at_creation: false,
                        });
                        gpu_bufs.insert(*buf, gpu_buf);
                    }
                }
                GpuStep::FillConstant { shader_idx, size, .. }
                | GpuStep::FillArange { shader_idx, size, .. } => {
                    let n_elements = size.eval(&dim_map) as u32;
                    let shader = &plan.shaders[*shader_idx];
                    let (_, layout) = &pipelines[*shader_idx];
                    let bind_group = self.create_bind_group(layout, shader, &gpu_bufs, dims_buf.as_ref());
                    let workgroups = (n_elements + shader.workgroup_size - 1) / shader.workgroup_size;
                    let bg_idx = bind_groups.len();
                    bind_groups.push(bind_group);
                    dispatches.push(DispatchInfo {
                        pipeline_idx: *shader_idx,
                        bind_group_idx: bg_idx,
                        wg_x: workgroups, wg_y: 1, wg_z: 1,
                    });
                }
                GpuStep::Dispatch { shader_idx, output_size, .. }
                | GpuStep::Pad { shader_idx, output_size, .. } => {
                    let n_elements = output_size.eval(&dim_map) as u32;
                    let shader = &plan.shaders[*shader_idx];
                    let (_, layout) = &pipelines[*shader_idx];
                    let bind_group = self.create_bind_group(layout, shader, &gpu_bufs, dims_buf.as_ref());
                    let workgroups = (n_elements + shader.workgroup_size - 1) / shader.workgroup_size;
                    let bg_idx = bind_groups.len();
                    bind_groups.push(bind_group);
                    dispatches.push(DispatchInfo {
                        pipeline_idx: *shader_idx,
                        bind_group_idx: bg_idx,
                        wg_x: workgroups, wg_y: 1, wg_z: 1,
                    });
                }
                GpuStep::DispatchMatmul { shader_idx, m_size, n_size, batch_size, .. } => {
                    let m = m_size.eval(&dim_map) as u32;
                    let n = n_size.eval(&dim_map) as u32;
                    let batch = batch_size.eval(&dim_map) as u32;
                    let shader = &plan.shaders[*shader_idx];
                    let (_, layout) = &pipelines[*shader_idx];
                    let bind_group = self.create_bind_group(layout, shader, &gpu_bufs, dims_buf.as_ref());
                    let tile_n = 16u32;
                    let tile_m = 16u32;
                    let bg_idx = bind_groups.len();
                    bind_groups.push(bind_group);
                    dispatches.push(DispatchInfo {
                        pipeline_idx: *shader_idx,
                        bind_group_idx: bg_idx,
                        wg_x: (n + tile_n - 1) / tile_n,
                        wg_y: (m + tile_m - 1) / tile_m,
                        wg_z: batch,
                    });
                }
            }
        }

        // Dispatch all operations in a single compute pass.
        // On Metal backend, dispatches within a compute pass are ordered and
        // writes from dispatch N are visible to dispatch N+1.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("all"),
                timestamp_writes: None,
            });
            for d in &dispatches {
                let (pipeline, _) = &pipelines[d.pipeline_idx];
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, Some(&bind_groups[d.bind_group_idx]), &[]);
                pass.dispatch_workgroups(d.wg_x, d.wg_y, d.wg_z);
            }
        }

        // Read back outputs — copy each output buffer into one staging buffer
        let total_byte_size = (output_size * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: total_byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut staging_offset = 0u64;
        for (buf_id, size_dim) in &plan.output_bufs {
            let n_elements = size_dim.eval(&dim_map);
            let byte_size = (n_elements * 4) as u64;
            let src_buf = &gpu_bufs[buf_id];
            encoder.copy_buffer_to_buffer(src_buf, 0, &staging, staging_offset, byte_size);
            staging_offset += byte_size;
        }

        self.queue.submit(Some(encoder.finish()));

        // Map and read
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        rx.recv().unwrap().expect("failed to map staging buffer");

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        result
    }

    fn compile_shader(&self, shader: &Shader) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(shader.source.as_str().into()),
        });

        // Build bind group layout entries
        let entries: Vec<wgpu::BindGroupLayoutEntry> = shader
            .bindings
            .iter()
            .map(|(binding, _buf, writable)| {
                wgpu::BindGroupLayoutEntry {
                    binding: *binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: !writable,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            })
            .chain(if shader.uses_dims {
                let binding = shader.bindings.len() as u32;
                vec![wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }]
            } else {
                vec![]
            })
            .collect();

        let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layout"),
            entries: &entries,
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&layout],
            immediate_size: 0,
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        (pipeline, layout)
    }

    fn create_bind_group(
        &self,
        layout: &wgpu::BindGroupLayout,
        shader: &Shader,
        gpu_bufs: &HashMap<BufId, wgpu::Buffer>,
        dims_buf: Option<&wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        let mut entries: Vec<wgpu::BindGroupEntry> = shader
            .bindings
            .iter()
            .map(|(binding, buf_id, _)| {
                wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: gpu_bufs[buf_id].as_entire_binding(),
                }
            })
            .collect();

        if shader.uses_dims {
            let binding = shader.bindings.len() as u32;
            entries.push(wgpu::BindGroupEntry {
                binding,
                resource: dims_buf.unwrap().as_entire_binding(),
            });
        }

        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group"),
            layout,
            entries: &entries,
        })
    }
}

/// Helper trait for buffer initialization (re-exported from wgpu::util).
trait CreateBufferInit {
    fn create_buffer_init(&self, desc: &wgpu::util::BufferInitDescriptor) -> wgpu::Buffer;
}

impl CreateBufferInit for wgpu::Device {
    fn create_buffer_init(&self, desc: &wgpu::util::BufferInitDescriptor) -> wgpu::Buffer {
        wgpu::util::DeviceExt::create_buffer_init(self, desc)
    }
}

/// A GPU runtime bundled with a compiled plan, implementing `TensorRuntime`.
pub struct BoundGpuRuntime {
    pub runtime: GpuRuntime,
    pub plan: GpuPlan,
}

impl BoundGpuRuntime {
    pub fn new(plan: GpuPlan) -> Self {
        BoundGpuRuntime {
            runtime: GpuRuntime::new(),
            plan,
        }
    }
}

impl TensorRuntime for BoundGpuRuntime {
    fn backend_name(&self) -> &str { "wgpu" }

    fn run(&mut self, inputs: &[&[f32]], output_size: usize) -> Vec<f32> {
        self.runtime.run(&self.plan, inputs, output_size)
    }

    fn run_with_dim_params(
        &mut self,
        dim_param_values: &[u32],
        inputs: &[&[f32]],
        output_size: usize,
    ) -> Vec<f32> {
        self.runtime.run_with_dim_params(&self.plan, dim_param_values, inputs, output_size)
    }
}
