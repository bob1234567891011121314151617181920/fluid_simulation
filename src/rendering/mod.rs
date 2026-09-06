mod camera;
mod egui_renderer;
mod sphere;
mod texture;

use self::{
    camera::create_static_camera, egui_renderer::EguiRenderer, sphere::SphereVertex,
    texture::Texture,
};
use crate::simulation::FlipSimulation;
use anyhow::Context;
use glam::UVec3;
use std::{sync::Arc, time::Duration, time::Instant};

use wgpu::util::DeviceExt;

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Fullscreen, Window},
};

const FIXED_DT_SECONDS: f32 = 1.0 / 120.0;
const MAX_STEPS_PER_FRAME: usize = 8;

const DIMENSIONS: UVec3 = UVec3::splat(16);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParticleInstance {
    position: [f32; 3],
}

impl ParticleInstance {
    fn describe() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x3,
            }],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Globals {
    view_projection: [[f32; 4]; 4],
    sphere_radius: f32,
    _padding: [f32; 7],
}

pub struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    depth_texture: Texture,
    simulation: FlipSimulation,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    index_count: u32,
    instance_count: u32,
    egui_renderer: EguiRenderer,
    fps: f32,
}

impl State {
    async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });

        let surface = instance
            .create_surface(window.clone())
            .context("failed to create the GPU surface")?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: true,
            })
            .await
            .context("no compatible GPU adapter was found")?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Fluid simulation device"),
                required_features: wgpu::Features::empty(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off, // Trace path
            })
            .await
            .context("failed to create the GPU device")?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };

        surface.configure(&device, &config);
        let density = 0.5;

        let simulation = FlipSimulation::new(DIMENSIONS, density, FIXED_DT_SECONDS);

        let view_projection =
            create_static_camera(DIMENSIONS, size.width, size.height).to_cols_array_2d();
        let sphere_radius = (density / DIMENSIONS.max_element() as f32) * 0.4;

        let globals = Globals {
            view_projection,
            sphere_radius,
            _padding: [0.0; 7],
        };

        let globals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Globals buffer"),
            contents: bytemuck::bytes_of(&globals),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let globals_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Globals Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Globals Bind Group"),
            layout: &globals_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buffer.as_entire_binding(),
            }],
        });

        let instances: Vec<[f32; 3]> = simulation
            .particle_positions()
            .map(|position| position.into())
            .collect();

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let depth_texture = Texture::create_depth_texture(&device, &config, "Depth Texture");

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[Some(&globals_bind_group_layout)],
                immediate_size: 0,
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                buffers: &[
                    Some(SphereVertex::describe()),
                    Some(ParticleInstance::describe()),
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });
        let uv_sphere = sphere::create_uv_sphere(16, 32);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&uv_sphere.0),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&uv_sphere.1),
            usage: wgpu::BufferUsages::INDEX,
        });

        let index_count = uv_sphere.1.len() as u32;
        let instance_count = instances.len() as u32;

        let egui_renderer = EguiRenderer::new(&device, surface_format, None, &window);
        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            is_surface_configured: true,
            depth_texture,
            simulation,
            render_pipeline,
            vertex_buffer,
            instance_buffer,
            index_buffer,
            globals_bind_group,
            index_count,
            instance_count,
            egui_renderer,
            fps: 0.0,
        })
    }

    fn update_fps(&mut self, frame_time: Duration) {
        let dt = frame_time.as_secs_f32().max(f32::EPSILON);
        let instant_fps = 1.0 / dt;
        let smoothing_time = 0.4;
        let alpha = 1.0 - (-dt / smoothing_time).exp();

        self.fps = if self.fps == 0.0 {
            instant_fps
        } else {
            self.fps + alpha * (instant_fps - self.fps)
        };
    }

    fn update(&mut self) {
        self.simulation.step();

        let instances: Vec<[f32; 3]> = self
            .simulation
            .particle_positions()
            .map(|position| position.into())
            .collect();
        self.queue
            .write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instances));
    }

    fn draw_ui(&mut self, ctx: &egui::Context) {
        egui::Window::new("Debug")
            .default_pos(egui::pos2(10.0, 10.0))
            .show(ctx, |ui| {
                ui.label(format!("FPS: {:.1}", self.fps));
                ui.separator();
                ui.heading("Fluid settings");
            });
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            self.is_surface_configured = false;
            return;
        }

        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth_texture =
            Texture::create_depth_texture(&self.device, &self.config, "Depth Texture");
        self.is_surface_configured = true;
    }

    fn render(&mut self) -> anyhow::Result<()> {
        if !self.is_surface_configured {
            return Ok(());
        }
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                anyhow::bail!("Lost device");
            }
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });

        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.set_bind_group(0, &self.globals_bind_group, &[]);
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.draw_indexed(0..self.index_count, 0, 0..self.instance_count);

        drop(render_pass);

        let egui_input = self.egui_renderer.take_input(&self.window);
        let egui_context = self.egui_renderer.context();

        let full_output = egui_context.run_ui(egui_input, |ui| {
            self.draw_ui(ui);
        });

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: self.window.scale_factor() as f32,
        };

        self.egui_renderer.end_frame_and_draw(
            &self.device,
            &self.queue,
            &mut encoder,
            &self.window,
            &view,
            screen_descriptor,
            full_output,
        );

        self.queue.submit([encoder.finish()]);
        self.queue.present(output);
        Ok(())
    }
}

pub struct App {
    state: Option<State>,
    last_frame: Instant,
    accumulator: Duration,
}

impl App {
    fn new() -> Self {
        let now = Instant::now();

        Self {
            state: None,
            last_frame: now,
            accumulator: Duration::ZERO,
        }
    }

    fn handle_frame_time(last_frame: &mut Instant, accumulator: &mut Duration) -> Duration {
        let now = Instant::now();
        let elapsed = now.saturating_duration_since(*last_frame);
        let frame_time = elapsed.min(Duration::from_millis(250));

        *last_frame = now;
        *accumulator += frame_time;

        elapsed
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes()
            .with_title("FLIP Fluid Simulation")
            .with_fullscreen(Some(Fullscreen::Borderless(None)));

        let window = match event_loop.create_window(window_attributes) {
            Ok(window) => Arc::new(window),

            Err(error) => {
                log::error!("Failed to create window: {error}");
                event_loop.exit();
                return;
            }
        };

        match pollster::block_on(State::new(window)) {
            Ok(state) => {
                state.window.request_redraw();

                self.state = Some(state);
                self.last_frame = Instant::now();
                self.accumulator = Duration::ZERO;
            }

            Err(error) => {
                log::error!("Failed to initialize renderer: {error:#}");
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(state) => state,
            None => return,
        };

        state.egui_renderer.handle_input(&state.window, &event);

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                state.resize(size.width, size.height);
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                event_loop.exit();
            }

            WindowEvent::RedrawRequested => {
                let frame_time =
                    Self::handle_frame_time(&mut self.last_frame, &mut self.accumulator);
                state.update_fps(frame_time);

                let fixed_dt = Duration::from_secs_f32(FIXED_DT_SECONDS);

                let mut steps = 0;

                while self.accumulator >= fixed_dt && steps < MAX_STEPS_PER_FRAME {
                    state.update();

                    self.accumulator -= fixed_dt;
                    steps += 1;
                }

                if steps == MAX_STEPS_PER_FRAME {
                    self.accumulator = Duration::ZERO;

                    log::warn!("Simulation could not keep up with real time");
                }

                match state.render() {
                    Ok(()) => {
                        state.window.request_redraw();
                    }
                    Err(error) => {
                        log::error!("Render error: {error:#}");
                        event_loop.exit();
                    }
                }
            }
            _ => {}
        }
    }
}

pub fn run() -> anyhow::Result<()> {
    env_logger::init();

    let event_loop = EventLoop::builder().build()?;
    let mut app = App::new();

    event_loop.run_app(&mut app)?;

    Ok(())
}
