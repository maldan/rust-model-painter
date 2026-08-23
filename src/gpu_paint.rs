//! GPU texture painting: UV-as-color + world-space brush (Substance-style).
//!
//! 1) Offscreen pass writes UV+facing and world position (depth-tested).
//! 2) Compute splat onto the **active layer** texture.
//! 3) Layer stack composite → material maps (albedo / MR channels).

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3};
use mega_render::{Handle, Mesh, Node, Scene};
use mega_ui::Rect;

use crate::brush_alpha::{BrushAlpha, NormalStamp};
use crate::paint::Brush;

/// Uniform slot stride for dynamic-offset passes (WebGPU min alignment).
const UBO_STRIDE: u64 = 256;

const UV_CLEAR: wgpu::Color = wgpu::Color {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 0.0,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UvVertex {
    pos: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    tangent: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniforms {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 3],
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ObjectUniforms {
    model: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BrushUniforms {
    /// Brush center in UV-map / pos-map pixel space.
    center: [f32; 2],
    /// Screen-space search radius (pixels).
    screen_radius: f32,
    /// World-space brush radius.
    world_radius: f32,
    hardness: f32,
    /// 0 = distance hardness, 1 = coverage stamp (noise).
    coverage_stamp: f32,
    /// WGSL uniform `vec4` aligns to 16 — pad 8 bytes after the f32 pair above.
    _pad_to_color: [f32; 2],
    color: [f32; 4],
    /// Per-channel write mask (always 1s when stamping into a layer).
    channel_mask: [f32; 4],
    opacity: f32,
    /// 0 = paint over, 1 = erase layer alpha.
    erase: f32,
    map_w: u32,
    map_h: u32,
    tex_w: u32,
    tex_h: u32,
    /// UDIM tile origin (`floor(uv)`), e.g. (1,0) for 1002.
    tile: [f32; 2],
    /// Stamp plane normal (world).
    normal: [f32; 3],
    /// 1 = tangent-space normal stamp (RNM / flatten erase).
    normal_mode: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StackUniforms {
    base: [f32; 4],
    channel_mask: [f32; 4],
    opacity: f32,
    /// 0 = apply base, 1 = blend paint tex, 2 = fill color.
    mode: u32,
    tex_w: u32,
    tex_h: u32,
    has_mask: u32,
    /// 0 = alpha mix, 1 = whiteout normals.
    blend: u32,
    _pad: [u32; 2],
}

/// One layer in `composite_stack` (bottom → top).
pub struct CompositeLayer<'a> {
    pub opacity: f32,
    /// 1 = paint texture, 2 = solid fill (`fill`).
    pub mode: u32,
    pub fill: [f32; 4],
    pub content: Option<&'a wgpu::Texture>,
    pub mask: Option<&'a wgpu::Texture>,
}

struct PaintMesh {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
    synced: u64,
}

pub struct GpuPaint {
    uv_format: wgpu::TextureFormat,
    pos_format: wgpu::TextureFormat,
    uv_tex: Option<wgpu::Texture>,
    uv_view: Option<wgpu::TextureView>,
    pos_tex: Option<wgpu::Texture>,
    pos_view: Option<wgpu::TextureView>,
    tbn_tex: Option<wgpu::Texture>,
    tbn_view: Option<wgpu::TextureView>,
    depth_tex: Option<wgpu::Texture>,
    depth_view: Option<wgpu::TextureView>,
    uv_size: (u32, u32),

    paint_read: Option<wgpu::Texture>,
    paint_read_view: Option<wgpu::TextureView>,
    stroke: Option<wgpu::Texture>,
    stroke_view: Option<wgpu::TextureView>,
    stroke_storage_view: Option<wgpu::TextureView>,
    stroke_zeros: Vec<u8>,
    paint_aux_size: (u32, u32),

    uv_pipeline: wgpu::RenderPipeline,
    frame_buf: wgpu::Buffer,
    frame_bg: wgpu::BindGroup,
    object_layout: wgpu::BindGroupLayout,

    splat_pipeline: wgpu::ComputePipeline,
    splat_bind_layout: wgpu::BindGroupLayout,
    comp_pipeline: wgpu::ComputePipeline,
    comp_bind_layout: wgpu::BindGroupLayout,
    brush_buf: wgpu::Buffer,
    brush_buf_slots: u32,
    /// Next free brush UBO slot within the current encoder batch.
    brush_slot: u32,

    alpha_texs: Vec<wgpu::Texture>,
    alpha_views: Vec<wgpu::TextureView>,
    nrm_texs: Vec<wgpu::Texture>,
    nrm_views: Vec<wgpu::TextureView>,
    alpha_sampler: wgpu::Sampler,

    stack_pipeline: wgpu::ComputePipeline,
    stack_bind_layout: wgpu::BindGroupLayout,
    /// Packed stack uniforms (256-byte stride) — written once per composite.
    stack_buf: wgpu::Buffer,
    stack_buf_slots: u32,
    /// 1×1 placeholder when a stack pass has no content/mask tex.
    #[allow(dead_code)]
    dummy_tex: wgpu::Texture,
    dummy_view: wgpu::TextureView,
    fill_px: [u8; 4],
    fill_bytes: Vec<u8>,

    meshes: HashMap<(u32, u32), PaintMesh>,
    /// Held until later frames so the paint encoder can submit them.
    models_buf: Option<wgpu::Buffer>,
    models_slots: u64,
    keep_bgs: [Vec<wgpu::BindGroup>; 2],
    keep_i: usize,
}

impl GpuPaint {
    pub fn new(device: &wgpu::Device) -> Self {
        let uv_format = wgpu::TextureFormat::Rgba16Float;
        let pos_format = wgpu::TextureFormat::Rgba16Float;

        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_paint_uv_frame_layout"),
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
        let object_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_paint_uv_object_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uv_shader = device.create_shader_module(wgpu::include_wgsl!("shaders/uv_map.wgsl"));
        let uv_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gpu_paint_uv_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("gpu_paint_uv_pl"),
                    bind_group_layouts: &[Some(&frame_layout), Some(&object_layout)],
                    immediate_size: 0,
                }),
            ),
            vertex: wgpu::VertexState {
                module: &uv_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<UvVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x3,
                        1 => Float32x3,
                        2 => Float32x2,
                        3 => Float32x4
                    ],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &uv_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: uv_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: pos_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: uv_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                front_face: wgpu::FrontFace::Cw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let frame_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_paint_frame_ubo"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_paint_frame_bg"),
            layout: &frame_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_buf.as_entire_binding(),
            }],
        });

        let splat_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_paint_splat_layout"),
            entries: &[
                brush_ubo_entry(0),
                tex_entry(1),
                tex_entry(2),
                storage_entry(3),
                sampled_tex_entry(4),
                sampler_entry(5),
                tex_entry(6),
            ],
        });
        let splat_shader = device.create_shader_module(wgpu::include_wgsl!("shaders/splat.wgsl"));
        let splat_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_paint_splat_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("gpu_paint_splat_pl"),
                    bind_group_layouts: &[Some(&splat_bind_layout)],
                    immediate_size: 0,
                }),
            ),
            module: &splat_shader,
            entry_point: Some("splat"),
            compilation_options: Default::default(),
            cache: None,
        });

        let comp_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_paint_comp_layout"),
            entries: &[
                brush_ubo_entry(0),
                tex_entry(1),
                tex_entry(2),
                storage_entry(3),
            ],
        });
        let comp_shader = device.create_shader_module(wgpu::include_wgsl!("shaders/composite.wgsl"));
        let comp_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_paint_comp_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("gpu_paint_comp_pl"),
                    bind_group_layouts: &[Some(&comp_bind_layout)],
                    immediate_size: 0,
                }),
            ),
            module: &comp_shader,
            entry_point: Some("composite"),
            compilation_options: Default::default(),
            cache: None,
        });

        let brush_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_paint_brush_ubo"),
            size: UBO_STRIDE * 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let stack_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_paint_stack_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<StackUniforms>() as u64,
                        ),
                    },
                    count: None,
                },
                tex_entry(1),
                tex_entry(2),
                storage_entry(3),
                tex_entry(4),
            ],
        });
        let stack_shader = device.create_shader_module(wgpu::include_wgsl!("shaders/stack.wgsl"));
        let stack_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_paint_stack_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("gpu_paint_stack_pl"),
                    bind_group_layouts: &[Some(&stack_bind_layout)],
                    immediate_size: 0,
                }),
            ),
            module: &stack_shader,
            entry_point: Some("stack"),
            compilation_options: Default::default(),
            cache: None,
        });
        let stack_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_paint_stack_ubo"),
            size: UBO_STRIDE * 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dummy_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gpu_paint_dummy"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let dummy_view = dummy_tex.create_view(&Default::default());

        let alpha_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("gpu_paint_alpha_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            uv_format,
            pos_format,
            uv_tex: None,
            uv_view: None,
            pos_tex: None,
            pos_view: None,
            tbn_tex: None,
            tbn_view: None,
            depth_tex: None,
            depth_view: None,
            uv_size: (0, 0),
            paint_read: None,
            paint_read_view: None,
            stroke: None,
            stroke_view: None,
            stroke_storage_view: None,
            stroke_zeros: Vec::new(),
            paint_aux_size: (0, 0),
            uv_pipeline,
            frame_buf,
            frame_bg,
            object_layout,
            splat_pipeline,
            splat_bind_layout,
            comp_pipeline,
            comp_bind_layout,
            brush_buf,
            brush_buf_slots: 8,
            brush_slot: 0,
            stack_pipeline,
            stack_bind_layout,
            stack_buf,
            stack_buf_slots: 8,
            dummy_tex,
            dummy_view,
            fill_px: [0, 0, 0, 0],
            fill_bytes: Vec::new(),
            meshes: HashMap::new(),
            models_buf: None,
            models_slots: 0,
            keep_bgs: [Vec::new(), Vec::new()],
            keep_i: 0,
            alpha_texs: Vec::new(),
            alpha_views: Vec::new(),
            nrm_texs: Vec::new(),
            nrm_views: Vec::new(),
            alpha_sampler,
        }
    }

    pub fn upload_alphas(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, alphas: &[BrushAlpha]) {
        self.alpha_texs.clear();
        self.alpha_views.clear();
        for alpha in alphas {
            let (tex, view) = upload_stamp_rgba(device, queue, alpha.size, &alpha.rgba_preview());
            self.alpha_texs.push(tex);
            self.alpha_views.push(view);
        }
    }

    pub fn upload_normals(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, stamps: &[NormalStamp]) {
        self.nrm_texs.clear();
        self.nrm_views.clear();
        for stamp in stamps {
            let (tex, view) = upload_stamp_rgba(device, queue, stamp.size, &stamp.rgba);
            self.nrm_texs.push(tex);
            self.nrm_views.push(view);
        }
    }

    pub fn alpha_views(&self) -> &[wgpu::TextureView] {
        &self.alpha_views
    }

    pub fn nrm_views(&self) -> &[wgpu::TextureView] {
        &self.nrm_views
    }

    pub fn ensure_uv_targets(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        let w = w.max(1);
        let h = h.max(1);
        if self.uv_size == (w, h)
            && self.uv_tex.is_some()
            && self.pos_tex.is_some()
            && self.tbn_tex.is_some()
        {
            return;
        }
        self.uv_size = (w, h);

        let mk = |label: &str, format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };

        let uv_tex = mk(
            "gpu_paint_uv_map",
            self.uv_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        self.uv_view = Some(uv_tex.create_view(&Default::default()));
        self.uv_tex = Some(uv_tex);

        let pos_tex = mk(
            "gpu_paint_pos_map",
            self.pos_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        self.pos_view = Some(pos_tex.create_view(&Default::default()));
        self.pos_tex = Some(pos_tex);

        let tbn_tex = mk(
            "gpu_paint_tbn_map",
            self.uv_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        self.tbn_view = Some(tbn_tex.create_view(&Default::default()));
        self.tbn_tex = Some(tbn_tex);

        let depth_tex = mk(
            "gpu_paint_uv_depth",
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        self.depth_view = Some(depth_tex.create_view(&Default::default()));
        self.depth_tex = Some(depth_tex);
    }

    fn ensure_aux(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        let w = w.max(1);
        let h = h.max(1);
        if self.paint_aux_size == (w, h) && self.paint_read.is_some() && self.stroke.is_some() {
            return;
        }
        self.paint_aux_size = (w, h);

        let paint_read = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gpu_paint_read_copy"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.paint_read_view = Some(paint_read.create_view(&Default::default()));
        self.paint_read = Some(paint_read);

        let stroke = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gpu_paint_stroke"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.stroke_view = Some(stroke.create_view(&Default::default()));
        self.stroke_storage_view = Some(stroke.create_view(&wgpu::TextureViewDescriptor {
            label: Some("gpu_paint_stroke_storage"),
            format: Some(wgpu::TextureFormat::Rgba8Unorm),
            ..Default::default()
        }));
        self.stroke = Some(stroke);
        self.stroke_zeros = vec![0u8; (w * h * 4) as usize];
    }

    fn sync_meshes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        paintable: &[Handle<Node>],
    ) {
        let mut live = HashMap::new();
        for &node_h in paintable {
            let Some(node) = scene.nodes.get(node_h) else {
                continue;
            };
            if !node.visible {
                continue;
            }
            let Some(mesh_h) = node.mesh else {
                continue;
            };
            let Some(mesh) = scene.meshes.get(mesh_h) else {
                continue;
            };
            let key = mesh_h.key();
            live.insert(key, ());
            if self.meshes.get(&key).map(|m| m.synced) == Some(mesh.version) {
                continue;
            }
            self.meshes.insert(key, upload_paint_mesh(device, queue, mesh));
        }
        self.meshes.retain(|k, _| live.contains_key(k));
    }

    pub fn render_uv_map(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scene: &Scene,
        paintable: &[Handle<Node>],
        aspect: f32,
    ) {
        self.keep_i ^= 1;
        self.keep_bgs[self.keep_i].clear();
        self.brush_slot = 0;
        // Reserve up front — recreating brush_buf mid-batch would drop earlier slot writes.
        self.ensure_brush_buf(device, 64);
        self.sync_meshes(device, queue, scene, paintable);

        let Some(uv_view) = self.uv_view.clone() else {
            return;
        };
        let Some(pos_view) = self.pos_view.clone() else {
            return;
        };
        let Some(tbn_view) = self.tbn_view.clone() else {
            return;
        };
        let Some(depth_view) = self.depth_view.clone() else {
            return;
        };

        let eye = scene.camera.eye;
        queue.write_buffer(
            &self.frame_buf,
            0,
            bytemuck::bytes_of(&FrameUniforms {
                view_proj: scene.camera.view_proj(aspect).to_cols_array_2d(),
                eye: eye.to_array(),
                _pad: 0.0,
            }),
        );

        struct Draw<'a> {
            mesh: &'a PaintMesh,
            model: Mat4,
        }
        let mut draws: Vec<Draw<'_>> = Vec::new();
        for &node_h in paintable {
            let Some(node) = scene.nodes.get(node_h) else {
                continue;
            };
            if !node.visible {
                continue;
            }
            let Some(mesh_h) = node.mesh else {
                continue;
            };
            let Some(gpu_mesh) = self.meshes.get(&mesh_h.key()) else {
                continue;
            };
            draws.push(Draw {
                mesh: gpu_mesh,
                model: scene.world_matrix(node_h),
            });
        }

        let color_atts = [
            Some(wgpu::RenderPassColorAttachment {
                view: &uv_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(UV_CLEAR),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &pos_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &tbn_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(UV_CLEAR),
                    store: wgpu::StoreOp::Store,
                },
            }),
        ];

        if draws.is_empty() {
            let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gpu_paint_uv_clear"),
                color_attachments: &color_atts,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            return;
        }

        const ALIGN: u64 = 256;
        let stride = ALIGN;
        let slots = (draws.len() as u64).max(1);
        if self.models_slots < slots {
            let cap = slots.next_power_of_two().max(8);
            self.models_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu_paint_models"),
                size: stride * cap,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.models_slots = cap;
        }
        let models_buf = self.models_buf.as_ref().expect("models_buf");
        for (i, d) in draws.iter().enumerate() {
            queue.write_buffer(
                models_buf,
                i as u64 * stride,
                bytemuck::bytes_of(&ObjectUniforms {
                    model: d.model.to_cols_array_2d(),
                }),
            );
        }

        let mut object_bgs = Vec::with_capacity(draws.len());
        for i in 0..draws.len() {
            object_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpu_paint_object_bg"),
                layout: &self.object_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: models_buf,
                        offset: i as u64 * stride,
                        size: std::num::NonZeroU64::new(
                            std::mem::size_of::<ObjectUniforms>() as u64,
                        ),
                    }),
                }],
            }));
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gpu_paint_uv_pass"),
                color_attachments: &color_atts,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.uv_pipeline);
            pass.set_bind_group(0, &self.frame_bg, &[]);
            for (i, d) in draws.iter().enumerate() {
                pass.set_bind_group(1, &object_bgs[i], &[]);
                pass.set_vertex_buffer(0, d.mesh.vertex_buf.slice(..));
                pass.set_index_buffer(d.mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..d.mesh.index_count, 0, 0..1);
            }
        }
        self.keep_bgs[self.keep_i].append(&mut object_bgs);
    }

    pub fn stamp(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        paint_tex: &wgpu::Texture,
        paint_size: (u32, u32),
        brush: &Brush,
        channel_mask: [f32; 4],
        erase: bool,
        center_px: Vec2,
        screen_radius_px: f32,
        world_radius: f32,
        alpha_index: usize,
        plane_normal: Vec3,
        coverage_stamp: bool,
        normal_mode: bool,
        tile: [f32; 2],
    ) {
        let (tw, th) = paint_size;
        let (mw, mh) = self.uv_size;
        if tw == 0 || th == 0 || mw == 0 || mh == 0 {
            return;
        }

        self.ensure_aux(device, tw, th);
        let slot = self.brush_slot;
        self.brush_slot += 1;
        self.ensure_brush_buf(device, self.brush_slot);
        let Some(uv_view) = self.uv_view.clone() else {
            return;
        };
        let Some(pos_view) = self.pos_view.clone() else {
            return;
        };
        let Some(tbn_view) = self.tbn_view.clone() else {
            return;
        };
        let Some(paint_read) = self.paint_read.clone() else {
            return;
        };
        let Some(paint_read_view) = self.paint_read_view.clone() else {
            return;
        };
        let Some(stroke) = self.stroke.clone() else {
            return;
        };
        let Some(stroke_view) = self.stroke_view.clone() else {
            return;
        };
        let Some(stroke_storage_view) = self.stroke_storage_view.clone() else {
            return;
        };

        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: paint_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &paint_read,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: tw,
                height: th,
                depth_or_array_layers: 1,
            },
        );

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &stroke,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.stroke_zeros,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * tw),
                rows_per_image: Some(th),
            },
            wgpu::Extent3d {
                width: tw,
                height: th,
                depth_or_array_layers: 1,
            },
        );

        let uniforms = BrushUniforms {
            center: center_px.to_array(),
            screen_radius: screen_radius_px.max(2.0),
            world_radius: world_radius.max(1e-4),
            hardness: brush.hardness.clamp(0.0, 1.0),
            coverage_stamp: if coverage_stamp { 1.0 } else { 0.0 },
            _pad_to_color: [0.0; 2],
            color: brush.color,
            channel_mask,
            opacity: brush.opacity.clamp(0.0, 1.0),
            erase: if erase { 1.0 } else { 0.0 },
            map_w: mw,
            map_h: mh,
            tex_w: tw,
            tex_h: th,
            tile,
            normal: plane_normal.normalize_or_zero().to_array(),
            normal_mode: if normal_mode { 1.0 } else { 0.0 },
        };
        queue.write_buffer(
            &self.brush_buf,
            slot as u64 * UBO_STRIDE,
            bytemuck::bytes_of(&uniforms),
        );
        let brush_off = slot * UBO_STRIDE as u32;
        let brush_binding = wgpu::BufferBinding {
            buffer: &self.brush_buf,
            offset: 0,
            size: std::num::NonZeroU64::new(std::mem::size_of::<BrushUniforms>() as u64),
        };

        let views = if normal_mode {
            &self.nrm_views
        } else {
            &self.alpha_views
        };
        let alpha_view = views
            .get(alpha_index)
            .or_else(|| views.first())
            .unwrap_or(&self.dummy_view);

        let splat_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_paint_splat_bg"),
            layout: &self.splat_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(brush_binding.clone()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&uv_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&pos_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&stroke_storage_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(alpha_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.alpha_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&tbn_view),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_paint_splat"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.splat_pipeline);
            pass.set_bind_group(0, &splat_bg, &[brush_off]);
            pass.dispatch_workgroups(mw.div_ceil(8), mh.div_ceil(8), 1);
        }

        let paint_view = paint_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("gpu_paint_storage_view"),
            format: Some(wgpu::TextureFormat::Rgba8Unorm),
            ..Default::default()
        });
        let comp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_paint_comp_bg"),
            layout: &self.comp_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(brush_binding),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&paint_read_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&stroke_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&paint_view),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_paint_composite"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.comp_pipeline);
            pass.set_bind_group(0, &comp_bg, &[brush_off]);
            pass.dispatch_workgroups(tw.div_ceil(8), th.div_ceil(8), 1);
        }
        self.keep_bgs[self.keep_i].push(splat_bg);
        self.keep_bgs[self.keep_i].push(comp_bg);
    }

    /// Clear a gpu-resident paint texture to transparent black.
    pub fn clear_texture(&mut self, queue: &wgpu::Queue, tex: &wgpu::Texture, size: (u32, u32)) {
        self.fill_texture(queue, tex, size, [0, 0, 0, 0]);
    }

    /// Fill a gpu-resident texture with a constant RGBA8 color.
    pub fn fill_texture(
        &mut self,
        queue: &wgpu::Queue,
        tex: &wgpu::Texture,
        size: (u32, u32),
        rgba: [u8; 4],
    ) {
        let (w, h) = (size.0.max(1), size.1.max(1));
        let n = (w * h * 4) as usize;
        if rgba == [0, 0, 0, 0] {
            if self.stroke_zeros.len() < n {
                self.stroke_zeros.resize(n, 0);
            }
            write_paint_rgba(queue, tex, &self.stroke_zeros[..n], w, h);
            return;
        }
        if self.fill_bytes.len() != n || self.fill_px != rgba {
            self.fill_bytes.clear();
            self.fill_bytes.resize(n, 0);
            for px in self.fill_bytes.chunks_exact_mut(4) {
                px.copy_from_slice(&rgba);
            }
            self.fill_px = rgba;
        }
        write_paint_rgba(queue, tex, &self.fill_bytes, w, h);
    }

    /// Composite layer stack into `dst` (material map).
    ///
    /// `channel_mask` selects which RGB channels are written (albedo all; MR G/B).
    /// Layers are bottom → top. Paint layers are straight RGBA (A = coverage).
    /// Fill layers use `fill` as RGB with full coverage, then the layer mask.
    ///
    /// Uniforms for every pass are written once into a strided UBO — `queue.write_buffer`
    /// is not recorded in the encoder, so per-dispatch overwrites would make every
    /// pass see only the last values (skipping base clear → layer stacks on itself).
    pub fn composite_stack(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        dst: &wgpu::Texture,
        size: (u32, u32),
        base: [f32; 4],
        channel_mask: [f32; 4],
        layers: &[CompositeLayer<'_>],
        nrm_blend: bool,
    ) {
        let (tw, th) = (size.0.max(1), size.1.max(1));
        self.ensure_aux(device, tw, th);
        let Some(paint_read) = self.paint_read.clone() else {
            return;
        };
        let Some(paint_read_view) = self.paint_read_view.clone() else {
            return;
        };

        let slots = 1 + layers.len() as u32;
        self.ensure_stack_buf(device, slots);

        // Pack all pass uniforms up-front (256-byte stride).
        let mut blob = vec![0u8; (UBO_STRIDE as usize) * slots as usize];
        let write_slot = |blob: &mut [u8], slot: u32, u: StackUniforms| {
            let off = (slot as usize) * UBO_STRIDE as usize;
            blob[off..off + std::mem::size_of::<StackUniforms>()]
                .copy_from_slice(bytemuck::bytes_of(&u));
        };
        write_slot(
            &mut blob,
            0,
            StackUniforms {
                base,
                channel_mask,
                opacity: 1.0,
                mode: 0,
                tex_w: tw,
                tex_h: th,
                has_mask: 0,
                blend: 0,
                _pad: [0; 2],
            },
        );
        let blend = u32::from(nrm_blend);
        for (i, layer) in layers.iter().enumerate() {
            write_slot(
                &mut blob,
                (i + 1) as u32,
                StackUniforms {
                    base: if layer.mode == 2 { layer.fill } else { base },
                    channel_mask,
                    opacity: layer.opacity.clamp(0.0, 1.0),
                    mode: layer.mode,
                    tex_w: tw,
                    tex_h: th,
                    has_mask: u32::from(layer.mask.is_some()),
                    blend,
                    _pad: [0; 2],
                },
            );
        }
        queue.write_buffer(&self.stack_buf, 0, &blob);

        let ubo_size = std::num::NonZeroU64::new(std::mem::size_of::<StackUniforms>() as u64);
        let dst_view = dst.create_view(&wgpu::TextureViewDescriptor {
            label: Some("gpu_paint_stack_dst"),
            format: Some(wgpu::TextureFormat::Rgba8Unorm),
            ..Default::default()
        });

        let dummy_view = &self.dummy_view;
        let stack_buf = &self.stack_buf;
        let stack_layout = &self.stack_bind_layout;
        let pipeline = &self.stack_pipeline;

        let dispatch = |encoder: &mut wgpu::CommandEncoder,
                            slot: u32,
                            layer_view: &wgpu::TextureView,
                            mask_view: &wgpu::TextureView| {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: dst,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &paint_read,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: tw,
                    height: th,
                    depth_or_array_layers: 1,
                },
            );

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpu_paint_stack_bg"),
                layout: stack_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: stack_buf,
                            offset: 0,
                            size: ubo_size,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&paint_read_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(layer_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&dst_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(mask_view),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpu_paint_stack"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bg, &[slot * UBO_STRIDE as u32]);
                pass.dispatch_workgroups(tw.div_ceil(8), th.div_ceil(8), 1);
            }
        };

        dispatch(encoder, 0, dummy_view, dummy_view);

        for (i, layer) in layers.iter().enumerate() {
            let content_view = layer.content.map(|t| t.create_view(&Default::default()));
            let mask_view = layer.mask.map(|t| t.create_view(&Default::default()));
            let lv = content_view.as_ref().unwrap_or(dummy_view);
            let mv = mask_view.as_ref().unwrap_or(dummy_view);
            dispatch(encoder, (i + 1) as u32, lv, mv);
        }
    }

    fn ensure_stack_buf(&mut self, device: &wgpu::Device, slots: u32) {
        let slots = slots.max(1);
        if slots <= self.stack_buf_slots {
            return;
        }
        self.stack_buf_slots = slots.next_power_of_two().max(8);
        self.stack_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_paint_stack_ubo"),
            size: UBO_STRIDE * self.stack_buf_slots as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }

    fn ensure_brush_buf(&mut self, device: &wgpu::Device, slots: u32) {
        let slots = slots.max(1);
        if slots <= self.brush_buf_slots {
            return;
        }
        self.brush_buf_slots = slots.next_power_of_two().max(8);
        self.brush_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_paint_brush_ubo"),
            size: UBO_STRIDE * self.brush_buf_slots as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }
}

fn brush_ubo_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: std::num::NonZeroU64::new(
                std::mem::size_of::<BrushUniforms>() as u64,
            ),
        },
        count: None,
    }
}

fn ubo_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn sampled_tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba8Unorm,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn upload_stamp_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    size: u32,
    rgba: &[u8],
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gpu_paint_stamp"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * size),
            rows_per_image: Some(size),
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );
    let view = tex.create_view(&Default::default());
    (tex, view)
}

fn upload_gpu_buf(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    bytes: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let size = (bytes.len() as u64).max(wgpu::COPY_BUFFER_ALIGNMENT);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !bytes.is_empty() {
        queue.write_buffer(&buf, 0, bytes);
    }
    buf
}

fn upload_paint_mesh(device: &wgpu::Device, queue: &wgpu::Queue, mesh: &Mesh) -> PaintMesh {
    let verts: Vec<UvVertex> = mesh
        .positions
        .iter()
        .enumerate()
        .map(|(i, p)| UvVertex {
            pos: *p,
            normal: mesh.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
            uv: mesh
                .uvs
                .first()
                .and_then(|c| c.get(i).copied())
                .unwrap_or([0.0, 0.0]),
            tangent: mesh
                .tangents
                .get(i)
                .copied()
                .unwrap_or([1.0, 0.0, 0.0, 1.0]),
        })
        .collect();
    PaintMesh {
        vertex_buf: upload_gpu_buf(
            device,
            queue,
            "gpu_paint_mesh_vb",
            bytemuck::cast_slice(&verts),
            wgpu::BufferUsages::VERTEX,
        ),
        index_buf: upload_gpu_buf(
            device,
            queue,
            "gpu_paint_mesh_ib",
            bytemuck::cast_slice(&mesh.indices),
            wgpu::BufferUsages::INDEX,
        ),
        index_count: mesh.indices.len() as u32,
        synced: mesh.version,
    }
}

pub fn write_paint_rgba(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    rgba: &[u8],
    width: u32,
    height: u32,
) {
    let w = width.max(1);
    let h = height.max(1);
    let need = (w * h * 4) as usize;
    if rgba.len() < need {
        return;
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &rgba[..need],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * w),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
}

pub fn cursor_to_map_px(screen: Vec2, viewport: Rect, map_w: u32, map_h: u32) -> Vec2 {
    let w = viewport.width().max(1.0);
    let h = viewport.height().max(1.0);
    let u = ((screen.x - viewport.min.x) / w).clamp(0.0, 1.0);
    let v = ((screen.y - viewport.min.y) / h).clamp(0.0, 1.0);
    Vec2::new(u * map_w as f32, v * map_h as f32)
}

pub fn world_radius_to_px(world_r: f32, distance: f32, fov_y: f32, map_h: u32) -> f32 {
    // `distance` = camera → surface (not orbit target — that under-sizes close hits).
    let dist = distance.max(0.05);
    let half = (fov_y * 0.5).tan().max(1e-4);
    let world_h = 2.0 * dist * half;
    (world_r / world_h * map_h as f32).clamp(2.0, map_h as f32 * 0.75)
}

