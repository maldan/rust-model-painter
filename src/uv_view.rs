//! 2D UV / UDIM preview: textured tiles + optional island wire overlay.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use mega_render::{Handle, Mesh, Node, Scene, Texture, WgpuVisualizer};
use wgpu::util::DeviceExt;

use crate::paint::PaintMap;
use crate::uv::udim_origin;

#[derive(Clone, Copy)]
pub struct UvView {
    pub mesh_idx: usize,
    pub show_uv: bool,
    pub pan: Vec2,
    pub zoom: f32,
    pub size: Vec2,
    pub needs_fit: bool,
}

impl Default for UvView {
    fn default() -> Self {
        Self {
            mesh_idx: 0,
            show_uv: true,
            pan: Vec2::new(0.5, 0.5),
            zoom: 1.25,
            size: Vec2::new(256.0, 256.0),
            needs_fit: true,
        }
    }
}

impl UvView {
    pub fn aspect(&self) -> f32 {
        (self.size.x / self.size.y.max(1.0)).max(0.05)
    }

    pub fn half_extent(&self) -> Vec2 {
        let h = (self.zoom * 0.5).max(0.02);
        Vec2::new(h * self.aspect(), h)
    }

    pub fn screen_to_uv(&self, local: Vec2) -> Vec2 {
        let size = self.size.max(Vec2::splat(1.0));
        let half = self.half_extent();
        let n = Vec2::new(local.x / size.x, local.y / size.y);
        Vec2::new(
            self.pan.x + (n.x - 0.5) * 2.0 * half.x,
            self.pan.y + (0.5 - n.y) * 2.0 * half.y,
        )
    }

    pub fn zoom_at(&mut self, factor: f32, pivot: Vec2) {
        let factor = factor.clamp(0.5, 2.0);
        let before = self.screen_to_uv(pivot);
        self.zoom = (self.zoom * factor).clamp(0.15, 64.0);
        let after = self.screen_to_uv(pivot);
        self.pan += before - after;
    }

    pub fn pan_px(&mut self, dx: f32, dy: f32) {
        let size = self.size.max(Vec2::splat(1.0));
        let half = self.half_extent();
        self.pan.x -= dx / size.x * half.x * 2.0;
        self.pan.y += dy / size.y * half.y * 2.0;
    }

    pub fn fit(&mut self, udims: &[u32]) {
        let mut min = Vec2::splat(f32::MAX);
        let mut max = Vec2::splat(f32::MIN);
        let ids = if udims.is_empty() {
            &[1001u32][..]
        } else {
            udims
        };
        for &id in ids {
            let o = udim_origin(id);
            let p = Vec2::new(o[0], o[1]);
            min = min.min(p);
            max = max.max(p + Vec2::ONE);
        }
        self.pan = (min + max) * 0.5;
        let span = (max - min).max(Vec2::splat(0.5));
        self.zoom = (span.y.max(span.x / self.aspect()) * 1.15).clamp(0.5, 64.0);
        self.needs_fit = false;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ViewUniform {
    pan: [f32; 2],
    half_extent: [f32; 2],
    channel: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TileVert {
    pos: [f32; 2],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LineVert {
    pos: [f32; 2],
    color: [f32; 4],
}

struct WireCache {
    buf: wgpu::Buffer,
    count: u32,
    version: u64,
}

pub struct UvPreview {
    tile_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    view_buf: wgpu::Buffer,
    view_bg: wgpu::BindGroup,
    tile_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    white_view: wgpu::TextureView,
    _white_tex: wgpu::Texture,
    wires: HashMap<(u32, u32), WireCache>,
}

impl UvPreview {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/uv_view.wgsl"));
        let view_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uv_view_layout"),
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
        let tile_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uv_tile_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let tile_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("uv_tile_pl"),
            bind_group_layouts: &[Some(&view_layout), Some(&tile_layout)],
            immediate_size: 0,
        });
        let line_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("uv_line_pl"),
            bind_group_layouts: &[Some(&view_layout)],
            immediate_size: 0,
        });
        let color = [Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let tile_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("uv_tile_pipe"),
            layout: Some(&tile_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_tile"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<TileVert>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_tile"),
                compilation_options: Default::default(),
                targets: &color,
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("uv_line_pipe"),
            layout: Some(&line_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_line"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineVert>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_line"),
                compilation_options: Default::default(),
                targets: &color,
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let view_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uv_view_ubo"),
            size: std::mem::size_of::<ViewUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uv_view_bg"),
            layout: &view_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: view_buf.as_entire_binding(),
            }],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("uv_tile_samp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let white_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("uv_white"),
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
        let white_view = white_tex.create_view(&Default::default());
        Self {
            tile_pipeline,
            line_pipeline,
            view_buf,
            view_bg,
            tile_layout,
            sampler,
            white_view,
            _white_tex: white_tex,
            wires: HashMap::new(),
        }
    }

    pub fn seed_white(&self, queue: &wgpu::Queue) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self._white_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[32u8, 32, 34, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        visualizer: &WgpuVisualizer,
        scene: &Scene,
        view: &UvView,
        map: PaintMap,
        tiles: &[(u32, Handle<Texture>)],
        mesh_node: Option<Handle<Node>>,
        udim_ids: &[u32],
    ) {
        let half = view.half_extent();
        let channel = match map {
            PaintMap::Roughness => 1,
            PaintMap::Metallic => 2,
            _ => 0,
        };
        queue.write_buffer(
            &self.view_buf,
            0,
            bytemuck::bytes_of(&ViewUniform {
                pan: view.pan.to_array(),
                half_extent: half.to_array(),
                channel,
                _pad: 0,
            }),
        );

        let mut tile_bufs: Vec<wgpu::Buffer> = Vec::new();
        let mut tile_views: Vec<wgpu::TextureView> = Vec::new();
        let ids: Vec<u32> = if udim_ids.is_empty() {
            vec![1001]
        } else {
            udim_ids.to_vec()
        };
        for &id in &ids {
            let origin = udim_origin(id);
            let o = Vec2::new(origin[0], origin[1]);
            let verts = tile_quad(o);
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uv_tile_vb"),
                contents: bytemuck::cast_slice(&verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let tex_view = tiles
                .iter()
                .find(|(tid, _)| *tid == id)
                .and_then(|(_, h)| visualizer.texture_view(*h).cloned())
                .unwrap_or_else(|| self.white_view.clone());
            tile_views.push(tex_view);
            tile_bufs.push(buf);
        }

        let border_verts = tile_borders(&ids);
        let border_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uv_border_vb"),
            contents: bytemuck::cast_slice(&border_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let border_count = border_verts.len() as u32;

        let mesh = mesh_node.and_then(|nh| {
            let n = scene.nodes.get(nh)?;
            let mh = n.mesh?;
            scene.meshes.get(mh).map(|m| (mh, m))
        });
        if let Some((mh, m)) = mesh {
            self.sync_wires(device, mh, m);
        }
        let wire = mesh.and_then(|(mh, _)| self.wires.get(&mh.key()));

        let mut tile_bgs = Vec::with_capacity(tile_views.len());
        for view in &tile_views {
            tile_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("uv_tile_bg"),
                layout: &self.tile_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            }));
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("uv_preview"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.07,
                            g: 0.07,
                            b: 0.08,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.tile_pipeline);
            pass.set_bind_group(0, &self.view_bg, &[]);
            for (buf, bg) in tile_bufs.iter().zip(tile_bgs.iter()) {
                pass.set_bind_group(1, bg, &[]);
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.draw(0..6, 0..1);
            }
            pass.set_pipeline(&self.line_pipeline);
            pass.set_bind_group(0, &self.view_bg, &[]);
            pass.set_vertex_buffer(0, border_buf.slice(..));
            pass.draw(0..border_count, 0..1);
            if view.show_uv {
                if let Some(w) = wire {
                    if w.count > 0 {
                        pass.set_vertex_buffer(0, w.buf.slice(..));
                        pass.draw(0..w.count, 0..1);
                    }
                }
            }
        }
    }

    fn sync_wires(&mut self, device: &wgpu::Device, mh: Handle<Mesh>, mesh: &Mesh) {
        let key = mh.key();
        if self
            .wires
            .get(&key)
            .is_some_and(|c| c.version == mesh.version)
        {
            return;
        }
        let verts = uv_wires(mesh);
        let count = verts.len() as u32;
        let dummy = [LineVert {
            pos: [0.0; 2],
            color: [0.0; 4],
        }];
        let bytes: &[u8] = if verts.is_empty() {
            bytemuck::bytes_of(&dummy)
        } else {
            bytemuck::cast_slice(&verts)
        };
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uv_wire_vb"),
            contents: bytes,
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.wires.insert(
            key,
            WireCache {
                buf,
                count,
                version: mesh.version,
            },
        );
    }
}

fn tile_quad(origin: Vec2) -> [TileVert; 6] {
    let a = origin;
    let b = origin + Vec2::X;
    let c = origin + Vec2::ONE;
    let d = origin + Vec2::Y;
    [
        TileVert {
            pos: a.to_array(),
            uv: [0.0, 0.0],
        },
        TileVert {
            pos: b.to_array(),
            uv: [1.0, 0.0],
        },
        TileVert {
            pos: c.to_array(),
            uv: [1.0, 1.0],
        },
        TileVert {
            pos: a.to_array(),
            uv: [0.0, 0.0],
        },
        TileVert {
            pos: c.to_array(),
            uv: [1.0, 1.0],
        },
        TileVert {
            pos: d.to_array(),
            uv: [0.0, 1.0],
        },
    ]
}

fn tile_borders(ids: &[u32]) -> Vec<LineVert> {
    let col = [0.55, 0.56, 0.6, 0.9];
    let mut v = Vec::with_capacity(ids.len() * 8);
    for &id in ids {
        let o = udim_origin(id);
        let a = Vec2::new(o[0], o[1]);
        let b = a + Vec2::X;
        let c = a + Vec2::ONE;
        let d = a + Vec2::Y;
        for (p, q) in [(a, b), (b, c), (c, d), (d, a)] {
            v.push(LineVert {
                pos: p.to_array(),
                color: col,
            });
            v.push(LineVert {
                pos: q.to_array(),
                color: col,
            });
        }
    }
    v
}

fn uv_wires(mesh: &Mesh) -> Vec<LineVert> {
    let Some(uvs) = mesh.uvs.first() else {
        return Vec::new();
    };
    let col = [0.15, 0.85, 1.0, 0.85];
    let mut v = Vec::with_capacity(mesh.indices.len() * 2);
    for tri in mesh.indices.chunks_exact(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        let Some(&u0) = uvs.get(i0) else { continue };
        let Some(&u1) = uvs.get(i1) else { continue };
        let Some(&u2) = uvs.get(i2) else { continue };
        for (a, b) in [(u0, u1), (u1, u2), (u2, u0)] {
            v.push(LineVert {
                pos: a,
                color: col,
            });
            v.push(LineVert {
                pos: b,
                color: col,
            });
        }
    }
    v
}
