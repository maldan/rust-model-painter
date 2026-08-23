mod app;
mod brush_alpha;
mod bvh;
mod gpu_paint;
mod paint;
mod pick;
mod segment;
mod uv;
mod uv_view;

use std::sync::Arc;
use std::time::{Duration, Instant};

use app::{Painter, SCENE_TEX, UV_TEX};
use brush_alpha::{ALPHA_TEX_BASE, NRM_TEX_BASE};
use glam::Vec2;
use gpu_paint::{cursor_to_map_px, write_paint_rgba, CompositeLayer, GpuPaint};
use mega_render::{Visualizer, WgpuVisualizer, WGPU_FEATURES};
use mega_ui::wgpu::UiRenderer;
use mega_ui::{CursorIcon, Ui, UiInput};
use paint::{luma, LayerKind, PaintMap, PaintTool, TEX_SIZE};
use pick::find_viewport_rect;
use uv_view::UvPreview;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

struct SceneTarget {
    _texture: wgpu::Texture,
    render_view: wgpu::TextureView,
    size: (u32, u32),
}

impl SceneTarget {
    fn ensure(device: &wgpu::Device, ui: &mut UiRenderer, size: (u32, u32)) -> Self {
        let (w, h) = (size.0.max(1), size.1.max(1));
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("painter scene color"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let render_view = texture.create_view(&Default::default());
        ui.bind_texture_view(device, SCENE_TEX, texture.create_view(&Default::default()));
        Self {
            _texture: texture,
            render_view,
            size: (w, h),
        }
    }

    fn resize(&mut self, device: &wgpu::Device, ui: &mut UiRenderer, w: u32, h: u32) {
        let w = w.max(1);
        let h = h.max(1);
        if self.size == (w, h) {
            return;
        }
        *self = Self::ensure(device, ui, (w, h));
    }
}

struct UvTarget {
    _texture: wgpu::Texture,
    render_view: wgpu::TextureView,
    size: (u32, u32),
}

impl UvTarget {
    fn ensure(device: &wgpu::Device, ui: &mut UiRenderer, size: (u32, u32)) -> Self {
        let (w, h) = (size.0.max(1), size.1.max(1));
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("painter uv color"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let render_view = texture.create_view(&Default::default());
        ui.bind_texture_view(device, UV_TEX, texture.create_view(&Default::default()));
        Self {
            _texture: texture,
            render_view,
            size: (w, h),
        }
    }

    fn resize(&mut self, device: &wgpu::Device, ui: &mut UiRenderer, w: u32, h: u32) {
        let w = w.max(1);
        let h = h.max(1);
        if self.size == (w, h) {
            return;
        }
        *self = Self::ensure(device, ui, (w, h));
    }
}

#[derive(Default)]
struct FrameInput {
    mouse_pos: Vec2,
    mouse_down: bool,
    mouse_pressed: bool,
    mouse_released: bool,
    mouse_right_down: bool,
    mouse_right_pressed: bool,
    mouse_right_released: bool,
    mouse_middle_down: bool,
    mouse_middle_pressed: bool,
    mouse_middle_released: bool,
    scroll_delta: Vec2,
    text: String,
    key_backspace: bool,
    key_delete: bool,
    key_enter: bool,
    key_left: bool,
    key_right: bool,
    key_up: bool,
    key_down: bool,
    key_home: bool,
    key_end: bool,
    key_shift: bool,
    key_ctrl: bool,
    key_copy: bool,
    key_paste: bool,
    key_cut: bool,
    key_select_all: bool,
    key_duplicate: bool,
    modifiers: winit::keyboard::ModifiersState,
    clipboard_paste: String,
    look_delta: Vec2,
}

impl FrameInput {
    fn clear_edges(&mut self) {
        self.mouse_pressed = false;
        self.mouse_released = false;
        self.mouse_right_pressed = false;
        self.mouse_right_released = false;
        self.mouse_middle_pressed = false;
        self.mouse_middle_released = false;
        self.scroll_delta = Vec2::ZERO;
        self.look_delta = Vec2::ZERO;
        self.text.clear();
        self.key_backspace = false;
        self.key_delete = false;
        self.key_enter = false;
        self.key_left = false;
        self.key_right = false;
        self.key_up = false;
        self.key_down = false;
        self.key_home = false;
        self.key_end = false;
        self.key_copy = false;
        self.key_paste = false;
        self.key_cut = false;
        self.key_select_all = false;
        self.key_duplicate = false;
        self.clipboard_paste.clear();
    }

    fn to_ui(&self, viewport: Vec2, dt: f32) -> UiInput {
        UiInput {
            mouse_pos: self.mouse_pos,
            mouse_down: self.mouse_down,
            mouse_pressed: self.mouse_pressed,
            mouse_released: self.mouse_released,
            mouse_right_down: self.mouse_right_down,
            mouse_right_pressed: self.mouse_right_pressed,
            mouse_right_released: self.mouse_right_released,
            mouse_middle_down: self.mouse_middle_down,
            mouse_middle_pressed: self.mouse_middle_pressed,
            mouse_middle_released: self.mouse_middle_released,
            viewport,
            scroll_delta: self.scroll_delta,
            dt,
            text: self.text.clone(),
            key_backspace: self.key_backspace,
            key_delete: self.key_delete,
            key_enter: self.key_enter,
            key_left: self.key_left,
            key_right: self.key_right,
            key_up: self.key_up,
            key_down: self.key_down,
            key_home: self.key_home,
            key_end: self.key_end,
            key_shift: self.key_shift,
            key_ctrl: self.key_ctrl || self.modifiers.super_key(),
            key_copy: self.key_copy,
            key_paste: self.key_paste,
            key_cut: self.key_cut,
            key_select_all: self.key_select_all,
            key_duplicate: self.key_duplicate,
            clipboard: self.clipboard_paste.clone(),
        }
    }
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    visualizer: WgpuVisualizer,
    ui_renderer: UiRenderer,
    scene_target: SceneTarget,
    uv_target: UvTarget,
    gpu_paint: GpuPaint,
    uv_preview: UvPreview,
}

struct Host {
    window: Option<Arc<Window>>,
    gpu: Option<Gpu>,
    painter: Painter,
    ui: Ui,
    input: FrameInput,
    last_frame: Instant,
    fps_accum_dt: f32,
    fps_frames: u32,
    fps: f32,
    animating: bool,
    cursor: CursorIcon,
    clipboard: Option<arboard::Clipboard>,
    want_capture_mouse: bool,
    looking: bool,
    panning: bool,
    uv_panning: bool,
    last_viewport_rect: Option<mega_ui::Rect>,
    last_uv_rect: Option<mega_ui::Rect>,
}

impl Host {
    fn new() -> Self {
        let mut ui = Ui::new();
        ui.load_builtin_icons();
        Self {
            window: None,
            gpu: None,
            painter: Painter::new(),
            ui,
            input: FrameInput::default(),
            last_frame: Instant::now(),
            fps_accum_dt: 0.0,
            fps_frames: 0,
            fps: 0.0,
            animating: true,
            cursor: CursorIcon::Default,
            clipboard: arboard::Clipboard::new().ok(),
            want_capture_mouse: false,
            looking: false,
            panning: false,
            uv_panning: false,
            last_viewport_rect: None,
            last_uv_rect: None,
        }
    }

    fn set_looking(&mut self, looking: bool) {
        if self.looking == looking {
            return;
        }
        self.looking = looking;
        if looking {
            self.painter.interrupt_orbit_snap(true);
        }
        let Some(window) = &self.window else {
            return;
        };
        let mode = if looking {
            CursorGrabMode::Locked
        } else {
            CursorGrabMode::None
        };
        if window.set_cursor_grab(mode).is_err() && looking {
            let _ = window.set_cursor_grab(CursorGrabMode::Confined);
        }
        window.set_cursor_visible(!looking);
    }

    fn init_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("no suitable GPU adapter");

        let mut limits = wgpu::Limits::default();
        let adapter_bytes = adapter.limits().max_color_attachment_bytes_per_sample;
        assert!(
            adapter_bytes >= 36,
            "GPU max_color_attachment_bytes_per_sample={adapter_bytes}, need ≥36"
        );
        limits.max_color_attachment_bytes_per_sample = adapter_bytes.min(64);

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("model-painter"),
            required_features: adapter.features() & WGPU_FEATURES,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .expect("request_device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| *f == wgpu::TextureFormat::Rgba8UnormSrgb)
            .expect("adapter must support Rgba8UnormSrgb");

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: caps
                .present_modes
                .iter()
                .copied()
                .find(|m| *m == wgpu::PresentMode::Fifo)
                .unwrap_or(caps.present_modes[0]),
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&device, &config);

        let mut visualizer = WgpuVisualizer::new(&device, &queue);
        let vp = (
            self.painter.viewport_size.x.max(1.0) as u32,
            self.painter.viewport_size.y.max(1.0) as u32,
        );
        visualizer.ensure_target(vp.0, vp.1);
        *visualizer.post_process() = self.painter.post.clone();

        let mut ui_renderer = UiRenderer::new(&device, &queue, format, &self.ui);
        ui_renderer.set_viewport(&queue, width as f32, height as f32);
        let scene_target = SceneTarget::ensure(&device, &mut ui_renderer, vp);
        let uv_size = (
            self.painter.uv.size.x.max(1.0) as u32,
            self.painter.uv.size.y.max(1.0) as u32,
        );
        let uv_target = UvTarget::ensure(&device, &mut ui_renderer, uv_size);
        let mut gpu_paint = GpuPaint::new(&device);
        gpu_paint.upload_alphas(&device, &queue, &self.painter.alphas);
        gpu_paint.upload_normals(&device, &queue, &self.painter.nrm_stamps);
        let uv_preview = UvPreview::new(&device);
        uv_preview.seed_white(&queue);
        for (i, view) in gpu_paint.alpha_views().iter().enumerate() {
            ui_renderer.bind_texture_view(
                &device,
                ALPHA_TEX_BASE + i as u32,
                view.clone(),
            );
        }
        for (i, view) in gpu_paint.nrm_views().iter().enumerate() {
            ui_renderer.bind_texture_view(
                &device,
                NRM_TEX_BASE + i as u32,
                view.clone(),
            );
        }

        self.gpu = Some(Gpu {
            device,
            queue,
            surface,
            config,
            visualizer,
            ui_renderer,
            scene_target,
            uv_target,
            gpu_paint,
            uv_preview,
        });
        self.window = Some(window);
    }

    fn resize(&mut self, width: u32, height: u32) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        if width == 0 || height == 0 {
            return;
        }
        gpu.config.width = width;
        gpu.config.height = height;
        gpu.surface.configure(&gpu.device, &gpu.config);
        gpu.ui_renderer
            .set_viewport(&gpu.queue, width as f32, height as f32);
    }

    fn begin_paste(&mut self) {
        self.input.key_paste = true;
        if !self.input.clipboard_paste.is_empty() {
            return;
        }
        if let Some(cb) = self.clipboard.as_mut() {
            if let Ok(text) = cb.get_text() {
                self.input.clipboard_paste = text;
            }
        }
    }

    fn apply_cursor(&mut self, window: &Window, cursor: CursorIcon) {
        if self.looking || cursor == self.cursor {
            return;
        }
        self.cursor = cursor;
        window.set_cursor(map_cursor(cursor));
    }

    fn redraw(&mut self) {
        let Some(window) = self.window.clone() else {
            return;
        };
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        self.fps_accum_dt += dt;
        self.fps_frames += 1;
        if self.fps_accum_dt >= 1.0 {
            self.fps = self.fps_frames as f32 / self.fps_accum_dt;
            self.fps_accum_dt = 0.0;
            self.fps_frames = 0;
        }

        if self.looking {
            const SENS: f32 = 0.005;
            self.painter.orbit_yaw += self.input.look_delta.x * SENS;
            self.painter.orbit_pitch += self.input.look_delta.y * SENS;
            self.painter.apply_camera();
        } else if self.uv_panning {
            self.painter
                .uv
                .pan_px(self.input.look_delta.x, self.input.look_delta.y);
        } else {
            self.painter.tick_orbit_snap(dt);
        }
        if self.panning {
            self.painter
                .pan_camera(self.input.look_delta.x, self.input.look_delta.y);
        }
        let over_uv = self
            .last_uv_rect
            .is_some_and(|r| r.contains(self.input.mouse_pos));
        if self.input.scroll_delta.y.abs() > 0.0 && over_uv {
            if let Some(rect) = self.last_uv_rect {
                let local = self.input.mouse_pos - rect.min;
                let factor = (1.0 - self.input.scroll_delta.y * 0.0015).clamp(0.5, 2.0);
                self.painter.uv.zoom_at(factor, local);
            }
        } else if self.input.scroll_delta.y.abs() > 0.0
            && self
                .last_viewport_rect
                .is_some_and(|r| r.contains(self.input.mouse_pos))
        {
            let zoom = 1.0 - self.input.scroll_delta.y * 0.0015;
            self.painter.orbit_dist *= zoom;
            self.painter.apply_camera();
        }

        let viewport = Vec2::new(size.width as f32, size.height as f32);
        let ui_input = if self.looking
            || self.panning
            || self.uv_panning
            || self.painter.orbit_snap_active()
        {
            let mut starved = self.input.to_ui(viewport, dt);
            starved.mouse_down = false;
            starved.mouse_pressed = false;
            starved.mouse_released = false;
            starved.mouse_right_down = false;
            starved.mouse_right_pressed = false;
            starved.mouse_right_released = false;
            starved.scroll_delta = Vec2::ZERO;
            starved
        } else {
            self.input.to_ui(viewport, dt)
        };

        let mouse_pos = self.input.mouse_pos;
        let mouse_down = self.input.mouse_down;
        let mouse_pressed = self.input.mouse_pressed;
        let mouse_released = self.input.mouse_released;

        self.ui.begin_frame(ui_input);
        let fps = self.fps;
        let keep_ui = self.painter.build_ui(&mut self.ui, viewport, fps);
        let out = self.ui.end_frame();
        self.want_capture_mouse = out.want_capture_mouse;
        self.last_viewport_rect = find_viewport_rect(&out.draw_list, SCENE_TEX);
        self.last_uv_rect = find_viewport_rect(&out.draw_list, UV_TEX);

        // Paint / view-gizmo snap
        if !self.looking && !self.panning && !self.uv_panning {
            if let Some(rect) = self.last_viewport_rect {
                let over = rect.contains(mouse_pos);
                let over_gizmo = over && self.painter.over_view_gizmo(mouse_pos, rect);
                if mouse_pressed && over && !out.want_capture_mouse {
                    if over_gizmo {
                        if let Some(axis) = self.painter.pick_view_gizmo(mouse_pos, rect) {
                            self.painter.snap_orbit_to_dir(axis.dir());
                            self.painter.status = format!("View {} · ortho", axis.label());
                        }
                    } else {
                        self.painter.painting = true;
                        self.painter.begin_stroke();
                        self.painter.viewport_interact(mouse_pos, rect);
                    }
                } else if self.painter.painting && mouse_down && over && !over_gizmo {
                    self.painter.viewport_interact(mouse_pos, rect);
                }
                if mouse_released || !mouse_down {
                    if self.painter.painting {
                        self.painter.end_stroke();
                    }
                    self.painter.painting = false;
                }
            }
        }

        let needs_composite = self.painter.docs.iter().any(|d| d.composite_dirty)
            || self.painter.needs_map_seed;
        self.animating = keep_ui
            || self.looking
            || self.panning
            || self.uv_panning
            || self.painter.painting
            || self.painter.orbit_snap_active()
            || out.needs_repaint
            || needs_composite;

        // Brush cursor (scene HUD on viewport texture).
        let vp_rect = self.last_viewport_rect.unwrap_or(mega_ui::Rect {
            min: Vec2::ZERO,
            max: Vec2::ZERO,
        });
        let show = self.last_viewport_rect.is_some()
            && !self.looking
            && !self.panning
            && !self.uv_panning
            && !out.want_capture_mouse
            && vp_rect.contains(mouse_pos);
        self.painter
            .update_brush_cursor(mouse_pos, vp_rect, show);
        self.painter.sync_segment_overlay();

        if let Some(text) = out.clipboard {
            if let Some(cb) = self.clipboard.as_mut() {
                let _ = cb.set_text(text);
            }
        }
        let cursor = out.cursor;
        let draw_list = out.draw_list;

        {
            let Some(gpu) = self.gpu.as_mut() else {
                return;
            };

            let vp_w = self.painter.viewport_size.x.round().max(1.0) as u32;
            let vp_h = self.painter.viewport_size.y.round().max(1.0) as u32;
            gpu.scene_target
                .resize(&gpu.device, &mut gpu.ui_renderer, vp_w, vp_h);
            gpu.visualizer.ensure_target(vp_w, vp_h);
            *gpu.visualizer.post_process() = self.painter.post.clone();

            gpu.visualizer.sync(&self.painter.scene);
            gpu.visualizer.set_debug_view(self.painter.debug_view);

            // gpu_resident maps keep GPU as source of truth — force a clean base upload
            // after create/load so the first frame isn't uninitialized garbage.
            if self.painter.needs_map_seed {
                let handles = self.painter.dst_map_handles();
                let mut ready = true;
                for h in &handles {
                    if gpu.visualizer.texture_gpu(*h).is_none() {
                        ready = false;
                        break;
                    }
                }
                if ready {
                    for h in handles {
                        let Some(gpu_tex) = gpu.visualizer.texture_gpu(h).cloned() else {
                            continue;
                        };
                        if let Some(cpu) = self.painter.scene.textures.get(h) {
                            write_paint_rgba(&gpu.queue, &gpu_tex, &cpu.rgba, cpu.width, cpu.height);
                        }
                    }
                    self.painter.needs_map_seed = false;
                    for doc in &mut self.painter.docs {
                        doc.mark_dirty();
                    }
                }
            }

            // Clear / init layer and mask textures flagged by UI (after sync so GPU tex exists).
            for doc in &mut self.painter.docs {
                for layer in &mut doc.layers {
                    if layer.needs_clear {
                        let tiles = layer.content_tiles(&self.painter.udim_ids);
                        if tiles.is_empty() {
                            layer.needs_clear = false;
                        } else {
                            let mut ok = true;
                            for (_, h) in &tiles {
                                let Some(tex) = gpu.visualizer.texture_gpu(*h).cloned() else {
                                    ok = false;
                                    break;
                                };
                                gpu.gpu_paint.clear_texture(&gpu.queue, &tex, (TEX_SIZE, TEX_SIZE));
                            }
                            if ok {
                                layer.needs_clear = false;
                                doc.composite_dirty = true;
                            } else {
                                doc.composite_dirty = true;
                            }
                        }
                    }
                    if layer.mask_init.is_some() {
                        let tiles = layer.mask_tiles(&self.painter.udim_ids);
                        if tiles.is_empty() {
                            layer.mask_init = None;
                        } else {
                            let rgba = layer.mask_init.unwrap_or([255, 255, 255, 255]);
                            let mut ok = true;
                            for (_, h) in &tiles {
                                let Some(tex) = gpu.visualizer.texture_gpu(*h).cloned() else {
                                    ok = false;
                                    break;
                                };
                                gpu.gpu_paint.fill_texture(
                                    &gpu.queue,
                                    &tex,
                                    (TEX_SIZE, TEX_SIZE),
                                    rgba,
                                );
                            }
                            if ok {
                                layer.mask_init = None;
                                doc.composite_dirty = true;
                            } else {
                                doc.composite_dirty = true;
                            }
                        }
                    }
                }
            }

            let stamps = self.painter.take_pending_stamps();
            let mut did_stamp = false;
            if !stamps.is_empty() {
                let aspect = vp_w as f32 / vp_h as f32;
                gpu.gpu_paint.ensure_uv_targets(&gpu.device, vp_w, vp_h);

                let mut encoder = gpu.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("gpu_paint"),
                    },
                );
                gpu.gpu_paint.render_uv_map(
                    &gpu.device,
                    &gpu.queue,
                    &mut encoder,
                    &self.painter.scene,
                    &self.painter.paintable,
                    aspect,
                );

                // Stamp into the active paint layer or its mask — never into the material composite.
                let stamp_tiles = self.painter.stamp_target_tiles();
                if !stamp_tiles.is_empty() {
                    let stamp_brush = self.painter.stamp_brush();
                    let erase = self.painter.tool == PaintTool::Eraser;
                    let nrm_stamp = self.painter.paint_map == PaintMap::Normal
                        && !self.painter.painting_mask();
                    let (stamp_idx, coverage_stamp) = if nrm_stamp {
                        (self.painter.active_nrm, false)
                    } else {
                        (
                            self.painter.active_alpha,
                            self.painter
                                .alphas
                                .get(self.painter.active_alpha)
                                .is_some_and(|a| a.coverage_stamp),
                        )
                    };
                    for &(udim, layer_h) in &stamp_tiles {
                        let Some(paint_tex) = gpu.visualizer.texture_gpu(layer_h).cloned() else {
                            continue;
                        };
                        let tile = crate::uv::udim_origin(udim);
                        for stamp in &stamps {
                            // All tiles: same-UDIM is free; cross-UDIM needs coplanar seam
                            // (see splat.wgsl) so a circle on the cut hits both without splash.
                            let center = cursor_to_map_px(stamp.screen, stamp.viewport, vp_w, vp_h);
                            let screen_r = stamp.screen_radius_px.max(4.0);
                            gpu.gpu_paint.stamp(
                                &gpu.device,
                                &gpu.queue,
                                &mut encoder,
                                &paint_tex,
                                (TEX_SIZE, TEX_SIZE),
                                &stamp_brush,
                                [1.0, 1.0, 1.0, 1.0],
                                erase,
                                center,
                                screen_r,
                                stamp_brush.radius,
                                stamp_idx,
                                stamp.plane_normal,
                                coverage_stamp,
                                nrm_stamp,
                                tile,
                            );
                            did_stamp = true;
                        }
                    }
                }
                gpu.queue.submit(Some(encoder.finish()));
                if did_stamp {
                    self.painter.doc_mut().mark_dirty();
                }
            }

            // If either MR stack is dirty, rebuild both channels from a clean seed.
            if self.painter.docs[PaintMap::Metallic.index()].composite_dirty
                || self.painter.docs[PaintMap::Roughness.index()].composite_dirty
            {
                self.painter.docs[PaintMap::Metallic.index()].mark_dirty();
                self.painter.docs[PaintMap::Roughness.index()].mark_dirty();
                for (_, h) in self.painter.material_map_tiles(PaintMap::Metallic) {
                    let Some(mr) = gpu.visualizer.texture_gpu(h).cloned() else {
                        continue;
                    };
                    if let Some(cpu) = self.painter.scene.textures.get(h) {
                        write_paint_rgba(&gpu.queue, &mr, &cpu.rgba, cpu.width, cpu.height);
                    }
                }
            }

            // Recomposite dirty paint maps into material textures.
            // One submit per map: `composite_stack` writes a shared UBO via
            // `queue.write_buffer`, which is not ordered inside the encoder — packing
            // Metallic then Roughness into one encoder made Metallic run with the
            // Roughness channel mask (paint landed in G, not B).
            let dirty_maps: Vec<_> = PaintMap::ALL
                .iter()
                .copied()
                .filter(|&m| self.painter.docs[m.index()].composite_dirty)
                .collect();
            for map in dirty_maps {
                let _ = self.painter.docs[map.index()].take_composite_dirty();
                let dst_tiles = self.painter.material_map_tiles(map);
                if dst_tiles.is_empty() {
                    self.painter.docs[map.index()].mark_dirty();
                    continue;
                }

                let mut any_missing_dst = false;
                for &(udim, dst_h) in &dst_tiles {
                    let Some(dst) = gpu.visualizer.texture_gpu(dst_h).cloned() else {
                        any_missing_dst = true;
                        continue;
                    };

                    let mut owned: Vec<(
                        f32,
                        u32,
                        [f32; 4],
                        Option<wgpu::Texture>,
                        Option<wgpu::Texture>,
                    )> = Vec::new();
                    let mut layers_ready = true;
                    for layer in self.painter.docs[map.index()]
                        .layers
                        .iter()
                        .filter(|l| l.visible && l.opacity > 0.001)
                    {
                        if layer.needs_clear || layer.mask_init.is_some() {
                            layers_ready = false;
                            break;
                        }
                        let content = match layer.kind {
                            LayerKind::Paint => {
                                let Some((_, h)) = layer
                                    .content_tiles(&self.painter.udim_ids)
                                    .into_iter()
                                    .find(|(id, _)| *id == udim)
                                else {
                                    layers_ready = false;
                                    break;
                                };
                                let Some(t) = gpu.visualizer.texture_gpu(h).cloned() else {
                                    layers_ready = false;
                                    break;
                                };
                                Some(t)
                            }
                            LayerKind::Fill => None,
                        };
                        let mask = if let Some((_, h)) = layer
                            .mask_tiles(&self.painter.udim_ids)
                            .into_iter()
                            .find(|(id, _)| *id == udim)
                        {
                            let Some(t) = gpu.visualizer.texture_gpu(h).cloned() else {
                                layers_ready = false;
                                break;
                            };
                            Some(t)
                        } else {
                            None
                        };
                        let fill = match map {
                            PaintMap::Albedo => layer.fill,
                            PaintMap::Normal => layer.fill,
                            PaintMap::Metallic | PaintMap::Roughness => {
                                let v = luma(layer.fill);
                                [v, v, v, 1.0]
                            }
                        };
                        let mode = match layer.kind {
                            LayerKind::Paint => 1,
                            LayerKind::Fill => 2,
                        };
                        owned.push((layer.opacity, mode, fill, content, mask));
                    }

                    let mut encoder = gpu.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("gpu_paint_composite_stack"),
                        },
                    );
                    if !layers_ready {
                        gpu.gpu_paint.composite_stack(
                            &gpu.device,
                            &gpu.queue,
                            &mut encoder,
                            &dst,
                            (TEX_SIZE, TEX_SIZE),
                            self.painter.docs[map.index()].base_rgba,
                            map.channel_mask(),
                            &[],
                            map == PaintMap::Normal,
                        );
                        any_missing_dst = true;
                    } else {
                        let layer_refs: Vec<CompositeLayer<'_>> = owned
                            .iter()
                            .map(|(opacity, mode, fill, content, mask)| CompositeLayer {
                                opacity: *opacity,
                                mode: *mode,
                                fill: *fill,
                                content: content.as_ref(),
                                mask: mask.as_ref(),
                            })
                            .collect();
                        gpu.gpu_paint.composite_stack(
                            &gpu.device,
                            &gpu.queue,
                            &mut encoder,
                            &dst,
                            (TEX_SIZE, TEX_SIZE),
                            self.painter.docs[map.index()].base_rgba,
                            map.channel_mask(),
                            &layer_refs,
                            map == PaintMap::Normal,
                        );
                    }
                    gpu.queue.submit(Some(encoder.finish()));
                }
                if any_missing_dst {
                    self.painter.docs[map.index()].mark_dirty();
                }
            }

            let aspect = vp_w as f32 / vp_h as f32;
            gpu.visualizer
                .render_to(&self.painter.scene, aspect, &gpu.scene_target.render_view);

            if self.painter.uv.needs_fit {
                self.painter.uv.fit(&self.painter.udim_ids);
            }
            let uv_w = self.painter.uv.size.x.round().max(1.0) as u32;
            let uv_h = self.painter.uv.size.y.round().max(1.0) as u32;
            gpu.uv_target
                .resize(&gpu.device, &mut gpu.ui_renderer, uv_w, uv_h);
            let tiles = self.painter.material_map_tiles(self.painter.paint_map);
            let mesh_node = self.painter.uv_mesh_node();
            let mut uv_enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("uv_preview"),
            });
            gpu.uv_preview.render(
                &gpu.device,
                &gpu.queue,
                &mut uv_enc,
                &gpu.uv_target.render_view,
                &gpu.visualizer,
                &self.painter.scene,
                &self.painter.uv,
                self.painter.paint_map,
                &tiles,
                mesh_node,
                &self.painter.udim_ids,
            );
            gpu.queue.submit(Some(uv_enc.finish()));

            gpu.ui_renderer
                .sync_atlases(&gpu.device, &gpu.queue, &mut self.ui);

            let frame = match gpu.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t)
                | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
                wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                    gpu.surface.configure(&gpu.device, &gpu.config);
                    window.request_redraw();
                    return;
                }
                _ => {
                    window.request_redraw();
                    return;
                }
            };

            let view = frame.texture.create_view(&Default::default());
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("painter frame"),
                });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ui pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.04,
                                g: 0.04,
                                b: 0.04,
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
                let _ = gpu.ui_renderer.draw(&gpu.queue, &mut pass, &draw_list);
            }

            gpu.queue.submit(Some(encoder.finish()));
            window.pre_present_notify();
            gpu.queue.present(frame);
        }

        self.input.clear_edges();
        self.apply_cursor(&window, cursor);
    }
}

impl ApplicationHandler for Host {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("model-painter")
            .with_inner_size(LogicalSize::new(1440.0, 900.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        self.init_gpu(window.clone());
        window.request_redraw();
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let ms = if self.animating
            || self.looking
            || self.panning
            || self.uv_panning
            || self.painter.painting
            || self.painter.orbit_snap_active()
        {
            8
        } else {
            33
        };
        event_loop.set_control_flow(ControlFlow::wait_duration(Duration::from_millis(ms)));
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.looking || self.panning || self.uv_panning {
                self.input.look_delta += Vec2::new(delta.0 as f32, delta.1 as f32);
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.resize(size.width, size.height);
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                if let Some(size) = self.window.as_ref().map(|w| w.inner_size()) {
                    self.resize(size.width, size.height);
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::CursorMoved { position, .. } => {
                self.input.mouse_pos = Vec2::new(position.x as f32, position.y as f32);
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        if down && !self.input.mouse_down {
                            self.input.mouse_pressed = true;
                        }
                        if !down && self.input.mouse_down {
                            self.input.mouse_released = true;
                        }
                        self.input.mouse_down = down;
                    }
                    MouseButton::Right => {
                        if down && !self.input.mouse_right_down {
                            self.input.mouse_right_pressed = true;
                        }
                        if !down && self.input.mouse_right_down {
                            self.input.mouse_right_released = true;
                        }
                        self.input.mouse_right_down = down;

                        if down && !self.want_capture_mouse && !self.looking && !self.panning && !self.uv_panning {
                            let over_uv = self
                                .last_uv_rect
                                .is_some_and(|r| r.contains(self.input.mouse_pos));
                            let over_vp = self
                                .last_viewport_rect
                                .is_some_and(|r| r.contains(self.input.mouse_pos));
                            let over_gizmo = self.last_viewport_rect.is_some_and(|r| {
                                self.painter.over_view_gizmo(self.input.mouse_pos, r)
                            });
                            if over_uv {
                                // Orbit stays on the 3D view.
                            } else if over_vp && !over_gizmo {
                                self.set_looking(true);
                            }
                        } else if !down {
                            self.set_looking(false);
                        }
                    }
                    MouseButton::Middle => {
                        if down && !self.input.mouse_middle_down {
                            self.input.mouse_middle_pressed = true;
                        }
                        if !down && self.input.mouse_middle_down {
                            self.input.mouse_middle_released = true;
                        }
                        self.input.mouse_middle_down = down;

                        if down && !self.want_capture_mouse && !self.looking {
                            let over_uv = self
                                .last_uv_rect
                                .is_some_and(|r| r.contains(self.input.mouse_pos));
                            let over_vp = self
                                .last_viewport_rect
                                .is_some_and(|r| r.contains(self.input.mouse_pos));
                            let over_gizmo = self.last_viewport_rect.is_some_and(|r| {
                                self.painter.over_view_gizmo(self.input.mouse_pos, r)
                            });
                            if over_uv {
                                self.uv_panning = true;
                                self.painter.painting = false;
                            } else if over_vp && !over_gizmo {
                                self.panning = true;
                                self.painter.painting = false;
                            }
                        } else if !down {
                            self.panning = false;
                            self.uv_panning = false;
                        }
                    }
                    _ => {}
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.input.scroll_delta = match delta {
                    MouseScrollDelta::LineDelta(x, y) => Vec2::new(x * 40.0, y * 40.0),
                    MouseScrollDelta::PixelDelta(p) => Vec2::new(p.x as f32, p.y as f32),
                };
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::ModifiersChanged(m) => {
                self.input.modifiers = m.state();
                self.input.key_shift = m.state().shift_key();
                self.input.key_ctrl = m.state().control_key() || m.state().super_key();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let down = event.state == ElementState::Pressed;
                if down {
                    if let Some(text) = event.text.as_ref() {
                        for ch in text.chars() {
                            if !ch.is_control() {
                                self.input.text.push(ch);
                            }
                        }
                    }
                }
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Backspace) => self.input.key_backspace = down,
                    PhysicalKey::Code(KeyCode::Delete) => self.input.key_delete = down,
                    PhysicalKey::Code(KeyCode::Enter) => self.input.key_enter = down,
                    PhysicalKey::Code(KeyCode::ArrowLeft) => self.input.key_left = down,
                    PhysicalKey::Code(KeyCode::ArrowRight) => self.input.key_right = down,
                    PhysicalKey::Code(KeyCode::ArrowUp) => self.input.key_up = down,
                    PhysicalKey::Code(KeyCode::ArrowDown) => self.input.key_down = down,
                    PhysicalKey::Code(KeyCode::Home) => self.input.key_home = down,
                    PhysicalKey::Code(KeyCode::End) => self.input.key_end = down,
                    PhysicalKey::Code(KeyCode::KeyC)
                        if down && (self.input.modifiers.control_key() || self.input.modifiers.super_key()) =>
                    {
                        self.input.key_copy = true;
                    }
                    PhysicalKey::Code(KeyCode::KeyV)
                        if down && (self.input.modifiers.control_key() || self.input.modifiers.super_key()) =>
                    {
                        self.begin_paste();
                    }
                    PhysicalKey::Code(KeyCode::KeyX)
                        if down && (self.input.modifiers.control_key() || self.input.modifiers.super_key()) =>
                    {
                        self.input.key_cut = true;
                    }
                    PhysicalKey::Code(KeyCode::KeyA)
                        if down && (self.input.modifiers.control_key() || self.input.modifiers.super_key()) =>
                    {
                        self.input.key_select_all = true;
                    }
                    PhysicalKey::Code(KeyCode::KeyD)
                        if down && (self.input.modifiers.control_key() || self.input.modifiers.super_key()) =>
                    {
                        self.input.key_duplicate = true;
                    }
                    _ => {}
                }
                if down {
                    if let Key::Named(NamedKey::Escape) = event.logical_key {
                        if self.looking {
                            self.set_looking(false);
                        } else {
                            event_loop.exit();
                        }
                    }
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn map_cursor(c: CursorIcon) -> winit::window::CursorIcon {
    match c {
        CursorIcon::Default => winit::window::CursorIcon::Default,
        CursorIcon::Pointer => winit::window::CursorIcon::Pointer,
        CursorIcon::Move => winit::window::CursorIcon::Move,
        CursorIcon::ResizeNwse => winit::window::CursorIcon::NwseResize,
        CursorIcon::ResizeEw => winit::window::CursorIcon::EwResize,
        CursorIcon::ResizeNs => winit::window::CursorIcon::NsResize,
        CursorIcon::Text => winit::window::CursorIcon::Text,
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut host = Host::new();
    event_loop.run_app(&mut host).expect("run app");
}
