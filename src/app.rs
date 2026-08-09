use std::collections::HashMap;
use std::f32::consts::TAU;
use std::path::Path;
use std::time::Instant;

use glam::{Mat4, Vec2, Vec3};
use mega_render::{
    cube, load_gltf, plane, sphere, Camera, DebugView, Handle, InputFrame, Light, Material, Mesh,
    Node, Scene, Texture, Transform,
};
use mega_ui::{DockNode, DockState, ScrollAxes, TextStyle, Ui};

use crate::paint::{Brush, PaintDocument, PaintMap, TEX_SIZE};
use crate::pick::{self, BvhCache};
use mega_ui::Rect;

pub const SCENE_TEX: u32 = 0;

const VIEW_MODES: &[DebugView] = &[
    DebugView::Final,
    DebugView::Albedo,
    DebugView::Metallic,
    DebugView::Roughness,
];

#[derive(Clone, Copy)]
pub struct PendingStamp {
    pub screen: Vec2,
    pub viewport: Rect,
    /// Screen-space search radius at stamp depth (viewport pixels).
    pub screen_radius_px: f32,
}

pub struct Painter {
    pub scene: Scene,
    pub docs: [PaintDocument; 3],
    pub paint_map: PaintMap,
    pub debug_view: DebugView,
    pub brush: Brush,
    pub albedo_tex: Handle<Texture>,
    pub mr_tex: Handle<Texture>,
    pub paintable: Vec<Handle<Node>>,
    pub dock: DockState,
    pub viewport_size: Vec2,
    pub orbit_yaw: f32,
    pub orbit_pitch: f32,
    pub orbit_dist: f32,
    pub orbit_target: Vec3,
    pub painting: bool,
    /// 0 sphere, 1 cube, 2 gltf model
    pub shape: usize,
    pub model_name: String,
    pub status: String,
    /// Layer stack changed — re-upload composite into gpu-resident albedo.
    pub needs_gpu_albedo_upload: bool,
    /// Layer stack changed — re-upload MR channels into gpu-resident map.
    pub needs_gpu_mr_upload: bool,
    pending_stamps: Vec<PendingStamp>,
    last_stamp_px: Option<Vec2>,
    root: Handle<Node>,
    paint_mat: Handle<Material>,
    sphere_node: Handle<Node>,
    cube_node: Handle<Node>,
    model_root: Option<Handle<Node>>,
    bvh_cache: BvhCache,
    /// Scratch for MR channel composite from a grayscale layer stack.
    mr_scratch: Vec<u8>,
}

impl Painter {
    pub fn new() -> Self {
        let mut scene = Scene::new();
        if let Some(Light::Directional(d)) = scene.lights.first_mut() {
            d.intensity = 2.8;
            d.color = [1.0, 0.98, 0.94];
            d.direction = Vec3::new(0.35, -0.55, 0.75).normalize();
        }
        scene.ambient = [0.08, 0.08, 0.09];

        let rough_u8 = (0.45 * 255.0) as u8;
        let docs = [
            PaintDocument::new(TEX_SIZE, TEX_SIZE),
            PaintDocument::with_base(TEX_SIZE, TEX_SIZE, [0, 0, 0]),
            PaintDocument::with_base(TEX_SIZE, TEX_SIZE, [rough_u8, rough_u8, rough_u8]),
        ];

        let mut albedo = Texture::gpu_resident(TEX_SIZE, TEX_SIZE, true);
        docs[0].composite_into(&mut albedo.rgba);
        let albedo_tex = scene.textures.insert(albedo);

        // glTF ORM: R unused, G=roughness, B=metallic. Scalars on material = 1 so map wins.
        let mut mr = Texture::gpu_resident(TEX_SIZE, TEX_SIZE, false);
        for px in mr.rgba.chunks_exact_mut(4) {
            px[0] = 255;
            px[1] = rough_u8;
            px[2] = 0;
            px[3] = 255;
        }
        let mr_tex = scene.textures.insert(mr);

        let mut paint_mat_data = Material::new([1.0, 1.0, 1.0, 1.0], 1.0, 1.0).with_map(albedo_tex);
        paint_mat_data.metallic_roughness_map = Some(mr_tex);
        let paint_mat = scene.materials.insert(paint_mat_data);
        let ground_mat = scene
            .materials
            .insert(Material::new([0.35, 0.36, 0.38, 1.0], 0.0, 0.85));

        let mesh_sphere = scene.meshes.insert(sphere(0.7, 48, 28));
        let mesh_cube = scene.meshes.insert(cube(1.1));
        let mesh_ground = scene.meshes.insert(plane(8.0, 8.0));

        let root = scene.nodes.insert(Node {
            name: "root".into(),
            parent: None,
            local: Transform::default(),
            mesh: None,
            material: None,
            skin: None,
            visible: true,
        });

        scene.nodes.insert(Node {
            name: "ground".into(),
            parent: Some(root),
            local: Transform::default(),
            mesh: Some(mesh_ground),
            material: Some(ground_mat),
            skin: None,
            visible: true,
        });

        let sphere_node = scene.nodes.insert(Node {
            name: "paint_sphere".into(),
            parent: Some(root),
            local: Transform::from_translation(Vec3::new(0.0, 0.7, 0.0)),
            mesh: Some(mesh_sphere),
            material: Some(paint_mat),
            skin: None,
            visible: true,
        });

        let cube_node = scene.nodes.insert(Node {
            name: "paint_cube".into(),
            parent: Some(root),
            local: Transform::from_translation(Vec3::new(0.0, 0.55, 0.0)),
            mesh: Some(mesh_cube),
            material: Some(paint_mat),
            skin: None,
            visible: false,
        });

        let mut painter = Self {
            scene,
            docs,
            paint_map: PaintMap::Albedo,
            debug_view: DebugView::Final,
            brush: Brush::default(),
            albedo_tex,
            mr_tex,
            paintable: vec![sphere_node],
            dock: DockState::new(DockNode::split_h(
                0.72,
                DockNode::leaf(&["Viewport"]),
                DockNode::split_v(
                    0.55,
                    DockNode::leaf(&["Brush", "Lights"]),
                    DockNode::leaf(&["Layers"]),
                ),
            )),
            viewport_size: Vec2::new(1280.0, 720.0),
            orbit_yaw: 0.7,
            orbit_pitch: 0.35,
            orbit_dist: 4.0,
            orbit_target: Vec3::new(0.0, 0.55, 0.0),
            painting: false,
            shape: 0,
            model_name: String::new(),
            status: "Ready · GPU brush".into(),
            needs_gpu_albedo_upload: false,
            needs_gpu_mr_upload: false,
            pending_stamps: Vec::new(),
            last_stamp_px: None,
            root,
            paint_mat,
            sphere_node,
            cube_node,
            model_root: None,
            bvh_cache: HashMap::new(),
            mr_scratch: vec![0; (TEX_SIZE * TEX_SIZE * 4) as usize],
        };
        painter.rebuild_bvhs();
        painter.apply_camera();
        painter
    }

    pub fn doc(&self) -> &PaintDocument {
        &self.docs[self.paint_map.index()]
    }

    pub fn doc_mut(&mut self) -> &mut PaintDocument {
        &mut self.docs[self.paint_map.index()]
    }

    pub fn paint_tex(&self) -> Handle<Texture> {
        match self.paint_map {
            PaintMap::Albedo => self.albedo_tex,
            PaintMap::Metallic | PaintMap::Roughness => self.mr_tex,
        }
    }

    /// Grayscale brush color for metallic / roughness stamps.
    pub fn stamp_brush(&self) -> Brush {
        match self.paint_map {
            PaintMap::Albedo => self.brush,
            PaintMap::Metallic | PaintMap::Roughness => {
                let c = self.brush.color;
                let v = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
                Brush {
                    color: [v, v, v, 1.0],
                    ..self.brush
                }
            }
        }
    }

    pub fn apply_camera(&mut self) {
        self.orbit_pitch = self.orbit_pitch.clamp(-1.4, 1.4);
        self.orbit_dist = self.orbit_dist.clamp(0.5, 80.0);
        self.scene.camera = Camera::orbit(
            self.orbit_yaw,
            self.orbit_pitch,
            self.orbit_dist,
            self.orbit_target,
        );
    }

    pub fn set_shape(&mut self, shape: usize) {
        self.shape = shape;
        let show_sphere = shape == 0;
        let show_cube = shape == 1;
        let show_model = shape == 2 && self.model_root.is_some();

        if let Some(n) = self.scene.nodes.get_mut(self.sphere_node) {
            n.visible = show_sphere;
        }
        if let Some(n) = self.scene.nodes.get_mut(self.cube_node) {
            n.visible = show_cube;
        }
        if let Some(root) = self.model_root {
            set_subtree_visible(&mut self.scene, root, show_model);
        }

        self.paintable = match shape {
            0 => vec![self.sphere_node],
            1 => vec![self.cube_node],
            _ => collect_mesh_nodes(&self.scene, self.model_root),
        };
        self.rebuild_bvhs();
    }

    fn rebuild_bvhs(&mut self) {
        let t0 = Instant::now();
        pick::ensure_bvhs(&self.scene, &self.paintable, &mut self.bvh_cache);
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        let tris: usize = self
            .paintable
            .iter()
            .filter_map(|&h| {
                let n = self.scene.nodes.get(h)?;
                let mh = n.mesh?;
                Some(self.scene.meshes.get(mh)?.indices.len() / 3)
            })
            .sum();
        if tris > 50_000 {
            self.status = format!("BVH ready · {tris} tris · {ms:.0} ms");
        }
    }

    pub fn open_model_dialog(&mut self) {
        let picked = rfd::FileDialog::new()
            .set_title("Open glTF / GLB")
            .add_filter("glTF / GLB", &["gltf", "glb"])
            .pick_file();
        let Some(path) = picked else {
            return;
        };
        match self.load_gltf_path(&path) {
            Ok(()) => {
                let name = path
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.display().to_string());
                self.model_name = name.clone();
                if !self.status.starts_with("BVH") {
                    self.status = format!("Loaded {name}");
                } else {
                    self.status = format!("Loaded {name} · {}", self.status);
                }
            }
            Err(e) => {
                self.status = format!("Load failed: {e}");
            }
        }
    }

    fn load_gltf_path(&mut self, path: &Path) -> Result<(), String> {
        if let Some(old) = self.model_root.take() {
            remove_subtree(&mut self.scene, old);
        }

        let root = load_gltf(&mut self.scene, path, Some(self.root))?;
        self.model_root = Some(root);

        // Wire paint maps onto every mesh material.
        let mesh_nodes = collect_mesh_nodes(&self.scene, Some(root));
        let mut touched = std::collections::HashSet::new();
        for &h in &mesh_nodes {
            let mat_h = {
                let Some(node) = self.scene.nodes.get_mut(h) else {
                    continue;
                };
                match node.material {
                    Some(m) => m,
                    None => {
                        node.material = Some(self.paint_mat);
                        continue;
                    }
                }
            };
            if !touched.insert(mat_h.key()) {
                continue;
            }
            if let Some(mat) = self.scene.materials.get_mut(mat_h) {
                mat.albedo = [1.0, 1.0, 1.0, 1.0];
                mat.albedo_map = Some(self.albedo_tex);
                mat.metallic = 1.0;
                mat.roughness = 1.0;
                mat.metallic_roughness_map = Some(self.mr_tex);
            }
        }

        self.reset_paint_docs();
        self.fit_to_nodes(&mesh_nodes);
        self.set_shape(2);
        Ok(())
    }

    fn reset_paint_docs(&mut self) {
        let rough_u8 = (0.45 * 255.0) as u8;
        self.docs = [
            PaintDocument::new(TEX_SIZE, TEX_SIZE),
            PaintDocument::with_base(TEX_SIZE, TEX_SIZE, [0, 0, 0]),
            PaintDocument::with_base(TEX_SIZE, TEX_SIZE, [rough_u8, rough_u8, rough_u8]),
        ];
        for d in &mut self.docs {
            d.mark_dirty();
        }
        self.needs_gpu_albedo_upload = true;
        self.needs_gpu_mr_upload = true;

        if let Some(mr) = self.scene.textures.get_mut(self.mr_tex) {
            for px in mr.rgba.chunks_exact_mut(4) {
                px[0] = 255;
                px[1] = rough_u8;
                px[2] = 0;
                px[3] = 255;
            }
        }
    }

    fn fit_to_nodes(&mut self, nodes: &[Handle<Node>]) {
        let Some((min, max)) = bounds_of_nodes(&self.scene, nodes) else {
            self.orbit_target = Vec3::new(0.0, 0.55, 0.0);
            self.orbit_dist = 4.0;
            self.apply_camera();
            return;
        };
        let center = (min + max) * 0.5;
        let radius = (max - min).length() * 0.5;
        self.orbit_target = center;
        self.orbit_dist = (radius * 2.4).clamp(1.2, 60.0);
        self.orbit_pitch = 0.35;
        self.apply_camera();
    }

    pub fn sync_paint_if_dirty(&mut self) {
        // Albedo stack
        if let Some(rect) = self.docs[0].take_dirty_rect() {
            if let Some(tex) = self.scene.textures.get_mut(self.albedo_tex) {
                self.docs[0].composite_rect(&mut tex.rgba, rect);
                self.needs_gpu_albedo_upload = true;
            }
        }

        // Metallic / roughness → MR channels (B / G)
        for map in [PaintMap::Metallic, PaintMap::Roughness] {
            let Some(channel) = map.mr_channel() else {
                continue;
            };
            let idx = map.index();
            let Some(rect) = self.docs[idx].take_dirty_rect() else {
                continue;
            };
            self.docs[idx].composite_rect(&mut self.mr_scratch, rect);
            let Some(tex) = self.scene.textures.get_mut(self.mr_tex) else {
                continue;
            };
            let w = TEX_SIZE;
            let x0 = rect.x0.min(w.saturating_sub(1));
            let y0 = rect.y0.min(w.saturating_sub(1));
            let x1 = rect.x1.min(w.saturating_sub(1));
            let y1 = rect.y1.min(w.saturating_sub(1));
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let i = ((y * w + x) * 4) as usize;
                    tex.rgba[i + channel] = self.mr_scratch[i];
                }
            }
            self.needs_gpu_mr_upload = true;
        }
    }

    pub fn take_pending_stamps(&mut self) -> Vec<PendingStamp> {
        std::mem::take(&mut self.pending_stamps)
    }

    pub fn paint_at(&mut self, screen: Vec2, viewport: mega_ui::Rect) {
        let local = Vec2::new(
            (screen.x - viewport.min.x).clamp(0.0, viewport.width().max(1.0)),
            (screen.y - viewport.min.y).clamp(0.0, viewport.height().max(1.0)),
        );
        let radius_px = self.brush_radius_px_at(screen, viewport);
        if let Some(prev) = self.last_stamp_px {
            if prev.distance(local) < radius_px * self.brush.spacing {
                return;
            }
        }
        self.last_stamp_px = Some(local);
        self.pending_stamps.push(PendingStamp {
            screen,
            viewport,
            screen_radius_px: radius_px,
        });
    }

    /// Face-on screen size of `brush.radius` at the surface under `screen` (or orbit fallback).
    pub fn brush_radius_px_at(&mut self, screen: Vec2, viewport: Rect) -> f32 {
        let map_h = self.viewport_size.y.round().max(1.0) as u32;
        let dist = self
            .pick_distance(screen, viewport)
            .unwrap_or(self.orbit_dist);
        crate::gpu_paint::world_radius_to_px(
            self.brush.radius,
            dist,
            self.scene.camera.fov_y,
            map_h,
        )
    }

    fn pick_distance(&mut self, screen: Vec2, viewport: Rect) -> Option<f32> {
        let aspect = (self.viewport_size.x / self.viewport_size.y.max(1.0)).max(1e-4);
        let view_proj = self.scene.camera.view_proj(aspect);
        let (origin, dir) =
            pick::screen_ray(screen, viewport, self.scene.camera.eye, view_proj.inverse());
        pick::ensure_bvhs(&self.scene, &self.paintable, &mut self.bvh_cache);
        let hit = pick::pick_mesh(
            &self.scene,
            origin,
            dir,
            &self.paintable,
            &self.bvh_cache,
        )?;
        Some(hit.position.distance(self.scene.camera.eye).max(0.05))
    }

    /// Brush cursor into `scene.hud` (viewport pixel space, drawn with the scene).
    pub fn update_brush_cursor(&mut self, screen: Vec2, viewport: Rect, show: bool) {
        let hud_size = Vec2::new(
            self.viewport_size.x.round().max(1.0),
            self.viewport_size.y.round().max(1.0),
        );
        let input = InputFrame {
            cursor: crate::gpu_paint::cursor_to_map_px(
                screen,
                viewport,
                hud_size.x as u32,
                hud_size.y as u32,
            ),
            ..Default::default()
        };
        self.scene.hud.begin(&input, hud_size);
        if show {
            self.draw_brush_cursor(screen, viewport, hud_size);
        }
        let _ = self.scene.hud.end();
    }

    fn draw_brush_cursor(&mut self, screen: Vec2, viewport: Rect, hud_size: Vec2) {
        const COLOR: [f32; 4] = [1.0, 1.0, 1.0, 0.92];
        const SEGMENTS: u32 = 48;
        const CROSS: f32 = 4.0;

        let aspect = hud_size.x / hud_size.y.max(1.0);
        let view_proj = self.scene.camera.view_proj(aspect);
        let (origin, dir) = pick::screen_ray(
            screen,
            viewport,
            self.scene.camera.eye,
            view_proj.inverse(),
        );
        pick::ensure_bvhs(&self.scene, &self.paintable, &mut self.bvh_cache);
        let hit = pick::pick_mesh(
            &self.scene,
            origin,
            dir,
            &self.paintable,
            &self.bvh_cache,
        );

        let (center_hud, ring_hud) = if let Some(hit) = hit {
            let center = hit.position + hit.normal * 0.002;
            let Some(c) = project_to_hud(center, view_proj, hud_size) else {
                return;
            };
            let ring = circle_on_plane(center, hit.normal, self.brush.radius, SEGMENTS)
                .into_iter()
                .filter_map(|p| project_to_hud(p, view_proj, hud_size))
                .collect::<Vec<_>>();
            (c, ring)
        } else {
            let c = crate::gpu_paint::cursor_to_map_px(
                screen,
                viewport,
                hud_size.x as u32,
                hud_size.y as u32,
            );
            let r = self.brush_radius_px_at(screen, viewport);
            let ring = (0..SEGMENTS)
                .map(|i| {
                    let a = TAU * i as f32 / SEGMENTS as f32;
                    c + Vec2::new(a.cos(), a.sin()) * r
                })
                .collect::<Vec<_>>();
            (c, ring)
        };

        if ring_hud.len() >= 2 {
            self.scene.hud.polyline(&ring_hud, COLOR, true);
        }
        self.scene.hud.line(
            center_hud + Vec2::new(-CROSS, 0.0),
            center_hud + Vec2::new(CROSS, 0.0),
            COLOR,
        );
        self.scene.hud.line(
            center_hud + Vec2::new(0.0, -CROSS),
            center_hud + Vec2::new(0.0, CROSS),
            COLOR,
        );
    }

    pub fn end_stroke(&mut self) {
        self.last_stamp_px = None;
        self.doc_mut().end_stroke();
    }

    pub fn pan_camera(&mut self, dx: f32, dy: f32) {
        let eye = self.scene.camera.eye;
        let f = (self.orbit_target - eye).normalize_or_zero();
        let mut r = Vec3::Y.cross(f).normalize_or_zero();
        if r.length_squared() < 1e-8 {
            let yaw = self.orbit_yaw;
            r = Vec3::new(yaw.cos(), 0.0, -yaw.sin());
        }
        let u = f.cross(r).normalize_or_zero();
        let scale = self.orbit_dist.max(0.5) * 0.0015;
        self.orbit_target += r * (-dx * scale) + u * (dy * scale);
        self.apply_camera();
    }

    pub fn build_ui(&mut self, ui: &mut Ui, window_size: Vec2, fps: f32) -> bool {
        let status_h = 24.0 * ui.scale();
        let dock_size = Vec2::new(window_size.x, (window_size.y - status_h).max(1.0));

        let mut keep = false;
        let mut shape = self.shape;
        let mut add_layer = false;
        let mut del_layer = false;
        let mut clear_layer = false;
        let mut move_up = false;
        let mut move_down = false;
        let mut active = self.doc().active;
        let mut dirty_layers = false;
        let mut open_model = false;
        let mut paint_map = self.paint_map;
        let mut debug_view = self.debug_view;
        let has_model = self.model_root.is_some();

        {
            let dock = &mut self.dock;
            let viewport_size = &mut self.viewport_size;
            let brush = &mut self.brush;
            let docs = &mut self.docs;
            let model_name = self.model_name.as_str();
            let scene = &mut self.scene;

            ui.dock_space("main", dock_size, dock, |ui, tab| match tab {
                "Viewport" => {
                    ui.label_styled(
                        &format!("LMB paint · MMB pan · RMB orbit · wheel zoom · {fps:.0} fps"),
                        TextStyle {
                            color: [0.7, 0.7, 0.72, 1.0],
                            size: 13.0,
                        },
                    );
                    ui.separator();
                    ui.label("View");
                    let mut view_idx = VIEW_MODES
                        .iter()
                        .position(|v| *v == debug_view)
                        .unwrap_or(0);
                    let view_labels: Vec<&str> = VIEW_MODES.iter().map(|v| v.label()).collect();
                    if ui
                        .toggle("debug_view", &mut view_idx, &view_labels)
                        .changed()
                    {
                        debug_view = VIEW_MODES[view_idx];
                        keep = true;
                    }
                    ui.separator();
                    let size = ui.available_size();
                    *viewport_size = size;
                    ui.texture(SCENE_TEX, size);
                }
                "Brush" => {
                    ui.label("Brush");
                    ui.separator();
                    ui.label("Color — stamp tint (MR maps use luminance)");
                    ui.color_edit("color", &mut brush.color);
                    ui.label("Radius — world-space stamp size");
                    ui.slider("radius", &mut brush.radius, 0.01..=0.35);
                    ui.label("Hardness — 0 soft falloff · 1 hard edge");
                    ui.slider("hardness", &mut brush.hardness, 0.0..=1.0);
                    ui.label("Opacity — stroke strength");
                    ui.slider("opacity", &mut brush.opacity, 0.05..=1.0);
                    ui.separator();
                    ui.label("Mesh");
                    if ui.button("Open…").clicked() {
                        open_model = true;
                    }
                    let options: &[&str] = if has_model {
                        &["Sphere", "Cube", "Model"]
                    } else {
                        &["Sphere", "Cube"]
                    };
                    let mut sel = shape.min(options.len().saturating_sub(1));
                    if ui.toggle("shape", &mut sel, options).changed() {
                        shape = sel;
                        keep = true;
                    }
                    if has_model && !model_name.is_empty() {
                        ui.label(model_name);
                    }
                }
                "Lights" => {
                    ui.label("Lights");
                    ui.separator();
                    ui.label("Ambient");
                    let mut amb = [scene.ambient[0], scene.ambient[1], scene.ambient[2], 1.0];
                    if ui.color_edit("ambient", &mut amb).changed() {
                        scene.ambient = [amb[0], amb[1], amb[2]];
                        keep = true;
                    }
                    if let Some(Light::Directional(d)) = scene.lights.first_mut() {
                        ui.separator();
                        ui.label("Directional");
                        if ui.checkbox("Cast shadows", &mut d.cast_shadows).changed() {
                            keep = true;
                        }
                        ui.label("Intensity");
                        if ui
                            .slider("light_intensity", &mut d.intensity, 0.0..=8.0)
                            .changed()
                        {
                            keep = true;
                        }
                        ui.label("Color");
                        let mut col = [d.color[0], d.color[1], d.color[2], 1.0];
                        if ui.color_edit("sun_color", &mut col).changed() {
                            d.color = [col[0], col[1], col[2]];
                            keep = true;
                        }
                        ui.label("Direction (world)");
                        if ui
                            .slider("dir_x", &mut d.direction.x, -1.0..=1.0)
                            .changed()
                            || ui
                                .slider("dir_y", &mut d.direction.y, -1.0..=1.0)
                                .changed()
                            || ui
                                .slider("dir_z", &mut d.direction.z, -1.0..=1.0)
                                .changed()
                        {
                            keep = true;
                        }
                    }
                }
                "Layers" => {
                    ui.label("Paint map");
                    let mut map_idx = paint_map.index();
                    let map_labels: Vec<&str> = PaintMap::ALL.iter().map(|m| m.label()).collect();
                    if ui.toggle("paint_map", &mut map_idx, &map_labels).changed() {
                        paint_map = PaintMap::ALL[map_idx];
                        active = docs[paint_map.index()].active;
                        keep = true;
                    }
                    ui.separator();
                    ui.row(|ui| {
                        if ui
                            .button_with("add_layer", |ui| {
                                ui.icon("plus", 14.0);
                            })
                            .clicked()
                        {
                            add_layer = true;
                        }
                        if ui
                            .button_with("del_layer", |ui| {
                                ui.icon("delete", 14.0);
                            })
                            .clicked()
                        {
                            del_layer = true;
                        }
                        if ui
                            .button_with("clear_layer", |ui| {
                                ui.icon("reset", 14.0);
                            })
                            .clicked()
                        {
                            clear_layer = true;
                        }
                        if ui
                            .button_with("layer_up", |ui| {
                                ui.icon("chevron_up", 14.0);
                            })
                            .clicked()
                        {
                            move_up = true;
                        }
                        if ui
                            .button_with("layer_dn", |ui| {
                                ui.icon("chevron_down", 14.0);
                            })
                            .clicked()
                        {
                            move_down = true;
                        }
                    });
                    ui.separator();
                    let layers = &mut docs[paint_map.index()].layers;
                    let size = ui.available_size();
                    ui.scroll_area("layers", size, ScrollAxes::Vertical, |ui| {
                        for i in (0..layers.len()).rev() {
                            let is_active = i == active;
                            let name = layers[i].name.clone();
                            ui.group(&name, |ui| {
                                let label = if is_active { "> active" } else { "Select" };
                                ui.row(|ui| {
                                    if ui.button(label).clicked() {
                                        active = i;
                                    }
                                    if ui
                                        .checkbox(&format!("vis{i}"), &mut layers[i].visible)
                                        .changed()
                                    {
                                        dirty_layers = true;
                                        keep = true;
                                    }
                                });
                                ui.label("Opacity");
                                if ui
                                    .slider(&format!("op{i}"), &mut layers[i].opacity, 0.0..=1.0)
                                    .changed()
                                {
                                    dirty_layers = true;
                                    keep = true;
                                }
                            });
                        }
                    });
                }
                _ => {}
            });
        }

        ui.status_bar(|ui| {
            ui.label(&self.status);
        });

        if paint_map != self.paint_map {
            self.paint_map = paint_map;
            active = self.doc().active;
        }
        self.debug_view = debug_view;

        if open_model {
            self.open_model_dialog();
            keep = true;
        } else if shape != self.shape {
            self.set_shape(shape);
        }
        if active != self.doc().active && active < self.doc().layers.len() {
            self.doc_mut().active = active;
        }
        if dirty_layers {
            self.doc_mut().mark_dirty();
        }
        if add_layer {
            self.doc_mut().add_layer();
        }
        if del_layer {
            self.doc_mut().remove_active();
        }
        if clear_layer {
            self.doc_mut().clear_active();
        }
        if move_up {
            self.doc_mut().move_active(1);
        }
        if move_down {
            self.doc_mut().move_active(-1);
        }

        keep || self.painting
    }
}

fn collect_mesh_nodes(scene: &Scene, root: Option<Handle<Node>>) -> Vec<Handle<Node>> {
    let Some(root) = root else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(h) = stack.pop() {
        let Some(node) = scene.nodes.get(h) else {
            continue;
        };
        if node.mesh.is_some() && node.visible {
            out.push(h);
        }
        for (child, n) in scene.nodes.iter() {
            if n.parent.is_some_and(|p| p.key() == h.key()) {
                stack.push(child);
            }
        }
    }
    out
}

fn set_subtree_visible(scene: &mut Scene, root: Handle<Node>, visible: bool) {
    let mut stack = vec![root];
    while let Some(h) = stack.pop() {
        if let Some(n) = scene.nodes.get_mut(h) {
            n.visible = visible;
        }
        let children: Vec<_> = scene
            .nodes
            .iter()
            .filter_map(|(c, n)| n.parent.is_some_and(|p| p.key() == h.key()).then_some(c))
            .collect();
        stack.extend(children);
    }
}

fn remove_subtree(scene: &mut Scene, root: Handle<Node>) {
    let mut order = Vec::new();
    let mut stack = vec![root];
    while let Some(h) = stack.pop() {
        order.push(h);
        let children: Vec<_> = scene
            .nodes
            .iter()
            .filter_map(|(c, n)| n.parent.is_some_and(|p| p.key() == h.key()).then_some(c))
            .collect();
        stack.extend(children);
    }
    for h in order.into_iter().rev() {
        scene.nodes.remove(h);
    }
}

fn bounds_of_nodes(scene: &Scene, nodes: &[Handle<Node>]) -> Option<(Vec3, Vec3)> {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    let mut any = false;
    for &h in nodes {
        let Some(node) = scene.nodes.get(h) else {
            continue;
        };
        let Some(mesh_h) = node.mesh else {
            continue;
        };
        let Some(mesh) = scene.meshes.get(mesh_h) else {
            continue;
        };
        let world = scene.world_matrix(h);
        let (a, b) = mesh_world_aabb(mesh, world);
        min = min.min(a);
        max = max.max(b);
        any = true;
    }
    any.then_some((min, max))
}

fn mesh_world_aabb(mesh: &Mesh, world: Mat4) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for p in &mesh.positions {
        let v = world.transform_point3(Vec3::from_array(*p));
        min = min.min(v);
        max = max.max(v);
    }
    if !min.is_finite() {
        (Vec3::ZERO, Vec3::ONE)
    } else {
        (min, max)
    }
}

fn project_to_hud(world: Vec3, view_proj: Mat4, hud_size: Vec2) -> Option<Vec2> {
    let clip = view_proj * world.extend(1.0);
    if clip.w <= 1e-5 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if !ndc.x.is_finite() || !ndc.y.is_finite() {
        return None;
    }
    Some(Vec2::new(
        (ndc.x * 0.5 + 0.5) * hud_size.x,
        (0.5 - ndc.y * 0.5) * hud_size.y,
    ))
}

fn circle_on_plane(center: Vec3, normal: Vec3, radius: f32, segments: u32) -> Vec<Vec3> {
    let n = normal.normalize_or_zero();
    if n.length_squared() < 1e-8 || radius <= 0.0 {
        return Vec::new();
    }
    let helper = if n.y.abs() < 0.99 { Vec3::Y } else { Vec3::X };
    let tangent = n.cross(helper).normalize_or_zero();
    let bitangent = n.cross(tangent).normalize_or_zero();
    let segs = segments.max(3);
    (0..segs)
        .map(|i| {
            let a = TAU * i as f32 / segs as f32;
            center + (tangent * a.cos() + bitangent * a.sin()) * radius
        })
        .collect()
}
