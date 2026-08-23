use std::collections::HashMap;
use std::f32::consts::TAU;
use std::path::Path;
use std::time::Instant;

use glam::{Mat4, Vec2, Vec3};
use mega_render::{
    cube, load_gltf, plane, sphere, view_gizmo, Camera, DebugView, Handle, HudRect, InputFrame,
    Light, Material, MaterialMaps, Mesh, Node, PolyOpts, PostProcessSettings, Projection, Scene,
    Texture, Transform,
};
use mega_ui::{DockNode, DockState, Rect};

use crate::brush_alpha::{generate_normal_stamps, generate_presets, BrushAlpha, NormalStamp};
use crate::paint::{
    luma, Brush, LayerKind, PaintDocument, PaintMap, PaintTarget, PaintTool, FLAT_NORMAL, TEX_SIZE,
};
use crate::pick::{self, BvhCache};
use crate::segment::{AppMode, SegOp, SegTool, Segmentation, UNASSIGNED};
use crate::uv::{self, TriPack, UnwrapAlgo};
use crate::uv_view::UvView;

mod post_ui;
mod ui;

pub const SCENE_TEX: u32 = 0;
pub const UV_TEX: u32 = 1;

struct OrbitSnap {
    from_yaw: f32,
    from_pitch: f32,
    to_yaw: f32,
    to_pitch: f32,
    t: f32,
}

#[derive(Clone, Copy)]
pub struct PendingStamp {
    pub screen: Vec2,
    pub viewport: Rect,
    /// Screen-space search radius at stamp depth (viewport pixels).
    pub screen_radius_px: f32,
    pub plane_normal: Vec3,
    /// Surface UV under the cursor (`None` if the pick missed).
    pub uv: Option<Vec2>,
}

pub struct Painter {
    pub scene: Scene,
    pub docs: [PaintDocument; 4],
    pub paint_map: PaintMap,
    pub debug_view: DebugView,
    pub brush: Brush,
    pub tool: PaintTool,
    pub alphas: Vec<BrushAlpha>,
    pub active_alpha: usize,
    pub nrm_stamps: Vec<NormalStamp>,
    pub active_nrm: usize,
    pub albedo_tex: Handle<Texture>,
    pub mr_tex: Handle<Texture>,
    pub nrm_tex: Handle<Texture>,
    /// Live UDIM ids, first is `albedo_tex` / `mr_tex` / `nrm_tex`.
    pub udim_ids: Vec<u32>,
    extra_albedo: Vec<(u32, Handle<Texture>)>,
    extra_mr: Vec<(u32, Handle<Texture>)>,
    extra_nrm: Vec<(u32, Handle<Texture>)>,
    pub paintable: Vec<Handle<Node>>,
    pub dock: DockState,
    pub viewport_size: Vec2,
    pub uv: UvView,
    pub orbit_yaw: f32,
    pub orbit_pitch: f32,
    pub orbit_dist: f32,
    pub orbit_target: Vec3,
    orbit_snap: Option<OrbitSnap>,
    /// View-gizmo axis snap uses ortho until the user orbits again.
    ortho_from_view: bool,
    pub painting: bool,
    pub mode: AppMode,
    pub seg_tool: SegTool,
    pub seg_op: SegOp,
    pub rect_through: bool,
    pub segmentation: Segmentation,
    /// 0 sphere, 1 cube, 2 gltf model
    pub shape: usize,
    pub model_name: String,
    pub status: String,
    pending_stamps: Vec<PendingStamp>,
    last_stamp_px: Option<Vec2>,
    root: Handle<Node>,
    paint_mat: Handle<Material>,
    sphere_node: Handle<Node>,
    cube_node: Handle<Node>,
    model_root: Option<Handle<Node>>,
    bvh_cache: BvhCache,
    /// After create/load: push CPU base pixels into GPU material maps once.
    pub needs_map_seed: bool,
    /// Post stack — synced to the visualizer each frame.
    pub post: PostProcessSettings,
    paint_debug_view: DebugView,
    rect_start: Option<Vec2>,
    last_seg_ptr: Option<(Vec2, Rect)>,
    last_seg_face: Option<((u32, u32), u32)>,
    seg_overlay_dirty: bool,
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

        let rough = 0.45;
        let mut docs = [
            PaintDocument::new(TEX_SIZE, TEX_SIZE, [220.0 / 255.0, 220.0 / 255.0, 225.0 / 255.0, 1.0]),
            PaintDocument::new(TEX_SIZE, TEX_SIZE, [0.0, 0.0, 0.0, 1.0]),
            PaintDocument::new(TEX_SIZE, TEX_SIZE, [0.0, rough, 0.0, 1.0]),
            PaintDocument::new(TEX_SIZE, TEX_SIZE, FLAT_NORMAL),
        ];

        // Material composite targets (filled by GPU layer composite).
        let mut albedo = Texture::gpu_resident(TEX_SIZE, TEX_SIZE, true);
        albedo.ensure_rgba();
        fill_rgba(&mut albedo.rgba, 220, 220, 225, 255);
        let albedo_tex = scene.textures.insert(albedo);

        let mut mr = Texture::gpu_resident(TEX_SIZE, TEX_SIZE, false);
        mr.ensure_rgba();
        let rough_u8 = (rough * 255.0) as u8;
        for px in mr.rgba.chunks_exact_mut(4) {
            px[0] = 255;
            px[1] = rough_u8;
            px[2] = 0;
            px[3] = 255;
        }
        let mr_tex = scene.textures.insert(mr);

        let mut nrm = Texture::gpu_resident(TEX_SIZE, TEX_SIZE, false);
        nrm.ensure_rgba();
        fill_rgba(&mut nrm.rgba, 128, 128, 255, 255);
        let nrm_tex = scene.textures.insert(nrm);

        // Seed each stack with one empty GPU layer.
        for doc in &mut docs {
            let tex = scene
                .textures
                .insert(Texture::gpu_resident(TEX_SIZE, TEX_SIZE, false));
            doc.layers.push(crate::paint::Layer::paint("Layer 1", tex));
            doc.active = 0;
            doc.mark_dirty();
        }

        let paint_mat_data = Material {
            maps: MaterialMaps::Single {
                albedo: Some(albedo_tex),
                normal: Some(nrm_tex),
                metallic_roughness: Some(mr_tex),
            },
            ..Material::new([1.0, 1.0, 1.0, 1.0], 1.0, 1.0)
        };
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
            tool: PaintTool::Paint,
            alphas: generate_presets(),
            active_alpha: 0,
            nrm_stamps: generate_normal_stamps(),
            active_nrm: 0,
            albedo_tex,
            mr_tex,
            nrm_tex,
            udim_ids: vec![1001],
            extra_albedo: Vec::new(),
            extra_mr: Vec::new(),
            extra_nrm: Vec::new(),
            paintable: vec![sphere_node],
            dock: DockState::new(DockNode::split_h(
                0.15,
                DockNode::leaf(&["Alphas"]),
                DockNode::split_h(
                    0.72,
                    DockNode::leaf(&["Viewport", "UV"]),
                    DockNode::split_v(
                        0.55,
                        DockNode::leaf(&["Brush", "Lights", "Effects"]),
                        DockNode::leaf(&["Layers", "Segments", "Meshes"]),
                    ),
                ),
            )),
            viewport_size: Vec2::new(1280.0, 720.0),
            uv: UvView::default(),
            orbit_yaw: 0.7,
            orbit_pitch: 0.35,
            orbit_dist: 4.0,
            orbit_target: Vec3::new(0.0, 0.55, 0.0),
            orbit_snap: None,
            ortho_from_view: false,
            painting: false,
            mode: AppMode::Paint,
            seg_tool: SegTool::Click,
            seg_op: SegOp::Select,
            rect_through: false,
            segmentation: Segmentation::default(),
            shape: 0,
            model_name: String::new(),
            status: "Ready · GPU brush".into(),
            pending_stamps: Vec::new(),
            last_stamp_px: None,
            root,
            paint_mat,
            sphere_node,
            cube_node,
            model_root: None,
            bvh_cache: HashMap::new(),
            needs_map_seed: true,
            post: {
                let mut post = PostProcessSettings::default();
                post.tonemap.exposure = 1.1;
                post
            },
            paint_debug_view: DebugView::Final,
            rect_start: None,
            last_seg_ptr: None,
            last_seg_face: None,
            seg_overlay_dirty: true,
        };
        painter.rebuild_bvhs();
        painter.segmentation.sync(&painter.scene, &painter.paintable);
        painter.apply_camera();
        painter
    }

    pub fn doc(&self) -> &PaintDocument {
        &self.docs[self.paint_map.index()]
    }

    pub fn doc_mut(&mut self) -> &mut PaintDocument {
        &mut self.docs[self.paint_map.index()]
    }

    pub fn uv_mesh_node(&self) -> Option<Handle<Node>> {
        self.paintable.get(self.uv.mesh_idx).copied()
    }

    pub fn stamp_target_tiles(&self) -> Vec<(u32, Handle<Texture>)> {
        let doc = self.doc();
        let Some(layer) = doc.layers.get(doc.active) else {
            return Vec::new();
        };
        if layer.kind == LayerKind::Fill || doc.paint_target == PaintTarget::Mask {
            layer.mask_tiles(&self.udim_ids)
        } else {
            layer.content_tiles(&self.udim_ids)
        }
    }

    pub fn stamp_target_tex(&self) -> Option<Handle<Texture>> {
        self.stamp_target_tiles().first().map(|(_, h)| *h)
    }

    pub fn painting_mask(&self) -> bool {
        let Some(layer) = self.doc().active_layer() else {
            return false;
        };
        (layer.kind == LayerKind::Fill || self.doc().paint_target == PaintTarget::Mask)
            && layer.mask.is_some()
    }

    /// Brush color for the current stamp target (layer pixels or mask).
    pub fn stamp_brush(&self) -> Brush {
        if self.painting_mask() {
            let v = luma(self.brush.color);
            return Brush {
                color: [v, v, v, 1.0],
                ..self.brush
            };
        }
        match self.paint_map {
            PaintMap::Albedo => self.brush,
            PaintMap::Normal => self.brush,
            PaintMap::Metallic | PaintMap::Roughness => {
                let v = luma(self.brush.color);
                Brush {
                    color: [v, v, v, 1.0],
                    ..self.brush
                }
            }
        }
    }

    fn alloc_layer_tex(&mut self) -> Handle<Texture> {
        self.scene
            .textures
            .insert(Texture::gpu_resident(TEX_SIZE, TEX_SIZE, false))
    }

    fn alloc_extra_tiles(&mut self) -> Vec<(u32, Handle<Texture>)> {
        let ids: Vec<u32> = self.udim_ids.iter().copied().skip(1).collect();
        ids.into_iter()
            .map(|id| (id, self.alloc_layer_tex()))
            .collect()
    }

    fn alloc_paint_layer(&mut self, name: impl Into<String>) -> crate::paint::Layer {
        let tex = self.alloc_layer_tex();
        let extra = self.alloc_extra_tiles();
        let mut layer = crate::paint::Layer::paint(name, tex);
        layer.extra_tex = extra;
        layer
    }

    pub fn add_layer(&mut self) {
        let name = {
            let n = self.doc().layers.len() + 1;
            format!("Layer {n}")
        };
        let layer = self.alloc_paint_layer(name);
        let doc = self.doc_mut();
        doc.layers.push(layer);
        doc.active = doc.layers.len() - 1;
        doc.paint_target = PaintTarget::Content;
        doc.mark_dirty();
    }

    pub fn add_fill_layer(&mut self) {
        let color = match self.paint_map {
            PaintMap::Albedo => self.brush.color,
            PaintMap::Normal => FLAT_NORMAL,
            PaintMap::Metallic | PaintMap::Roughness => {
                let v = luma(self.brush.color);
                [v, v, v, 1.0]
            }
        };
        let n = self
            .doc()
            .layers
            .iter()
            .filter(|l| l.kind == LayerKind::Fill)
            .count()
            + 1;
        let doc = self.doc_mut();
        doc.layers
            .push(crate::paint::Layer::fill(format!("Fill {n}"), color));
        doc.active = doc.layers.len() - 1;
        doc.paint_target = PaintTarget::Content;
        doc.mark_dirty();
    }

    fn add_mask_to(&mut self, i: usize) {
        if self
            .doc()
            .layers
            .get(i)
            .is_some_and(|l| l.mask.is_some())
        {
            let doc = self.doc_mut();
            doc.active = i;
            doc.paint_target = PaintTarget::Mask;
            return;
        }
        let tex = self.alloc_layer_tex();
        let extra = self.alloc_extra_tiles();
        let doc = self.doc_mut();
        let Some(layer) = doc.layers.get_mut(i) else {
            self.scene.textures.remove(tex);
            for (_, h) in extra {
                self.scene.textures.remove(h);
            }
            return;
        };
        let hide = layer.kind == LayerKind::Fill;
        layer.mask = Some(tex);
        layer.extra_mask = extra;
        // Fill: black (paint to reveal). Paint: white (keep existing strokes).
        layer.mask_init = Some(if hide {
            [0, 0, 0, 255]
        } else {
            [255, 255, 255, 255]
        });
        doc.active = i;
        doc.paint_target = PaintTarget::Mask;
        doc.mark_dirty();
    }

    fn toggle_mask_target(&mut self, i: usize) {
        if i >= self.doc().layers.len() {
            return;
        }
        let has_mask = self.doc().layers[i].mask.is_some();
        if !has_mask {
            self.add_mask_to(i);
            return;
        }
        let doc = self.doc_mut();
        if doc.active == i && doc.paint_target == PaintTarget::Mask {
            doc.paint_target = PaintTarget::Content;
        } else {
            doc.active = i;
            doc.paint_target = PaintTarget::Mask;
        }
    }

    fn remove_mask_from_active(&mut self) {
        let active = self.doc().active;
        let (tex, extra) = {
            let doc = self.doc_mut();
            let Some(layer) = doc.layers.get_mut(active) else {
                return;
            };
            let tex = layer.mask.take();
            let extra = std::mem::take(&mut layer.extra_mask);
            layer.mask_init = None;
            doc.paint_target = PaintTarget::Content;
            doc.mark_dirty();
            (tex, extra)
        };
        if let Some(tex) = tex {
            self.scene.textures.remove(tex);
        }
        for (_, h) in extra {
            self.scene.textures.remove(h);
        }
    }

    pub fn remove_active_layer(&mut self) {
        let targeting_mask = self.doc().paint_target == PaintTarget::Mask
            && self
                .doc()
                .active_layer()
                .is_some_and(|l| l.mask.is_some());
        if targeting_mask {
            self.remove_mask_from_active();
            return;
        }
        let active = self.doc().active;
        if self.doc().layers.len() <= 1 {
            let doc = self.doc_mut();
            if let Some(l) = doc.layers.get_mut(0) {
                if l.kind == LayerKind::Paint {
                    l.needs_clear = true;
                }
            }
            doc.paint_target = PaintTarget::Content;
            doc.mark_dirty();
            return;
        }
        let handles: Vec<_> = {
            let doc = self.doc_mut();
            let layer = doc.layers.remove(active);
            doc.active = active.min(doc.layers.len() - 1);
            doc.paint_target = PaintTarget::Content;
            doc.clamp_paint_target();
            doc.mark_dirty();
            layer.gpu_handles().collect()
        };
        for tex in handles {
            self.scene.textures.remove(tex);
        }
    }

    pub fn clear_active_layer(&mut self) {
        let targeting_mask = self.doc().paint_target == PaintTarget::Mask;
        let doc = self.doc_mut();
        let active = doc.active;
        if let Some(l) = doc.layers.get_mut(active) {
            if targeting_mask && l.mask.is_some() {
                l.mask_init = Some(if l.kind == LayerKind::Fill {
                    [0, 0, 0, 255]
                } else {
                    [255, 255, 255, 255]
                });
            } else if l.kind == LayerKind::Paint {
                l.needs_clear = true;
            }
        }
        doc.mark_dirty();
    }

    pub fn apply_camera(&mut self) {
        self.orbit_pitch = self.orbit_pitch.clamp(-1.55, 1.55);
        self.orbit_dist = self.orbit_dist.clamp(0.02, 80.0);
        self.scene.camera = Camera::orbit(
            self.orbit_yaw,
            self.orbit_pitch,
            self.orbit_dist,
            self.orbit_target,
        );
        // Default near is 0.1 — closer than that and the model gets clipped.
        self.scene.camera.near = (self.orbit_dist * 0.05).clamp(0.001, 0.1);
        if self.ortho_from_view {
            self.scene.camera.projection = Projection::Orthographic;
            self.scene.camera.sync_ortho_from_distance();
        }
    }

    /// Place the orbit cam on `dir` (world), looking at target. Short tween + ortho.
    pub fn snap_orbit_to_dir(&mut self, dir: Vec3) {
        let d = dir.normalize_or_zero();
        if d.length_squared() < 1e-8 {
            return;
        }
        let to_pitch = d.y.clamp(-1.0, 1.0).asin().clamp(-1.55, 1.55);
        let to_yaw = d.x.atan2(d.z);
        self.orbit_snap = Some(OrbitSnap {
            from_yaw: self.orbit_yaw,
            from_pitch: self.orbit_pitch,
            to_yaw,
            to_pitch,
            t: 0.0,
        });
        self.ortho_from_view = true;
        self.apply_camera();
    }

    pub fn tick_orbit_snap(&mut self, dt: f32) {
        let Some(snap) = self.orbit_snap.as_mut() else {
            return;
        };
        snap.t = (snap.t + dt / 0.22).min(1.0);
        let t = snap.t;
        let s = t * t * (3.0 - 2.0 * t);
        self.orbit_yaw = lerp_angle(snap.from_yaw, snap.to_yaw, s);
        self.orbit_pitch = snap.from_pitch + (snap.to_pitch - snap.from_pitch) * s;
        if t >= 1.0 {
            self.orbit_yaw = snap.to_yaw;
            self.orbit_pitch = snap.to_pitch;
            self.orbit_snap = None;
        }
        self.apply_camera();
    }

    pub fn orbit_snap_active(&self) -> bool {
        self.orbit_snap.is_some()
    }

    /// Drop an in-flight snap. `restore_perspective` after a free orbit (Blender-style).
    pub fn interrupt_orbit_snap(&mut self, restore_perspective: bool) {
        self.orbit_snap = None;
        if restore_perspective && self.ortho_from_view {
            self.ortho_from_view = false;
            self.apply_camera();
        }
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
        self.segmentation.sync(&self.scene, &self.paintable);
        self.seg_overlay_dirty = true;
    }

    pub fn unwrap_uv(&mut self) {
        self.segmentation.sync(&self.scene, &self.paintable);
        let algo: &dyn UnwrapAlgo = &TriPack;
        let mut tiles = std::collections::BTreeSet::new();
        let mut seen = std::collections::HashSet::new();
        let meshes: Vec<Handle<Mesh>> = self
            .paintable
            .iter()
            .filter_map(|&nh| self.scene.nodes.get(nh)?.mesh)
            .collect();
        for mh in meshes {
            if !seen.insert(mh.key()) {
                continue;
            }
            let Some(mesh) = self.scene.meshes.get(mh) else {
                continue;
            };
            let n = mesh.indices.len() / 3;
            let face_udim = uv::face_udims(&self.segmentation, mh, n);
            tiles.extend(face_udim.iter().copied());
            let Some(mesh) = self.scene.meshes.get_mut(mh) else {
                continue;
            };
            uv::apply_unwrap(mesh, &face_udim, algo);
        }
        let mut ids: Vec<u32> = tiles.into_iter().collect();
        if ids.is_empty() {
            ids.push(1001);
        }
        self.udim_ids = ids;
        self.rebuild_extra_maps();
        self.reset_paint_docs();
        self.bind_paint_maps();
        self.rebuild_bvhs();
        self.status = format!(
            "Unwrapped · {} · {} UDIM",
            algo.name(),
            self.udim_ids.len()
        );
        self.uv.needs_fit = true;
    }

    fn reset_udim(&mut self) {
        self.drop_extra_maps();
        self.udim_ids = vec![1001];
    }

    fn drop_extra_maps(&mut self) {
        let extra: Vec<_> = self
            .extra_albedo
            .drain(..)
            .chain(self.extra_mr.drain(..))
            .chain(self.extra_nrm.drain(..))
            .collect();
        for (_, h) in extra {
            self.scene.textures.remove(h);
        }
    }

    fn rebuild_extra_maps(&mut self) {
        self.drop_extra_maps();
        let rough = 0.45;
        let rough_u8 = (rough * 255.0) as u8;
        for &id in self.udim_ids.iter().skip(1) {
            let mut albedo = Texture::gpu_resident(TEX_SIZE, TEX_SIZE, true);
            albedo.ensure_rgba();
            fill_rgba(&mut albedo.rgba, 220, 220, 225, 255);
            self.extra_albedo
                .push((id, self.scene.textures.insert(albedo)));

            let mut mr = Texture::gpu_resident(TEX_SIZE, TEX_SIZE, false);
            mr.ensure_rgba();
            for px in mr.rgba.chunks_exact_mut(4) {
                px[0] = 255;
                px[1] = rough_u8;
                px[2] = 0;
                px[3] = 255;
            }
            self.extra_mr.push((id, self.scene.textures.insert(mr)));

            let mut nrm = Texture::gpu_resident(TEX_SIZE, TEX_SIZE, false);
            nrm.ensure_rgba();
            fill_rgba(&mut nrm.rgba, 128, 128, 255, 255);
            self.extra_nrm.push((id, self.scene.textures.insert(nrm)));
        }
    }

    fn paint_maps(&self) -> MaterialMaps {
        if self.udim_ids.len() <= 1 {
            return MaterialMaps::Single {
                albedo: Some(self.albedo_tex),
                normal: Some(self.nrm_tex),
                metallic_roughness: Some(self.mr_tex),
            };
        }
        let id0 = self.udim_ids[0];
        let mut albedo = vec![(id0, self.albedo_tex)];
        albedo.extend(self.extra_albedo.iter().copied());
        let mut normal = vec![(id0, self.nrm_tex)];
        normal.extend(self.extra_nrm.iter().copied());
        let mut metallic_roughness = vec![(id0, self.mr_tex)];
        metallic_roughness.extend(self.extra_mr.iter().copied());
        MaterialMaps::Udim {
            albedo,
            normal,
            metallic_roughness,
        }
    }

    fn bind_paint_maps(&mut self) {
        let maps = self.paint_maps();
        let mut handles = vec![self.paint_mat];
        for &nh in &self.paintable {
            if let Some(m) = self.scene.nodes.get(nh).and_then(|n| n.material) {
                handles.push(m);
            }
        }
        handles.sort_by_key(|h| h.key());
        handles.dedup_by_key(|h| h.key());
        for h in handles {
            if let Some(mat) = self.scene.materials.get_mut(h) {
                mat.maps = maps.clone();
            }
        }
    }

    pub fn material_map_tiles(&self, map: PaintMap) -> Vec<(u32, Handle<Texture>)> {
        let id0 = *self.udim_ids.first().unwrap_or(&1001);
        match map {
            PaintMap::Albedo => {
                let mut v = vec![(id0, self.albedo_tex)];
                v.extend(self.extra_albedo.iter().copied());
                v
            }
            PaintMap::Metallic | PaintMap::Roughness => {
                let mut v = vec![(id0, self.mr_tex)];
                v.extend(self.extra_mr.iter().copied());
                v
            }
            PaintMap::Normal => {
                let mut v = vec![(id0, self.nrm_tex)];
                v.extend(self.extra_nrm.iter().copied());
                v
            }
        }
    }

    pub fn dst_map_handles(&self) -> Vec<Handle<Texture>> {
        let mut v = vec![self.albedo_tex, self.mr_tex, self.nrm_tex];
        v.extend(self.extra_albedo.iter().map(|(_, h)| *h));
        v.extend(self.extra_mr.iter().map(|(_, h)| *h));
        v.extend(self.extra_nrm.iter().map(|(_, h)| *h));
        v
    }

    fn rebuild_bvhs(&mut self) {
        let t0 = Instant::now();
        self.bvh_cache.clear();
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
                self.uv.needs_fit = true;
                self.uv.needs_fit = true;
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

        self.reset_udim();
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
            let maps = self.paint_maps();
            if let Some(mat) = self.scene.materials.get_mut(mat_h) {
                mat.albedo = [1.0, 1.0, 1.0, 1.0];
                mat.metallic = 1.0;
                mat.roughness = 1.0;
                mat.maps = maps;
            }
        }

        self.reset_paint_docs();
        self.fit_to_nodes(&mesh_nodes);
        self.set_shape(2);
        Ok(())
    }

    fn reset_paint_docs(&mut self) {
        let mut old = Vec::new();
        for doc in &mut self.docs {
            for layer in doc.layers.drain(..) {
                old.extend(layer.gpu_handles());
            }
        }
        for tex in old {
            self.scene.textures.remove(tex);
        }

        let rough = 0.45;
        let rough_u8 = (rough * 255.0) as u8;
        self.docs = [
            PaintDocument::new(
                TEX_SIZE,
                TEX_SIZE,
                [220.0 / 255.0, 220.0 / 255.0, 225.0 / 255.0, 1.0],
            ),
            PaintDocument::new(TEX_SIZE, TEX_SIZE, [0.0, 0.0, 0.0, 1.0]),
            PaintDocument::new(TEX_SIZE, TEX_SIZE, [0.0, rough, 0.0, 1.0]),
            PaintDocument::new(TEX_SIZE, TEX_SIZE, FLAT_NORMAL),
        ];
        for i in 0..self.docs.len() {
            let layer = self.alloc_paint_layer("Layer 1");
            self.docs[i].layers.push(layer);
            self.docs[i].active = 0;
            self.docs[i].mark_dirty();
        }

        // Reset CPU mirrors (gpu_resident won't re-upload on version bump alone).
        if let Some(tex) = self.scene.textures.get_mut(self.albedo_tex) {
            tex.ensure_rgba();
            fill_rgba(&mut tex.rgba, 220, 220, 225, 255);
        }
        if let Some(tex) = self.scene.textures.get_mut(self.mr_tex) {
            tex.ensure_rgba();
            for px in tex.rgba.chunks_exact_mut(4) {
                px[0] = 255;
                px[1] = rough_u8;
                px[2] = 0;
                px[3] = 255;
            }
        }
        if let Some(tex) = self.scene.textures.get_mut(self.nrm_tex) {
            tex.ensure_rgba();
            fill_rgba(&mut tex.rgba, 128, 128, 255, 255);
        }
        for (_, h) in &self.extra_albedo {
            if let Some(tex) = self.scene.textures.get_mut(*h) {
                tex.ensure_rgba();
                fill_rgba(&mut tex.rgba, 220, 220, 225, 255);
            }
        }
        for (_, h) in &self.extra_mr {
            if let Some(tex) = self.scene.textures.get_mut(*h) {
                tex.ensure_rgba();
                for px in tex.rgba.chunks_exact_mut(4) {
                    px[0] = 255;
                    px[1] = rough_u8;
                    px[2] = 0;
                    px[3] = 255;
                }
            }
        }
        for (_, h) in &self.extra_nrm {
            if let Some(tex) = self.scene.textures.get_mut(*h) {
                tex.ensure_rgba();
                fill_rgba(&mut tex.rgba, 128, 128, 255, 255);
            }
        }
        self.needs_map_seed = true;
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

    pub fn take_pending_stamps(&mut self) -> Vec<PendingStamp> {
        std::mem::take(&mut self.pending_stamps)
    }

    pub fn set_mode(&mut self, mode: AppMode) {
        if self.mode == mode {
            return;
        }
        match mode {
            AppMode::Segment => {
                self.paint_debug_view = self.debug_view;
                self.debug_view = DebugView::Wireframe;
                self.seg_overlay_dirty = true;
            }
            AppMode::Paint => {
                self.debug_view = self.paint_debug_view;
                self.scene.debug.clear();
            }
        }
        self.mode = mode;
        self.painting = false;
        self.last_stamp_px = None;
        self.rect_start = None;
    }

    pub fn begin_stroke(&mut self) {
        self.last_stamp_px = None;
        self.last_seg_face = None;
        self.rect_start = None;
        if self.mode == AppMode::Paint {
            self.doc_mut().end_stroke();
        }
    }

    pub fn end_stroke(&mut self) {
        if self.mode == AppMode::Segment {
            self.commit_rect_select();
        } else {
            self.doc_mut().end_stroke();
        }
        self.last_stamp_px = None;
        self.last_seg_face = None;
        self.rect_start = None;
    }

    pub fn viewport_interact(&mut self, screen: Vec2, viewport: Rect) {
        if self.mode == AppMode::Segment {
            self.segment_at(screen, viewport);
        } else {
            self.paint_at(screen, viewport);
        }
    }

    pub fn paint_at(&mut self, screen: Vec2, viewport: mega_ui::Rect) {
        if self
            .doc()
            .active_layer()
            .is_some_and(|l| l.kind == LayerKind::Fill && l.mask.is_none())
        {
            let i = self.doc().active;
            self.add_mask_to(i);
        }
        if self.stamp_target_tex().is_none() {
            return;
        }
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
        let hit = self.pick_hit(screen, viewport);
        let plane_normal = hit.map(|h| h.normal).unwrap_or(Vec3::Y);
        let uv = hit.map(|h| h.uv);
        self.pending_stamps.push(PendingStamp {
            screen,
            viewport,
            screen_radius_px: radius_px,
            plane_normal,
            uv,
        });
        self.doc_mut().mark_dirty();
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
        let hit = self.pick_hit(screen, viewport)?;
        Some(hit.position.distance(self.scene.camera.eye).max(0.05))
    }

    fn pick_hit(&mut self, screen: Vec2, viewport: Rect) -> Option<pick::Hit> {
        let aspect = (self.viewport_size.x / self.viewport_size.y.max(1.0)).max(1e-4);
        let view_proj = self.scene.camera.view_proj(aspect);
        let (origin, dir) = pick::screen_ray(screen, viewport, view_proj.inverse());
        pick::ensure_bvhs(&self.scene, &self.paintable, &mut self.bvh_cache);
        pick::pick_mesh(
            &self.scene,
            origin,
            dir,
            &self.paintable,
            &self.bvh_cache,
        )
    }

    fn apply_seg_faces(&mut self, faces: &[(Handle<Mesh>, u32)]) {
        if faces.is_empty() {
            return;
        }
        let id = match self.seg_op {
            SegOp::Select => {
                let Some(id) = self.segmentation.active else {
                    return;
                };
                id
            }
            SegOp::Deselect => UNASSIGNED,
        };
        self.segmentation.set_faces(faces, id);
        self.seg_overlay_dirty = true;
    }

    fn segment_at(&mut self, screen: Vec2, viewport: Rect) {
        self.last_seg_ptr = Some((screen, viewport));
        match self.seg_tool {
            SegTool::Rect => {
                if self.rect_start.is_none() {
                    self.rect_start = Some(screen);
                }
            }
            SegTool::Click => {
                let Some(hit) = self.pick_hit(screen, viewport) else {
                    return;
                };
                let key = (hit.mesh.key(), hit.tri_index);
                if self.last_seg_face == Some(key) {
                    return;
                }
                self.last_seg_face = Some(key);
                self.apply_seg_faces(&[(hit.mesh, hit.tri_index)]);
            }
            SegTool::Brush => {
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
                let Some(hit) = self.pick_hit(screen, viewport) else {
                    return;
                };
                self.last_stamp_px = Some(local);
                let faces = pick::faces_in_sphere(
                    &self.scene,
                    &self.paintable,
                    &self.bvh_cache,
                    self.scene.camera.eye,
                    hit.position,
                    self.brush.radius,
                );
                self.apply_seg_faces(&faces);
            }
        }
    }

    fn commit_rect_select(&mut self) {
        let Some(start) = self.rect_start else {
            return;
        };
        let Some((end, viewport)) = self.last_seg_ptr else {
            return;
        };
        let aspect = (self.viewport_size.x / self.viewport_size.y.max(1.0)).max(1e-4);
        let view_proj = self.scene.camera.view_proj(aspect);
        if start.distance(end) < 4.0 {
            if let Some(hit) = self.pick_hit(end, viewport) {
                self.apply_seg_faces(&[(hit.mesh, hit.tri_index)]);
            }
            return;
        }
        let faces = pick::faces_in_rect(
            &self.scene,
            &self.paintable,
            self.scene.camera.eye,
            view_proj,
            viewport,
            start,
            end,
            self.rect_through,
        );
        self.apply_seg_faces(&faces);
    }

    pub fn sync_segment_overlay(&mut self) {
        if self.mode != AppMode::Segment {
            return;
        }
        if !self.seg_overlay_dirty {
            return;
        }
        self.seg_overlay_dirty = false;
        self.scene.debug.clear();
        let faces = self.segmentation.overlay_faces();
        let mut worlds = HashMap::new();
        for &nh in &self.paintable {
            let Some(n) = self.scene.nodes.get(nh) else {
                continue;
            };
            if !n.visible {
                continue;
            }
            let Some(mh) = n.mesh else {
                continue;
            };
            worlds.entry(mh.key()).or_insert_with(|| self.scene.world_matrix(nh));
        }
        let bias = (self.orbit_dist * 0.0008).clamp(0.001, 0.03);
        for (mesh_h, ti, color) in faces {
            let Some(mesh) = self.scene.meshes.get(mesh_h) else {
                continue;
            };
            let Some(&world) = worlds.get(&mesh_h.key()) else {
                continue;
            };
            let base = ti as usize * 3;
            if base + 2 >= mesh.indices.len() {
                continue;
            }
            let i0 = mesh.indices[base] as usize;
            let i1 = mesh.indices[base + 1] as usize;
            let i2 = mesh.indices[base + 2] as usize;
            let a = world.transform_point3(Vec3::from_array(mesh.positions[i0]));
            let b = world.transform_point3(Vec3::from_array(mesh.positions[i1]));
            let c = world.transform_point3(Vec3::from_array(mesh.positions[i2]));
            let n = (b - a).cross(c - a).normalize_or_zero();
            let o = n * bias;
            self.scene.debug.tri(
                a + o,
                b + o,
                c + o,
                PolyOpts {
                    color,
                    depth_test: true,
                },
            );
        }
    }

    fn view_gizmo_cursor(&self, screen: Vec2, viewport: Rect) -> (Vec2, Vec2) {
        let hud_size = Vec2::new(
            self.viewport_size.x.round().max(1.0),
            self.viewport_size.y.round().max(1.0),
        );
        let local = crate::gpu_paint::cursor_to_map_px(
            screen,
            viewport,
            hud_size.x as u32,
            hud_size.y as u32,
        );
        (hud_size, local)
    }

    pub fn over_view_gizmo(&self, screen: Vec2, viewport: Rect) -> bool {
        if viewport.width() <= 1.0 || !viewport.contains(screen) {
            return false;
        }
        let (size, local) = self.view_gizmo_cursor(screen, viewport);
        view_gizmo::contains_cursor(size, local)
    }

    pub fn pick_view_gizmo(&self, screen: Vec2, viewport: Rect) -> Option<view_gizmo::ViewAxis> {
        let (size, local) = self.view_gizmo_cursor(screen, viewport);
        view_gizmo::hit_test(&self.scene.camera, size, local)
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
        let local = input.cursor;
        if viewport.width() > 1.0 && viewport.height() > 1.0 {
            view_gizmo::draw(&mut self.scene.hud, &self.scene.camera, hud_size, local);
        }
        let over_gizmo = view_gizmo::contains_cursor(hud_size, local);
        if show && !over_gizmo {
            if self.mode == AppMode::Segment {
                self.draw_segment_cursor(screen, viewport, hud_size);
            } else {
                self.draw_brush_cursor(screen, viewport, hud_size);
            }
        }
        let _ = self.scene.hud.end();
    }

    fn draw_segment_cursor(&mut self, screen: Vec2, viewport: Rect, hud_size: Vec2) {
        let color = match self.seg_op {
            SegOp::Select => [1.0, 0.92, 0.35, 0.92],
            SegOp::Deselect => [0.55, 0.75, 1.0, 0.92],
        };
        match self.seg_tool {
            SegTool::Rect => {
                let a = self.rect_start.unwrap_or(screen);
                let pa = crate::gpu_paint::cursor_to_map_px(
                    a,
                    viewport,
                    hud_size.x as u32,
                    hud_size.y as u32,
                );
                let pb = crate::gpu_paint::cursor_to_map_px(
                    screen,
                    viewport,
                    hud_size.x as u32,
                    hud_size.y as u32,
                );
                let min = pa.min(pb);
                let max = pa.max(pb);
                self.scene.hud.fill(
                    HudRect { min, max },
                    [color[0], color[1], color[2], 0.12],
                );
                let corners = [
                    Vec2::new(min.x, min.y),
                    Vec2::new(max.x, min.y),
                    Vec2::new(max.x, max.y),
                    Vec2::new(min.x, max.y),
                ];
                self.scene.hud.polyline(&corners, color, true);
            }
            SegTool::Click => {
                self.draw_brush_cursor_ring(screen, viewport, hud_size, color, 0.0);
            }
            SegTool::Brush => {
                self.draw_brush_cursor_ring(screen, viewport, hud_size, color, self.brush.radius);
            }
        }
    }

    fn draw_brush_cursor(&mut self, screen: Vec2, viewport: Rect, hud_size: Vec2) {
        let color = match self.tool {
            PaintTool::Paint => [1.0, 1.0, 1.0, 0.92],
            PaintTool::Eraser => [0.55, 0.75, 1.0, 0.92],
        };
        self.draw_brush_cursor_ring(screen, viewport, hud_size, color, self.brush.radius);
    }

    fn draw_brush_cursor_ring(
        &mut self,
        screen: Vec2,
        viewport: Rect,
        hud_size: Vec2,
        color: [f32; 4],
        radius: f32,
    ) {
        const SEGMENTS: u32 = 48;
        const CROSS: f32 = 4.0;

        let aspect = hud_size.x / hud_size.y.max(1.0);
        let view_proj = self.scene.camera.view_proj(aspect);
        let (origin, dir) = pick::screen_ray(screen, viewport, view_proj.inverse());
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
            let ring = if radius > 1e-5 {
                circle_on_plane(center, hit.normal, radius, SEGMENTS)
                    .into_iter()
                    .filter_map(|p| project_to_hud(p, view_proj, hud_size))
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            (c, ring)
        } else {
            let c = crate::gpu_paint::cursor_to_map_px(
                screen,
                viewport,
                hud_size.x as u32,
                hud_size.y as u32,
            );
            let ring = if radius > 1e-5 {
                let r = self.brush_radius_px_at(screen, viewport);
                (0..SEGMENTS)
                    .map(|i| {
                        let a = TAU * i as f32 / SEGMENTS as f32;
                        c + Vec2::new(a.cos(), a.sin()) * r
                    })
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            (c, ring)
        };

        if ring_hud.len() >= 2 {
            self.scene.hud.polyline(&ring_hud, color, true);
        }
        self.scene.hud.line(
            center_hud + Vec2::new(-CROSS, 0.0),
            center_hud + Vec2::new(CROSS, 0.0),
            color,
        );
        self.scene.hud.line(
            center_hud + Vec2::new(0.0, -CROSS),
            center_hud + Vec2::new(0.0, CROSS),
            color,
        );
    }

    pub fn pan_camera(&mut self, dx: f32, dy: f32) {
        self.interrupt_orbit_snap(false);
        let eye = self.scene.camera.eye;
        let f = (self.orbit_target - eye).normalize_or_zero();
        let mut r = Vec3::Y.cross(f).normalize_or_zero();
        if r.length_squared() < 1e-8 {
            let yaw = self.orbit_yaw;
            r = Vec3::new(yaw.cos(), 0.0, -yaw.sin());
        }
        let u = f.cross(r).normalize_or_zero();
        let scale = self.orbit_dist.max(0.02) * 0.0015;
        self.orbit_target += r * (-dx * scale) + u * (dy * scale);
        self.apply_camera();
    }
}

fn lerp_angle(a: f32, b: f32, t: f32) -> f32 {
    let mut d = b - a;
    while d > std::f32::consts::PI {
        d -= TAU;
    }
    while d < -std::f32::consts::PI {
        d += TAU;
    }
    a + d * t
}

fn fill_rgba(buf: &mut [u8], r: u8, g: u8, b: u8, a: u8) {
    for px in buf.chunks_exact_mut(4) {
        px[0] = r;
        px[1] = g;
        px[2] = b;
        px[3] = a;
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
        if node.mesh.is_some() {
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
