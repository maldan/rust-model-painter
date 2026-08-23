use glam::Vec2;
use mega_render::{DebugView, Handle, Light, Node, Scene};
use mega_ui::{ScrollAxes, TextStyle, Ui};

use crate::brush_alpha::{ALPHA_TEX_BASE, NRM_TEX_BASE};
use crate::paint::{luma, Brush, LayerKind, PaintDocument, PaintMap, PaintTarget, PaintTool};
use crate::segment::{AppMode, SegOp, SegTool, Segmentation};
use super::{post_ui, Painter, SCENE_TEX, UV_TEX};
use crate::uv_view::UvView;

const VIEW_MODES: &[DebugView] = &[
    DebugView::Final,
    DebugView::Albedo,
    DebugView::Metallic,
    DebugView::Roughness,
    DebugView::Normals,
];

/// Clicks and edits collected while the dock borrows painter fields.
struct Actions {
    keep: bool,
    shape: usize,
    add_layer: bool,
    add_fill: bool,
    del_layer: bool,
    clear_layer: bool,
    move_up: bool,
    move_down: bool,
    active: usize,
    mask_click: Option<usize>,
    content_click: Option<usize>,
    dirty_layers: bool,
    open_model: bool,
    paint_map: PaintMap,
    debug_view: DebugView,
    tool: PaintTool,
    mode: AppMode,
    seg_tool: SegTool,
    seg_op: SegOp,
    rect_through: bool,
    add_segment: bool,
    del_segment: bool,
    unwrap_uv: bool,
    seg_overlay: bool,
    active_alpha: usize,
    active_nrm: usize,
}

impl Painter {
    pub fn build_ui(&mut self, ui: &mut Ui, window_size: Vec2, fps: f32) -> bool {
        let status_h = 24.0 * ui.scale();
        let dock_size = Vec2::new(window_size.x, (window_size.y - status_h).max(1.0));

        let alpha_names: Vec<&'static str> = self.alphas.iter().map(|a| a.name).collect();
        let nrm_names: Vec<&'static str> = self.nrm_stamps.iter().map(|a| a.name).collect();
        let has_model = self.model_root.is_some();

        let mut a = Actions {
            keep: false,
            shape: self.shape,
            add_layer: false,
            add_fill: false,
            del_layer: false,
            clear_layer: false,
            move_up: false,
            move_down: false,
            active: self.doc().active,
            mask_click: None,
            content_click: None,
            dirty_layers: false,
            open_model: false,
            paint_map: self.paint_map,
            debug_view: self.debug_view,
            tool: self.tool,
            mode: self.mode,
            seg_tool: self.seg_tool,
            seg_op: self.seg_op,
            rect_through: self.rect_through,
            add_segment: false,
            del_segment: false,
            unwrap_uv: false,
            seg_overlay: false,
            active_alpha: self.active_alpha,
            active_nrm: self.active_nrm,
        };

        {
            let dock = &mut self.dock;
            let viewport_size = &mut self.viewport_size;
            let brush = &mut self.brush;
            let docs = &mut self.docs;
            let post = &mut self.post;
            let model_name = self.model_name.as_str();
            let scene = &mut self.scene;
            let segmentation = &mut self.segmentation;
            let mesh_nodes = self.paintable.clone();
            let uv = &mut self.uv;

            ui.dock_space("main", dock_size, dock, |ui, tab| match tab {
                "Alphas" => alphas_panel(ui, &mut a, docs, &alpha_names, &nrm_names),
                "Viewport" => viewport_panel(ui, &mut a, docs, viewport_size),
                "UV" => uv_panel(ui, &mut a, docs, scene, &mesh_nodes, uv),
                "Brush" => brush_panel(ui, &mut a, docs, brush, model_name, has_model),
                "Lights" => lights_panel(ui, &mut a, scene),
                "Effects" => {
                    if post_ui::post_effects_panel(ui, post, scene) {
                        a.keep = true;
                    }
                }
                "Layers" => layers_panel(ui, &mut a, docs),
                "Segments" => segments_panel(ui, &mut a, segmentation),
                "Meshes" => meshes_panel(ui, &mut a, scene, &mesh_nodes),
                _ => {}
            });
        }

        let status = format!("{:.0} fps  ·  {}", fps, self.status);
        ui.status_bar(|ui| {
            ui.label(&status);
        });

        apply_actions(self, a)
    }
}

fn apply_actions(painter: &mut Painter, a: Actions) -> bool {
    if a.paint_map != painter.paint_map {
        painter.paint_map = a.paint_map;
    }
    if a.mode == AppMode::Paint && painter.mode == AppMode::Paint {
        painter.debug_view = a.debug_view;
    }
    if a.mode != painter.mode {
        painter.set_mode(a.mode);
    }
    painter.tool = a.tool;
    painter.seg_tool = a.seg_tool;
    painter.seg_op = a.seg_op;
    painter.rect_through = a.rect_through;
    painter.active_alpha = a.active_alpha;
    painter.active_nrm = a.active_nrm;

    let mut keep = a.keep;
    if a.open_model {
        painter.open_model_dialog();
        keep = true;
    } else if a.shape != painter.shape {
        painter.set_shape(a.shape);
    }
    if a.add_segment {
        painter.segmentation.add_segment();
        painter.seg_overlay_dirty = true;
        keep = true;
    }
    if a.del_segment {
        painter.segmentation.remove_active();
        painter.seg_overlay_dirty = true;
        keep = true;
    }
    if a.unwrap_uv {
        painter.unwrap_uv();
        keep = true;
    }
    if a.seg_overlay {
        painter.seg_overlay_dirty = true;
    }
    if let Some(i) = a.mask_click {
        painter.toggle_mask_target(i);
    } else if let Some(i) = a.content_click {
        if i < painter.doc().layers.len() {
            let doc = painter.doc_mut();
            doc.active = i;
            doc.paint_target = PaintTarget::Content;
        }
    }
    if a.dirty_layers {
        painter.doc_mut().mark_dirty();
    }
    if a.add_layer {
        painter.add_layer();
    }
    if a.add_fill {
        painter.add_fill_layer();
    }
    if a.del_layer {
        painter.remove_active_layer();
    }
    if a.clear_layer {
        painter.clear_active_layer();
    }
    if a.move_up {
        painter.doc_mut().move_active(1);
    }
    if a.move_down {
        painter.doc_mut().move_active(-1);
    }

    keep || painter.painting
}

fn alphas_panel(
    ui: &mut Ui,
    a: &mut Actions,
    docs: &[PaintDocument; 4],
    alpha_names: &[&str],
    nrm_names: &[&str],
) {
    let nrm_panel = a.paint_map == PaintMap::Normal
        && docs[a.paint_map.index()].paint_target != PaintTarget::Mask;
    ui.label(if nrm_panel { "Normals" } else { "Alphas" });
    ui.separator();
    let size = ui.available_size();
    if nrm_panel {
        ui.scroll_area("nrm_stamps", size, ScrollAxes::Vertical, |ui| {
            ui.grid(2, |ui| {
                for i in 0..nrm_names.len() {
                    ui.grid_cell(|ui| {
                        if ui
                            .selectable(&format!("nrm{i}"), i == a.active_nrm, |ui| {
                                let w = ui.available_size().x.max(24.0);
                                ui.texture(NRM_TEX_BASE + i as u32, Vec2::new(w, w));
                                ui.label(nrm_names[i]);
                            })
                            .clicked()
                        {
                            a.active_nrm = i;
                            a.keep = true;
                        }
                    });
                }
            });
        });
    } else {
        ui.scroll_area("alphas", size, ScrollAxes::Vertical, |ui| {
            ui.grid(2, |ui| {
                for i in 0..alpha_names.len() {
                    ui.grid_cell(|ui| {
                        if ui
                            .selectable(&format!("alpha{i}"), i == a.active_alpha, |ui| {
                                let w = ui.available_size().x.max(24.0);
                                ui.texture(ALPHA_TEX_BASE + i as u32, Vec2::new(w, w));
                                ui.label(alpha_names[i]);
                            })
                            .clicked()
                        {
                            a.active_alpha = i;
                            a.keep = true;
                        }
                    });
                }
            });
        });
    }
}

fn viewport_panel(
    ui: &mut Ui,
    a: &mut Actions,
    docs: &[PaintDocument; 4],
    viewport_size: &mut Vec2,
) {
    ui.label("Mode");
    let mut mode_idx = AppMode::ALL
        .iter()
        .position(|m| *m == a.mode)
        .unwrap_or(0);
    let mode_labels: Vec<&str> = AppMode::ALL.iter().map(|m| m.label()).collect();
    if ui.toggle("app_mode", &mut mode_idx, &mode_labels).changed() {
        a.mode = AppMode::ALL[mode_idx];
        a.keep = true;
    }
    let tool_hint = if a.mode == AppMode::Segment {
        match (a.seg_tool, a.seg_op) {
            (SegTool::Click, SegOp::Select) => "click faces",
            (SegTool::Click, SegOp::Deselect) => "click to unassign",
            (SegTool::Brush, SegOp::Select) => "brush faces",
            (SegTool::Brush, SegOp::Deselect) => "brush unassign",
            (SegTool::Rect, SegOp::Select) => "rect faces",
            (SegTool::Rect, SegOp::Deselect) => "rect unassign",
        }
    } else {
        match (a.tool, docs[a.paint_map.index()].paint_target) {
            (_, PaintTarget::Mask) => "paint mask",
            (PaintTool::Paint, _) => "paint",
            (PaintTool::Eraser, _) => "erase",
        }
    };
    ui.label_styled(
        &format!("LMB {tool_hint} · MMB pan · RMB orbit · wheel zoom"),
        TextStyle {
            color: [0.7, 0.7, 0.72, 1.0],
            size: 13.0,
        },
    );
    ui.separator();
    if a.mode == AppMode::Paint {
        ui.label("View");
        let mut view_idx = VIEW_MODES
            .iter()
            .position(|v| *v == a.debug_view)
            .unwrap_or(0);
        let view_labels: Vec<&str> = VIEW_MODES.iter().map(|v| v.label()).collect();
        if ui
            .toggle("debug_view", &mut view_idx, &view_labels)
            .changed()
        {
            a.debug_view = VIEW_MODES[view_idx];
            a.keep = true;
        }
        ui.separator();
    }
    let size = ui.available_size();
    *viewport_size = size;
    ui.texture(SCENE_TEX, size);
}

fn uv_panel(
    ui: &mut Ui,
    a: &mut Actions,
    docs: &mut [PaintDocument; 4],
    scene: &Scene,
    nodes: &[Handle<Node>],
    uv: &mut UvView,
) {
    let names: Vec<String> = nodes
        .iter()
        .map(|&h| {
            scene
                .nodes
                .get(h)
                .map(|n| {
                    if n.name.is_empty() {
                        let k = h.key();
                        format!("Mesh {}/{}", k.0, k.1)
                    } else {
                        n.name.clone()
                    }
                })
                .unwrap_or_else(|| "Mesh".into())
        })
        .collect();
    let map_labels: Vec<&str> = PaintMap::ALL.iter().map(|m| m.label()).collect();
    ui.row(|ui| {
        if names.is_empty() {
            ui.label("No meshes");
        } else {
            let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
            let mut idx = uv.mesh_idx.min(names.len() - 1);
            uv.mesh_idx = idx;
            if ui.select("uv_mesh", &mut idx, &name_refs).changed() {
                uv.mesh_idx = idx;
                a.keep = true;
            }
        }
        let mut map_idx = a.paint_map.index();
        if ui.select("uv_paint_map", &mut map_idx, &map_labels).changed() {
            a.paint_map = PaintMap::ALL[map_idx];
            a.active = docs[a.paint_map.index()].active;
            a.keep = true;
        }
        if ui.checkbox("Show UV", &mut uv.show_uv).changed() {
            a.keep = true;
        }
    });
    ui.separator();
    let size = ui.available_size();
    uv.size = size;
    ui.texture(UV_TEX, size);
}

fn brush_panel(
    ui: &mut Ui,
    a: &mut Actions,
    docs: &mut [PaintDocument; 4],
    brush: &mut Brush,
    model_name: &str,
    has_model: bool,
) {
    ui.label("Brush");
    ui.separator();
    if a.mode == AppMode::Segment {
        ui.label("Select tool");
        let mut tool_idx = SegTool::ALL
            .iter()
            .position(|t| *t == a.seg_tool)
            .unwrap_or(0);
        let tool_labels: Vec<&str> = SegTool::ALL.iter().map(|t| t.label()).collect();
        if ui.toggle("seg_tool", &mut tool_idx, &tool_labels).changed() {
            a.seg_tool = SegTool::ALL[tool_idx];
            a.keep = true;
        }
        ui.label("Action");
        let mut op_idx = SegOp::ALL.iter().position(|t| *t == a.seg_op).unwrap_or(0);
        let op_labels: Vec<&str> = SegOp::ALL.iter().map(|t| t.label()).collect();
        if ui.toggle("seg_op", &mut op_idx, &op_labels).changed() {
            a.seg_op = SegOp::ALL[op_idx];
            a.keep = true;
        }
        if a.seg_tool == SegTool::Rect {
            if ui
                .checkbox("Through mesh", &mut a.rect_through)
                .changed()
            {
                a.keep = true;
            }
        }
        if a.seg_tool == SegTool::Brush {
            ui.label("Radius — world-space brush");
            ui.slider("radius", &mut brush.radius, 0.01..=0.35);
        }
        ui.separator();
        ui.label("Mesh");
        if ui.button("Open…").clicked() {
            a.open_model = true;
        }
        let options: &[&str] = if has_model {
            &["Sphere", "Cube", "Model"]
        } else {
            &["Sphere", "Cube"]
        };
        let mut sel = a.shape.min(options.len().saturating_sub(1));
        if ui.toggle("shape", &mut sel, options).changed() {
            a.shape = sel;
            a.keep = true;
        }
        if has_model && !model_name.is_empty() {
            ui.label(model_name);
        }
        return;
    }
    ui.label("Tool");
    let mut tool_idx = PaintTool::ALL
        .iter()
        .position(|t| *t == a.tool)
        .unwrap_or(0);
    let tool_labels: Vec<&str> = PaintTool::ALL.iter().map(|t| t.label()).collect();
    if ui.toggle("paint_tool", &mut tool_idx, &tool_labels).changed() {
        a.tool = PaintTool::ALL[tool_idx];
        a.keep = true;
    }
    ui.separator();
    let targeting_mask = docs[a.paint_map.index()].paint_target == PaintTarget::Mask;
    let fill_content = docs[a.paint_map.index()]
        .layers
        .get(a.active)
        .is_some_and(|l| l.kind == LayerKind::Fill);
    if fill_content {
        if a.paint_map == PaintMap::Albedo {
            ui.label("Fill color");
            if let Some(layer) = docs[a.paint_map.index()].layers.get_mut(a.active) {
                if ui.color_edit("fill_color", &mut layer.fill).changed() {
                    a.dirty_layers = true;
                    a.keep = true;
                }
            }
        } else if a.paint_map == PaintMap::Normal {
            ui.label("Fill — flat normal (128, 128, 255)");
        } else if let Some(layer) = docs[a.paint_map.index()].layers.get_mut(a.active) {
            ui.label("Fill value");
            let mut v = luma(layer.fill);
            if ui.slider("fill_value", &mut v, 0.0..=1.0).changed() {
                layer.fill = [v, v, v, 1.0];
                a.dirty_layers = true;
                a.keep = true;
            }
        }
    }
    if targeting_mask {
        ui.label("Brush — luminance on mask (white pass, black block)");
    } else if a.paint_map == PaintMap::Normal {
        ui.label("Normal stamp — pick from the left panel");
    } else if fill_content {
        ui.label("Brush — luminance on mask (white pass, black block)");
    } else {
        ui.label("Color — stamp tint (MR maps use luminance)");
    }
    if a.paint_map != PaintMap::Normal && (a.tool == PaintTool::Paint || targeting_mask || fill_content)
    {
        ui.color_edit("color", &mut brush.color);
    }
    ui.label("Radius — world-space stamp size");
    ui.slider("radius", &mut brush.radius, 0.01..=0.35);
    ui.label("Hardness — 0 soft falloff · 1 hard edge");
    ui.slider("hardness", &mut brush.hardness, 0.0..=1.0);
    ui.label("Opacity — stroke strength");
    ui.slider("opacity", &mut brush.opacity, 0.05..=1.0);
    ui.separator();
    ui.label("Mesh");
    if ui.button("Open…").clicked() {
        a.open_model = true;
    }
    let options: &[&str] = if has_model {
        &["Sphere", "Cube", "Model"]
    } else {
        &["Sphere", "Cube"]
    };
    let mut sel = a.shape.min(options.len().saturating_sub(1));
    if ui.toggle("shape", &mut sel, options).changed() {
        a.shape = sel;
        a.keep = true;
    }
    if has_model && !model_name.is_empty() {
        ui.label(model_name);
    }
}

fn lights_panel(ui: &mut Ui, a: &mut Actions, scene: &mut Scene) {
    ui.label("Lights");
    ui.separator();
    ui.label("Ambient");
    let mut amb = [scene.ambient[0], scene.ambient[1], scene.ambient[2], 1.0];
    if ui.color_edit("ambient", &mut amb).changed() {
        scene.ambient = [amb[0], amb[1], amb[2]];
        a.keep = true;
    }
    if let Some(Light::Directional(d)) = scene.lights.first_mut() {
        ui.separator();
        ui.label("Directional");
        if ui.checkbox("Cast shadows", &mut d.cast_shadows).changed() {
            a.keep = true;
        }
        ui.label("Intensity");
        if ui
            .slider("light_intensity", &mut d.intensity, 0.0..=8.0)
            .changed()
        {
            a.keep = true;
        }
        ui.label("Color");
        let mut col = [d.color[0], d.color[1], d.color[2], 1.0];
        if ui.color_edit("sun_color", &mut col).changed() {
            d.color = [col[0], col[1], col[2]];
            a.keep = true;
        }
        ui.label("Direction (world)");
        if ui.slider("dir_x", &mut d.direction.x, -1.0..=1.0).changed()
            || ui.slider("dir_y", &mut d.direction.y, -1.0..=1.0).changed()
            || ui.slider("dir_z", &mut d.direction.z, -1.0..=1.0).changed()
        {
            a.keep = true;
        }
    }
}

fn layers_panel(ui: &mut Ui, a: &mut Actions, docs: &mut [PaintDocument; 4]) {
    ui.label("Paint map");
    let mut map_idx = a.paint_map.index();
    let map_labels: Vec<&str> = PaintMap::ALL.iter().map(|m| m.label()).collect();
    if ui.toggle("paint_map", &mut map_idx, &map_labels).changed() {
        a.paint_map = PaintMap::ALL[map_idx];
        a.active = docs[a.paint_map.index()].active;
        a.keep = true;
    }
    ui.separator();
    ui.row(|ui| {
        if ui
            .button_with("add_layer", |ui| {
                ui.icon("plus", 14.0);
            })
            .clicked()
        {
            a.add_layer = true;
        }
        if ui.button("Fill").clicked() {
            a.add_fill = true;
        }
        if ui
            .button_with("del_layer", |ui| {
                ui.icon("delete", 14.0);
            })
            .clicked()
        {
            a.del_layer = true;
        }
        if ui
            .button_with("clear_layer", |ui| {
                ui.icon("reset", 14.0);
            })
            .clicked()
        {
            a.clear_layer = true;
        }
        if ui
            .button_with("layer_up", |ui| {
                ui.icon("chevron_up", 14.0);
            })
            .clicked()
        {
            a.move_up = true;
        }
        if ui
            .button_with("layer_dn", |ui| {
                ui.icon("chevron_down", 14.0);
            })
            .clicked()
        {
            a.move_down = true;
        }
    });
    ui.label("Pencil = mask · fill paints mask (white reveals)");
    ui.separator();
    let paint_target = docs[a.paint_map.index()].paint_target;
    let layers = &mut docs[a.paint_map.index()].layers;
    let size = ui.available_size();
    ui.scroll_area("layers", size, ScrollAxes::Vertical, |ui| {
        for i in (0..layers.len()).rev() {
            let is_active = i == a.active;
            let name = layers[i].name.clone();
            let kind = layers[i].kind;
            let has_mask = layers[i].mask.is_some();
            let mask_selected = is_active && paint_target == PaintTarget::Mask && has_mask;
            if ui
                .selectable(&format!("layer{i}"), is_active, |ui| {
                    ui.row(|ui| {
                        let eye = if layers[i].visible {
                            "visibility"
                        } else {
                            "visibility_off"
                        };
                        if ui.icon_button(&format!("vis{i}"), eye, layers[i].visible) {
                            layers[i].visible = !layers[i].visible;
                            a.dirty_layers = true;
                            a.keep = true;
                        }
                        ui.label(&name);
                        if ui.icon_button(&format!("mask{i}"), "edit", mask_selected) {
                            a.mask_click = Some(i);
                            a.keep = true;
                        }
                    });
                    if kind == LayerKind::Fill {
                        ui.label("Fill");
                        if a.paint_map == PaintMap::Albedo {
                            if ui
                                .color_edit(&format!("fill{i}"), &mut layers[i].fill)
                                .changed()
                            {
                                a.dirty_layers = true;
                                a.keep = true;
                            }
                        } else if a.paint_map == PaintMap::Normal {
                            ui.label("Flat nrm");
                        } else {
                            let mut v = luma(layers[i].fill);
                            if ui
                                .slider(&format!("fillv{i}"), &mut v, 0.0..=1.0)
                                .changed()
                            {
                                layers[i].fill = [v, v, v, 1.0];
                                a.dirty_layers = true;
                                a.keep = true;
                            }
                        }
                    }
                    ui.label("Opacity");
                    if ui
                        .slider(&format!("op{i}"), &mut layers[i].opacity, 0.0..=1.0)
                        .changed()
                    {
                        a.dirty_layers = true;
                        a.keep = true;
                    }
                })
                .clicked()
            {
                a.active = i;
                a.content_click = Some(i);
                a.keep = true;
            }
        }
    });
}

fn segments_panel(ui: &mut Ui, a: &mut Actions, segmentation: &mut Segmentation) {
    ui.label("Segments");
    ui.separator();
    ui.row(|ui| {
        if ui
            .button_with("add_seg", |ui| {
                ui.icon("plus", 14.0);
            })
            .clicked()
        {
            a.add_segment = true;
            a.keep = true;
        }
        if ui
            .button_with("del_seg", |ui| {
                ui.icon("delete", 14.0);
            })
            .clicked()
        {
            a.del_segment = true;
            a.keep = true;
        }
    });
    if ui.button("Unwrap UV").clicked() {
        a.unwrap_uv = true;
        a.keep = true;
    }
    ui.label("Color is viewport only · one triangle, one segment");
    ui.separator();
    let active = segmentation.active;
    let size = ui.available_size();
    ui.scroll_area("segments", size, ScrollAxes::Vertical, |ui| {
        for i in 0..segmentation.segments.len() {
            let id = segmentation.segments[i].id;
            let is_active = active == Some(id);
            let count = segmentation.face_count(id);
            if ui
                .selectable(&format!("seg{id}"), is_active, |ui| {
                    ui.row(|ui| {
                        ui.label(&format!("#{id}"));
                        ui.label(&format!("{count} faces"));
                    });
                    if ui
                        .text_input(&format!("seg_name{id}"), &mut segmentation.segments[i].name)
                        .changed()
                    {
                        a.keep = true;
                    }
                    if ui
                        .color_edit(&format!("seg_col{id}"), &mut segmentation.segments[i].color)
                        .changed()
                    {
                        a.seg_overlay = true;
                        a.keep = true;
                    }
                })
                .clicked()
            {
                segmentation.active = Some(id);
                a.keep = true;
            }
        }
    });
}

fn meshes_panel(ui: &mut Ui, a: &mut Actions, scene: &mut Scene, nodes: &[Handle<Node>]) {
    ui.label("Meshes");
    ui.separator();
    if nodes.is_empty() {
        ui.label("No meshes");
        return;
    }
    let size = ui.available_size();
    ui.scroll_area("meshes", size, ScrollAxes::Vertical, |ui| {
        for &h in nodes {
            let key = h.key();
            let (visible, name) = {
                let Some(node) = scene.nodes.get(h) else {
                    continue;
                };
                let name = if node.name.is_empty() {
                    format!("Mesh {}/{}", key.0, key.1)
                } else {
                    node.name.clone()
                };
                (node.visible, name)
            };
            ui.row(|ui| {
                let eye = if visible {
                    "visibility"
                } else {
                    "visibility_off"
                };
                if ui.icon_button(&format!("mesh_vis{}_{}", key.0, key.1), eye, visible) {
                    if let Some(n) = scene.nodes.get_mut(h) {
                        n.visible = !n.visible;
                    }
                    a.seg_overlay = true;
                    a.keep = true;
                }
                ui.label(&name);
            });
        }
    });
}
