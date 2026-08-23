use std::collections::{HashMap, HashSet};

use mega_render::{Handle, Mesh, Node, Scene};

pub type SegmentId = u16;
pub const UNASSIGNED: SegmentId = 0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AppMode {
    #[default]
    Paint,
    Segment,
}

impl AppMode {
    pub const ALL: &'static [AppMode] = &[AppMode::Paint, AppMode::Segment];

    pub fn label(self) -> &'static str {
        match self {
            AppMode::Paint => "Paint",
            AppMode::Segment => "Segment",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SegTool {
    #[default]
    Click,
    Brush,
    Rect,
}

impl SegTool {
    pub const ALL: &'static [SegTool] = &[SegTool::Click, SegTool::Brush, SegTool::Rect];

    pub fn label(self) -> &'static str {
        match self {
            SegTool::Click => "Click",
            SegTool::Brush => "Brush",
            SegTool::Rect => "Rect",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SegOp {
    #[default]
    Select,
    Deselect,
}

impl SegOp {
    pub const ALL: &'static [SegOp] = &[SegOp::Select, SegOp::Deselect];

    pub fn label(self) -> &'static str {
        match self {
            SegOp::Select => "Select",
            SegOp::Deselect => "Deselect",
        }
    }
}

const PALETTE: [[f32; 4]; 8] = [
    [0.92, 0.25, 0.22, 0.50],
    [0.22, 0.76, 0.34, 0.50],
    [0.24, 0.48, 0.95, 0.50],
    [0.95, 0.78, 0.18, 0.50],
    [0.86, 0.28, 0.78, 0.50],
    [0.18, 0.82, 0.82, 0.50],
    [0.95, 0.52, 0.16, 0.50],
    [0.58, 0.38, 0.92, 0.50],
];

#[derive(Clone)]
pub struct Segment {
    pub id: SegmentId,
    pub name: String,
    pub color: [f32; 4],
}

struct MeshSeg {
    mesh: Handle<Mesh>,
    labels: Vec<SegmentId>,
}

pub struct Segmentation {
    pub segments: Vec<Segment>,
    pub active: Option<SegmentId>,
    next_id: SegmentId,
    meshes: HashMap<(u32, u32), MeshSeg>,
    assigned: HashSet<((u32, u32), u32)>,
    counts: HashMap<SegmentId, usize>,
}

impl Default for Segmentation {
    fn default() -> Self {
        let mut s = Self {
            segments: Vec::new(),
            active: None,
            next_id: 1,
            meshes: HashMap::new(),
            assigned: HashSet::new(),
            counts: HashMap::new(),
        };
        s.add_segment();
        s
    }
}

impl Segmentation {
    pub fn add_segment(&mut self) -> SegmentId {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1).max(1);
        let n = self.segments.len();
        self.segments.push(Segment {
            id,
            name: format!("Segment {}", n + 1),
            color: PALETTE[n % PALETTE.len()],
        });
        self.active = Some(id);
        id
    }

    pub fn remove_active(&mut self) {
        let Some(id) = self.active else {
            return;
        };
        self.segments.retain(|s| s.id != id);
        let doomed: Vec<_> = self
            .assigned
            .iter()
            .copied()
            .filter(|&(mk, ti)| {
                self.meshes
                    .get(&mk)
                    .and_then(|m| m.labels.get(ti as usize).copied())
                    == Some(id)
            })
            .collect();
        for (mk, ti) in doomed {
            if let Some(ms) = self.meshes.get_mut(&mk) {
                if let Some(slot) = ms.labels.get_mut(ti as usize) {
                    *slot = UNASSIGNED;
                }
            }
            self.assigned.remove(&(mk, ti));
        }
        self.counts.remove(&id);
        self.active = self.segments.last().map(|s| s.id);
    }

    pub fn color_of(&self, id: SegmentId) -> Option<[f32; 4]> {
        self.segments.iter().find(|s| s.id == id).map(|s| s.color)
    }

    pub fn face_count(&self, id: SegmentId) -> usize {
        self.counts.get(&id).copied().unwrap_or(0)
    }

    pub fn leftover_faces(&self) -> usize {
        self.meshes
            .values()
            .map(|ms| ms.labels.iter().filter(|&&id| id == UNASSIGNED).count())
            .sum()
    }

    pub fn label_of(&self, mesh: Handle<Mesh>, tri: u32) -> SegmentId {
        self.meshes
            .get(&mesh.key())
            .and_then(|m| m.labels.get(tri as usize).copied())
            .unwrap_or(UNASSIGNED)
    }

    pub fn sync(&mut self, scene: &Scene, paintable: &[Handle<Node>]) {
        let mut keep = HashSet::new();
        for &nh in paintable {
            let Some(node) = scene.nodes.get(nh) else {
                continue;
            };
            let Some(mh) = node.mesh else {
                continue;
            };
            let Some(mesh) = scene.meshes.get(mh) else {
                continue;
            };
            let n = mesh.indices.len() / 3;
            keep.insert(mh.key());
            match self.meshes.get_mut(&mh.key()) {
                Some(ms) if ms.labels.len() == n => {
                    ms.mesh = mh;
                }
                _ => {
                    self.assigned.retain(|(k, _)| *k != mh.key());
                    self.meshes.insert(
                        mh.key(),
                        MeshSeg {
                            mesh: mh,
                            labels: vec![UNASSIGNED; n],
                        },
                    );
                }
            }
        }
        self.meshes.retain(|k, _| keep.contains(k));
        self.assigned.retain(|(k, _)| keep.contains(k));
        self.rebuild_counts();
    }

    fn rebuild_counts(&mut self) {
        self.counts.clear();
        for &(mk, ti) in &self.assigned {
            let Some(id) = self
                .meshes
                .get(&mk)
                .and_then(|m| m.labels.get(ti as usize).copied())
            else {
                continue;
            };
            if id != UNASSIGNED {
                *self.counts.entry(id).or_insert(0) += 1;
            }
        }
    }

    pub fn set_face(&mut self, mesh: Handle<Mesh>, tri: u32, id: SegmentId) {
        let key = mesh.key();
        let Some(ms) = self.meshes.get_mut(&key) else {
            return;
        };
        let Some(slot) = ms.labels.get_mut(tri as usize) else {
            return;
        };
        let old = *slot;
        if old == id {
            return;
        }
        *slot = id;
        if old != UNASSIGNED {
            if let Some(c) = self.counts.get_mut(&old) {
                *c = c.saturating_sub(1);
            }
        }
        if id == UNASSIGNED {
            self.assigned.remove(&(key, tri));
        } else {
            self.assigned.insert((key, tri));
            *self.counts.entry(id).or_insert(0) += 1;
        }
    }

    pub fn set_faces(&mut self, faces: &[(Handle<Mesh>, u32)], id: SegmentId) {
        for &(mesh, tri) in faces {
            self.set_face(mesh, tri, id);
        }
    }

    /// Assigned faces for overlay: `(mesh, tri, color)`.
    pub fn overlay_faces(&self) -> Vec<(Handle<Mesh>, u32, [f32; 4])> {
        let mut out = Vec::with_capacity(self.assigned.len());
        for &(mk, ti) in &self.assigned {
            let Some(ms) = self.meshes.get(&mk) else {
                continue;
            };
            let Some(&id) = ms.labels.get(ti as usize) else {
                continue;
            };
            if id == UNASSIGNED {
                continue;
            }
            let Some(color) = self.color_of(id) else {
                continue;
            };
            out.push((ms.mesh, ti, color));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mega_render::Mesh;

    fn tri_mesh() -> Mesh {
        Mesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            vec![[0.0, 0.0, 1.0]; 4],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            vec![0, 1, 2, 1, 3, 2],
        )
    }

    #[test]
    fn steal_and_delete() {
        let mut scene = Scene::new();
        let mh = scene.meshes.insert(tri_mesh());
        let node = scene.nodes.insert(mega_render::Node {
            name: "n".into(),
            parent: None,
            local: mega_render::Transform::default(),
            mesh: Some(mh),
            material: None,
            skin: None,
            visible: true,
        });
        let mut seg = Segmentation::default();
        let a = seg.active.unwrap();
        seg.sync(&scene, &[node]);
        seg.set_face(mh, 0, a);
        assert_eq!(seg.face_count(a), 1);

        let b = seg.add_segment();
        seg.set_face(mh, 0, b);
        assert_eq!(seg.face_count(a), 0);
        assert_eq!(seg.face_count(b), 1);

        seg.active = Some(b);
        seg.remove_active();
        assert_eq!(seg.face_count(b), 0);
        assert!(seg.segments.iter().all(|s| s.id != b));
    }
}
