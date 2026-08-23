//! Automatic UV unwrap. Algorithms are swappable via [`UnwrapAlgo`].

mod tri_pack;

use std::collections::HashMap;

use glam::Vec2;
use mega_render::Mesh;

use crate::segment::{SegmentId, Segmentation, UNASSIGNED};

pub use tri_pack::TriPack;

/// Padding inside each UDIM tile (UV units).
pub const PAD: f32 = 0.012;

/// Mari UDIM origin: 1001 → (0,0), 1002 → (1,0), 1011 → (0,1).
pub fn udim_origin(udim: u32) -> [f32; 2] {
    let n = udim.saturating_sub(1001);
    [(n % 10) as f32, (n / 10) as f32]
}

/// Inverse of [`udim_origin`]: `floor(uv)` → Mari id.
pub fn udim_from_uv(uv: Vec2) -> u32 {
    let u = uv.x.floor().clamp(0.0, 9.0) as u32;
    let v = uv.y.floor().clamp(0.0, 9.0) as u32;
    1001 + u + v * 10
}

/// Local UVs in `[0,1]` for the given triangle indices, same order as `tris`.
pub trait UnwrapAlgo {
    fn name(&self) -> &'static str;
    fn unwrap_tris(
        &self,
        positions: &[[f32; 3]],
        indices: &[u32],
        tris: &[u32],
    ) -> Vec<[Vec2; 3]>;
}

/// Leftover faces first (if any), then non-empty named segments.
pub fn segment_udims(seg: &Segmentation) -> HashMap<SegmentId, u32> {
    let mut map = HashMap::new();
    let mut next = 1001u32;
    if seg.leftover_faces() > 0 {
        map.insert(UNASSIGNED, next);
        next += 1;
    }
    for s in &seg.segments {
        if seg.face_count(s.id) > 0 {
            map.insert(s.id, next);
            next += 1;
        }
    }
    if map.is_empty() {
        map.insert(UNASSIGNED, 1001);
    }
    map
}

pub fn face_udims(seg: &Segmentation, mesh: mega_render::Handle<Mesh>, tri_count: usize) -> Vec<u32> {
    let map = segment_udims(seg);
    let fallback = *map.get(&UNASSIGNED).unwrap_or(&1001);
    (0..tri_count)
        .map(|ti| {
            let label = seg.label_of(mesh, ti as u32);
            map.get(&label).copied().unwrap_or(fallback)
        })
        .collect()
}

/// Split vertices on every corner and write UDIM UVs. Triangle order is kept.
pub fn apply_unwrap(mesh: &mut Mesh, face_udim: &[u32], algo: &dyn UnwrapAlgo) {
    let tri_count = mesh.indices.len() / 3;
    if tri_count == 0 {
        return;
    }
    let n = tri_count.min(face_udim.len());
    let mut buckets: HashMap<u32, Vec<u32>> = HashMap::new();
    for ti in 0..n as u32 {
        buckets.entry(face_udim[ti as usize]).or_default().push(ti);
    }

    let mut local = vec![[Vec2::splat(0.5); 3]; tri_count];
    for (udim, tris) in &buckets {
        let packed = algo.unwrap_tris(&mesh.positions, &mesh.indices, tris);
        let origin = Vec2::from_array(udim_origin(*udim));
        for (i, &ti) in tris.iter().enumerate() {
            let Some(uv) = packed.get(i) else {
                continue;
            };
            local[ti as usize] = [uv[0] + origin, uv[1] + origin, uv[2] + origin];
        }
    }

    let mut corner_uv = Vec::with_capacity(n * 3);
    for ti in 0..n {
        for k in 0..3 {
            corner_uv.push(local[ti][k].to_array());
        }
    }
    expand_mesh(mesh, &corner_uv);
}

fn expand_mesh(mesh: &mut Mesh, corner_uv: &[[f32; 2]]) {
    let idx = mesh.indices.clone();
    let n = corner_uv.len().min(idx.len());
    if n == 0 {
        return;
    }
    let idx = &idx[..n];

    let positions = expand(&mesh.positions, idx, [0.0; 3]);
    let normals = expand(&mesh.normals, idx, [0.0, 1.0, 0.0]);
    let tangents = expand(&mesh.tangents, idx, [1.0, 0.0, 0.0, 1.0]);
    let uvs: Vec<Vec<[f32; 2]>> = mesh
        .uvs
        .iter()
        .enumerate()
        .map(|(ci, col)| {
            if ci == 0 {
                corner_uv[..n].to_vec()
            } else {
                expand(col, idx, [0.0, 0.0])
            }
        })
        .collect();
    let uvs = if uvs.is_empty() {
        vec![corner_uv[..n].to_vec()]
    } else {
        uvs
    };
    let colors: Vec<_> = mesh
        .colors
        .iter()
        .map(|c| expand(c, idx, [1.0, 1.0, 1.0, 1.0]))
        .collect();
    let joints: Vec<_> = mesh.joints.iter().map(|c| expand(c, idx, [0; 4])).collect();
    let weights: Vec<_> = mesh
        .weights
        .iter()
        .map(|c| expand(c, idx, [0.0; 4]))
        .collect();

    if !mesh.basis_positions.is_empty() {
        mesh.basis_positions = expand(&mesh.basis_positions, idx, [0.0; 3]);
    }
    if !mesh.basis_normals.is_empty() {
        mesh.basis_normals = expand(&mesh.basis_normals, idx, [0.0, 1.0, 0.0]);
    }
    for t in &mut mesh.morph_targets {
        if !t.position_deltas.is_empty() {
            t.position_deltas = expand(&t.position_deltas, idx, [0.0; 3]);
        }
        if !t.normal_deltas.is_empty() {
            t.normal_deltas = expand(&t.normal_deltas, idx, [0.0; 3]);
        }
    }

    mesh.positions = positions;
    mesh.normals = normals;
    mesh.tangents = tangents;
    mesh.uvs = uvs;
    mesh.colors = colors;
    mesh.joints = joints;
    mesh.weights = weights;
    mesh.indices = (0..n as u32).collect();
    mesh.mark_changed();
}

fn expand<T: Copy>(src: &[T], indices: &[u32], fallback: T) -> Vec<T> {
    indices
        .iter()
        .map(|&i| src.get(i as usize).copied().unwrap_or(fallback))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mega_render::Mesh;

    fn tri_mesh() -> Mesh {
        Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            vec![[0.0, 0.0, 1.0]; 4],
            vec![[0.0, 0.0]; 4],
            vec![0, 1, 2, 1, 3, 2],
        )
    }

    #[test]
    fn mari_id_from_uv() {
        assert_eq!(udim_from_uv(Vec2::new(0.2, 0.3)), 1001);
        assert_eq!(udim_from_uv(Vec2::new(1.2, 0.3)), 1002);
        assert_eq!(udim_from_uv(Vec2::new(0.2, 1.1)), 1011);
    }

    #[test]
    fn two_tiles_offset() {
        let mut mesh = tri_mesh();
        apply_unwrap(&mut mesh, &[1001, 1002], &TriPack);
        assert_eq!(mesh.positions.len(), 6);
        let uv = &mesh.uvs[0];
        assert!(uv[0][0] < 1.0 && uv[1][0] < 1.0 && uv[2][0] < 1.0);
        assert!(uv[3][0] >= 1.0 && uv[4][0] >= 1.0 && uv[5][0] >= 1.0);
    }
}
