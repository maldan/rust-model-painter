use std::collections::HashMap;

use glam::{Mat4, Vec2, Vec3};
use mega_render::{Handle, Mesh, Node, Scene};
use mega_ui::Rect;

use crate::bvh::MeshBvh;

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct Hit {
    pub node: Handle<Node>,
    pub t: f32,
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
    /// World-space triangle corners (for brush radius projection).
    pub tri_world: [Vec3; 3],
    pub tri_uv: [Vec2; 3],
}

pub type BvhCache = HashMap<(u32, u32), MeshBvh>;

pub fn screen_ray(screen: Vec2, viewport: Rect, cam_eye: Vec3, view_proj_inv: Mat4) -> (Vec3, Vec3) {
    let w = viewport.width().max(1.0);
    let h = viewport.height().max(1.0);
    let u = ((screen.x - viewport.min.x) / w).clamp(0.0, 1.0);
    let v = ((screen.y - viewport.min.y) / h).clamp(0.0, 1.0);
    let ndc = Vec3::new(u * 2.0 - 1.0, 1.0 - v * 2.0, 1.0);
    let far = view_proj_inv.project_point3(ndc);
    let dir = (far - cam_eye).normalize_or_zero();
    (cam_eye, dir)
}

/// Ensure BVHs exist for every mesh referenced by `paintable` nodes.
pub fn ensure_bvhs(scene: &Scene, paintable: &[Handle<Node>], cache: &mut BvhCache) {
    let mut needed = std::collections::HashSet::new();
    for &node_h in paintable {
        let Some(node) = scene.nodes.get(node_h) else {
            continue;
        };
        let Some(mesh_h) = node.mesh else {
            continue;
        };
        needed.insert(mesh_h.key());
        if cache.contains_key(&mesh_h.key()) {
            continue;
        }
        if let Some(mesh) = scene.meshes.get(mesh_h) {
            cache.insert(mesh_h.key(), MeshBvh::build(mesh));
        }
    }
    cache.retain(|k, _| needed.contains(k));
}

pub fn pick_mesh(
    scene: &Scene,
    origin: Vec3,
    dir: Vec3,
    paintable: &[Handle<Node>],
    cache: &BvhCache,
) -> Option<Hit> {
    let mut best: Option<Hit> = None;
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
        let world = scene.world_matrix(node_h);
        let bvh = cache.get(&mesh_h.key());
        if let Some(hit) = ray_mesh_uv(origin, dir, mesh, world, node_h, bvh) {
            if best.is_none_or(|b| hit.t < b.t) {
                best = Some(hit);
            }
        }
    }
    best
}

fn ray_mesh_uv(
    origin: Vec3,
    dir: Vec3,
    mesh: &Mesh,
    world: Mat4,
    node: Handle<Node>,
    bvh: Option<&MeshBvh>,
) -> Option<Hit> {
    let inv_world = world.inverse();
    let local_o = inv_world.transform_point3(origin);
    let local_d = inv_world.transform_vector3(dir);
    if local_d.length_squared() < 1e-16 {
        return None;
    }

    let (best_t, i0, i1, i2, p0, p1, p2, u, v) = if let Some(bvh) = bvh {
        let (t, ti, u, v) = bvh.raycast(mesh, local_o, local_d)?;
        let base = ti as usize * 3;
        let i0 = mesh.indices[base] as usize;
        let i1 = mesh.indices[base + 1] as usize;
        let i2 = mesh.indices[base + 2] as usize;
        let p0 = Vec3::from_array(mesh.positions[i0]);
        let p1 = Vec3::from_array(mesh.positions[i1]);
        let p2 = Vec3::from_array(mesh.positions[i2]);
        (t, i0, i1, i2, p0, p1, p2, u, v)
    } else {
        // Fallback brute-force (primitives without cache yet).
        let mut best_t = f32::MAX;
        let mut best = None;
        for tri in mesh.indices.chunks_exact(3) {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;
            let p0 = Vec3::from_array(mesh.positions[i0]);
            let p1 = Vec3::from_array(mesh.positions[i1]);
            let p2 = Vec3::from_array(mesh.positions[i2]);
            if let Some((t, u, v)) = ray_triangle(local_o, local_d, p0, p1, p2) {
                if t < best_t {
                    best_t = t;
                    best = Some((i0, i1, i2, p0, p1, p2, u, v));
                }
            }
        }
        let (i0, i1, i2, p0, p1, p2, u, v) = best?;
        (best_t, i0, i1, i2, p0, p1, p2, u, v)
    };

    let uv0 = Vec2::from_array(mesh.uvs.get(i0).copied().unwrap_or([0.0, 0.0]));
    let uv1 = Vec2::from_array(mesh.uvs.get(i1).copied().unwrap_or([0.0, 0.0]));
    let uv2 = Vec2::from_array(mesh.uvs.get(i2).copied().unwrap_or([0.0, 0.0]));
    let uv = uv0 + (uv1 - uv0) * u + (uv2 - uv0) * v;

    let n0 = Vec3::from_array(mesh.normals.get(i0).copied().unwrap_or([0.0, 1.0, 0.0]));
    let n1 = Vec3::from_array(mesh.normals.get(i1).copied().unwrap_or([0.0, 1.0, 0.0]));
    let n2 = Vec3::from_array(mesh.normals.get(i2).copied().unwrap_or([0.0, 1.0, 0.0]));
    let n_local = (n0 + (n1 - n0) * u + (n2 - n0) * v).normalize_or_zero();
    let normal_matrix = inv_world.transpose();
    let normal = normal_matrix.transform_vector3(n_local).normalize_or_zero();

    let local_hit = local_o + local_d * best_t;
    let position = world.transform_point3(local_hit);
    let world_t = (position - origin).dot(dir);

    Some(Hit {
        node,
        t: world_t.max(0.0),
        position,
        normal,
        uv,
        tri_world: [
            world.transform_point3(p0),
            world.transform_point3(p1),
            world.transform_point3(p2),
        ],
        tri_uv: [uv0, uv1, uv2],
    })
}

fn ray_triangle(origin: Vec3, dir: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<(f32, f32, f32)> {
    const EPS: f32 = 1e-6;
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let pvec = dir.cross(e2);
    let det = e1.dot(pvec);
    if det.abs() < EPS {
        return None;
    }
    let inv_det = 1.0 / det;
    let tvec = origin - v0;
    let u = tvec.dot(pvec) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qvec = tvec.cross(e1);
    let v = dir.dot(qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(qvec) * inv_det;
    (t > EPS).then_some((t, u, v))
}

/// Approximate UV-space radius — unused after 3D surface stamps; kept for experiments.
#[allow(dead_code)]
pub fn world_radius_to_uv(hit: &Hit, world_radius: f32) -> f32 {
    let e1 = hit.tri_world[1] - hit.tri_world[0];
    let e2 = hit.tri_world[2] - hit.tri_world[0];
    let du1 = hit.tri_uv[1] - hit.tri_uv[0];
    let du2 = hit.tri_uv[2] - hit.tri_uv[0];
    let world_area = e1.cross(e2).length();
    let uv_area = (du1.x * du2.y - du1.y * du2.x).abs();
    if world_area < 1e-12 || uv_area < 1e-12 {
        return 0.02;
    }
    let scale = (uv_area / world_area).sqrt();
    (world_radius * scale).clamp(0.0005, 0.5)
}

pub fn find_viewport_rect(draw_list: &[mega_ui::DrawCommand], scene_tex: u32) -> Option<Rect> {
    draw_list
        .iter()
        .rev()
        .find(|c| (c.kind - 1.0).abs() < 0.1 && c.tex == scene_tex)
        .map(|c| c.rect)
}
