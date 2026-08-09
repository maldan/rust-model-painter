//! Simple midpoint-split BVH for local-space mesh picking.

use glam::Vec3;
use mega_render::Mesh;

const LEAF_TRIS: usize = 8;

#[derive(Clone, Copy)]
struct BvhNode {
    min: Vec3,
    max: Vec3,
    /// Leaf: first index into `tri_indices`. Interior: left child.
    left_first: u32,
    /// 0 = interior. >0 = leaf triangle count.
    tri_count: u32,
    /// Interior: right child. Leaf: unused.
    right: u32,
}

/// Acceleration structure over mesh triangles (local space).
pub struct MeshBvh {
    nodes: Vec<BvhNode>,
    /// Triangle indices (index into `mesh.indices` chunks of 3).
    tri_indices: Vec<u32>,
}

impl MeshBvh {
    pub fn build(mesh: &Mesh) -> Self {
        let tri_count = mesh.indices.len() / 3;
        let mut tri_indices: Vec<u32> = (0..tri_count as u32).collect();
        let mut centroids = Vec::with_capacity(tri_count);
        let mut tri_bounds = Vec::with_capacity(tri_count);

        for ti in 0..tri_count {
            let base = ti * 3;
            let i0 = mesh.indices[base] as usize;
            let i1 = mesh.indices[base + 1] as usize;
            let i2 = mesh.indices[base + 2] as usize;
            let p0 = Vec3::from_array(mesh.positions[i0]);
            let p1 = Vec3::from_array(mesh.positions[i1]);
            let p2 = Vec3::from_array(mesh.positions[i2]);
            let min = p0.min(p1).min(p2);
            let max = p0.max(p1).max(p2);
            tri_bounds.push((min, max));
            centroids.push((p0 + p1 + p2) * (1.0 / 3.0));
        }

        let mut nodes = Vec::with_capacity(tri_count * 2 + 1);
        if tri_count == 0 {
            nodes.push(BvhNode {
                min: Vec3::ZERO,
                max: Vec3::ZERO,
                left_first: 0,
                tri_count: 0,
                right: 0,
            });
            return Self { nodes, tri_indices };
        }

        build_recursive(
            &mut nodes,
            &mut tri_indices,
            &centroids,
            &tri_bounds,
            0,
            tri_count,
        );

        Self { nodes, tri_indices }
    }

    /// Closest hit in local space. Returns `(t, tri_index, u, v)`.
    pub fn raycast(
        &self,
        mesh: &Mesh,
        origin: Vec3,
        dir: Vec3,
    ) -> Option<(f32, u32, f32, f32)> {
        if self.nodes.is_empty() || self.tri_indices.is_empty() {
            return None;
        }
        let inv_dir = Vec3::new(
            if dir.x.abs() > 1e-12 {
                1.0 / dir.x
            } else {
                f32::INFINITY.copysign(dir.x)
            },
            if dir.y.abs() > 1e-12 {
                1.0 / dir.y
            } else {
                f32::INFINITY.copysign(dir.y)
            },
            if dir.z.abs() > 1e-12 {
                1.0 / dir.z
            } else {
                f32::INFINITY.copysign(dir.z)
            },
        );

        let mut best_t = f32::MAX;
        let mut best: Option<(u32, f32, f32)> = None;
        let mut stack = Vec::with_capacity(64);
        stack.push(0u32);

        while let Some(ni) = stack.pop() {
            let ni = ni as usize;
            let node = self.nodes[ni];
            if !ray_aabb_inv(origin, inv_dir, node.min, node.max, best_t) {
                continue;
            }

            if node.tri_count > 0 {
                let first = node.left_first as usize;
                let count = node.tri_count as usize;
                for k in first..first + count {
                    let ti = self.tri_indices[k] as usize;
                    let base = ti * 3;
                    let i0 = mesh.indices[base] as usize;
                    let i1 = mesh.indices[base + 1] as usize;
                    let i2 = mesh.indices[base + 2] as usize;
                    let p0 = Vec3::from_array(mesh.positions[i0]);
                    let p1 = Vec3::from_array(mesh.positions[i1]);
                    let p2 = Vec3::from_array(mesh.positions[i2]);
                    if let Some((t, u, v)) = ray_triangle(origin, dir, p0, p1, p2) {
                        if t < best_t {
                            best_t = t;
                            best = Some((ti as u32, u, v));
                        }
                    }
                }
            } else {
                let left = node.left_first;
                let right = node.right;
                let t_l = aabb_tmin(
                    origin,
                    inv_dir,
                    self.nodes[left as usize].min,
                    self.nodes[left as usize].max,
                );
                let t_r = aabb_tmin(
                    origin,
                    inv_dir,
                    self.nodes[right as usize].min,
                    self.nodes[right as usize].max,
                );
                if t_l <= t_r {
                    if t_r < best_t {
                        stack.push(right);
                    }
                    if t_l < best_t {
                        stack.push(left);
                    }
                } else {
                    if t_l < best_t {
                        stack.push(left);
                    }
                    if t_r < best_t {
                        stack.push(right);
                    }
                }
            }
        }

        best.map(|(ti, u, v)| (best_t, ti, u, v))
    }

    /// Collect triangle indices whose AABB intersects the sphere `(center, radius)` (local space).
    pub fn gather_sphere(&self, center: Vec3, radius: f32, out: &mut Vec<u32>) {
        out.clear();
        if self.nodes.is_empty() || radius <= 0.0 {
            return;
        }
        let r2 = radius * radius;
        let mut stack = Vec::with_capacity(64);
        stack.push(0u32);
        while let Some(ni) = stack.pop() {
            let node = self.nodes[ni as usize];
            if !sphere_aabb(center, r2, node.min, node.max) {
                continue;
            }
            if node.tri_count > 0 {
                let first = node.left_first as usize;
                let count = node.tri_count as usize;
                out.extend_from_slice(&self.tri_indices[first..first + count]);
            } else {
                stack.push(node.left_first);
                stack.push(node.right);
            }
        }
    }
}

#[inline]
fn sphere_aabb(c: Vec3, r2: f32, bmin: Vec3, bmax: Vec3) -> bool {
    let mut d2 = 0.0f32;
    for i in 0..3 {
        let v = c[i];
        if v < bmin[i] {
            let d = bmin[i] - v;
            d2 += d * d;
        } else if v > bmax[i] {
            let d = v - bmax[i];
            d2 += d * d;
        }
    }
    d2 <= r2
}

fn build_recursive(
    nodes: &mut Vec<BvhNode>,
    tri_indices: &mut [u32],
    centroids: &[Vec3],
    tri_bounds: &[(Vec3, Vec3)],
    start: usize,
    count: usize,
) -> u32 {
    let node_idx = nodes.len() as u32;
    nodes.push(BvhNode {
        min: Vec3::ZERO,
        max: Vec3::ZERO,
        left_first: 0,
        tri_count: 0,
        right: 0,
    });

    let (bmin, bmax) = bounds_of_range(tri_indices, tri_bounds, start, count);
    if count <= LEAF_TRIS {
        nodes[node_idx as usize] = BvhNode {
            min: bmin,
            max: bmax,
            left_first: start as u32,
            tri_count: count as u32,
            right: 0,
        };
        return node_idx;
    }

    let extent = bmax - bmin;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };

    // Split by centroid midpoint of this node’s triangle set.
    let mut cmin = f32::MAX;
    let mut cmax = f32::MIN;
    for i in start..start + count {
        let c = centroids[tri_indices[i] as usize][axis];
        cmin = cmin.min(c);
        cmax = cmax.max(c);
    }
    let mid_val = 0.5 * (cmin + cmax);

    let mut mid = start;
    for i in start..start + count {
        if centroids[tri_indices[i] as usize][axis] < mid_val {
            tri_indices.swap(i, mid);
            mid += 1;
        }
    }

    // Degenerate split → force half
    if mid == start || mid == start + count {
        mid = start + count / 2;
    }

    let left_count = mid - start;
    let right_count = count - left_count;
    let left = build_recursive(nodes, tri_indices, centroids, tri_bounds, start, left_count);
    let right = build_recursive(nodes, tri_indices, centroids, tri_bounds, mid, right_count);

    nodes[node_idx as usize] = BvhNode {
        min: bmin,
        max: bmax,
        left_first: left,
        tri_count: 0,
        right,
    };
    node_idx
}

fn bounds_of_range(
    tri_indices: &[u32],
    tri_bounds: &[(Vec3, Vec3)],
    start: usize,
    count: usize,
) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for i in start..start + count {
        let (a, b) = tri_bounds[tri_indices[i] as usize];
        min = min.min(a);
        max = max.max(b);
    }
    (min, max)
}

#[inline]
fn ray_aabb_inv(origin: Vec3, inv_dir: Vec3, bmin: Vec3, bmax: Vec3, t_max: f32) -> bool {
    let mut tmin = 0.0f32;
    let mut tmax = t_max;
    for i in 0..3 {
        let t0 = (bmin[i] - origin[i]) * inv_dir[i];
        let t1 = (bmax[i] - origin[i]) * inv_dir[i];
        let (t0, t1) = if t0 < t1 { (t0, t1) } else { (t1, t0) };
        tmin = tmin.max(t0);
        tmax = tmax.min(t1);
        if tmin > tmax {
            return false;
        }
    }
    tmax >= 0.0
}

#[inline]
fn aabb_tmin(origin: Vec3, inv_dir: Vec3, bmin: Vec3, bmax: Vec3) -> f32 {
    let mut tmin = 0.0f32;
    let mut tmax = f32::MAX;
    for i in 0..3 {
        let t0 = (bmin[i] - origin[i]) * inv_dir[i];
        let t1 = (bmax[i] - origin[i]) * inv_dir[i];
        let (t0, t1) = if t0 < t1 { (t0, t1) } else { (t1, t0) };
        tmin = tmin.max(t0);
        tmax = tmax.min(t1);
        if tmin > tmax {
            return f32::MAX;
        }
    }
    if tmax < 0.0 {
        f32::MAX
    } else {
        tmin.max(0.0)
    }
}

#[inline]
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
