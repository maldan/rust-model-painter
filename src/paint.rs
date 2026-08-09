use glam::{Mat4, Vec2, Vec3};
use mega_render::Mesh;

use crate::bvh::MeshBvh;
use crate::pick::Hit;

pub const TEX_SIZE: u32 = 1024;

#[derive(Clone, Copy, Debug)]
pub struct DirtyRect {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

impl DirtyRect {
    pub fn full(w: u32, h: u32) -> Self {
        Self {
            x0: 0,
            y0: 0,
            x1: w.saturating_sub(1),
            y1: h.saturating_sub(1),
        }
    }

    pub fn merge(self, other: Self) -> Self {
        Self {
            x0: self.x0.min(other.x0),
            y0: self.y0.min(other.y0),
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
        }
    }
}

#[derive(Clone)]
pub struct Layer {
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    /// Straight RGBA8.
    pub pixels: Vec<u8>,
}

impl Layer {
    pub fn new(name: impl Into<String>, w: u32, h: u32) -> Self {
        let n = (w * h * 4) as usize;
        Self {
            name: name.into(),
            visible: true,
            opacity: 1.0,
            pixels: vec![0; n],
        }
    }

    pub fn clear(&mut self) {
        self.pixels.fill(0);
    }
}

/// Which material map the brush / layer stack targets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PaintMap {
    #[default]
    Albedo,
    Metallic,
    Roughness,
}

impl PaintMap {
    pub const ALL: &'static [PaintMap] =
        &[PaintMap::Albedo, PaintMap::Metallic, PaintMap::Roughness];

    pub fn label(self) -> &'static str {
        match self {
            PaintMap::Albedo => "Albedo",
            PaintMap::Metallic => "Metallic",
            PaintMap::Roughness => "Roughness",
        }
    }

    pub fn index(self) -> usize {
        match self {
            PaintMap::Albedo => 0,
            PaintMap::Metallic => 1,
            PaintMap::Roughness => 2,
        }
    }

    /// RGB write mask for GPU composite (glTF MR: G=roughness, B=metallic).
    pub fn channel_mask(self) -> [f32; 4] {
        match self {
            PaintMap::Albedo => [1.0, 1.0, 1.0, 1.0],
            PaintMap::Roughness => [0.0, 1.0, 0.0, 0.0],
            PaintMap::Metallic => [0.0, 0.0, 1.0, 0.0],
        }
    }

    /// MR texture channel for CPU layer sync (None = full albedo RGBA).
    pub fn mr_channel(self) -> Option<usize> {
        match self {
            PaintMap::Albedo => None,
            PaintMap::Roughness => Some(1),
            PaintMap::Metallic => Some(2),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Brush {
    pub color: [f32; 4],
    /// World-space radius.
    pub radius: f32,
    /// 0 = soft, 1 = hard.
    pub hardness: f32,
    pub opacity: f32,
    pub spacing: f32,
}

impl Default for Brush {
    fn default() -> Self {
        Self {
            color: [0.9, 0.15, 0.12, 1.0],
            radius: 0.08,
            hardness: 0.55,
            opacity: 0.85,
            spacing: 0.35,
        }
    }
}

pub struct PaintDocument {
    pub width: u32,
    pub height: u32,
    /// Underpaint when layers are transparent.
    pub base_rgb: [u8; 3],
    pub layers: Vec<Layer>,
    pub active: usize,
    /// Last stamp world position for stroke spacing.
    pub last_pos: Option<Vec3>,
    dirty_rect: Option<DirtyRect>,
    /// Scratch buffer for nearby triangle indices.
    tri_scratch: Vec<u32>,
}

impl PaintDocument {
    pub fn new(width: u32, height: u32) -> Self {
        Self::with_base(width, height, [220, 220, 225])
    }

    pub fn with_base(width: u32, height: u32, base_rgb: [u8; 3]) -> Self {
        Self {
            width,
            height,
            base_rgb,
            layers: vec![Layer::new("Layer 1", width, height)],
            active: 0,
            last_pos: None,
            dirty_rect: Some(DirtyRect::full(width, height)),
            tri_scratch: Vec::new(),
        }
    }

    pub fn add_layer(&mut self) {
        let name = format!("Layer {}", self.layers.len() + 1);
        self.layers.push(Layer::new(name, self.width, self.height));
        self.active = self.layers.len() - 1;
        self.mark_dirty();
    }

    pub fn remove_active(&mut self) {
        if self.layers.len() <= 1 {
            if let Some(l) = self.layers.get_mut(0) {
                l.clear();
            }
            self.mark_dirty();
            return;
        }
        self.layers.remove(self.active);
        self.active = self.active.min(self.layers.len() - 1);
        self.mark_dirty();
    }

    pub fn clear_active(&mut self) {
        if let Some(l) = self.layers.get_mut(self.active) {
            l.clear();
            self.mark_dirty();
        }
    }

    pub fn move_active(&mut self, delta: isize) {
        let n = self.layers.len() as isize;
        if n <= 1 {
            return;
        }
        let to = (self.active as isize + delta).clamp(0, n - 1) as usize;
        if to == self.active {
            return;
        }
        self.layers.swap(self.active, to);
        self.active = to;
        self.mark_dirty();
    }

    pub fn end_stroke(&mut self) {
        self.last_pos = None;
    }

    /// Stamp brush in **3D surface space**: all nearby triangles get painted by
    /// world-space distance to the hit, so UV seams don't cut the brush in half.
    pub fn stamp_surface(
        &mut self,
        hit: &Hit,
        brush: &Brush,
        view_dir: Vec3,
        mesh: &Mesh,
        world: Mat4,
        bvh: &MeshBvh,
    ) -> bool {
        let facing = (-view_dir).dot(hit.normal);
        if facing < 0.15 {
            return false;
        }
        let facing_fade = ((facing - 0.15) / 0.85).clamp(0.0, 1.0);

        if let Some(prev) = self.last_pos {
            if prev.distance(hit.position) < brush.radius * brush.spacing {
                return false;
            }
        }
        self.last_pos = Some(hit.position);

        // Local-space query radius (handle non-uniform scale conservatively).
        let sx = world.transform_vector3(Vec3::X).length().max(1e-6);
        let sy = world.transform_vector3(Vec3::Y).length().max(1e-6);
        let sz = world.transform_vector3(Vec3::Z).length().max(1e-6);
        let min_scale = sx.min(sy).min(sz);
        let local_r = brush.radius / min_scale;
        let inv_world = world.inverse();
        let local_center = inv_world.transform_point3(hit.position);

        bvh.gather_sphere(local_center, local_r * 1.05, &mut self.tri_scratch);
        if self.tri_scratch.is_empty() {
            return false;
        }
        let tris = std::mem::take(&mut self.tri_scratch);

        let strength = brush.opacity * facing_fade;
        let w = self.width;
        let h = self.height;
        let Some(layer) = self.layers.get_mut(self.active) else {
            self.tri_scratch = tris;
            return false;
        };

        let mut dirty: Option<DirtyRect> = None;
        let r = brush.radius;
        let r2 = r * r;
        let hard = brush.hardness.clamp(0.0, 1.0);
        let inner = r * hard;
        let inv_outer = 1.0 / (r - inner).max(1e-4);
        let cr = (brush.color[0].clamp(0.0, 1.0) * 255.0) as u8;
        let cg = (brush.color[1].clamp(0.0, 1.0) * 255.0) as u8;
        let cb = (brush.color[2].clamp(0.0, 1.0) * 255.0) as u8;
        let center = hit.position;
        let hit_n = hit.normal;

        for &ti in &tris {
            let base = ti as usize * 3;
            let i0 = mesh.indices[base] as usize;
            let i1 = mesh.indices[base + 1] as usize;
            let i2 = mesh.indices[base + 2] as usize;

            let lp0 = Vec3::from_array(mesh.positions[i0]);
            let lp1 = Vec3::from_array(mesh.positions[i1]);
            let lp2 = Vec3::from_array(mesh.positions[i2]);
            let p0 = world.transform_point3(lp0);
            let p1 = world.transform_point3(lp1);
            let p2 = world.transform_point3(lp2);

            // Skip back-facing tris relative to brush/hit normal (reduces bleed-through).
            let tn = (p1 - p0).cross(p2 - p0);
            if tn.dot(hit_n) < 0.0 {
                continue;
            }

            // Quick reject: triangle too far from brush sphere.
            let closest = closest_point_on_tri(center, p0, p1, p2);
            if closest.distance_squared(center) > r2 {
                continue;
            }

            let uv0 = Vec2::from_array(mesh.uvs.get(i0).copied().unwrap_or([0.0, 0.0]));
            let uv1 = Vec2::from_array(mesh.uvs.get(i1).copied().unwrap_or([0.0, 0.0]));
            let uv2 = Vec2::from_array(mesh.uvs.get(i2).copied().unwrap_or([0.0, 0.0]));

            // UV AABB in texels
            let u_min = uv0.x.min(uv1.x).min(uv2.x);
            let u_max = uv0.x.max(uv1.x).max(uv2.x);
            let v_min = uv0.y.min(uv1.y).min(uv2.y);
            let v_max = uv0.y.max(uv1.y).max(uv2.y);
            // Skip degenerate / wrapped UV islands that span most of the atlas
            if (u_max - u_min) > 0.95 || (v_max - v_min) > 0.95 {
                continue;
            }

            let x0 = ((u_min * w as f32).floor() as i32 - 1).max(0) as u32;
            let y0 = ((v_min * h as f32).floor() as i32 - 1).max(0) as u32;
            let x1 = ((u_max * w as f32).ceil() as i32 + 1).min(w as i32 - 1) as u32;
            let y1 = ((v_max * h as f32).ceil() as i32 + 1).min(h as i32 - 1) as u32;
            if x0 > x1 || y0 > y1 {
                continue;
            }

            let mut touched = false;
            for y in y0..=y1 {
                let row = (y * w) as usize;
                let fv = (y as f32 + 0.5) / h as f32;
                for x in x0..=x1 {
                    let fu = (x as f32 + 0.5) / w as f32;
                    let Some((bu, bv, bw)) = barycentric_uv(Vec2::new(fu, fv), uv0, uv1, uv2)
                    else {
                        continue;
                    };
                    // bu,bv,bw are weights for v0,v1,v2
                    let pos = p0 * bu + p1 * bv + p2 * bw;
                    let d2 = pos.distance_squared(center);
                    if d2 > r2 {
                        continue;
                    }
                    let d = d2.sqrt();
                    let cover = if d <= inner {
                        1.0
                    } else {
                        let t = (1.0 - (d - inner) * inv_outer).clamp(0.0, 1.0);
                        t * t * (3.0 - 2.0 * t)
                    };
                    let a = cover * strength;
                    if a < 1e-4 {
                        continue;
                    }

                    let i = (row + x as usize) * 4;
                    let px = &mut layer.pixels[i..i + 4];
                    let da = px[3] as f32 * (1.0 / 255.0);
                    let out_a = a + da * (1.0 - a);
                    if out_a < 1e-4 {
                        continue;
                    }
                    let inv_a = 1.0 / out_a;
                    let keep = da * (1.0 - a);
                    px[0] = ((cr as f32 * a + px[0] as f32 * keep) * inv_a) as u8;
                    px[1] = ((cg as f32 * a + px[1] as f32 * keep) * inv_a) as u8;
                    px[2] = ((cb as f32 * a + px[2] as f32 * keep) * inv_a) as u8;
                    px[3] = (out_a * 255.0).clamp(0.0, 255.0) as u8;
                    touched = true;
                }
            }

            if touched {
                let rect = DirtyRect { x0, y0, x1, y1 };
                dirty = Some(match dirty {
                    Some(prev) => prev.merge(rect),
                    None => rect,
                });
            }
        }

        if let Some(rect) = dirty {
            self.dirty_rect = Some(match self.dirty_rect {
                Some(prev) => prev.merge(rect),
                None => rect,
            });
            self.tri_scratch = tris;
            true
        } else {
            self.tri_scratch = tris;
            false
        }
    }

    pub fn take_dirty_rect(&mut self) -> Option<DirtyRect> {
        self.dirty_rect.take()
    }

    pub fn mark_dirty(&mut self) {
        self.dirty_rect = Some(DirtyRect::full(self.width, self.height));
    }

    /// Recomposite only `rect` into `out` (full RGBA buffer).
    pub fn composite_rect(&self, out: &mut [u8], rect: DirtyRect) {
        let w = self.width;
        let h = self.height;
        let n = (w * h * 4) as usize;
        assert_eq!(out.len(), n);
        let x0 = rect.x0.min(w.saturating_sub(1));
        let y0 = rect.y0.min(h.saturating_sub(1));
        let x1 = rect.x1.min(w.saturating_sub(1));
        let y1 = rect.y1.min(h.saturating_sub(1));

        for y in y0..=y1 {
            for x in x0..=x1 {
                let i = ((y * w + x) * 4) as usize;
                let mut r = self.base_rgb[0];
                let mut g = self.base_rgb[1];
                let mut b = self.base_rgb[2];

                for layer in &self.layers {
                    if !layer.visible || layer.opacity <= 0.001 {
                        continue;
                    }
                    let lo = layer.opacity.clamp(0.0, 1.0);
                    let s = &layer.pixels[i..i + 4];
                    let sa = (s[3] as f32 * (1.0 / 255.0)) * lo;
                    if sa < 1e-4 {
                        continue;
                    }
                    let inv = 1.0 - sa;
                    r = (s[0] as f32 * sa + r as f32 * inv) as u8;
                    g = (s[1] as f32 * sa + g as f32 * inv) as u8;
                    b = (s[2] as f32 * sa + b as f32 * inv) as u8;
                }

                out[i] = r;
                out[i + 1] = g;
                out[i + 2] = b;
                out[i + 3] = 255;
            }
        }
    }

    pub fn composite_into(&self, out: &mut [u8]) {
        self.composite_rect(out, DirtyRect::full(self.width, self.height));
    }
}

/// Barycentric weights (w0,w1,w2) for point in UV triangle, or None if outside.
fn barycentric_uv(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> Option<(f32, f32, f32)> {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-20 {
        return None;
    }
    let inv = 1.0 / denom;
    let v = (d11 * d20 - d01 * d21) * inv;
    let w = (d00 * d21 - d01 * d20) * inv;
    let u = 1.0 - v - w;
    // small epsilon for texel centers on edges
    const EPS: f32 = -1e-4;
    if u >= EPS && v >= EPS && w >= EPS {
        Some((u, v, w))
    } else {
        None
    }
}

fn closest_point_on_tri(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    // Ericson — Real-Time Collision Detection
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}
