//! Per-triangle isometric projection + shelf pack. No shared charts, no seam search.

use glam::{Vec2, Vec3};

use super::{UnwrapAlgo, PAD};

pub struct TriPack;

impl UnwrapAlgo for TriPack {
    fn name(&self) -> &'static str {
        "tri-pack"
    }

    fn unwrap_tris(
        &self,
        positions: &[[f32; 3]],
        indices: &[u32],
        tris: &[u32],
    ) -> Vec<[Vec2; 3]> {
        let mut islands: Vec<Island> = Vec::with_capacity(tris.len());
        for &ti in tris {
            let Some(pts) = tri_pts(positions, indices, ti) else {
                islands.push(Island::empty());
                continue;
            };
            islands.push(project(pts));
        }
        pack(&mut islands);
        islands
            .into_iter()
            .map(|i| [i.uv[0], i.uv[1], i.uv[2]])
            .collect()
    }
}

struct Island {
    uv: [Vec2; 3],
    size: Vec2,
}

impl Island {
    fn empty() -> Self {
        Self {
            uv: [Vec2::splat(0.5); 3],
            size: Vec2::ZERO,
        }
    }
}

fn tri_pts(positions: &[[f32; 3]], indices: &[u32], ti: u32) -> Option<[Vec3; 3]> {
    let base = (ti as usize).checked_mul(3)?;
    let i0 = *indices.get(base)? as usize;
    let i1 = *indices.get(base + 1)? as usize;
    let i2 = *indices.get(base + 2)? as usize;
    Some([
        Vec3::from_array(*positions.get(i0)?),
        Vec3::from_array(*positions.get(i1)?),
        Vec3::from_array(*positions.get(i2)?),
    ])
}

fn project(p: [Vec3; 3]) -> Island {
    let e1 = p[1] - p[0];
    let e2 = p[2] - p[0];
    let n = e1.cross(e2);
    let t = e1.normalize_or_zero();
    let b = n.normalize_or_zero().cross(t);
    if t.length_squared() < 1e-12 || b.length_squared() < 1e-12 {
        return Island::empty();
    }
    let mut uv = [
        Vec2::ZERO,
        Vec2::new(e1.dot(t), e1.dot(b)),
        Vec2::new(e2.dot(t), e2.dot(b)),
    ];
    // Keep corner order (needed for tangents); flip V if the 2D winding is reversed.
    let area = (uv[1].x * uv[2].y - uv[2].x * uv[1].y) * 0.5;
    if area < 0.0 {
        for u in &mut uv {
            u.y = -u.y;
        }
    }
    let min = uv[0].min(uv[1]).min(uv[2]);
    for u in &mut uv {
        *u -= min;
    }
    let max = uv[0].max(uv[1]).max(uv[2]);
    let pad = max.max_element().max(1e-6) * 0.08 + 1e-4;
    Island {
        uv,
        size: max + Vec2::splat(pad),
    }
}

fn pack(islands: &mut [Island]) {
    if islands.is_empty() {
        return;
    }
    let mut order: Vec<usize> = (0..islands.len()).collect();
    order.sort_by(|&a, &b| {
        islands[b]
            .size
            .y
            .partial_cmp(&islands[a].size.y)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let area: f32 = islands.iter().map(|i| i.size.x * i.size.y).sum();
    let max_w = islands
        .iter()
        .map(|i| i.size.x)
        .fold(0.0_f32, f32::max)
        .max(1e-6);
    let bin_w = max_w.max(area.sqrt());

    let mut origin = vec![Vec2::ZERO; islands.len()];
    let mut x = 0.0;
    let mut y = 0.0;
    let mut row_h = 0.0;
    let mut used = Vec2::ZERO;
    for &i in &order {
        let sz = islands[i].size;
        if sz.max_element() < 1e-12 {
            origin[i] = Vec2::splat(0.5);
            continue;
        }
        if x > 0.0 && x + sz.x > bin_w {
            x = 0.0;
            y += row_h;
            row_h = 0.0;
        }
        origin[i] = Vec2::new(x, y);
        x += sz.x;
        row_h = row_h.max(sz.y);
        used.x = used.x.max(x);
        used.y = used.y.max(y + row_h);
    }
    let span = used.x.max(used.y).max(1e-8);
    let inner = (1.0 - 2.0 * PAD).max(0.1);
    let scale = inner / span;
    for (i, island) in islands.iter_mut().enumerate() {
        if island.size.max_element() < 1e-12 {
            island.uv = [Vec2::splat(0.5); 3];
            continue;
        }
        let o = origin[i];
        for u in &mut island.uv {
            *u = (*u + o) * scale + Vec2::splat(PAD);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_tris_fit_unit_square() {
        let pos = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let idx = [0, 1, 2, 1, 3, 2];
        let out = TriPack.unwrap_tris(&pos, &idx, &[0, 1]);
        assert_eq!(out.len(), 2);
        for tri in &out {
            for p in tri {
                assert!(p.x >= 0.0 && p.x <= 1.0);
                assert!(p.y >= 0.0 && p.y <= 1.0);
            }
        }
    }
}
