//! Runtime-generated brush coverage maps (Substance-style alphas).

pub const ALPHA_SIZE: u32 = 256;
/// UI texture slots: 0 = viewport, 2 = color-picker SV.
pub const ALPHA_TEX_BASE: u32 = 16;

pub struct BrushAlpha {
    pub name: &'static str,
    pub size: u32,
    /// Row-major coverage 0..=255, `size * size`.
    pub coverage: Vec<u8>,
    /// True: stamp the texels as coverage (noise). False: `1 - a` is a distance for hardness.
    pub coverage_stamp: bool,
}

impl BrushAlpha {
    pub fn rgba_preview(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.coverage.len() * 4);
        for &c in &self.coverage {
            out.extend_from_slice(&[c, c, c, 255]);
        }
        out
    }
}

pub fn generate_presets() -> Vec<BrushAlpha> {
    vec![
        bake_dist("Circle", circle),
        bake_dist("Hard", hard_circle),
        bake_dist("Square", square),
        bake_dist("Diamond", diamond),
        bake_dist("Cross", cross),
        bake_solid("Ring", ring),
        bake_solid("Dots", dots),
        bake_map("Noise", noise_coverage, true),
    ]
}

fn pack(name: &'static str, coverage: Vec<u8>, coverage_stamp: bool) -> BrushAlpha {
    BrushAlpha {
        name,
        size: ALPHA_SIZE,
        coverage,
        coverage_stamp,
    }
}

fn raster(sample: impl Fn(f32, f32) -> f32) -> Vec<u8> {
    let n = ALPHA_SIZE;
    let mut coverage = vec![0u8; (n * n) as usize];
    let inv = 2.0 / n as f32;
    for y in 0..n {
        for x in 0..n {
            let px = (x as f32 + 0.5) * inv - 1.0;
            let py = (y as f32 + 0.5) * inv - 1.0;
            let a = sample(px, py).clamp(0.0, 1.0);
            coverage[(y * n + x) as usize] = (a * 255.0 + 0.5) as u8;
        }
    }
    coverage
}

/// Distance-style: `a = saturate(-sd)` so circle is `1 - r` (old brush hardness).
fn bake_dist(name: &'static str, sdf: fn(f32, f32) -> f32) -> BrushAlpha {
    pack(name, raster(|x, y| (-sdf(x, y)).clamp(0.0, 1.0)), false)
}

/// Solid shape: 1 inside, 0 outside, small AA band. Ring/dots need this or they stay ~0.2 grey.
fn bake_solid(name: &'static str, sdf: fn(f32, f32) -> f32) -> BrushAlpha {
    pack(name, raster(|x, y| sdf_cover(sdf(x, y), 0.07)), false)
}

fn bake_map(name: &'static str, sample: fn(f32, f32) -> f32, coverage_stamp: bool) -> BrushAlpha {
    pack(name, raster(sample), coverage_stamp)
}

fn sdf_cover(sd: f32, edge: f32) -> f32 {
    let t = ((edge - sd) / (2.0 * edge).max(1e-4)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn circle(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt() - 1.0
}

fn hard_circle(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt() / 0.92 - 1.0
}

fn square(x: f32, y: f32) -> f32 {
    x.abs().max(y.abs()) - 1.0
}

fn diamond(x: f32, y: f32) -> f32 {
    x.abs() + y.abs() - 1.0
}

fn cross(x: f32, y: f32) -> f32 {
    let arm = 0.28;
    let a = x.abs().max(y.abs() / arm) - 1.0;
    let b = (x.abs() / arm).max(y.abs()) - 1.0;
    a.min(b)
}

fn ring(x: f32, y: f32) -> f32 {
    ((x * x + y * y).sqrt() - 0.58).abs() - 0.20
}

fn dots(x: f32, y: f32) -> f32 {
    let r = 0.20;
    let offs = [
        [0.0, 0.0],
        [0.55, 0.0],
        [-0.48, 0.32],
        [-0.22, -0.52],
        [0.38, 0.48],
        [0.28, -0.50],
    ];
    let mut d: f32 = 1.0e9;
    for o in offs {
        let dx = x - o[0];
        let dy = y - o[1];
        d = d.min((dx * dx + dy * dy).sqrt() - r);
    }
    d
}

fn noise_coverage(x: f32, y: f32) -> f32 {
    let r = (x * x + y * y).sqrt();
    if r >= 1.0 {
        return 0.0;
    }
    let t = (1.0 - r).clamp(0.0, 1.0);
    let window = t * t * (3.0 - 2.0 * t);
    fbm(x, y) * window
}

fn fbm(x: f32, y: f32) -> f32 {
    let mut v = 0.0;
    let mut amp = 0.5;
    let mut freq = 5.0;
    let mut sum = 0.0;
    for _ in 0..5 {
        v += amp * value_noise(x * freq + 17.3, y * freq - 9.1);
        sum += amp;
        amp *= 0.5;
        freq *= 2.05;
    }
    (v / sum).clamp(0.0, 1.0)
}

fn value_noise(x: f32, y: f32) -> f32 {
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let u = fx * fx * (3.0 - 2.0 * fx);
    let v = fy * fy * (3.0 - 2.0 * fy);
    let a = hash_i(x0, y0);
    let b = hash_i(x0 + 1.0, y0);
    let c = hash_i(x0, y0 + 1.0);
    let d = hash_i(x0 + 1.0, y0 + 1.0);
    a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v
}

fn hash_i(x: f32, y: f32) -> f32 {
    let mut n = (x as i32).wrapping_mul(374761393) ^ (y as i32).wrapping_mul(668265263);
    n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    (n as u32 as f32) * (1.0 / 4294967295.0)
}

/// UI slots for tangent-space normal stamps.
pub const NRM_TEX_BASE: u32 = 48;

pub struct NormalStamp {
    pub name: &'static str,
    pub size: u32,
    pub rgba: Vec<u8>,
}

pub fn generate_normal_stamps() -> Vec<NormalStamp> {
    vec![
        bake_height("Dome", dome_h),
        bake_height("Groove", groove_h),
        bake_height("Bevel", bevel_h),
        bake_height("Rivet", rivet_h),
        bake_height("Ridge", ridge_h),
        bake_height("Noise", noise_h),
    ]
}

fn bake_height(name: &'static str, height: fn(f32, f32) -> f32) -> NormalStamp {
    let n = ALPHA_SIZE as usize;
    let mut h = vec![0.0f32; n * n];
    let inv = 2.0 / n as f32;
    for y in 0..n {
        for x in 0..n {
            let px = (x as f32 + 0.5) * inv - 1.0;
            let py = (y as f32 + 0.5) * inv - 1.0;
            h[y * n + x] = height(px, py);
        }
    }
    let mut rgba = vec![0u8; n * n * 4];
    let strength = 5.5;
    for y in 0..n {
        for x in 0..n {
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(n - 1);
            let ym = y.saturating_sub(1);
            let yp = (y + 1).min(n - 1);
            let dx = (h[y * n + xp] - h[y * n + xm]) * strength;
            let dy = (h[yp * n + x] - h[ym * n + x]) * strength;
            let len = (dx * dx + dy * dy + 1.0).sqrt().max(1e-6);
            let nx = 0.5 * (-dx / len) + 0.5;
            let ny = 0.5 * (-dy / len) + 0.5;
            let nz = 0.5 * (1.0 / len) + 0.5;
            let px = (x as f32 + 0.5) * inv - 1.0;
            let py = (y as f32 + 0.5) * inv - 1.0;
            let a = (1.0 - (px * px + py * py).sqrt()).clamp(0.0, 1.0);
            let i = (y * n + x) * 4;
            rgba[i] = (nx.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            rgba[i + 1] = (ny.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            rgba[i + 2] = (nz.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            rgba[i + 3] = (a * 255.0 + 0.5) as u8;
        }
    }
    NormalStamp {
        name,
        size: ALPHA_SIZE,
        rgba,
    }
}

fn dome_h(x: f32, y: f32) -> f32 {
    (1.0 - x * x - y * y).max(0.0)
}

fn groove_h(x: f32, y: f32) -> f32 {
    let r = (x * x + y * y).sqrt();
    let d = (r - 0.55).abs();
    -(1.0 - (d / 0.18).clamp(0.0, 1.0)).powi(2)
}

fn bevel_h(x: f32, y: f32) -> f32 {
    let r = (x * x + y * y).sqrt();
    let d = (r - 0.55).abs();
    (1.0 - (d / 0.16).clamp(0.0, 1.0)).powi(2)
}

fn rivet_h(x: f32, y: f32) -> f32 {
    let r2 = x * x + y * y;
    if r2 > 0.12 {
        return 0.0;
    }
    (1.0 - r2 / 0.12).max(0.0).powi(2)
}

fn ridge_h(x: f32, y: f32) -> f32 {
    let across = (1.0 - x.abs() * 1.15).max(0.0);
    let along = (1.0 - y.abs()).max(0.0);
    across * across * along
}

fn noise_h(x: f32, y: f32) -> f32 {
    let r = (x * x + y * y).sqrt();
    if r >= 1.0 {
        return 0.0;
    }
    let win = (1.0 - r).clamp(0.0, 1.0);
    (fbm(x, y) * 2.0 - 1.0) * win * win
}
