struct Brush {
    center: vec2<f32>,
    screen_radius: f32,
    world_radius: f32,
    hardness: f32,
    /// 0 = `1-a` is distance (shapes), 1 = stamp `a` as coverage (noise).
    coverage_stamp: f32,
    _pad_to_color: vec2<f32>,
    color: vec4<f32>,
    channel_mask: vec4<f32>,
    opacity: f32,
    erase: f32,
    map_w: u32,
    map_h: u32,
    tex_w: u32,
    tex_h: u32,
    tile: vec2<f32>,
    normal: vec3<f32>,
    normal_mode: f32,
}

@group(0) @binding(0) var<uniform> brush: Brush;
@group(0) @binding(1) var uv_map: texture_2d<f32>;
@group(0) @binding(2) var pos_map: texture_2d<f32>;
@group(0) @binding(3) var stroke_dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var alpha_tex: texture_2d<f32>;
@group(0) @binding(5) var alpha_samp: sampler;
@group(0) @binding(6) var tbn_map: texture_2d<f32>;

fn oct_sign(v: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(select(-1.0, 1.0, v.x >= 0.0), select(-1.0, 1.0, v.y >= 0.0));
}

fn oct_decode(f: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    if (n.z < 0.0) {
        let xy = (1.0 - abs(n.yx)) * oct_sign(n.xy);
        n = vec3<f32>(xy, n.z);
    }
    return normalize(n);
}

fn unpack_n(c: vec3<f32>) -> vec3<f32> {
    return normalize(c * 2.0 - 1.0);
}

fn pack_n(n: vec3<f32>) -> vec3<f32> {
    return n * 0.5 + 0.5;
}

fn brush_cover(d: f32, radius: f32, hardness: f32) -> f32 {
    if (d >= radius) { return 0.0; }
    let inner = radius * hardness;
    if (d <= inner) { return 1.0; }
    let t = (1.0 - (d - inner) / max(radius - inner, 1e-4));
    let s = clamp(t, 0.0, 1.0);
    return s * s * (3.0 - 2.0 * s);
}

@compute @workgroup_size(8, 8)
fn splat(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= brush.map_w || id.y >= brush.map_h) { return; }

    let p = vec2<f32>(f32(id.x) + 0.5, f32(id.y) + 0.5);
    // Screen window only culls work; falloff is world-space. Extra pad for square corners.
    if (distance(p, brush.center) > brush.screen_radius * 1.5) { return; }

    let cx = i32(floor(brush.center.x));
    let cy = i32(floor(brush.center.y));
    if (cx < 0 || cy < 0 || cx >= i32(brush.map_w) || cy >= i32(brush.map_h)) {
        return;
    }

    let center_uv = textureLoad(uv_map, vec2<i32>(cx, cy), 0);
    let center_pos = textureLoad(pos_map, vec2<i32>(cx, cy), 0);
    // a=valid (uv), pos.a = bitangent sign
    if (center_uv.a < 0.5 || abs(center_pos.a) < 0.5 || center_uv.b < 0.15) { return; }

    let sample_uv = textureLoad(uv_map, vec2<i32>(i32(id.x), i32(id.y)), 0);
    let sample_pos = textureLoad(pos_map, vec2<i32>(i32(id.x), i32(id.y)), 0);
    if (sample_uv.a < 0.5 || abs(sample_pos.a) < 0.5) { return; }

    // Only this tile's UVs — other tiles get their own stamp pass.
    let center_tile = floor(center_uv.xy);
    let sample_tile = floor(sample_uv.xy);
    if (abs(sample_tile.x - brush.tile.x) > 0.1 || abs(sample_tile.y - brush.tile.y) > 0.1) {
        return;
    }

    let facing = sample_uv.b;
    if (facing < 0.15) { return; }
    let facing_fade = clamp((facing - 0.15) / 0.85, 0.0, 1.0);

    let offset = sample_pos.xyz - center_pos.xyz;
    let dist = length(offset);
    // Same UDIM as cursor: allow square corners. Other UDIM (seam): strict radius
    // so painting one island does not splash the neighbor across the cut.
    let cross_udim = abs(sample_tile.x - center_tile.x) > 0.1
        || abs(sample_tile.y - center_tile.y) > 0.1;
    let max_dist = select(brush.world_radius * 1.5, brush.world_radius, cross_udim);
    if (dist > max_dist) { return; }

    var n = brush.normal;
    if (dot(n, n) < 1e-8) {
        n = vec3<f32>(0.0, 1.0, 0.0);
    }
    n = normalize(n);
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);
    let u = dot(offset, tangent) / max(brush.world_radius, 1e-4);
    let v = dot(offset, bitangent) / max(brush.world_radius, 1e-4);
    if (max(abs(u), abs(v)) > 1.0) { return; }

    let alpha_uv = vec2<f32>(u * 0.5 + 0.5, 0.5 - v * 0.5);
    let texel = textureSampleLevel(alpha_tex, alpha_samp, alpha_uv, 0.0);
    var shape = texel.r;
    var rgb = brush.color.rgb;
    if (brush.normal_mode > 0.5) {
        let ns = unpack_n(texel.rgb);
        let n_world = normalize(tangent * ns.x + bitangent * ns.y + n * ns.z);
        let packed = textureLoad(tbn_map, vec2<i32>(i32(id.x), i32(id.y)), 0);
        let n0 = oct_decode(packed.xy);
        var t = oct_decode(packed.zw);
        t = normalize(t);
        let b = cross(n0, t) * sign(sample_pos.a);
        let n_ts = vec3<f32>(dot(n_world, t), dot(n_world, b), dot(n_world, n0));
        rgb = pack_n(normalize(n_ts));
        shape = texel.a;
    }
    var cover: f32;
    if (brush.coverage_stamp > 0.5) {
        let lo = 0.2 * brush.hardness;
        let hi = 1.0 - 0.2 * brush.hardness;
        cover = smoothstep(lo, hi, shape);
    } else {
        cover = brush_cover(1.0 - shape, 1.0, brush.hardness);
    }
    cover *= facing_fade;
    if (cover < 0.001) { return; }

    let uv = min(fract(sample_uv.xy), vec2<f32>(0.99999));
    let tc = vec2<i32>(
        i32(floor(uv.x * f32(brush.tex_w))),
        i32(floor(uv.y * f32(brush.tex_h))),
    );
    let out_a = cover * brush.opacity;
    textureStore(stroke_dst, tc, vec4<f32>(rgb, out_a));
}
