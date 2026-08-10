struct Brush {
    center: vec2<f32>,
    screen_radius: f32,
    world_radius: f32,
    hardness: f32,
    _pad0: f32,
    _pad_to_color: vec2<f32>,
    color: vec4<f32>,
    channel_mask: vec4<f32>,
    opacity: f32,
    erase: f32,
    map_w: u32,
    map_h: u32,
    tex_w: u32,
    tex_h: u32,
    _pad_end: vec2<u32>,
}

@group(0) @binding(0) var<uniform> brush: Brush;
@group(0) @binding(1) var uv_map: texture_2d<f32>;
@group(0) @binding(2) var pos_map: texture_2d<f32>;
@group(0) @binding(3) var stroke_dst: texture_storage_2d<rgba8unorm, write>;

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
    // Screen window only culls work; falloff is world-space.
    if (distance(p, brush.center) > brush.screen_radius * 1.25) { return; }

    let cx = i32(floor(brush.center.x));
    let cy = i32(floor(brush.center.y));
    if (cx < 0 || cy < 0 || cx >= i32(brush.map_w) || cy >= i32(brush.map_h)) {
        return;
    }

    let center_uv = textureLoad(uv_map, vec2<i32>(cx, cy), 0);
    let center_pos = textureLoad(pos_map, vec2<i32>(cx, cy), 0);
    // a=valid, b=facing
    if (center_uv.a < 0.5 || center_pos.a < 0.5 || center_uv.b < 0.15) { return; }

    let sample_uv = textureLoad(uv_map, vec2<i32>(i32(id.x), i32(id.y)), 0);
    let sample_pos = textureLoad(pos_map, vec2<i32>(i32(id.x), i32(id.y)), 0);
    if (sample_uv.a < 0.5 || sample_pos.a < 0.5) { return; }

    let facing = sample_uv.b;
    if (facing < 0.15) { return; }
    let facing_fade = clamp((facing - 0.15) / 0.85, 0.0, 1.0);

    // Reject depth jumps (front leaking onto far/back geometry in the same brush window).
    let world_d = distance(sample_pos.xyz, center_pos.xyz);
    if (world_d > brush.world_radius) { return; }

    let cover = brush_cover(world_d, brush.world_radius, brush.hardness) * facing_fade;
    if (cover < 0.001) { return; }

    let uv = clamp(sample_uv.xy, vec2<f32>(0.0), vec2<f32>(0.99999));
    let tc = vec2<i32>(
        i32(floor(uv.x * f32(brush.tex_w))),
        i32(floor(uv.y * f32(brush.tex_h))),
    );
    let a = cover * brush.opacity;
    textureStore(stroke_dst, tc, vec4<f32>(brush.color.rgb, a));
}
