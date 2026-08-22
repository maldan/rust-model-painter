struct Brush {
    center: vec2<f32>,
    screen_radius: f32,
    world_radius: f32,
    hardness: f32,
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
    _pad_end: vec2<u32>,
    normal: vec3<f32>,
    normal_mode: f32,
}

@group(0) @binding(0) var<uniform> brush: Brush;
@group(0) @binding(1) var paint_src: texture_2d<f32>;
@group(0) @binding(2) var stroke_src: texture_2d<f32>;
@group(0) @binding(3) var paint_dst: texture_storage_2d<rgba8unorm, write>;

fn unpack_n(c: vec3<f32>) -> vec3<f32> {
    return normalize(c * 2.0 - 1.0);
}

fn pack_n(n: vec3<f32>) -> vec3<f32> {
    return n * 0.5 + 0.5;
}

fn blend_whiteout(n1: vec3<f32>, n2: vec3<f32>) -> vec3<f32> {
    return normalize(vec3<f32>(n1.xy + n2.xy, n1.z * n2.z));
}

@compute @workgroup_size(8, 8)
fn composite(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= brush.tex_w || id.y >= brush.tex_h) { return; }
    let tc = vec2<i32>(i32(id.x), i32(id.y));
    let src = textureLoad(paint_src, tc, 0);

    let R = 1;
    var best_a = 0.0;
    var best_rgb = vec3<f32>(0.0);
    for (var dy = -R; dy <= R; dy++) {
        for (var dx = -R; dx <= R; dx++) {
            let q = tc + vec2(dx, dy);
            if (q.x < 0 || q.y < 0 || q.x >= i32(brush.tex_w) || q.y >= i32(brush.tex_h)) {
                continue;
            }
            let s = textureLoad(stroke_src, q, 0);
            if (s.a > best_a) {
                best_a = s.a;
                best_rgb = s.rgb;
            }
        }
    }

    if (best_a < 0.001) {
        textureStore(paint_dst, tc, src);
        return;
    }

    let flat = vec3<f32>(0.5, 0.5, 1.0);

    if (brush.normal_mode > 0.5) {
        if (brush.erase > 0.5) {
            let out_rgb = mix(src.rgb, flat, best_a);
            textureStore(paint_dst, tc, vec4<f32>(out_rgb, src.a));
            return;
        }
        let base_rgb = select(flat, src.rgb, src.a > 0.001);
        let n_dst = unpack_n(base_rgb);
        let n_src = unpack_n(best_rgb);
        let n_b = blend_whiteout(n_dst, n_src);
        let n_out = normalize(mix(n_dst, n_b, best_a));
        let out_a = src.a + best_a * (1.0 - src.a);
        textureStore(paint_dst, tc, vec4<f32>(pack_n(n_out), out_a));
        return;
    }

    if (brush.erase > 0.5) {
        let out_a = src.a * (1.0 - best_a);
        textureStore(paint_dst, tc, vec4<f32>(src.rgb, out_a));
        return;
    }

    let mask = brush.channel_mask.rgb;
    let sa = best_a;
    let da = src.a;
    let out_a = sa + da * (1.0 - sa);
    var blended = src.rgb;
    if (out_a > 1e-5) {
        blended = (best_rgb * sa + src.rgb * da * (1.0 - sa)) / out_a;
    }
    let out_rgb = mix(src.rgb, blended, mask);
    let out_alpha = mix(src.a, out_a, max(mask.r, max(mask.g, mask.b)));
    textureStore(paint_dst, tc, vec4<f32>(out_rgb, out_alpha));
}
