struct Stack {
    base: vec4<f32>,
    channel_mask: vec4<f32>,
    opacity: f32,
    mode: u32,
    tex_w: u32,
    tex_h: u32,
    has_mask: u32,
    blend: u32,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<uniform> u: Stack;
@group(0) @binding(1) var dst_src: texture_2d<f32>;
@group(0) @binding(2) var layer_src: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var mask_src: texture_2d<f32>;

fn mask_weight(tc: vec2<i32>) -> f32 {
    if (u.has_mask == 0u) {
        return 1.0;
    }
    let mk = textureLoad(mask_src, tc, 0);
    return dot(mk.rgb, vec3<f32>(0.2126, 0.7152, 0.0722)) * mk.a;
}

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
fn stack(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= u.tex_w || id.y >= u.tex_h) { return; }
    let tc = vec2<i32>(i32(id.x), i32(id.y));
    let src = textureLoad(dst_src, tc, 0);
    let mask = u.channel_mask.rgb;

    if (u.mode == 0u) {
        let out_rgb = mix(src.rgb, u.base.rgb, mask);
        textureStore(dst, tc, vec4<f32>(out_rgb, 1.0));
        return;
    }

    var rgb = vec3<f32>(0.0);
    var a = 0.0;
    if (u.mode == 2u) {
        rgb = u.base.rgb;
        a = u.opacity * mask_weight(tc);
    } else {
        let lay = textureLoad(layer_src, tc, 0);
        rgb = lay.rgb;
        a = lay.a * u.opacity * mask_weight(tc);
    }

    var out_rgb: vec3<f32>;
    if (u.blend == 1u && a > 0.001) {
        let n1 = unpack_n(src.rgb);
        let n2 = unpack_n(rgb);
        out_rgb = pack_n(blend_whiteout(n1, n2));
        out_rgb = mix(src.rgb, out_rgb, a);
    } else {
        out_rgb = mix(src.rgb, rgb, a * mask);
    }
    textureStore(dst, tc, vec4<f32>(out_rgb, 1.0));
}
