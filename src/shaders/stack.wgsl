struct Stack {
    base: vec4<f32>,
    channel_mask: vec4<f32>,
    opacity: f32,
    mode: u32,
    tex_w: u32,
    tex_h: u32,
}

@group(0) @binding(0) var<uniform> u: Stack;
@group(0) @binding(1) var dst_src: texture_2d<f32>;
@group(0) @binding(2) var layer_src: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba8unorm, write>;

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

    let lay = textureLoad(layer_src, tc, 0);
    let a = lay.a * u.opacity;
    let out_rgb = mix(src.rgb, lay.rgb, a * mask);
    textureStore(dst, tc, vec4<f32>(out_rgb, 1.0));
}
