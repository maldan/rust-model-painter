struct Frame {
    view_proj: mat4x4<f32>,
    eye: vec3<f32>,
    _pad: f32,
}
struct Object { model: mat4x4<f32>, }
@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<uniform> object: Object;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
}
struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) world_n: vec3<f32>,
    @location(3) world_t: vec3<f32>,
    @location(4) tan_sign: f32,
}

struct FsOut {
    @location(0) uv_facing: vec4<f32>,
    @location(1) world_pos: vec4<f32>,
    @location(2) tbn: vec4<f32>,
}

fn oct_sign(v: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(select(-1.0, 1.0, v.x >= 0.0), select(-1.0, 1.0, v.y >= 0.0));
}

fn oct_encode(v: vec3<f32>) -> vec2<f32> {
    let n = v / max(abs(v.x) + abs(v.y) + abs(v.z), 1e-8);
    var p = n.xy;
    if (n.z < 0.0) {
        p = (1.0 - abs(p.yx)) * oct_sign(p);
    }
    return p;
}

@vertex
fn vs_main(v: VsIn) -> VsOut {
    var o: VsOut;
    let world = object.model * vec4<f32>(v.pos, 1.0);
    o.clip = frame.view_proj * world;
    o.uv = v.uv;
    o.world_pos = world.xyz;
    o.world_n = normalize((object.model * vec4<f32>(v.normal, 0.0)).xyz);
    let tw = (object.model * vec4<f32>(v.tangent.xyz, 0.0)).xyz;
    o.world_t = normalize(tw);
    o.tan_sign = v.tangent.w;
    return o;
}

@fragment
fn fs_main(i: VsOut) -> FsOut {
    var o: FsOut;
    let n = normalize(i.world_n);
    let t = normalize(i.world_t);
    let vdir = normalize(frame.eye - i.world_pos);
    let facing = max(dot(n, vdir), 0.0);
    o.uv_facing = vec4<f32>(i.uv, facing, 1.0);
    // w = bitangent sign (±1). Validity is uv_facing.a.
    o.world_pos = vec4<f32>(i.world_pos, i.tan_sign);
    o.tbn = vec4<f32>(oct_encode(n), oct_encode(t));
    return o;
}
