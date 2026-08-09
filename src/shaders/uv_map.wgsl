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
}
struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) world_n: vec3<f32>,
}

struct FsOut {
    @location(0) uv_facing: vec4<f32>,
    @location(1) world_pos: vec4<f32>,
}

@vertex
fn vs_main(v: VsIn) -> VsOut {
    var o: VsOut;
    let world = object.model * vec4<f32>(v.pos, 1.0);
    o.clip = frame.view_proj * world;
    o.uv = v.uv;
    o.world_pos = world.xyz;
    // Rigid / uniform-scale friendly.
    o.world_n = normalize((object.model * vec4<f32>(v.normal, 0.0)).xyz);
    return o;
}

@fragment
fn fs_main(i: VsOut) -> FsOut {
    var o: FsOut;
    let vdir = normalize(frame.eye - i.world_pos);
    let facing = max(dot(normalize(i.world_n), vdir), 0.0);
    o.uv_facing = vec4<f32>(i.uv, facing, 1.0);
    o.world_pos = vec4<f32>(i.world_pos, 1.0);
    return o;
}
