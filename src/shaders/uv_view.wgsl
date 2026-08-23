struct View {
    pan: vec2<f32>,
    half_extent: vec2<f32>,
    channel: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> view: View;
@group(1) @binding(0) var tile_tex: texture_2d<f32>;
@group(1) @binding(1) var tile_samp: sampler;

struct TileIn {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct TileOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

fn to_clip(uv: vec2<f32>) -> vec4<f32> {
    let p = (uv - view.pan) / view.half_extent;
    return vec4<f32>(p.x, p.y, 0.0, 1.0);
}

@vertex
fn vs_tile(in: TileIn) -> TileOut {
    var out: TileOut;
    out.clip = to_clip(in.pos);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_tile(in: TileOut) -> @location(0) vec4<f32> {
    var c = textureSample(tile_tex, tile_samp, in.uv).rgb;
    if view.channel == 1u {
        c = vec3<f32>(c.g);
    } else if view.channel == 2u {
        c = vec3<f32>(c.b);
    }
    return vec4<f32>(c, 1.0);
}

struct LineIn {
    @location(0) pos: vec2<f32>,
    @location(1) color: vec4<f32>,
}

struct LineOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_line(in: LineIn) -> LineOut {
    var out: LineOut;
    out.clip = to_clip(in.pos);
    out.color = in.color;
    return out;
}

@fragment
fn fs_line(in: LineOut) -> @location(0) vec4<f32> {
    return in.color;
}
