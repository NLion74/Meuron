// Softmax forward.
@group(0) @binding(0) var<storage, read>       inp : array<f32>;
@group(0) @binding(1) var<storage, read_write> out : array<f32>;

struct Dims { batch: u32, last_dim: u32, pad0: u32, pad1: u32 }
@group(0) @binding(2) var<uniform> d: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if b >= d.batch { return; }
    let off = b * d.last_dim;

    var mx = inp[off];
    for (var i = 1u; i < d.last_dim; i++) {
        if inp[off + i] > mx { mx = inp[off + i]; }
    }
    var s: f32 = 0.0;
    for (var i = 0u; i < d.last_dim; i++) {
        s = s + exp(inp[off + i] - mx);
    }
    for (var i = 0u; i < d.last_dim; i++) {
        out[off + i] = exp(inp[off + i] - mx) / s;
    }
}
