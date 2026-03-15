// Add two tensors together, matches dims of a to b along the last dimension of b
@group(0) @binding(0) var<storage, read>       a   : array<f32>;
@group(0) @binding(1) var<storage, read>       b   : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;

struct Dims { total: u32, last_dim: u32, pad0: u32, pad1: u32 }
@group(0) @binding(3) var<uniform> d: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= d.total { return; }
    out[i] = a[i] + b[i % d.last_dim];
}
