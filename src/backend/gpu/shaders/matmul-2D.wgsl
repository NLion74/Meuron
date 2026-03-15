// matmul by 2D
@group(0) @binding(0) var<storage, read>       a   : array<f32>;
@group(0) @binding(1) var<storage, read>       b   : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;

struct Dims { m: u32, k: u32, n: u32, pad: u32 }
@group(0) @binding(3) var<uniform> d: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if row >= d.m || col >= d.n { return; }
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < d.k; k = k + 1u) {
        sum = sum + a[row * d.k + k] * b[k * d.n + col];
    }
    out[row * d.n + col] = sum;
}
