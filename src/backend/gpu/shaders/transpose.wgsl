// 2D matrix transpose: out[col][row] = in[row][col]
@group(0) @binding(0) var<storage, read>       inp : array<f32>;
@group(0) @binding(1) var<storage, read_write> out : array<f32>;

struct Dims { rows: u32, cols: u32, pad0: u32, pad1: u32 }
@group(0) @binding(2) var<uniform> d: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    let c = gid.y;
    if r >= d.rows || c >= d.cols { return; }
    out[c * d.rows + r] = inp[r * d.cols + c];
}
