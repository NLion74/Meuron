// op: 0=add  1=sub  2=mul  3=div
@group(0) @binding(0) var<storage, read>       a   : array<f32>;
@group(0) @binding(1) var<storage, read>       b   : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;

struct P { size: u32, op: u32, pad0: u32, pad1: u32 }
@group(0) @binding(3) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= p.size { return; }
    if      p.op == 0u { out[i] = a[i] + b[i]; }
    else if p.op == 1u { out[i] = a[i] - b[i]; }
    else if p.op == 2u { out[i] = a[i] * b[i]; }
    else               { out[i] = a[i] / b[i]; }
}
