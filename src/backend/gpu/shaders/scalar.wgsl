// op: 0=scale (x*s)  1=scalar_sub (s-x)  2=max(x,s)  3=min(x,s)
@group(0) @binding(0) var<storage, read>       inp : array<f32>;
@group(0) @binding(1) var<storage, read_write> out : array<f32>;

struct P { size: u32, op: u32, scalar: f32, pad: u32 }
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= p.size { return; }
    if      p.op == 0u { out[i] = inp[i] * p.scalar; }
    else if p.op == 1u { out[i] = p.scalar - inp[i]; }
    else if p.op == 2u { out[i] = max(inp[i], p.scalar); }
    else               { out[i] = min(inp[i], p.scalar); }
}
