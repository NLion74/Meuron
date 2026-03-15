// matmul covers (2,2), (3,2), (3,3)

struct Dims {
    batch:    u32,
    m:        u32,
    k:        u32,
    n:        u32,
    b_shared: u32,
    pad0:     u32,
    pad1:     u32,
    pad2:     u32,
}

@group(0) @binding(0) var<storage, read>       a:    array<f32>;
@group(0) @binding(1) var<storage, read>       b:    array<f32>;
@group(0) @binding(2) var<storage, read_write> c:    array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bi  = gid.z;
    let row = gid.y;
    let col = gid.x;

    if bi >= dims.batch || row >= dims.m || col >= dims.n { return; }

    let a_base = bi * dims.m * dims.k;
    let b_base = select(bi * dims.k * dims.n, 0u, dims.b_shared == 1u);
    let c_base = bi * dims.m * dims.n;

    var acc = 0.0f;
    for (var ki = 0u; ki < dims.k; ki++) {
        acc += a[a_base + row * dims.k + ki]
             * b[b_base + ki  * dims.n + col];
    }
    c[c_base + row * dims.n + col] = acc;
}
