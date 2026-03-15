// N-D axis-swap transpose — supports up to 6 dimensions.
struct Params {
    ndim:       u32,
    total:      u32,
    axis1:      u32,
    axis2:      u32,
    out_shape:  array<u32, 6>,
    out_stride: array<u32, 6>,
    in_stride:  array<u32, 6>,
    pad:        array<u32, 2>,
}

@group(0) @binding(0) var<storage, read>       src:    array<f32>;
@group(0) @binding(1) var<storage, read_write> dst:    array<f32>;
@group(0) @binding(2) var<storage, read>       params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat_out = gid.x;
    if flat_out >= params.total { return; }

    var remaining = flat_out;
    var out_idx: array<u32, 6>;
    for (var i = 0u; i < params.ndim; i++) {
        out_idx[i] = remaining / params.out_stride[i];
        remaining   = remaining % params.out_stride[i];
    }

    var in_idx: array<u32, 6>;
    for (var i = 0u; i < params.ndim; i++) {
        in_idx[i] = out_idx[i];
    }
    let tmp = in_idx[params.axis1];
    in_idx[params.axis1] = in_idx[params.axis2];
    in_idx[params.axis2] = tmp;

    var flat_in = 0u;
    for (var i = 0u; i < params.ndim; i++) {
        flat_in += in_idx[i] * params.in_stride[i];
    }

    dst[flat_out] = src[flat_in];
}
