// op:
//  0=tanh        1=sigmoid       2=relu
//  3=tanh_deriv  4=sigmoid_deriv 5=relu_deriv
//  6=exp         7=ln            8=abs
//  9=neg         10=sqrt
struct Params { size: u32, op: u32, pad0: u32, pad1: u32 }

@group(0) @binding(0) var<storage, read>       input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out:   array<f32>;
@group(0) @binding(2) var<uniform>             p:     Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= p.size { return; }
    let x = input[i];
    switch p.op {
        case 0u: { out[i] = tanh(x); }
        case 1u: { out[i] = 1.0 / (1.0 + exp(-x)); }
        case 2u: { out[i] = max(x, 0.0); }
        case 3u: { let t = tanh(x); out[i] = 1.0 - t * t; }
        case 4u: { let s = 1.0 / (1.0 + exp(-x)); out[i] = s * (1.0 - s); }
        case 5u: { out[i] = select(0.0, 1.0, x > 0.0); }
        case 6u: { out[i] = exp(x); }
        case 7u: { out[i] = log(x); }
        case 8u: { out[i] = abs(x); }
        case 9u: { out[i] = -x; }
        case 10u: { out[i] = sqrt(x); }
        default: { out[i] = 0.0; }
    }
}
