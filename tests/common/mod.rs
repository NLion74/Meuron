use ndarray::{Array, Dimension};

pub fn assert_close<D: Dimension>(name: &str, a: &Array<f32, D>, b: &Array<f32, D>, eps: f32) {
    assert_eq!(a.shape(), b.shape(), "{name}: shape mismatch");

    for (idx, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (*va - *vb).abs();
        assert!(
            diff <= eps,
            "{name}: mismatch at flat index {idx}: left={va}, right={vb}, diff={diff}, eps={eps}"
        );
    }
}

#[allow(dead_code)]
pub fn assert_all_finite<D: Dimension>(name: &str, a: &Array<f32, D>) {
    for (idx, v) in a.iter().enumerate() {
        assert!(
            v.is_finite(),
            "{name}: non-finite value at flat index {idx}: {v}"
        );
    }
}
