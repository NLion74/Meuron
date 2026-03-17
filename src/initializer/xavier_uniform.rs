use crate::backend::Backend;
use crate::initializer::Initializer;
use ndarray::Dimension;

#[derive(Clone, Copy, Debug, Default)]
pub struct XavierUniform;

impl<B: Backend> Initializer<B> for XavierUniform {
    fn init<D: Dimension>(&self, shape: D) -> B::Tensor<D> {
        let dims = shape.slice();
        assert!(
            dims.len() == 2,
            "XavierUniform expects a 2D shape (fan_in, fan_out)"
        );

        let fan_in = dims[0] as f32;
        let fan_out = dims[1] as f32;
        let limit = (6.0 / (fan_in + fan_out)).sqrt();

        B::random_uniform(shape, -limit, limit)
    }
}
