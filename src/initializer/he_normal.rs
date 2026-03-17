use crate::backend::Backend;
use crate::initializer::Initializer;
use ndarray::Dimension;

#[derive(Clone, Copy, Debug, Default)]
pub struct HeNormal;

impl<B: Backend> Initializer<B> for HeNormal {
    fn init<D: Dimension>(&self, shape: D) -> B::Tensor<D> {
        let dims = shape.slice();
        assert!(dims.len() == 2, "HeNormal expects a 2D shape");

        let fan_in = dims[0] as f32;
        let std = (2.0 / fan_in).sqrt();

        B::random_normal(shape, 0.0, std)
    }
}
