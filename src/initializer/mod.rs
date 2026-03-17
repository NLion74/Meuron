use crate::backend::Backend;
use ndarray::Dimension;

pub mod constant;
pub mod he_normal;
pub mod xavier_uniform;
pub mod zeros;

pub use constant::Constant;
pub use he_normal::HeNormal;
pub use xavier_uniform::XavierUniform;
pub use zeros::Zeros;

pub trait Initializer<B: Backend> {
    fn init<D: Dimension>(&self, shape: D) -> B::Tensor<D>;
}
