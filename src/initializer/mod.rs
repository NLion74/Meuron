use crate::backend::Backend;
use ndarray::Dimension;

pub mod zeros;
pub mod xavier_uniform;
pub mod he_normal;
pub mod constant;

pub use zeros::Zeros;
pub use xavier_uniform::XavierUniform;
pub use he_normal::HeNormal;
pub use constant::Constant;

pub trait Initializer<B: Backend> {
    fn init<D: Dimension>(&self, shape: D) -> B::Tensor<D>;
}
