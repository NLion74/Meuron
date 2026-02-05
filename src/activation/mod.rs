use ndarray::Array2;

pub trait Activation: Clone {
    fn activate(&self, x: &Array2<f32>) -> Array2<f32>;
    fn derivative(&self, x: &Array2<f32>) -> Array2<f32>;
}

#[derive(Clone, Copy)]
pub struct ReLU;

impl Activation for ReLU {
    fn activate(&self, x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| v.max(0.0))
    }

    fn derivative(&self, x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
    }
}

#[derive(Clone, Copy)]
pub struct Sigmoid;

impl Activation for Sigmoid {
    fn activate(&self, x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn derivative(&self, x: &Array2<f32>) -> Array2<f32> {
        let sigmoid = self.activate(x);
        &sigmoid * &(1.0 - &sigmoid)
    }
}

#[derive(Clone, Copy)]
pub struct Softmax;

impl Activation for Softmax {
    fn activate(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut result = Array2::zeros(x.dim());
        for (i, row) in x.axis_iter(ndarray::Axis(0)).enumerate() {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp: ndarray::Array1<f32> = row.mapv(|v| (v - max).exp());
            let sum: f32 = exp.sum();
            let normalized = &exp / sum;
            result.row_mut(i).assign(&normalized);
        }
        result
    }

    fn derivative(&self, x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|_| 1.0)
    }
}
