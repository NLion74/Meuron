use ndarray::{ArrayBase, Dimension, OwnedRepr};

pub trait Cost<D: Dimension> {
    fn loss(
        &self,
        predicted: &ArrayBase<OwnedRepr<f32>, D>,
        target: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> f32;
    fn gradient(
        &self,
        predicted: &ArrayBase<OwnedRepr<f32>, D>,
        target: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> ArrayBase<OwnedRepr<f32>, D>;
}

#[derive(Clone, Copy)]
pub struct MSE;

impl<D: Dimension> Cost<D> for MSE {
    fn loss(
        &self,
        predicted: &ArrayBase<OwnedRepr<f32>, D>,
        target: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> f32 {
        let diff = predicted - target;
        (&diff * &diff).mean().unwrap()
    }

    fn gradient(
        &self,
        predicted: &ArrayBase<OwnedRepr<f32>, D>,
        target: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> ArrayBase<OwnedRepr<f32>, D> {
        predicted - target
    }
}

#[derive(Clone, Copy)]
pub struct CrossEntropy;

impl<D: Dimension> Cost<D> for CrossEntropy {
    fn loss(
        &self,
        predicted: &ArrayBase<OwnedRepr<f32>, D>,
        target: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> f32 {
        let epsilon = 1e-15;
        let predicted_clipped = predicted.mapv(|v| v.max(epsilon).min(1.0 - epsilon));
        -(target * &predicted_clipped.mapv(|v| v.ln()))
            .mean()
            .unwrap()
    }

    fn gradient(
        &self,
        predicted: &ArrayBase<OwnedRepr<f32>, D>,
        target: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> ArrayBase<OwnedRepr<f32>, D> {
        predicted - target
    }
}

#[derive(Clone, Copy)]
pub struct BinaryCrossEntropy;

impl<D: Dimension> Cost<D> for BinaryCrossEntropy {
    fn loss(
        &self,
        predicted: &ArrayBase<OwnedRepr<f32>, D>,
        target: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> f32 {
        let epsilon = 1e-15;
        let predicted_clipped = predicted.mapv(|v| v.max(epsilon).min(1.0 - epsilon));
        let loss = -(target * &predicted_clipped.mapv(|v| v.ln())
            + (1.0 - target) * &(1.0 - &predicted_clipped).mapv(|v| v.ln()));
        loss.mean().unwrap()
    }

    fn gradient(
        &self,
        predicted: &ArrayBase<OwnedRepr<f32>, D>,
        target: &ArrayBase<OwnedRepr<f32>, D>,
    ) -> ArrayBase<OwnedRepr<f32>, D> {
        let epsilon = 1e-15;
        let predicted_clipped = predicted.mapv(|v| v.max(epsilon).min(1.0 - epsilon));
        (&predicted_clipped - target) / (&predicted_clipped * &(1.0 - &predicted_clipped))
    }
}
