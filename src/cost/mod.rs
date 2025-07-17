use ndarray::{Array2, ArrayBase, ArrayD, Dimension, IxDyn, OwnedRepr};
use crate::activation::{Activation, Softmax};

pub trait Cost<T: Dimension> {
    fn loss(&self, predictions: &ArrayBase<OwnedRepr<f32>, T>, targets: &ArrayBase<OwnedRepr<f32>, T>) -> f32;
    fn gradient(&self, predictions: &ArrayBase<OwnedRepr<f32>, T>, targets: &ArrayBase<OwnedRepr<f32>, T>) -> ArrayBase<OwnedRepr<f32>, T>;
}

pub struct MSE;
pub struct CrossEntropy;
pub struct BinaryCrossEntropy;

impl<T: Dimension> Cost<T> for MSE {
    fn loss(&self, predictions: &ArrayBase<OwnedRepr<f32>, T>, targets: &ArrayBase<OwnedRepr<f32>, T>) -> f32 {
        let diff = predictions - targets;
        let squared_diff = &diff * &diff;
        squared_diff.sum() / targets.len() as f32 * predictions.shape()[0] as f32
    }

    fn gradient(&self, predictions: &ArrayBase<OwnedRepr<f32>, T>, targets: &ArrayBase<OwnedRepr<f32>, T>) -> ArrayBase<OwnedRepr<f32>, T> {
        2.0 * (predictions - targets) / (predictions.shape()[0] as f32)
    }
}

// impl Cost for BinaryCrossEntropy {
//     fn loss(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
//         let epsilon = 1e-12; // To avoid log(0)
//         let predictions = predictions.mapv(|x| x.clamp(epsilon, 1.0 - epsilon)); // Clipping
//         let bce = -targets * predictions.mapv(|x| x.ln()) - (1.0 - targets) * (1.0 - &predictions).mapv(|x| x.ln());
//         bce.sum() / predictions.shape()[0] as f32 // Average over the batch
//     }
    

//     fn gradient(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Array2<f32> {
//         let epsilon = 1e-12; // To avoid division by zero
//         let predictions = predictions.mapv(|x| x.clamp(epsilon, 1.0 - epsilon)); // Clipping
//         let derivatives = (&predictions - targets) / (&predictions * (1.0 - &predictions));
//         derivatives
//     }
// }

// impl Cost for BinaryCrossEntropy {
//     fn loss(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
//         let epsilon = 1e-7; // A small value to prevent log(0) or log(1)
//         let clipped_preds = predictions.mapv(|p| p.clamp(epsilon, 1.0 - epsilon));

//         // Compute the binary cross-entropy loss
//         let bce_loss = targets * clipped_preds.mapv(f32::ln)
//             + (1.0 - targets) * (1.0 - clipped_preds).mapv(f32::ln);

//         // Take the mean loss over all examples
//         -bce_loss.mean().unwrap()
//     }

//     fn gradient(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Array2<f32> {
//         let epsilon = 1e-7; // To prevent NaN gradients
//         let clipped_preds = predictions.mapv(|p| p.clamp(epsilon, 1.0 - epsilon));

//         // Compute the gradient of the loss with respect to the predictions
//         (&clipped_preds - targets) / (&clipped_preds * (1.0 - &clipped_preds))
//     }
// }

// impl Cost for CrossEntropy {
//     fn loss(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
//         // Use the softmax from the activation module
//         let softmax = Softmax;
//         let softmax_predictions = softmax.activate(predictions); // Apply Softmax to logits
        
//         let mut loss = 0.0;

//         for (pred_row, target_row) in softmax_predictions.outer_iter().zip(targets.outer_iter()) {
//             // Find the index of the maximum element in the target (i.e., the class label)
//             let target_idx = target_row
//                 .iter()
//                 .enumerate()
//                 .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
//                 .map(|(idx, _)| idx)
//                 .unwrap();

//             // Get the predicted probability for the target class
//             let predicted_prob = pred_row[target_idx];

//             // Compute cross-entropy loss
//             loss -= predicted_prob.ln();
//         }
        
//         loss / predictions.shape()[0] as f32
//     }

//     fn gradient(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Array2<f32> {
//         // Use the softmax from the activation module
//         let softmax = Softmax;
//         let softmax_logits = softmax.activate(predictions); // Apply Softmax

//         // Gradient calculation
//         let num_samples = targets.shape()[0];
//         let grad = softmax_logits - targets; // Softmax probabilities - true labels
        
//         grad / num_samples as f32 // Average gradient
//     }
// }
