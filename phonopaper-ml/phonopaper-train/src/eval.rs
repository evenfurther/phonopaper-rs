//! Evaluation of a trained detector on the validation split.

use std::path::Path;

use burn::data::dataset::Dataset;
use burn::prelude::*;

use crate::data::{Item, Split, load_split};
use crate::model::{Detector, decode_output, image_tensor};

/// Aggregate metrics over a split.
#[derive(Debug, Clone, PartialEq)]
pub struct Metrics {
    /// Number of evaluated images.
    pub count: usize,
    /// Fraction of images whose presence was classified correctly (threshold 0.5).
    pub accuracy: f64,
    /// Precision of the "present" class.
    pub precision: f64,
    /// Recall of the "present" class.
    pub recall: f64,
    /// Mean Euclidean corner error, in pixels, over true positives.
    pub mean_corner_error_px: f64,
    /// Fraction of true-positive corners within 3 pixels of the truth.
    pub within_3px: f64,
    /// Fraction of true-positive corners within 6 pixels of the truth.
    pub within_6px: f64,
}

impl std::fmt::Display for Metrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "images evaluated:       {}", self.count)?;
        writeln!(f, "presence accuracy:      {:.2} %", self.accuracy * 100.0)?;
        writeln!(f, "presence precision:     {:.2} %", self.precision * 100.0)?;
        writeln!(f, "presence recall:        {:.2} %", self.recall * 100.0)?;
        writeln!(
            f,
            "mean corner error:      {:.2} px",
            self.mean_corner_error_px
        )?;
        writeln!(
            f,
            "corners within 3 px:    {:.2} %",
            self.within_3px * 100.0
        )?;
        write!(
            f,
            "corners within 6 px:    {:.2} %",
            self.within_6px * 100.0
        )
    }
}

/// Evaluate `model` on one split of `dataset_dir`.
///
/// # Errors
///
/// Returns a message when the dataset cannot be loaded or does not match the
/// model input size.
pub fn evaluate<B: Backend>(
    model: &Detector<B>,
    dataset_dir: &Path,
    split: Split,
    batch_size: usize,
    device: &B::Device,
) -> Result<Metrics, String> {
    let (valid, size) = load_split(dataset_dir, split)?;
    if size != model.input_size() {
        return Err(format!(
            "dataset images are {size} px but the model expects {} px",
            model.input_size()
        ));
    }
    let items: Vec<Item> = valid.iter().collect();
    #[expect(clippy::cast_precision_loss, reason = "image side is a small integer")]
    let px_per_unit = size as f64;

    let (mut correct, mut tp, mut fp, mut fn_) = (0usize, 0usize, 0usize, 0usize);
    let mut corner_err_sum = 0.0;
    let (mut corners_total, mut within3, mut within6) = (0usize, 0usize, 0usize);

    for chunk in items.chunks(batch_size.max(1)) {
        let tensors: Vec<Tensor<B, 3>> = chunk
            .iter()
            .map(|item| image_tensor::<B>(&item.pixels, item.size, device))
            .collect();
        let output = model.forward(Tensor::stack(tensors, 0));
        for (det, item) in decode_output(&output).iter().zip(chunk) {
            let predicted = det.probability >= 0.5;
            let truth = item.present();
            correct += usize::from(predicted == truth);
            match (predicted, truth) {
                (true, true) => {
                    tp += 1;
                    // Either orientation of the sheet is a correct answer;
                    // score against the closer one (mirrors the loss).
                    let truth: [[f32; 2]; 4] =
                        std::array::from_fn(|k| [item.target[1 + 2 * k], item.target[2 + 2 * k]]);
                    let errors = |pred: &[[f32; 2]; 4]| -> [f64; 4] {
                        std::array::from_fn(|k| {
                            let dx = f64::from(pred[k][0] - truth[k][0]) * px_per_unit;
                            let dy = f64::from(pred[k][1] - truth[k][1]) * px_per_unit;
                            (dx * dx + dy * dy).sqrt()
                        })
                    };
                    let direct = errors(&det.corners);
                    let rotated = errors(&det.rotated_180().corners);
                    let best = if direct.iter().sum::<f64>() <= rotated.iter().sum::<f64>() {
                        direct
                    } else {
                        rotated
                    };
                    for err in best {
                        corner_err_sum += err;
                        corners_total += 1;
                        within3 += usize::from(err <= 3.0);
                        within6 += usize::from(err <= 6.0);
                    }
                }
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => {}
            }
        }
    }

    let ratio = |num: usize, den: usize| {
        if den == 0 {
            0.0
        } else {
            #[expect(clippy::cast_precision_loss, reason = "sample counts are small")]
            let r = num as f64 / den as f64;
            r
        }
    };
    Ok(Metrics {
        count: items.len(),
        accuracy: ratio(correct, items.len()),
        precision: ratio(tp, tp + fp),
        recall: ratio(tp, tp + fn_),
        mean_corner_error_px: if corners_total == 0 {
            0.0
        } else {
            #[expect(clippy::cast_precision_loss, reason = "corner counts are small")]
            let m = corner_err_sum / corners_total as f64;
            m
        },
        within_3px: ratio(within3, corners_total),
        within_6px: ratio(within6, corners_total),
    })
}
