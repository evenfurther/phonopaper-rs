//! Evaluation of a trained detector on the validation split.

use std::path::Path;

use burn::data::dataset::Dataset;
use burn::prelude::*;

use crate::data::{DetectionBatcher, Item, Split, load_split};
use crate::model::{Detection, Detector, decode_output};

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
    /// Mean corner error relative to the pattern's longest edge, over true
    /// positives (a 6 px error on a 30 px pattern is not the same as on a
    /// 120 px one).
    pub mean_relative_error: f64,
    /// Fraction of true-positive corners within 2 % of the pattern's longest
    /// edge.
    pub within_2pct: f64,
    /// Fraction of true-positive corners within 5 % of the pattern's longest
    /// edge.
    pub within_5pct: f64,
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
        writeln!(
            f,
            "corners within 6 px:    {:.2} %",
            self.within_6px * 100.0
        )?;
        writeln!(
            f,
            "mean relative error:    {:.2} % of pattern size",
            self.mean_relative_error * 100.0
        )?;
        writeln!(
            f,
            "corners within 2 %:     {:.2} %",
            self.within_2pct * 100.0
        )?;
        write!(
            f,
            "corners within 5 %:     {:.2} %",
            self.within_5pct * 100.0
        )
    }
}

/// Per-corner Euclidean errors (pixels) of a detection against an item's
/// truth, plus the pattern's longest edge in pixels (≥ 1).
///
/// Either orientation of the sheet is a correct answer, so the errors are
/// those of the closer of the two equivalent orderings (mirrors the loss).
fn corner_errors(det: &Detection, item: &Item, px_per_unit: f64) -> ([f64; 4], f64) {
    let truth: [[f32; 2]; 4] =
        std::array::from_fn(|k| [item.target[1 + 2 * k], item.target[2 + 2 * k]]);
    let distance = |a: [f32; 2], b: [f32; 2]| {
        let dx = f64::from(a[0] - b[0]) * px_per_unit;
        let dy = f64::from(a[1] - b[1]) * px_per_unit;
        (dx * dx + dy * dy).sqrt()
    };
    let errors =
        |pred: &[[f32; 2]; 4]| -> [f64; 4] { std::array::from_fn(|k| distance(pred[k], truth[k])) };
    let direct = errors(&det.corners);
    let rotated = errors(&det.rotated_180().corners);
    let best = if direct.iter().sum::<f64>() <= rotated.iter().sum::<f64>() {
        direct
    } else {
        rotated
    };
    let pattern_size = (0..4)
        .map(|k| distance(truth[k], truth[(k + 1) % 4]))
        .fold(0.0_f64, f64::max)
        .max(1.0);
    (best, pattern_size)
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
    let mut relative_err_sum = 0.0;
    let (mut corners_total, mut within3, mut within6) = (0usize, 0usize, 0usize);
    let (mut within2pct, mut within5pct) = (0usize, 0usize);

    for chunk in items.chunks(batch_size.max(1)) {
        let output = model.forward(
            DetectionBatcher::assemble(chunk)
                .to_device::<B>(device)
                .images,
        );
        for (det, item) in decode_output(&output).iter().zip(chunk) {
            let predicted = det.probability >= 0.5;
            let truth = item.present();
            correct += usize::from(predicted == truth);
            match (predicted, truth) {
                (true, true) => {
                    tp += 1;
                    let (errors, pattern_size) = corner_errors(det, item, px_per_unit);
                    for err in errors {
                        corner_err_sum += err;
                        corners_total += 1;
                        within3 += usize::from(err <= 3.0);
                        within6 += usize::from(err <= 6.0);
                        let rel = err / pattern_size;
                        relative_err_sum += rel;
                        within2pct += usize::from(rel <= 0.02);
                        within5pct += usize::from(rel <= 0.05);
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
    let mean = |sum: f64, den: usize| {
        if den == 0 {
            0.0
        } else {
            #[expect(clippy::cast_precision_loss, reason = "corner counts are small")]
            let m = sum / den as f64;
            m
        }
    };
    Ok(Metrics {
        count: items.len(),
        accuracy: ratio(correct, items.len()),
        precision: ratio(tp, tp + fp),
        recall: ratio(tp, tp + fn_),
        mean_corner_error_px: mean(corner_err_sum, corners_total),
        within_3px: ratio(within3, corners_total),
        within_6px: ratio(within6, corners_total),
        mean_relative_error: mean(relative_err_sum, corners_total),
        within_2pct: ratio(within2pct, corners_total),
        within_5pct: ratio(within5pct, corners_total),
    })
}
