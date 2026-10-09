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
    /// Median Euclidean corner error, in pixels, over true positives.
    pub median_corner_error_px: f64,
    /// 90th-percentile Euclidean corner error, in pixels, over true positives.
    pub p90_corner_error_px: f64,
    /// 95th-percentile Euclidean corner error, in pixels, over true positives.
    pub p95_corner_error_px: f64,
    /// Worst Euclidean corner error, in pixels, over true positives.
    pub worst_corner_error_px: f64,
    /// Mean exact intersection-over-union of the predicted and true convex
    /// quadrilaterals over true positives. Degenerate or non-convex predictions
    /// have zero `IoU`.
    pub mean_quad_iou: f64,
    /// 10th-percentile quadrilateral `IoU` over true positives.
    pub p10_quad_iou: f64,
    /// Worst (minimum) quadrilateral `IoU` over true positives.
    pub worst_quad_iou: f64,
    /// Mean signed width scale error (`predicted / truth - 1`) over true
    /// positives. Width is the mean length of the two marker-band edges.
    pub mean_width_scale_bias: f64,
    /// Mean absolute width scale error over true positives.
    pub mean_abs_width_scale_error: f64,
    /// Mean signed height scale error (`predicted / truth - 1`) over true
    /// positives. Height is the mean length of the two marker-band-to-marker-band
    /// edges.
    pub mean_height_scale_bias: f64,
    /// Mean absolute height scale error over true positives.
    pub mean_abs_height_scale_error: f64,
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
        writeln!(
            f,
            "corners within 5 %:     {:.2} %",
            self.within_5pct * 100.0
        )?;
        writeln!(
            f,
            "corner error p50/90/95/worst: {:.2} / {:.2} / {:.2} / {:.2} px",
            self.median_corner_error_px,
            self.p90_corner_error_px,
            self.p95_corner_error_px,
            self.worst_corner_error_px
        )?;
        writeln!(
            f,
            "quadrilateral IoU mean/p10/worst: {:.2} / {:.2} / {:.2} %",
            self.mean_quad_iou * 100.0,
            self.p10_quad_iou * 100.0,
            self.worst_quad_iou * 100.0
        )?;
        writeln!(
            f,
            "width scale bias / abs error:  {:+.2} / {:.2} %",
            self.mean_width_scale_bias * 100.0,
            self.mean_abs_width_scale_error * 100.0
        )?;
        write!(
            f,
            "height scale bias / abs error: {:+.2} / {:.2} %",
            self.mean_height_scale_bias * 100.0,
            self.mean_abs_height_scale_error * 100.0
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

type Point = [f64; 2];

fn cross(a: Point, b: Point, c: Point) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn signed_area(polygon: &[Point]) -> f64 {
    polygon
        .iter()
        .zip(polygon.iter().cycle().skip(1))
        .map(|(a, b)| a[0] * b[1] - a[1] * b[0])
        .sum::<f64>()
        / 2.0
}

fn is_convex_quad(quad: &[Point; 4]) -> bool {
    let area = signed_area(quad);
    if area.abs() <= f64::EPSILON {
        return false;
    }
    let orientation = area.signum();
    (0..4).all(|i| cross(quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]) * orientation > 0.0)
}

fn line_intersection(segment_start: Point, segment_end: Point, a: Point, b: Point) -> Point {
    let segment_cross = cross(a, b, segment_start);
    let end_cross = cross(a, b, segment_end);
    let t = segment_cross / (segment_cross - end_cross);
    [
        segment_start[0] + t * (segment_end[0] - segment_start[0]),
        segment_start[1] + t * (segment_end[1] - segment_start[1]),
    ]
}

fn convex_intersection(subject: &[Point], clip: &[Point; 4]) -> Vec<Point> {
    let orientation = signed_area(clip).signum();
    let mut output = subject.to_vec();
    for i in 0..4 {
        let (a, b) = (clip[i], clip[(i + 1) % 4]);
        let input = std::mem::take(&mut output);
        let Some(mut previous) = input.last().copied() else {
            break;
        };
        let mut previous_inside = cross(a, b, previous) * orientation >= 0.0;
        for current in input {
            let current_inside = cross(a, b, current) * orientation >= 0.0;
            if current_inside != previous_inside {
                output.push(line_intersection(previous, current, a, b));
            }
            if current_inside {
                output.push(current);
            }
            previous = current;
            previous_inside = current_inside;
        }
    }
    output
}

fn quadrilateral_iou(predicted: &[[f32; 2]; 4], truth: &[[f32; 2]; 4]) -> f64 {
    let predicted = predicted.map(|[x, y]| [f64::from(x), f64::from(y)]);
    let truth = truth.map(|[x, y]| [f64::from(x), f64::from(y)]);
    if !is_convex_quad(&predicted) || !is_convex_quad(&truth) {
        return 0.0;
    }
    let predicted_area = signed_area(&predicted).abs();
    let truth_area = signed_area(&truth).abs();
    let intersection_area = signed_area(&convex_intersection(&predicted, &truth)).abs();
    intersection_area / (predicted_area + truth_area - intersection_area)
}

fn edge_length(a: [f32; 2], b: [f32; 2]) -> f64 {
    f64::from((a[0] - b[0]).hypot(a[1] - b[1]))
}

fn quad_dimensions(quad: &[[f32; 2]; 4]) -> (f64, f64) {
    let width = f64::midpoint(edge_length(quad[0], quad[1]), edge_length(quad[2], quad[3]));
    let height = f64::midpoint(edge_length(quad[1], quad[2]), edge_length(quad[3], quad[0]));
    (width, height)
}

fn percentile(values: &mut [f64], fraction: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    #[expect(clippy::cast_precision_loss, reason = "metric vectors fit in memory")]
    let index = fraction * (values.len() - 1) as f64;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the percentile fraction and vector-derived index are nonnegative"
    )]
    let lower = index.floor() as usize;
    let upper = (lower + 1).min(values.len() - 1);
    values[lower] + (values[upper] - values[lower]) * index.fract()
}

/// Evaluate `model` on one split of `dataset_dir`.
///
/// # Errors
///
/// Returns a message when the dataset cannot be loaded or does not match the
/// model input size.
#[expect(
    clippy::too_many_lines,
    reason = "keeping evaluation aggregation in one loop avoids duplicating model inference"
)]
pub fn evaluate(
    model: &Detector,
    dataset_dir: &Path,
    split: Split,
    batch_size: usize,
    device: &Device,
) -> Result<Metrics, String> {
    let (valid, size) = load_split(dataset_dir, split)?;
    if size != model.input_size() {
        return Err(format!(
            "dataset images are {size} px but the model expects {} px",
            model.input_size()
        ));
    }
    let items: Vec<Item> = valid
        .iter()
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    #[expect(clippy::cast_precision_loss, reason = "image side is a small integer")]
    let px_per_unit = size as f64;

    let (mut correct, mut tp, mut fp, mut fn_) = (0usize, 0usize, 0usize, 0usize);
    let mut corner_err_sum = 0.0;
    let mut relative_err_sum = 0.0;
    let (mut corners_total, mut within3, mut within6) = (0usize, 0usize, 0usize);
    let (mut within2pct, mut within5pct) = (0usize, 0usize);
    let mut corner_errors_px = Vec::new();
    let mut quad_ious = Vec::new();
    let mut width_scale_errors = Vec::new();
    let mut height_scale_errors = Vec::new();

    for chunk in items.chunks(batch_size.max(1)) {
        let output = model.forward(DetectionBatcher::assemble(chunk).to_device(device).images);
        for (det, item) in decode_output(&output).iter().zip(chunk) {
            let predicted = det.probability >= 0.5;
            let truth = item.present();
            correct += usize::from(predicted == truth);
            match (predicted, truth) {
                (true, true) => {
                    tp += 1;
                    let (errors, pattern_size) = corner_errors(det, item, px_per_unit);
                    let truth: [[f32; 2]; 4] =
                        std::array::from_fn(|k| [item.target[1 + 2 * k], item.target[2 + 2 * k]]);
                    quad_ious.push(quadrilateral_iou(&det.corners, &truth));
                    let (pred_width, pred_height) = quad_dimensions(&det.corners);
                    let (truth_width, truth_height) = quad_dimensions(&truth);
                    width_scale_errors.push(pred_width / truth_width - 1.0);
                    height_scale_errors.push(pred_height / truth_height - 1.0);
                    for err in errors {
                        corner_err_sum += err;
                        corner_errors_px.push(err);
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
    let mean_values = |values: &[f64]| mean(values.iter().sum(), values.len());
    let mean_abs =
        |values: &[f64]| mean(values.iter().map(|value| value.abs()).sum(), values.len());
    let worst_corner_error_px = corner_errors_px.iter().copied().fold(0.0, f64::max);
    let worst_quad_iou = quad_ious.iter().copied().reduce(f64::min).unwrap_or(0.0);
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
        median_corner_error_px: percentile(&mut corner_errors_px.clone(), 0.5),
        p90_corner_error_px: percentile(&mut corner_errors_px.clone(), 0.9),
        p95_corner_error_px: percentile(&mut corner_errors_px, 0.95),
        worst_corner_error_px,
        mean_quad_iou: mean_values(&quad_ious),
        p10_quad_iou: percentile(&mut quad_ious, 0.1),
        worst_quad_iou,
        mean_width_scale_bias: mean_values(&width_scale_errors),
        mean_abs_width_scale_error: mean_abs(&width_scale_errors),
        mean_height_scale_bias: mean_values(&height_scale_errors),
        mean_abs_height_scale_error: mean_abs(&height_scale_errors),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad(points: [[f32; 2]; 4]) -> [[f32; 2]; 4] {
        points
    }

    #[test]
    fn convex_iou_known_quadrilaterals() {
        let truth = quad([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]]);
        assert_eq!(quadrilateral_iou(&truth, &truth), 1.0);
        assert_eq!(
            quadrilateral_iou(
                &quad([[4.0, 2.0], [0.0, 2.0], [0.0, 0.0], [4.0, 0.0]]),
                &truth
            ),
            1.0
        );
        assert!(
            (quadrilateral_iou(
                &quad([[-1.0, -1.0], [5.0, -1.0], [5.0, 3.0], [-1.0, 3.0]]),
                &truth
            ) - 1.0 / 3.0)
                .abs()
                < 1e-12
        );
        assert!(
            (quadrilateral_iou(
                &quad([[2.0, 0.0], [6.0, 0.0], [6.0, 2.0], [2.0, 2.0]]),
                &truth
            ) - 1.0 / 3.0)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn convex_iou_handles_clipping_and_invalid_quads() {
        let square = quad([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]);
        let diamond = quad([[1.0, -1.0], [3.0, 1.0], [1.0, 3.0], [-1.0, 1.0]]);
        assert!((quadrilateral_iou(&diamond, &square) - 0.5).abs() < 1e-12);
        let bow_tie = quad([[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]]);
        assert_eq!(quadrilateral_iou(&bow_tie, &square), 0.0);
        let flat = quad([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]);
        assert_eq!(quadrilateral_iou(&flat, &square), 0.0);
    }

    #[test]
    fn corner_errors_accept_rotated_equivalent_order() {
        let item = Item {
            pixels: Vec::new(),
            size: 100,
            target: [1.0, 0.1, 0.2, 0.8, 0.2, 0.8, 0.7, 0.1, 0.7],
        };
        let detection = Detection {
            probability: 1.0,
            corners: [[0.8, 0.7], [0.1, 0.7], [0.1, 0.2], [0.8, 0.2]],
        };
        assert_eq!(corner_errors(&detection, &item, 100.0).0, [0.0; 4]);
    }

    #[test]
    fn dimensions_expose_moscow_like_expansion_without_translation() {
        let truth = quad([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]]);
        let expanded = quad([[-1.0, -0.5], [5.0, -0.5], [5.0, 2.5], [-1.0, 2.5]]);
        let shifted = quad([[3.0, 4.0], [7.0, 4.0], [7.0, 6.0], [3.0, 6.0]]);
        let (truth_width, truth_height) = quad_dimensions(&truth);
        let (width, height) = quad_dimensions(&expanded);
        assert_eq!((width / truth_width, height / truth_height), (1.5, 1.5));
        assert_eq!(quad_dimensions(&shifted), (truth_width, truth_height));
    }

    #[test]
    fn percentile_interpolates_and_handles_empty_input() {
        assert_eq!(percentile(&mut [], 0.5), 0.0);
        assert_eq!(percentile(&mut [0.0, 10.0, 20.0], 0.75), 15.0);
    }
}
