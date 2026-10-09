//! Conservative classical refinement of coarse `PhonoPaper` corners.
//!
//! This module intentionally handles only upright, axis-aligned or already
//! approximately rectified patterns. It uses the marker stripe topology in
//! image columns and does not estimate a homography. Rotated or strongly
//! perspective-distorted inputs return `None` so callers can safely retain
//! their original corners.

use image::DynamicImage;

use super::detect_marker_geometry_at_column;

const MAX_EDGE_SLOPE: f32 = 0.12;
const SEARCH_PADDING_FRACTION: f32 = 0.12;
const MIN_SUPPORT_COLUMNS: u32 = 3;
const MIN_SUPPORT_RATIO: f32 = 0.55;

/// A marker-supported refinement of a coarse pattern quadrilateral.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CornerRefinement {
    /// Refined outer marker ink-box corners in pixel-boundary coordinates,
    /// ordered `[TL, TR, BR, BL]`.
    pub corners: [[f32; 2]; 4],
    /// Conservative confidence in `[0, 1]`, combining marker-column support
    /// with agreement between detected and proposed horizontal extents.
    pub confidence: f32,
    /// Number of consecutive columns whose marker topology supported the
    /// refined bounds.
    pub support_columns: u32,
    /// Number of columns in the coarse horizontal span against which support
    /// was evaluated.
    pub sampled_columns: u32,
}

/// Refine an upright coarse quadrilateral to the outer marker ink box.
///
/// `coarse` is expressed in image pixels and ordered `[TL, TR, BR, BL]`. The
/// function searches a small horizontal neighborhood, requires the existing
/// thick/thin marker topology in a contiguous majority of the coarse span,
/// and returns pixel-boundary corners. In particular, the vertical result is
/// the outer marker extent, **not** the inner audio [`super::DataBounds`].
///
/// This is a conservative axis-aligned first pass rather than a projective
/// optimizer. It supports clean generated images and upright or approximately
/// rectified camera crops. It returns `None` for rotated, strongly skewed,
/// degenerate, non-finite, weakly supported, or topology-free inputs; callers
/// should then retain `coarse`.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "image pixel coordinates are represented by the public f32 corner API"
)]
pub fn refine_pattern_corners(
    image: &DynamicImage,
    coarse: [[f32; 2]; 4],
) -> Option<CornerRefinement> {
    let [tl, tr, br, bl] = coarse;
    if !coarse.iter().flatten().all(|value| value.is_finite()) {
        return None;
    }

    let left = f32::midpoint(tl[0], bl[0]);
    let right = f32::midpoint(tr[0], br[0]);
    let top = f32::midpoint(tl[1], tr[1]);
    let bottom = f32::midpoint(bl[1], br[1]);
    let coarse_width = right - left;
    let coarse_height = bottom - top;
    if coarse_width < 2.0 || coarse_height < 2.0 || image.width() == 0 || image.height() == 0 {
        return None;
    }

    let horizontal_edge_tolerance = coarse_width * MAX_EDGE_SLOPE;
    let vertical_edge_tolerance = coarse_height * MAX_EDGE_SLOPE;
    if (tl[1] - tr[1]).abs() > vertical_edge_tolerance
        || (bl[1] - br[1]).abs() > vertical_edge_tolerance
        || (tl[0] - bl[0]).abs() > horizontal_edge_tolerance
        || (tr[0] - br[0]).abs() > horizontal_edge_tolerance
    {
        return None;
    }

    let padding = (coarse_width * SEARCH_PADDING_FRACTION).max(2.0);
    let image_right = image.width().saturating_sub(1);
    let search_left = floor_to_u32((left - padding).max(0.0)).min(image_right);
    let search_right = ceil_to_u32((right + padding).max(0.0)).min(image_right);

    let mut best: Vec<(u32, super::MarkerColumnGeometry)> = Vec::new();
    let mut current = Vec::new();
    for x in search_left..=search_right {
        if let Ok(geometry) = detect_marker_geometry_at_column(image, x) {
            current.push((x, geometry));
        } else {
            keep_longer(&mut best, &mut current);
        }
    }
    keep_longer(&mut best, &mut current);

    let sampled_columns = ceil_to_u32(coarse_width).max(1);
    let support_columns = u32::try_from(best.len()).ok()?;
    let support_ratio = support_columns as f32 / sampled_columns as f32;
    if support_columns < MIN_SUPPORT_COLUMNS || support_ratio < MIN_SUPPORT_RATIO {
        return None;
    }

    let detected_left = best.first()?.0;
    let detected_right = best.last()?.0 + 1;
    let detected_width = detected_right - detected_left;
    let width_agreement =
        1.0 - ((detected_width as f32 - coarse_width).abs() / coarse_width).clamp(0.0, 1.0);
    if width_agreement < 0.5 {
        return None;
    }

    let left_geometry = best.first()?.1;
    let right_geometry = best.last()?.1;
    let corners = [
        [detected_left as f32, left_geometry.outer_top as f32],
        [detected_right as f32, right_geometry.outer_top as f32],
        [detected_right as f32, right_geometry.outer_bottom as f32],
        [detected_left as f32, left_geometry.outer_bottom as f32],
    ];
    Some(CornerRefinement {
        corners,
        confidence: support_ratio.min(1.0) * width_agreement,
        support_columns,
        sampled_columns,
    })
}

fn keep_longer<T>(best: &mut Vec<T>, current: &mut Vec<T>) {
    if current.len() > best.len() {
        *best = std::mem::take(current);
    } else {
        current.clear();
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "coordinates are finite, non-negative, and clamped to image dimensions"
)]
fn floor_to_u32(value: f32) -> u32 {
    value.floor() as u32
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "coordinates are finite, non-negative, and clamped to image dimensions"
)]
fn ceil_to_u32(value: f32) -> u32 {
    value.ceil() as u32
}
