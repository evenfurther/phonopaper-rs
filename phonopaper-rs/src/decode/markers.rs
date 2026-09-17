//! `PhonoPaper` marker band detection.
//!
//! Provides [`detect_markers`], [`detect_markers_at_column`], and the
//! [`DataBounds`] type that locates the audio data area within an image by
//! scanning a chosen column for the characteristic thick/thin stripe pattern.

use image::{DynamicImage, GenericImageView};

use crate::error::{PhonoPaperError, Result};

// ─── Luminance helper ─────────────────────────────────────────────────────────

/// Convert an RGBA pixel to BT.601 luminance (0–255).
pub(super) fn pixel_luma(p: image::Rgba<u8>) -> u8 {
    let r = u32::from(p[0]);
    let g = u32::from(p[1]);
    let b = u32::from(p[2]);
    // Maximum value: 255*299 + 255*587 + 255*114 = 255*1000 = 255_000,
    // divided by 1000 = 255.  The cast to u8 is always safe.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "BT.601 sum divided by 1000 is always in [0, 255]"
    )]
    let luma = ((r * 299 + g * 587 + b * 114) / 1000) as u8;
    luma
}

// ─── DataBounds ───────────────────────────────────────────────────────────────

/// The boundaries of the `PhonoPaper` data area within an image.
///
/// Both values are pixel row indices: `data_top` is the first row of audio
/// data and `data_bottom` is one past the last row (exclusive).
#[derive(Debug, Clone, Copy)]
pub struct DataBounds {
    /// First pixel row of the audio data area (inclusive).
    pub data_top: u32,
    /// Last pixel row of the audio data area (exclusive).
    pub data_bottom: u32,
}

impl DataBounds {
    /// Height of the data area in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.data_bottom - self.data_top
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

// Threshold: a pixel is "dark" if its luminance is below this value.
const DARK_THRESHOLD: u8 = 128;

#[derive(Debug, Clone, Copy)]
enum MarkerSide {
    Top,
    Bottom,
}

#[derive(Debug, Clone, Copy)]
struct MarkerCandidate {
    data_edge: u32,
    thick_len: u32,
    thin_ref: u32,
    gap_ref: u32,
}

fn all_light(runs: &[(bool, u32, u32)], indices: &[usize]) -> bool {
    indices.iter().all(|&idx| !runs[idx].0)
}

fn all_dark(runs: &[(bool, u32, u32)], indices: &[usize]) -> bool {
    indices.iter().all(|&idx| runs[idx].0)
}

fn max3(a: u32, b: u32, c: u32) -> u32 {
    a.max(b).max(c)
}

fn min3(a: u32, b: u32, c: u32) -> u32 {
    a.min(b).min(c)
}

fn is_consistent_run_family(lengths: &[u32]) -> bool {
    let (Some(&min_len), Some(&max_len)) = (lengths.iter().min(), lengths.iter().max()) else {
        return false;
    };
    min_len > 0 && max_len <= min_len * 3
}

fn matches_top_marker_pattern(runs: &[(bool, u32, u32)], idx: usize) -> Option<MarkerCandidate> {
    if idx < 5 || idx + 2 >= runs.len() {
        return None;
    }

    let light_runs = [idx - 5, idx - 3, idx - 1, idx + 1];
    let dark_runs = [idx - 4, idx - 2, idx, idx + 2];
    if !all_light(runs, &light_runs) || !all_dark(runs, &dark_runs) {
        return None;
    }

    let outer_thin_1 = runs[idx - 4].2;
    let outer_thin_2 = runs[idx - 2].2;
    let gap_1 = runs[idx - 3].2;
    let gap_2 = runs[idx - 1].2;
    let gap_3 = runs[idx + 1].2;
    let thick = runs[idx].2;

    if !is_consistent_run_family(&[outer_thin_1, outer_thin_2])
        || !is_consistent_run_family(&[gap_1, gap_2, gap_3])
    {
        return None;
    }

    let thin_ref = outer_thin_1.max(outer_thin_2);
    if thin_ref == 0 || gap_1.min(gap_2).min(gap_3) == 0 {
        return None;
    }

    if thick.saturating_mul(2) < 3 * (outer_thin_1 + outer_thin_2) {
        return None;
    }

    let max_gap = max3(gap_1, gap_2, gap_3);
    let min_gap = min3(gap_1, gap_2, gap_3);
    if max_gap > thin_ref * 4 || min_gap * 2 < thin_ref {
        return None;
    }

    let inner_dark = runs[idx + 2];
    Some(MarkerCandidate {
        data_edge: inner_dark.1 + inner_dark.2,
        thick_len: thick,
        thin_ref,
        gap_ref: max_gap,
    })
}

fn matches_bottom_marker_pattern(runs: &[(bool, u32, u32)], idx: usize) -> Option<MarkerCandidate> {
    if idx < 2 || idx + 5 >= runs.len() {
        return None;
    }

    let light_runs = [idx - 1, idx + 1, idx + 3, idx + 5];
    let dark_runs = [idx - 2, idx, idx + 2, idx + 4];
    if !all_light(runs, &light_runs) || !all_dark(runs, &dark_runs) {
        return None;
    }

    let outer_thin_1 = runs[idx + 2].2;
    let outer_thin_2 = runs[idx + 4].2;
    let gap_1 = runs[idx - 1].2;
    let gap_2 = runs[idx + 1].2;
    let gap_3 = runs[idx + 3].2;
    let thick = runs[idx].2;

    if !is_consistent_run_family(&[outer_thin_1, outer_thin_2])
        || !is_consistent_run_family(&[gap_1, gap_2, gap_3])
    {
        return None;
    }

    let thin_ref = outer_thin_1.max(outer_thin_2);
    if thin_ref == 0 || gap_1.min(gap_2).min(gap_3) == 0 {
        return None;
    }

    if thick.saturating_mul(2) < 3 * (outer_thin_1 + outer_thin_2) {
        return None;
    }

    let max_gap = max3(gap_1, gap_2, gap_3);
    let min_gap = min3(gap_1, gap_2, gap_3);
    if max_gap > thin_ref * 4 || min_gap * 2 < thin_ref {
        return None;
    }

    Some(MarkerCandidate {
        data_edge: runs[idx - 2].1,
        thick_len: thick,
        thin_ref,
        gap_ref: max_gap,
    })
}

fn find_marker_candidate(
    runs: &[(bool, u32, u32)],
    side: MarkerSide,
    zone_start: u32,
    zone_end: u32,
) -> Option<MarkerCandidate> {
    runs.iter()
        .enumerate()
        .filter(|(_, (is_dark, start, _))| *is_dark && *start >= zone_start && *start < zone_end)
        .filter_map(|(idx, _)| match side {
            MarkerSide::Top => matches_top_marker_pattern(runs, idx),
            MarkerSide::Bottom => matches_bottom_marker_pattern(runs, idx),
        })
        .max_by_key(|candidate| candidate.thick_len)
}

fn evenly_spaced_columns(width: u32, requested_samples: u32) -> Vec<u32> {
    if width == 0 {
        return Vec::new();
    }
    let n_samples = requested_samples.min(width).max(1);
    let mut sample_xs: Vec<u32> = (0..n_samples)
        .map(|i| {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "value is rounded and clamped to [0, width-1]; fits in u32"
            )]
            #[expect(
                clippy::cast_sign_loss,
                reason = ".round() on a non-negative f64 product is always non-negative"
            )]
            let x = (f64::from(i) / f64::from(n_samples - 1).max(1.0) * f64::from(width - 1))
                .round() as u32;
            x.min(width - 1)
        })
        .collect();
    sample_xs.dedup();
    sample_xs
}

fn bounds_are_consistent(prev_x: u32, prev: DataBounds, next_x: u32, next: DataBounds) -> bool {
    let dx = next_x.abs_diff(prev_x);
    let max_boundary_step = dx / 4 + 4;
    let prev_height = prev.height();
    let next_height = next.height();
    let min_height = prev_height.min(next_height);
    let max_height_delta = min_height / 10 + 6;

    prev.data_top.abs_diff(next.data_top) <= max_boundary_step
        && prev.data_bottom.abs_diff(next.data_bottom) <= max_boundary_step
        && prev_height.abs_diff(next_height) <= max_height_delta
}

fn detect_markers_in_column_cluster(image: &DynamicImage, sample_xs: &[u32]) -> Result<DataBounds> {
    let min_cluster_len = sample_xs.len().min(3);
    let mut best_cluster: Vec<(u32, DataBounds)> = Vec::new();
    let mut current_cluster: Vec<(u32, DataBounds)> = Vec::new();

    for &col_x in sample_xs {
        match detect_markers_at_column(image, col_x) {
            Ok(bounds) => {
                let continues_cluster =
                    current_cluster
                        .last()
                        .is_some_and(|&(prev_x, prev_bounds)| {
                            bounds_are_consistent(prev_x, prev_bounds, col_x, bounds)
                        });

                if !continues_cluster {
                    if current_cluster.len() > best_cluster.len() {
                        best_cluster = std::mem::take(&mut current_cluster);
                    } else {
                        current_cluster.clear();
                    }
                }

                current_cluster.push((col_x, bounds));
            }
            Err(_) => {
                if current_cluster.len() > best_cluster.len() {
                    best_cluster = std::mem::take(&mut current_cluster);
                } else {
                    current_cluster.clear();
                }
            }
        }
    }

    if current_cluster.len() > best_cluster.len() {
        best_cluster = current_cluster;
    }

    if best_cluster.len() < min_cluster_len {
        return Err(PhonoPaperError::MarkerNotFound(
            "no sufficiently wide cluster of consistent marker detections found",
        ));
    }

    Ok(best_cluster[best_cluster.len() / 2].1)
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Scan a specific vertical column of the image to locate the `PhonoPaper`
/// marker bands.
///
/// This is the column-parametric version of [`detect_markers`]; it scans
/// column `col_x` instead of the image centre column.  Use this when
/// processing a perspective-distorted image where the marker bands are not
/// perfectly horizontal: call this function for a set of evenly-spaced
/// columns and interpolate `data_top` / `data_bottom` per column to
/// compensate for keystone distortion, paper curl, and tilt without requiring
/// an explicit de-warp step.
///
/// See [`detect_markers`] for a description of the algorithm and the marker
/// band layout.
///
/// # Errors
///
/// Returns [`PhonoPaperError::MarkerNotFound`] if no valid marker pattern is
/// detected in column `col_x`, or if `col_x` is out of bounds for the image.
pub fn detect_markers_at_column(image: &DynamicImage, col_x: u32) -> Result<DataBounds> {
    let (width, height) = image.dimensions();

    if col_x >= width {
        return Err(PhonoPaperError::MarkerNotFound("column out of bounds"));
    }

    // Build a grayscale column at col_x.
    let luma: Vec<u8> = (0..height)
        .map(|y| pixel_luma(image.get_pixel(col_x, y)))
        .collect();

    let is_dark: Vec<bool> = luma.iter().map(|&v| v < DARK_THRESHOLD).collect();

    // Run-length encode the dark/light sequence.
    // Each entry: (is_dark: bool, start_row: u32, length: u32)
    let mut runs: Vec<(bool, u32, u32)> = Vec::new();
    let mut i = 0usize;
    while i < is_dark.len() {
        let dark = is_dark[i];
        // Image heights are at most u32::MAX; the usize→u32 cast is safe for
        // any real image.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "loop index comes from iterating 0..height where height is u32"
        )]
        let start = i as u32;
        while i < is_dark.len() && is_dark[i] == dark {
            i += 1;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "i - start ≤ height which is u32, so the difference fits in u32"
        )]
        let len = i as u32 - start;
        runs.push((dark, start, len));
    }

    // Search for a locally valid marker pattern in the top and bottom zones.
    // Looking for the full light/dark stripe topology is much stricter than
    // merely finding a long dark run, which avoids false positives on
    // arbitrary photographs and random noise.
    let top_limit = height * 3 / 10;
    let bot_limit = height * 7 / 10;

    let top_marker = find_marker_candidate(&runs, MarkerSide::Top, 0, top_limit).ok_or(
        PhonoPaperError::MarkerNotFound("no valid top marker pattern found"),
    )?;
    let bottom_marker = find_marker_candidate(&runs, MarkerSide::Bottom, bot_limit, height).ok_or(
        PhonoPaperError::MarkerNotFound("no valid bottom marker pattern found"),
    )?;

    let data_top = top_marker.data_edge;
    let data_bottom = bottom_marker.data_edge;

    if data_bottom <= data_top {
        return Err(PhonoPaperError::MarkerNotFound("data area has zero height"));
    }

    let max_thin = top_marker.thin_ref.max(bottom_marker.thin_ref);
    let min_thin = top_marker.thin_ref.min(bottom_marker.thin_ref);
    if min_thin == 0 || max_thin > min_thin * 3 {
        return Err(PhonoPaperError::MarkerNotFound(
            "top and bottom marker stripe widths disagree too much",
        ));
    }

    let max_gap = top_marker.gap_ref.max(bottom_marker.gap_ref);
    let min_gap = top_marker.gap_ref.min(bottom_marker.gap_ref);
    if min_gap == 0 || max_gap > min_gap * 3 {
        return Err(PhonoPaperError::MarkerNotFound(
            "top and bottom marker gap widths disagree too much",
        ));
    }

    Ok(DataBounds {
        data_top,
        data_bottom,
    })
}

/// Scan multiple evenly-spaced columns of the image to locate the `PhonoPaper`
/// marker bands.
///
/// This is a more robust wrapper around [`detect_markers_at_column`] that
/// samples up to nine evenly-spaced columns across the image, keeps only
/// detections that form a horizontally consistent cluster, and returns the
/// cluster's median bounds.  Isolated single-column hits are rejected so
/// arbitrary photographs and noise do not decode as false `PhonoPaper`
/// patterns.
///
/// For clean, axis-aligned images all sampled columns produce identical
/// `DataBounds`.  For mildly distorted images (slight tilt or uneven
/// illumination) the accepted cluster can drift smoothly across columns.  For
/// images with severe perspective distortion use [`detect_markers_at_column`]
/// directly across many columns.
///
/// The `PhonoPaper` marker pattern consists of alternating black and white
/// horizontal stripes.  The key identifying feature is a **thick black stripe**
/// that is at least 3× wider than the surrounding thin stripes.  This pattern
/// appears at both the top and bottom of the image, bounding the audio data
/// area.
///
/// # Algorithm
///
/// 1. For each sampled column, collect a run-length-encoded sequence of
///    dark/light runs.
/// 2. Search only in the **top 30%** and **bottom 30%** of the image, and
///    require the full `PhonoPaper` light/dark stripe topology around each
///    thick stripe candidate.
/// 3. Compare the top and bottom marker proportions within the same column and
///    reject columns whose stripe or gap widths disagree too much.
/// 4. Keep only detections that remain horizontally consistent across adjacent
///    sampled columns, then return the median bounds from the widest such
///    cluster.
///
/// # Errors
///
/// Returns [`PhonoPaperError::MarkerNotFound`] if no valid marker pattern is
/// detected in any of the sampled columns.
pub fn detect_markers(image: &DynamicImage) -> Result<DataBounds> {
    let (width, _) = image.dimensions();
    if width == 0 {
        return Err(PhonoPaperError::MarkerNotFound("image has zero width"));
    }
    let sample_xs = evenly_spaced_columns(width, 9);
    detect_markers_in_column_cluster(image, &sample_xs)
}
