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

/// Find the index (within `dark_runs`) of the thick stripe in the slice
/// `dark_runs[from..to]`.
///
/// The thick stripe is the dark run whose length is ≥ 3× the average of its
/// immediate dark neighbours.  Falls back to the longest run if none qualifies.
fn find_thick_stripe(dark_runs: &[(u32, u32)], from: usize, to: usize) -> Option<usize> {
    let slice = &dark_runs[from..to];
    if slice.len() < 3 {
        return None;
    }
    // Find the run that is ≥ 3× all its neighbours.
    for i in 1..slice.len().saturating_sub(1) {
        let cur_len = slice[i].1;
        let prev_len = slice[i - 1].1;
        let next_len = slice[i + 1].1;
        let avg_neighbours = f64::from(prev_len + next_len) / 2.0;
        if f64::from(cur_len) >= 3.0 * avg_neighbours && avg_neighbours > 0.0 {
            return Some(from + i);
        }
    }
    // Fallback: pick the longest run in the slice.
    let max_idx = slice.iter().enumerate().max_by_key(|(_, (_, len))| *len)?.0;
    Some(from + max_idx)
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

    // Keep only dark runs; we need to find a thick one.
    let dark_runs: Vec<(u32, u32)> = runs
        .iter()
        .filter(|(dark, _, _)| *dark)
        .map(|(_, start, len)| (*start, *len))
        .collect();

    if dark_runs.len() < 3 {
        return Err(PhonoPaperError::MarkerNotFound(
            "fewer than 3 dark runs found",
        ));
    }

    // Split the dark-run list by pixel row, not by list index.
    //
    // Splitting by index (n/2) breaks when the audio data area contains
    // many dark pixels (e.g. a loud sine wave): those dark runs fill the
    // middle of the list and push the real marker runs into the wrong half.
    //
    // Instead, search for the top marker only among runs that start in the
    // top 30% of the image, and the bottom marker only among runs that start
    // in the bottom 30%.  The marker bands occupy ≈ 184/1088 ≈ 17% of the
    // image height at default settings, so 30% gives comfortable headroom.
    let top_limit = height * 3 / 10;
    let bot_limit = height * 7 / 10;

    let top_end = dark_runs.partition_point(|&(start, _)| start < top_limit);
    let bot_start_idx = dark_runs.partition_point(|&(start, _)| start < bot_limit);

    let top_idx = find_thick_stripe(&dark_runs, 0, top_end).ok_or(
        PhonoPaperError::MarkerNotFound("no thick stripe in top 30% of image"),
    )?;
    let bot_idx = find_thick_stripe(&dark_runs, bot_start_idx, dark_runs.len()).ok_or(
        PhonoPaperError::MarkerNotFound("no thick stripe in bottom 30% of image"),
    )?;

    let (top_start, top_len) = dark_runs[top_idx];
    let (bot_start, _) = dark_runs[bot_idx];

    // The marker band layout around the data area is:
    //   top:    ... THICK_STRIPE → white gap → thin_stripe → [DATA]
    //   bottom: [DATA] → thin_stripe → white gap → THICK_STRIPE ...
    //
    // So `top_thick_start + top_thick_len` lands in the white gap, not at the
    // data area yet.  The next dark run after the thick stripe is the trailing
    // thin stripe; the data area begins immediately after that thin stripe.
    // Symmetrically, the dark run immediately before the bottom thick stripe is
    // the leading thin stripe; the data area ends at the start of that run.
    //
    // If no such adjacent thin stripe exists (malformed image), fall back to
    // the thick stripe edge itself.
    let data_top = if top_idx + 1 < dark_runs.len() {
        let (inner_start, inner_len) = dark_runs[top_idx + 1];
        inner_start + inner_len
    } else {
        top_start + top_len
    };

    let data_bottom = if bot_idx > 0 {
        let (inner_start, _) = dark_runs[bot_idx - 1];
        inner_start
    } else {
        bot_start
    };

    if data_bottom <= data_top {
        return Err(PhonoPaperError::MarkerNotFound("data area has zero height"));
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
/// samples up to three evenly-spaced columns (`width/4`, `width/2`,
/// `3*width/4`) and returns the result from the first column that succeeds.
/// If all three agree the result is unambiguous; if fewer than three succeed
/// the earliest success is returned.
///
/// For clean, axis-aligned images all three columns produce identical
/// `DataBounds`.  For mildly distorted images (slight tilt or uneven
/// illumination) at least one column typically succeeds where the centre
/// column alone might fail.  For images with severe perspective distortion
/// use [`detect_markers_at_column`] directly across many columns.
///
/// The `PhonoPaper` marker pattern consists of alternating black and white
/// horizontal stripes.  The key identifying feature is a **thick black stripe**
/// that is at least 3× wider than the surrounding thin stripes.  This pattern
/// appears at both the top and bottom of the image, bounding the audio data
/// area.
///
/// # Algorithm
///
/// 1. Collect a run-length-encoded sequence of dark/light runs along the
///    sampled column.
/// 2. Look for a run of dark pixels that is ≥ 3× the length of its immediate
///    dark neighbours, searching only in the **top 30%** of the image for the
///    top marker and the **bottom 30%** for the bottom marker.  This prevents
///    dark pixels in the audio data area from being misidentified as marker
///    stripes.
/// 3. The data area starts immediately after the **thin stripe that follows**
///    the top thick stripe (not at the thick stripe's inner edge), and ends
///    immediately before the **thin stripe that precedes** the bottom thick
///    stripe.  This correctly excludes the inner gap and thin stripe of each
///    marker band from the decoded audio data.
///
/// # Errors
///
/// Returns [`PhonoPaperError::MarkerNotFound`] if no valid marker pattern is
/// detected in any of the sampled columns.
pub fn detect_markers(image: &DynamicImage) -> Result<DataBounds> {
    let (width, _) = image.dimensions();
    // Try up to three evenly-spaced columns; return the first success.
    // The candidates [width/4, width/2, 3*width/4] are computed by integer
    // division and are already in non-decreasing order, so dedup() correctly
    // removes consecutive duplicates (which can occur for very narrow images
    // where width/4 == width/2 or width/2 == 3*width/4).
    let candidates: Vec<u32> = {
        let mut cs = vec![width / 4, width / 2, width * 3 / 4];
        cs.dedup();
        cs
    };

    let mut last_err =
        PhonoPaperError::MarkerNotFound("no valid marker pattern in any sampled column");
    for col in candidates {
        match detect_markers_at_column(image, col) {
            Ok(bounds) => return Ok(bounds),
            Err(e) => last_err = e,
        }
    }
    Err(last_err)
}
