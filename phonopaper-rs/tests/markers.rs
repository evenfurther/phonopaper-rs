//! Tests for [`phonopaper_rs::decode::detect_markers`] and
//! [`phonopaper_rs::decode::detect_markers_at_column`].
//!
//! All test images are generated programmatically — no binary image files are
//! committed to the repository.  The canonical way to obtain a valid
//! `PhonoPaper` image with correct marker bands is via
//! [`phonopaper_rs::render::spectrogram_to_image`].

use image::{DynamicImage, GenericImageView as _, GrayImage, Luma, RgbImage};
use phonopaper_rs::decode::{DataBounds, detect_markers, detect_markers_at_column};
use phonopaper_rs::render::{RenderOptions, spectrogram_to_image};
use phonopaper_rs::spectrogram::SpectrogramVec;

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Generate a valid `PhonoPaper` image using the default [`RenderOptions`].
///
/// The spectrogram is all-zero (silence), so the data area is all-white.
/// Only the marker bands matter for these tests.
fn default_phonopaper_image() -> (DynamicImage, RenderOptions) {
    let opts = RenderOptions {
        draw_octave_lines: false,
        ..RenderOptions::default()
    };
    let spec = SpectrogramVec::new(4); // 4 columns of silence
    let rgb = spectrogram_to_image(&spec, &opts);
    (DynamicImage::ImageRgb8(rgb), opts)
}

/// Expected `data_top` for a default-options render: the height of one marker
/// band (margin + 3 thin stripes + thick stripe + gaps = 184 px).
fn expected_top(opts: &RenderOptions) -> u32 {
    opts.marker_band_height()
}

/// Expected `data_bottom` for a default-options render:
/// `marker_band_height + data_height`.
fn expected_bottom(opts: &RenderOptions) -> u32 {
    let data_height = opts.px_per_octave * 8;
    opts.marker_band_height() + data_height
}

// ─── DataBounds unit tests ────────────────────────────────────────────────────

/// `DataBounds::height()` returns `data_bottom − data_top`.
#[test]
fn data_bounds_height() {
    let b = DataBounds {
        data_top: 184,
        data_bottom: 904,
    };
    assert_eq!(b.height(), 720);
}

/// `DataBounds::height()` is zero when `data_top == data_bottom`.
#[test]
fn data_bounds_height_zero() {
    let b = DataBounds {
        data_top: 50,
        data_bottom: 50,
    };
    assert_eq!(b.height(), 0);
}

// ─── detect_markers on a clean rendered image ─────────────────────────────────

/// `detect_markers` on a programmatically rendered `PhonoPaper` image returns
/// `data_top` equal to `marker_band_height` (184 px with default options).
#[test]
fn detect_markers_data_top_default() {
    let (img, opts) = default_phonopaper_image();
    let bounds = detect_markers(&img).expect("marker detection should succeed");
    assert_eq!(
        bounds.data_top,
        expected_top(&opts),
        "data_top should equal marker_band_height"
    );
}

/// `detect_markers` on a clean rendered image returns `data_bottom` equal to
/// `marker_band_height + data_height` (904 px with default options).
#[test]
fn detect_markers_data_bottom_default() {
    let (img, opts) = default_phonopaper_image();
    let bounds = detect_markers(&img).expect("marker detection should succeed");
    assert_eq!(
        bounds.data_bottom,
        expected_bottom(&opts),
        "data_bottom should equal marker_band_height + data_height"
    );
}

/// `detect_markers` reports the correct data-area height (720 px at default
/// 90 px/octave × 8 octaves).
#[test]
fn detect_markers_data_height_default() {
    let (img, opts) = default_phonopaper_image();
    let bounds = detect_markers(&img).expect("marker detection should succeed");
    let expected = opts.px_per_octave * 8;
    assert_eq!(
        bounds.height(),
        expected,
        "detected height {}, expected {expected}",
        bounds.height()
    );
}

/// `detect_markers` succeeds on images with non-default (but still valid)
/// `RenderOptions` and returns the correspondingly adjusted boundaries.
#[test]
fn detect_markers_custom_options() {
    let opts = RenderOptions {
        px_per_octave: 45, // half-height data area
        thin_stripe: 5,
        thick_stripe: 21,
        marker_gap: 6,
        margin: 40,
        draw_octave_lines: false,
        gamma: 1.0,
    };
    let spec = SpectrogramVec::new(2);
    let rgb = spectrogram_to_image(&spec, &opts);
    let img = DynamicImage::ImageRgb8(rgb);

    let bounds = detect_markers(&img).expect("marker detection should succeed with custom opts");
    assert_eq!(bounds.data_top, opts.marker_band_height());
    assert_eq!(
        bounds.data_bottom,
        opts.marker_band_height() + opts.px_per_octave * 8
    );
}

/// `detect_markers` succeeds even when the data area contains many dark runs
/// — alternating loud and silent frequency bins create a striped pattern in
/// the data area that could confuse a naive detector.
///
/// Only the *odd-indexed* bins are set to maximum amplitude so that the data
/// area has interleaved dark and white rows (one pixel each), producing a
/// large number of short dark runs that must not be mistaken for marker
/// stripes.
#[test]
fn detect_markers_with_dark_data_area() {
    let opts = RenderOptions {
        draw_octave_lines: false,
        ..RenderOptions::default()
    };
    let mut spec = SpectrogramVec::new(8);
    for col in 0..8 {
        for bin in (1..phonopaper_rs::format::TOTAL_BINS).step_by(2) {
            spec.set(col, bin, 1.0); // every other bin at maximum → alternating dark rows
        }
    }
    let rgb = spectrogram_to_image(&spec, &opts);
    let img = DynamicImage::ImageRgb8(rgb);

    let bounds = detect_markers(&img).expect("should detect markers with striped data area");
    assert_eq!(bounds.data_top, expected_top(&opts));
    // Allow a 1-pixel tolerance on data_bottom: when the bottommost data bin
    // is dark, it can merge with the adjacent inner thin stripe, shifting the
    // detected boundary by 1 row.  The important property is that the data
    // area height is within 1 pixel of the expected value.
    let expected_b = expected_bottom(&opts);
    assert!(
        bounds.data_bottom.abs_diff(expected_b) <= 1,
        "data_bottom {} should be within 1 pixel of expected {expected_b}",
        bounds.data_bottom,
    );
}

// ─── detect_markers failure cases ────────────────────────────────────────────

/// An all-white image has no dark stripes, so `detect_markers` returns an error.
#[test]
fn detect_markers_all_white_is_error() {
    let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(
        200,
        1088,
        image::Rgb([255u8, 255, 255]),
    ));
    assert!(
        detect_markers(&img).is_err(),
        "all-white image should return an error"
    );
}

/// An all-black image has no pattern of thin/thick stripes, so detection
/// should fail rather than panic or produce a nonsensical result.
#[test]
fn detect_markers_all_black_is_error() {
    let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(200, 1088, image::Rgb([0u8, 0, 0])));
    assert!(
        detect_markers(&img).is_err(),
        "all-black image (no thick/thin contrast) should return an error"
    );
}

/// An image that is too small to contain a valid marker band (height < 2 ×
/// `marker_band_height`) returns an error rather than panicking.
#[test]
fn detect_markers_too_small_is_error() {
    // A 1×10 image — far too small for the default 184-px marker bands.
    let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(1, 10, Luma([255u8])));
    assert!(
        detect_markers(&img).is_err(),
        "tiny image should return an error"
    );
}

// ─── detect_markers_at_column ────────────────────────────────────────────────

/// `detect_markers_at_column` with the centre column returns the same result
/// as `detect_markers` (which also uses the centre column).
#[test]
fn detect_markers_at_column_matches_centre() {
    let (img, _opts) = default_phonopaper_image();
    let (width, _) = img.dimensions();
    let cx = width / 2;

    let bounds_detect = detect_markers(&img).expect("detect_markers should succeed");
    let bounds_at_col =
        detect_markers_at_column(&img, cx).expect("detect_markers_at_column should succeed");

    assert_eq!(
        bounds_detect.data_top, bounds_at_col.data_top,
        "data_top must match between detect_markers and detect_markers_at_column(width/2)"
    );
    assert_eq!(
        bounds_detect.data_bottom, bounds_at_col.data_bottom,
        "data_bottom must match"
    );
}

/// `detect_markers_at_column` returns correct bounds for every column in a
/// clean rendered image (all columns have identical marker bands).
#[test]
fn detect_markers_at_column_all_columns_consistent() {
    let (img, opts) = default_phonopaper_image();
    let (width, _) = img.dimensions();

    let expected_t = expected_top(&opts);
    let expected_b = expected_bottom(&opts);

    for col in 0..width {
        let bounds = detect_markers_at_column(&img, col)
            .unwrap_or_else(|_| panic!("marker detection should succeed for column {col}"));
        assert_eq!(
            bounds.data_top, expected_t,
            "column {col}: data_top {}, expected {expected_t}",
            bounds.data_top
        );
        assert_eq!(
            bounds.data_bottom, expected_b,
            "column {col}: data_bottom {}, expected {expected_b}",
            bounds.data_bottom
        );
    }
}

/// `detect_markers_at_column` with an out-of-bounds column index returns
/// `Err(MarkerNotFound)` rather than panicking.
#[test]
fn detect_markers_at_column_out_of_bounds_is_error() {
    let (img, _) = default_phonopaper_image();
    let (width, _) = img.dimensions();

    // col_x == width is one past the last valid column.
    assert!(
        detect_markers_at_column(&img, width).is_err(),
        "out-of-bounds col_x should return an error"
    );
    // A wildly large value should also be an error.
    assert!(
        detect_markers_at_column(&img, u32::MAX).is_err(),
        "u32::MAX col_x should return an error"
    );
}

/// `detect_markers_at_column` succeeds on the leftmost (0) and rightmost
/// (width − 1) columns of a rendered image.
#[test]
fn detect_markers_at_column_edge_columns() {
    let (img, opts) = default_phonopaper_image();
    let (width, _) = img.dimensions();

    let left = detect_markers_at_column(&img, 0).expect("should detect markers in leftmost column");
    let right = detect_markers_at_column(&img, width - 1)
        .expect("should detect markers in rightmost column");

    assert_eq!(left.data_top, expected_top(&opts));
    assert_eq!(left.data_bottom, expected_bottom(&opts));
    assert_eq!(right.data_top, expected_top(&opts));
    assert_eq!(right.data_bottom, expected_bottom(&opts));
}

// ─── Perspective / tilt simulation ───────────────────────────────────────────

/// Simulate a perspective-distorted image by constructing it column-by-column:
/// the marker bands shift downward linearly from left to right (i.e. the top
/// edge of the image is tilted).
///
/// `detect_markers_at_column` is called at the left edge and right edge and
/// must return `data_top` values that reflect the tilt — the right edge should
/// have a larger `data_top` than the left edge.
#[test]
fn detect_markers_at_column_perspective_tilt() {
    use phonopaper_rs::render::RenderOptions;

    // We build the tilted image manually by placing the marker bands at a
    // column-dependent vertical offset.
    let opts = RenderOptions {
        draw_octave_lines: false,
        ..RenderOptions::default()
    };

    // Generate the canonical flat image first (4 columns wide).
    let spec = SpectrogramVec::new(4);
    let flat_rgb = spectrogram_to_image(&spec, &opts);
    let (width, height) = flat_rgb.dimensions();

    // Tilt: each column is shifted down by `col * shift_per_col` pixels.
    // We use a small shift so the bands remain well within the image.
    let shift_per_col: u32 = 2; // total tilt = 2 × (width-1) pixels
    let max_shift = shift_per_col * (width - 1);

    // The tilted image height must be tall enough to contain the shifted bands.
    let tilted_height = height + max_shift;

    let mut tilted = RgbImage::from_pixel(width, tilted_height, image::Rgb([255u8, 255, 255]));

    for col in 0..width {
        let shift = shift_per_col * col;
        for row in 0..height {
            let src_pixel = flat_rgb.get_pixel(col, row);
            tilted.put_pixel(col, row + shift, *src_pixel);
        }
    }

    let tilted_dyn = DynamicImage::ImageRgb8(tilted);

    // Left column: no shift, so data_top = marker_band_height.
    let left_bounds =
        detect_markers_at_column(&tilted_dyn, 0).expect("left column detection should succeed");

    // Right column: shifted down by max_shift, so data_top = marker_band_height + max_shift.
    let right_bounds = detect_markers_at_column(&tilted_dyn, width - 1)
        .expect("right column detection should succeed");

    assert_eq!(
        left_bounds.data_top,
        opts.marker_band_height(),
        "left column data_top should equal marker_band_height (no shift)"
    );
    assert_eq!(
        right_bounds.data_top,
        opts.marker_band_height() + max_shift,
        "right column data_top should equal marker_band_height + max_shift"
    );

    // The data-area height should be the same in both columns.
    assert_eq!(
        left_bounds.height(),
        right_bounds.height(),
        "data-area height should be identical regardless of vertical shift"
    );
}

/// A minimal single-column `PhonoPaper` image is correctly decoded by both
/// `detect_markers` and `detect_markers_at_column`.
#[test]
fn detect_markers_single_column_image() {
    let opts = RenderOptions {
        draw_octave_lines: false,
        ..RenderOptions::default()
    };
    let spec = SpectrogramVec::new(1);
    let rgb = spectrogram_to_image(&spec, &opts);
    let img = DynamicImage::ImageRgb8(rgb);

    let b_detect = detect_markers(&img).expect("single-column image: detect_markers should work");
    let b_at_col =
        detect_markers_at_column(&img, 0).expect("single-column image: at_column(0) should work");

    assert_eq!(b_detect.data_top, opts.marker_band_height());
    assert_eq!(b_at_col.data_top, opts.marker_band_height());
    assert_eq!(b_detect.data_bottom, b_at_col.data_bottom);
}

// ─── find_thick_stripe fallback (no dominant run → pick longest) ──────────────

/// When no dark run in the marker zone is ≥ 3× the average of its neighbours,
/// `find_thick_stripe` falls back to the longest run.  Verify this by
/// constructing a column where all stripes have the same width (no clear
/// dominant), and checking that marker detection still succeeds and returns
/// self-consistent bounds.
///
/// The image is painted by hand: a 200-pixel-tall grayscale image with two
/// groups of equal-width stripes in the top and bottom 30 % respectively, plus
/// a white data area in between.
#[test]
fn detect_markers_equal_stripe_widths_fallback_to_longest() {
    // Layout (1 wide × 200 tall image):
    //  rows   0..10  — black (run A)
    //  rows  10..20  — white
    //  rows  20..30  — black (run B)
    //  rows  30..40  — white
    //  rows  40..50  — black (run C)   ← will be picked as "thick" (first/longest tie)
    //  rows  50..140 — white (data area)
    //  rows 140..150 — black (run D)   ← will be picked as "thick" in bottom zone
    //  rows 150..160 — white
    //  rows 160..170 — black (run E)
    //  rows 170..180 — white
    //  rows 180..190 — black (run F)
    //  rows 190..200 — white

    let width = 1u32;
    let height = 200u32;
    let mut img = GrayImage::from_pixel(width, height, Luma([255u8]));

    // Paint the six equal-width (10 px) black stripes.
    for &(start, end) in &[
        (0u32, 10u32),
        (20, 30),
        (40, 50),
        (140, 150),
        (160, 170),
        (180, 190),
    ] {
        for row in start..end {
            img.put_pixel(0, row, Luma([0u8]));
        }
    }

    let dyn_img = DynamicImage::ImageLuma8(img);

    // With height = 200: top_limit = 60, bot_limit = 140.
    // Top zone dark runs: A(0,10), B(20,10), C(40,10) → none is ≥ 3× avg → fallback picks first.
    // Bottom zone dark runs: D(140,10), E(160,10), F(180,10) → same fallback → picks D.
    let bounds = detect_markers_at_column(&dyn_img, 0)
        .expect("detect_markers_at_column should succeed with equal-width stripes via fallback");

    // With all stripes of equal width, max_by_key picks the LAST maximum (Rust tie-breaking).
    // Top zone: runs A(0,10), B(20,10), C(40,10) → picks C (index 2) → top_idx = 2
    //   data_top: dark_runs[top_idx + 1] = dark_runs[3] = D(140,10) → 140 + 10 = 150
    // Bottom zone: runs D(140,10), E(160,10), F(180,10) → picks F (index 2) → bot_idx = 5
    //   data_bottom: dark_runs[bot_idx - 1] = dark_runs[4] = E(160,10) → 160
    assert!(
        bounds.data_top < bounds.data_bottom,
        "data_top ({}) must be less than data_bottom ({})",
        bounds.data_top,
        bounds.data_bottom
    );
    assert_eq!(
        bounds.data_top, 150,
        "data_top should be 150 (end of run D)"
    );
    assert_eq!(
        bounds.data_bottom, 160,
        "data_bottom should be 160 (start of run E)"
    );
}

/// When the top zone contains fewer than 3 dark runs, `find_thick_stripe`
/// returns `None` (hitting the `slice.len() < 3` early exit) and
/// `detect_markers_at_column` propagates a [`MarkerNotFound`] error.
///
/// Layout: 2 black stripes in the top 30 % of a 200-pixel column, and 3
/// black stripes in the bottom 70 %.  The bottom zone finds a thick stripe
/// (fallback to longest), but the top zone fails because it only has 2 runs.
#[test]
fn detect_markers_fewer_than_three_top_runs_is_error() {
    let width = 1u32;
    let height = 200u32;
    let mut img = GrayImage::from_pixel(width, height, Luma([255u8]));

    // Top zone (rows 0..60): only 2 black stripes → find_thick_stripe returns None.
    for &(start, end) in &[(0u32, 10u32), (20u32, 30u32)] {
        for row in start..end {
            img.put_pixel(0, row, Luma([0u8]));
        }
    }

    // Bottom zone (rows 140..200): 3 black stripes → enough for find_thick_stripe.
    for &(start, end) in &[(140u32, 150u32), (160u32, 170u32), (180u32, 190u32)] {
        for row in start..end {
            img.put_pixel(0, row, Luma([0u8]));
        }
    }

    let dyn_img = DynamicImage::ImageLuma8(img);

    // With only 2 runs in the top zone, find_thick_stripe(slice.len()=2) → None
    // → detect_markers_at_column should return an error.
    let result = detect_markers_at_column(&dyn_img, 0);
    assert!(
        result.is_err(),
        "fewer than 3 dark runs in the top zone should return an error"
    );
}

/// The `Display` impl for `PhonoPaperError::MarkerNotFound` includes the
/// reason string so callers can diagnose failures without inspecting internals.
#[test]
fn marker_not_found_display_includes_reason() {
    use phonopaper_rs::PhonoPaperError;
    let err = PhonoPaperError::MarkerNotFound("test reason");
    let msg = err.to_string();
    assert!(
        msg.contains("test reason"),
        "Display should include the reason string, got: {msg}"
    );
    assert!(
        msg.contains("marker"),
        "Display should mention markers, got: {msg}"
    );
}
