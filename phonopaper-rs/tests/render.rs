//! Tests for the `render` module:
//! [`RenderOptions::image_height`], [`image_buf_size`],
//! [`spectrogram_to_image_buf`], [`RenderOptions::draw_octave_lines`],
//! and [`RenderOptions::gamma`].

use phonopaper_rs::format::OCTAVES;
use phonopaper_rs::render::{
    RenderOptions, image_buf_size, spectrogram_to_image, spectrogram_to_image_buf,
};
use phonopaper_rs::spectrogram::SpectrogramVec;

// ─── helpers ─────────────────────────────────────────────────────────────────

/// A silent (all-zero) spectrogram with `n` columns.
fn silent_spec(n: usize) -> SpectrogramVec {
    SpectrogramVec::new(n)
}

// ─── RenderOptions::image_height ─────────────────────────────────────────────

/// `image_height()` equals `2 * marker_band_height() + px_per_octave * OCTAVES`.
#[test]
fn image_height_equals_two_bands_plus_data() {
    let opts = RenderOptions::default();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "OCTAVES is the compile-time constant 8, which fits trivially in u32"
    )]
    let expected = 2 * opts.marker_band_height() + opts.px_per_octave * OCTAVES as u32;
    assert_eq!(
        opts.image_height(),
        expected,
        "image_height should be 2 × marker_band_height + data_height"
    );
}

/// `image_height()` is correct for non-default `RenderOptions`.
#[test]
fn image_height_custom_options() {
    let opts = RenderOptions {
        px_per_octave: 45,
        thin_stripe: 5,
        thick_stripe: 21,
        marker_gap: 6,
        margin: 40,
        ..RenderOptions::default()
    };
    #[expect(
        clippy::cast_possible_truncation,
        reason = "OCTAVES is the compile-time constant 8, which fits trivially in u32"
    )]
    let expected = 2 * opts.marker_band_height() + opts.px_per_octave * OCTAVES as u32;
    assert_eq!(opts.image_height(), expected);
}

// ─── image_buf_size ───────────────────────────────────────────────────────────

/// `image_buf_size` returns `num_columns * opts.image_height()`.
#[test]
fn image_buf_size_matches_formula() {
    let opts = RenderOptions::default();
    let n = 7;
    assert_eq!(
        image_buf_size(n, &opts),
        n * opts.image_height() as usize,
        "image_buf_size should equal num_columns * image_height"
    );
}

/// `image_buf_size` for zero columns is zero.
#[test]
fn image_buf_size_zero_columns() {
    assert_eq!(image_buf_size(0, &RenderOptions::default()), 0);
}

// ─── spectrogram_to_image_buf ────────────────────────────────────────────────

/// The white margin rows of the top marker band are all 255 (white).
#[test]
fn image_buf_top_margin_is_white() {
    let opts = RenderOptions::default();
    let n = 3;
    let mut buf = vec![0u8; image_buf_size(n, &opts)];
    spectrogram_to_image_buf(&silent_spec(n), &opts, &mut buf);

    // The first `margin` rows (row-major) should be entirely white.
    let margin = opts.margin as usize;
    for row in 0..margin {
        for col in 0..n {
            assert_eq!(
                buf[row * n + col],
                255,
                "top margin row {row} col {col} should be 255"
            );
        }
    }
}

/// A bin set to amplitude 1.0 renders as pixel 0 (black);
/// a bin set to amplitude 0.0 renders as pixel 255 (white).
#[test]
fn image_buf_amplitude_to_pixel_mapping() {
    let opts = RenderOptions::default();
    let mut spec = SpectrogramVec::new(1);

    // Set bin 0 (top of data area, highest frequency) to full amplitude.
    spec.set(0, 0, 1.0);

    let mut buf = vec![0u8; image_buf_size(1, &opts)];
    spectrogram_to_image_buf(&spec, &opts, &mut buf);

    // Find the data area start row.
    let data_y = opts.marker_band_height() as usize;

    // The very first data row maps to bin 0.  With amplitude 1.0 and gamma
    // 1.0, amplitude_to_luma = round((1 - 1.0) * 255) = 0.
    assert_eq!(
        buf[data_y], 0,
        "amplitude 1.0 should render as pixel 0 (black)"
    );

    // The last bin (lowest frequency) was not set, so it stays at 0.0 →
    // pixel 255.
    let last_data_row = data_y + opts.px_per_octave as usize * OCTAVES - 1;
    assert_eq!(
        buf[last_data_row], 255,
        "amplitude 0.0 should render as pixel 255 (white)"
    );
}

/// The output buffer length must equal `image_buf_size`; passing a correctly-
/// sized buffer does not panic.
#[test]
fn image_buf_correct_size_does_not_panic() {
    let opts = RenderOptions::default();
    let n = 5;
    let mut buf = vec![0u8; image_buf_size(n, &opts)];
    // Should not panic.
    spectrogram_to_image_buf(&silent_spec(n), &opts, &mut buf);
}

/// Passing a wrongly-sized buffer panics.
#[test]
#[should_panic(expected = "must equal")]
fn image_buf_wrong_size_panics() {
    let opts = RenderOptions::default();
    let mut buf = vec![0u8; 1]; // too short
    spectrogram_to_image_buf(&silent_spec(1), &opts, &mut buf);
}

// ─── draw_octave_lines ───────────────────────────────────────────────────────

/// When `draw_octave_lines` is `true`, every octave-boundary row in the data
/// area has luma value 200 (the separator colour).
#[test]
fn draw_octave_lines_sets_separator_rows() {
    let opts = RenderOptions {
        draw_octave_lines: true,
        ..RenderOptions::default()
    };
    let n = 4;
    let mut buf = vec![0u8; image_buf_size(n, &opts)];
    spectrogram_to_image_buf(&silent_spec(n), &opts, &mut buf);

    let data_y = opts.marker_band_height() as usize;

    for octave in 1..OCTAVES {
        let sep_row = data_y + octave * opts.px_per_octave as usize;
        for col in 0..n {
            assert_eq!(
                buf[sep_row * n + col],
                200,
                "octave separator at row {sep_row} col {col} should be luma 200"
            );
        }
    }
}

/// When `draw_octave_lines` is `false` (the default), no row in the data area
/// has luma 200.
#[test]
fn no_octave_lines_by_default() {
    let opts = RenderOptions {
        draw_octave_lines: false,
        ..RenderOptions::default()
    };
    let n = 4;
    let mut buf = vec![0u8; image_buf_size(n, &opts)];
    spectrogram_to_image_buf(&silent_spec(n), &opts, &mut buf);

    let data_y = opts.marker_band_height() as usize;
    let data_end = data_y + opts.px_per_octave as usize * OCTAVES;
    let data_pixels = &buf[data_y * n..data_end * n];

    assert!(
        data_pixels.iter().all(|&p| p != 200),
        "no pixel should have luma 200 when draw_octave_lines is false"
    );
}

// ─── gamma encoding ───────────────────────────────────────────────────────────

/// The gamma power curve affects mid-level amplitudes in the expected direction:
///
/// - `gamma < 1.0` (e.g. `0.5`): `amp^0.5 > amp` for `amp ∈ (0,1)`, so
///   `1 − amp^0.5 < 1 − amp`, yielding a *darker* pixel (lower luma).
///   Quiet bins appear louder in the printed image.
///
/// - `gamma > 1.0` (e.g. `2.0`): `amp^2.0 < amp` for `amp ∈ (0,1)`, so
///   `1 − amp^2.0 > 1 − amp`, yielding a *lighter* pixel (higher luma).
///   Mid-level bins are compressed toward silence/white.
#[test]
fn gamma_power_curve_affects_midtone_pixels() {
    let amplitude = 0.4_f32;

    let render = |gamma: f32| -> u8 {
        let opts = RenderOptions {
            gamma,
            ..RenderOptions::default()
        };
        let mut spec = SpectrogramVec::new(1);
        spec.set(0, 0, amplitude);
        let mut buf = vec![0u8; image_buf_size(1, &opts)];
        spectrogram_to_image_buf(&spec, &opts, &mut buf);
        buf[opts.marker_band_height() as usize] // first data row = bin 0
    };

    let luma_compress = render(2.0); // gamma > 1: amp^2 = 0.16 → luma ≈ 219
    let luma_flat = render(1.0); //     gamma = 1: amp^1 = 0.40 → luma ≈ 153
    let luma_expand = render(0.5); //   gamma < 1: amp^½ ≈ 0.63 → luma ≈ 97

    // Darker pixel = lower luma = louder perceived signal.
    assert!(
        luma_expand < luma_flat,
        "gamma=0.5 (expand) should give a darker pixel ({luma_expand}) than gamma=1.0 ({luma_flat})"
    );
    assert!(
        luma_flat < luma_compress,
        "gamma=1.0 should give a darker pixel ({luma_flat}) than gamma=2.0 (compress) ({luma_compress})"
    );
}

/// With `gamma = 1.0`, the pixel value equals `round((1 - amplitude) * 255)`.
#[test]
fn gamma_one_pixel_formula() {
    let opts = RenderOptions {
        gamma: 1.0,
        ..RenderOptions::default()
    };

    let amplitude = 0.6_f32;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "value is in [0.0, 255.0] after clamp, fits in u8"
    )]
    #[expect(
        clippy::cast_sign_loss,
        reason = "(1.0 - amplitude) is always non-negative for amplitude in [0.0, 1.0]"
    )]
    let expected_luma = ((1.0 - amplitude) * 255.0).round() as u8;

    let mut spec = SpectrogramVec::new(1);
    spec.set(0, 0, amplitude);

    let mut buf = vec![0u8; image_buf_size(1, &opts)];
    spectrogram_to_image_buf(&spec, &opts, &mut buf);

    let data_y = opts.marker_band_height() as usize;
    assert_eq!(
        buf[data_y], expected_luma,
        "gamma=1 pixel should equal round((1-amp)*255)"
    );
}

/// `spectrogram_to_image` produces an image whose dimensions match
/// `image_height()` and `num_columns`.
#[test]
fn spectrogram_to_image_dimensions() {
    let opts = RenderOptions::default();
    let n = 5;
    let img = spectrogram_to_image(&silent_spec(n), &opts);
    assert_eq!(img.width() as usize, n);
    assert_eq!(img.height(), opts.image_height());
}
