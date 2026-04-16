//! # robust\_decode — perspective-aware `PhonoPaper` decoder
//!
//! Decodes a `PhonoPaper` image taken by a camera, compensating for rotation,
//! tilt, keystone distortion, and paper curl.
//!
//! ## Strategy
//!
//! Rather than requiring a perfect axis-aligned scan, this tool uses
//! **per-column marker detection** (recommended in `PHONOPAPER_SPEC.md §11`):
//!
//! 1. Sample a set of evenly-spaced columns across the image width.
//! 2. For each sample column, call [`detect_markers_at_column`] to find that
//!    column's `data_top` and `data_bottom` independently.
//! 3. Linearly interpolate `data_top` / `data_bottom` for every image column.
//! 4. For each column, call [`column_amplitudes_from_image_into`] with the
//!    interpolated [`DataBounds`], so the pixel sampling automatically follows
//!    the slanted or curved marker lines.
//! 5. Synthesise audio from the resulting spectrogram and write a WAV file.
//!
//! This approach handles arbitrary keystone / trapezoid perspective, paper
//! curl, and rotation without requiring an explicit homography de-warp.
//!
//! ## Optional outputs
//!
//! * **`--debug-image <path.png>`** – saves a copy of the input image with the
//!   detected `data_top` and `data_bottom` polylines drawn in red, to allow
//!   visual inspection of the marker detection quality.
//!
//! * **`--rectified <path.png>`** – uses [`imageproc`]'s perspective warp to
//!   produce a flat rectangular image from the four corner points of the
//!   detected data area.  Useful for comparing the result to a canonical
//!   `PhonoPaper` image.
//!
//! ## Usage
//!
//! ```text
//! cargo run --example robust_decode -- \
//!     --input photo.jpg --output decoded.wav
//!
//! cargo run --example robust_decode -- \
//!     --input photo.jpg --output decoded.wav \
//!     --debug-image debug.png --rectified rectified.png
//! ```

use clap::Parser;
use image::{DynamicImage, GenericImageView as _, Rgba};
use imageproc::geometric_transformations::{Interpolation, Projection, warp};
use phonopaper_rs::decode::{
    AmplitudeMode, DataBounds, SynthesisOptions, column_amplitudes_from_image_into,
    detect_markers_at_column, spectrogram_to_audio,
};
use phonopaper_rs::format::TOTAL_BINS;
use phonopaper_rs::spectrogram::SpectrogramVec;
use std::io::{BufWriter, Write as _};
use std::path::Path;
use std::process;

// ─── CLI ─────────────────────────────────────────────────────────────────────

/// Decode a (possibly skewed or perspective-distorted) `PhonoPaper` image into
/// a WAV audio file.
///
/// Uses per-column marker detection so that keystone distortion, camera tilt,
/// and paper curl are handled without an explicit de-warp step.
#[derive(Parser)]
#[command(name = "robust_decode", version)]
struct Args {
    /// Input `PhonoPaper` image (PNG or JPEG), possibly taken by a camera.
    #[arg(short, long, value_name = "FILE")]
    input: String,

    /// Output WAV file path.
    #[arg(short, long, value_name = "FILE")]
    output: String,

    /// How many evenly-spaced columns to sample for marker detection.
    ///
    /// More samples give better compensation for perspective and curl at the
    /// cost of more [`detect_markers_at_column`] calls.  Defaults to `50`.
    #[arg(long, default_value_t = 50, value_name = "N")]
    sample_columns: u32,

    /// Master output gain (linear).
    #[arg(long, default_value_t = 3.0, value_name = "GAIN")]
    gain: f32,

    /// Output audio sample rate in Hz.
    #[arg(long, default_value_t = 44_100, value_name = "HZ")]
    sample_rate: u32,

    /// Number of PCM samples to synthesise per image column.
    #[arg(long, default_value_t = 353, value_name = "N")]
    samples_per_column: usize,

    /// Lower bound of the dB window (maps to white / silence).
    #[arg(long, default_value_t = -60.0, value_name = "DB")]
    min_db: f32,

    /// Upper bound of the dB window (maps to black / maximum amplitude).
    #[arg(long, default_value_t = -10.0, value_name = "DB")]
    max_db: f32,

    /// Save a debug PNG with the detected `data_top`/`data_bottom` lines
    /// overlaid on the input image.
    #[arg(long, value_name = "FILE")]
    debug_image: Option<String>,

    /// Save a perspective-rectified PNG of the data area.
    ///
    /// The four corner points of the detected data area are mapped to a
    /// canonical rectangle and warped using a projective transformation.
    #[arg(long, value_name = "FILE")]
    rectified: Option<String>,
}

// ─── Per-column marker detection ─────────────────────────────────────────────

/// Detect markers at a set of evenly-spaced sample columns and interpolate
/// `data_top` / `data_bottom` for every column.
///
/// Returns a vector of length `width`, where each entry is the
/// `(data_top, data_bottom)` pixel rows for that column.
///
/// # Errors
///
/// Returns an error if no single sample column succeeds — the image has no
/// detectable `PhonoPaper` marker pattern at all.
fn interpolate_bounds(
    image: &DynamicImage,
    sample_columns: u32,
) -> Result<Vec<(f32, f32)>, String> {
    let (width, _) = image.dimensions();

    // Choose the actual number of sample columns, clamped to the image width.
    let n_samples = sample_columns.min(width).max(2);

    // Build evenly-spaced sample x-coordinates: 0, step, 2*step, …, width-1.
    // Always include both the leftmost and rightmost column.
    let mut sample_xs: Vec<u32> = (0..n_samples)
        .map(|i| {
            // Maps i ∈ [0, n_samples-1] → x ∈ [0, width-1].
            // The result of `.round()` is always in [0.0, (width-1) as f64],
            // which fits in u32 without sign loss or truncation.
            #[expect(
                clippy::cast_possible_truncation,
                reason = "value is rounded and clamped to [0, width-1]; fits in u32"
            )]
            #[expect(
                clippy::cast_sign_loss,
                reason = ".round() on a non-negative f64 product is always ≥ 0.0"
            )]
            let x = (f64::from(i) / f64::from(n_samples - 1) * f64::from(width - 1)).round() as u32;
            x.min(width - 1)
        })
        .collect();

    // Deduplicate (may happen for very narrow images).
    sample_xs.dedup();

    // Run marker detection for each sample column; record successes.
    let mut detected: Vec<(u32, f32, f32)> = Vec::with_capacity(sample_xs.len()); // (x, top, bottom)
    let mut n_failed: usize = 0;

    for col_x in &sample_xs {
        match detect_markers_at_column(image, *col_x) {
            Ok(bounds) => {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "pixel coordinates converted to f32 for interpolation; \
                              loss is at most 1 ULP and has no perceptible audio effect"
                )]
                detected.push((*col_x, bounds.data_top as f32, bounds.data_bottom as f32));
            }
            Err(_) => {
                n_failed += 1;
            }
        }
    }

    if detected.is_empty() {
        return Err("No PhonoPaper marker pattern found in any sample column.".to_string());
    }

    // Warn if many columns failed (likely bad perspective or partial occlusion).
    let fail_pct = n_failed * 100 / sample_xs.len();
    if fail_pct > 20 {
        eprintln!(
            "  Warning: marker detection failed in {n_failed}/{} sample columns ({fail_pct}%).  \
             Results may be inaccurate.",
            sample_xs.len()
        );
    }

    // Interpolate / extrapolate bounds for every image column using piecewise
    // linear interpolation over the detected anchor points.
    //
    // For columns to the left of the first detected anchor, or to the right of
    // the last, we extrapolate using the nearest edge anchor.
    #[expect(
        clippy::cast_precision_loss,
        reason = "column index converted to f32 for interpolation arithmetic"
    )]
    let result: Vec<(f32, f32)> = (0..width)
        .map(|x| {
            let xf = x as f32;
            // Find the two anchors that bracket xf.
            let pos = detected.partition_point(|&(ax, _, _)| ax <= x);

            match pos {
                0 => {
                    // x is before the first anchor — use the first anchor's value.
                    (detected[0].1, detected[0].2)
                }
                p if p >= detected.len() => {
                    // x is after the last anchor — use the last anchor's value.
                    let last = detected.len() - 1;
                    (detected[last].1, detected[last].2)
                }
                p => {
                    // Linear interpolation between anchors[p-1] and anchors[p].
                    let (x0, t0, b0) = detected[p - 1];
                    let (x1, t1, b1) = detected[p];
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "anchor x-coordinates converted to f32 for interpolation"
                    )]
                    let t = (xf - x0 as f32) / (x1 as f32 - x0 as f32);
                    (t0 + t * (t1 - t0), b0 + t * (b1 - b0))
                }
            }
        })
        .collect();

    Ok(result)
}

// ─── Audio synthesis ─────────────────────────────────────────────────────────

/// Build a [`SpectrogramVec`] from the image using per-column [`DataBounds`].
///
/// # Errors
///
/// Returns an error if amplitude extraction fails for any column.
fn build_spectrogram(
    image: &DynamicImage,
    col_bounds: &[(f32, f32)],
) -> Result<SpectrogramVec, phonopaper_rs::PhonoPaperError> {
    let width = col_bounds.len();
    let mut spec = SpectrogramVec::new(width);

    let mut amp_buf = [0.0_f32; TOTAL_BINS];

    for (col_x, &(top_f, bottom_f)) in col_bounds.iter().enumerate() {
        // Round the interpolated floats to the nearest pixel row.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "interpolated pixel row, clamped below to be non-negative; \
                      truncation towards zero is correct rounding for pixel indices"
        )]
        #[expect(
            clippy::cast_sign_loss,
            reason = "values are clamped to ≥ 0.0 before conversion"
        )]
        let data_top = top_f.max(0.0).round() as u32;
        #[expect(clippy::cast_possible_truncation, reason = "same as data_top")]
        #[expect(
            clippy::cast_sign_loss,
            reason = "values are clamped to ≥ 0.0 before conversion"
        )]
        let data_bottom = bottom_f.max(0.0).round() as u32;

        let bounds = DataBounds {
            data_top,
            data_bottom,
        };

        #[expect(
            clippy::cast_possible_truncation,
            reason = "col_x iterates over 0..width; width comes from col_bounds.len() which \
                      was derived from image.width() (a u32), so col_x fits in u32"
        )]
        column_amplitudes_from_image_into(image, Some(bounds), col_x as u32, &mut amp_buf)?;
        spec.column_mut(col_x).copy_from_slice(&amp_buf);
    }

    Ok(spec)
}

// ─── WAV writer ──────────────────────────────────────────────────────────────

/// Write a mono 16-bit PCM WAV file.
///
/// # Errors
///
/// Returns an `std::io::Error` if any I/O operation fails.
fn write_wav(path: impl AsRef<Path>, samples: &[f32], sample_rate: u32) -> std::io::Result<()> {
    let num_samples = u32::try_from(samples.len()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "sample buffer too large for WAV",
        )
    })?;
    let data_bytes = num_samples * 2;
    let fmt_chunk_size: u32 = 16;
    let file_size = 4 + (8 + fmt_chunk_size) + (8 + data_bytes);

    let file = std::fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    w.write_all(b"RIFF")?;
    w.write_all(&file_size.to_le_bytes())?;
    w.write_all(b"WAVE")?;
    w.write_all(b"fmt ")?;
    w.write_all(&fmt_chunk_size.to_le_bytes())?;
    w.write_all(&1u16.to_le_bytes())?; // PCM
    w.write_all(&1u16.to_le_bytes())?; // mono
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&(sample_rate * 2).to_le_bytes())?;
    w.write_all(&2u16.to_le_bytes())?; // block align
    w.write_all(&16u16.to_le_bytes())?; // bits per sample
    w.write_all(b"data")?;
    w.write_all(&data_bytes.to_le_bytes())?;

    for &s in samples {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "rounded value is clamped to [-32767.5, 32767.5] and rounded, so the result fits in i16"
        )]
        let word: i16 = (s.clamp(-1.0, 1.0) * 32_767.5).round() as i16;
        w.write_all(&word.to_le_bytes())?;
    }

    Ok(())
}

// ─── Debug image ─────────────────────────────────────────────────────────────

/// Overlay the per-column `data_top` and `data_bottom` lines onto the image
/// and save the result as a PNG.
///
/// The detected top boundary is drawn in red and the bottom boundary in blue.
///
/// # Errors
///
/// Returns an error if the image cannot be saved.
fn save_debug_image(
    image: &DynamicImage,
    col_bounds: &[(f32, f32)],
    path: impl AsRef<Path>,
) -> image::ImageResult<()> {
    let mut rgba = image.to_rgba8();

    for (col_x, &(top_f, bottom_f)) in col_bounds.iter().enumerate() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "col_x ≤ image.width() which fits in u32"
        )]
        let x = col_x as u32;
        let (_, height) = image.dimensions();

        #[expect(
            clippy::cast_possible_truncation,
            reason = "top_f is an interpolated pixel row derived from u32 image height; \
                      rounded and clamped, so it fits in u32"
        )]
        #[expect(
            clippy::cast_sign_loss,
            reason = "top_f comes from u32 pixel rows; the rounded value is always ≥ 0"
        )]
        let top_y = (top_f.max(0.0).round() as u32).min(height.saturating_sub(1));
        #[expect(
            clippy::cast_possible_truncation,
            reason = "bottom_f is an interpolated pixel row; rounded and clamped to image bounds"
        )]
        #[expect(
            clippy::cast_sign_loss,
            reason = "clamped via .max(0.0) before conversion"
        )]
        let bottom_y = (bottom_f.max(0.0).round() as u32).min(height.saturating_sub(1));

        rgba.put_pixel(x, top_y, Rgba([255, 0, 0, 255])); // red = top
        rgba.put_pixel(x, bottom_y, Rgba([0, 0, 255, 255])); // blue = bottom
    }

    DynamicImage::ImageRgba8(rgba).save(path)
}

// ─── Perspective rectification ───────────────────────────────────────────────

/// Save a perspective-rectified PNG of the data area.
///
/// The four corners of the detected data area are:
/// - top-left: column 0, `data_top[0]`
/// - top-right: column width-1, `data_top[width-1]`
/// - bottom-right: column width-1, `data_bottom[width-1]`
/// - bottom-left: column 0, `data_bottom[0]`
///
/// These are mapped to a canonical rectangle of the same width and the median
/// data-area height via a projective transformation.
///
/// # Errors
///
/// Returns a string describing the error if the projection cannot be computed
/// or the image cannot be saved.
fn save_rectified_image(
    image: &DynamicImage,
    col_bounds: &[(f32, f32)],
    path: impl AsRef<Path>,
) -> Result<(), String> {
    if col_bounds.len() < 2 {
        return Err("Image too narrow to rectify.".to_string());
    }

    let (width, _) = image.dimensions();
    #[expect(
        clippy::cast_precision_loss,
        reason = "width - 1 is a pixel coordinate; converted to f32 for imageproc projection \
                  arithmetic.  Images wide enough to lose precision (> 16M px) are not a \
                  practical concern."
    )]
    let w_f = (width - 1) as f32;

    let last = col_bounds.len() - 1;
    let (top_left_y, bot_left_y) = col_bounds[0];
    let (top_right_y, bot_right_y) = col_bounds[last];

    // Compute the median height from all detected column heights.
    let mut heights: Vec<f32> = col_bounds.iter().map(|&(t, b)| (b - t).max(0.0)).collect();
    heights.sort_by(f32::total_cmp);
    let median_height = heights[heights.len() / 2];
    if median_height < 1.0 {
        return Err("Detected data area has zero height; cannot rectify.".to_string());
    }

    // Output canvas size.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "median_height is a pixel row count derived from u32 image dimensions, \
                  so it fits safely in u32"
    )]
    #[expect(
        clippy::cast_sign_loss,
        reason = "median_height is clamped to ≥ 1.0 above"
    )]
    let out_h = median_height.round() as u32;
    let out_w = width;

    // The projection maps FROM the output (destination) space TO the input
    // (source) space — imageproc's warp function uses an *inverse* mapping.
    //
    // Destination corners (flat canonical rectangle):
    //   TL=(0, 0), TR=(out_w-1, 0), BR=(out_w-1, out_h-1), BL=(0, out_h-1)
    //
    // Source corners (distorted data area in the original image):
    //   TL=(0, top_left_y), TR=(w_f, top_right_y),
    //   BR=(w_f, bot_right_y), BL=(0, bot_left_y)
    //
    // from_control_points(from, to) builds a projection that maps each
    // `from[i]` point to `to[i]`.  Here `from` = destination corners and
    // `to` = source corners, giving us the inverse (source-lookup) warp.
    #[expect(
        clippy::cast_precision_loss,
        reason = "output canvas dimensions converted to f32 for imageproc projection points; \
                  images larger than 16M px are not a practical concern"
    )]
    let dst_right = (out_w - 1) as f32;
    #[expect(
        clippy::cast_precision_loss,
        reason = "same rationale as dst_right; output height converted to f32 for projection"
    )]
    let dst_bottom = (out_h - 1) as f32;

    let from_pts = [
        (0.0_f32, 0.0_f32),
        (dst_right, 0.0_f32),
        (dst_right, dst_bottom),
        (0.0_f32, dst_bottom),
    ];
    let to_pts = [
        (0.0_f32, top_left_y),
        (w_f, top_right_y),
        (w_f, bot_right_y),
        (0.0_f32, bot_left_y),
    ];

    let projection = Projection::from_control_points(from_pts, to_pts).ok_or_else(|| {
        "Could not compute perspective projection (degenerate geometry).".to_string()
    })?;

    let rgb_input = image.to_rgb8();
    let warped = warp(
        &rgb_input,
        &projection,
        Interpolation::Bilinear,
        image::Rgb([255u8, 255, 255]),
    );
    warped.save(path).map_err(|e| e.to_string())
}

// ─── Entry point ─────────────────────────────────────────────────────────────

fn run(args: &Args) -> Result<(), String> {
    // 1. Load the input image.
    eprintln!("Loading  {} …", args.input);
    let image = image::open(&args.input).map_err(|e| format!("Cannot open image: {e}"))?;
    let (width, height) = image.dimensions();
    eprintln!("  {width} × {height} pixels");

    // 2. Detect marker bands with per-column sampling.
    eprintln!(
        "Detecting markers (sampling {} columns) …",
        args.sample_columns
    );
    let col_bounds = interpolate_bounds(&image, args.sample_columns)?;

    // Print a summary of the detected bounds.
    let tops: Vec<f32> = col_bounds.iter().map(|&(t, _)| t).collect();
    let bots: Vec<f32> = col_bounds.iter().map(|&(_, b)| b).collect();
    #[expect(
        clippy::cast_precision_loss,
        reason = "col_bounds.len() is image width (u32); f64 precision is sufficient for a mean"
    )]
    let mean_top = tops.iter().copied().map(f64::from).sum::<f64>() / col_bounds.len() as f64;
    #[expect(clippy::cast_precision_loss, reason = "same as mean_top")]
    let mean_bot = bots.iter().copied().map(f64::from).sum::<f64>() / col_bounds.len() as f64;
    eprintln!(
        "  mean data_top={mean_top:.1}px  mean data_bottom={mean_bot:.1}px  \
         mean height={:.1}px",
        mean_bot - mean_top
    );

    // 3. Optionally save the debug image.
    if let Some(ref debug_path) = args.debug_image {
        eprintln!("Saving debug image → {debug_path} …");
        save_debug_image(&image, &col_bounds, debug_path)
            .map_err(|e| format!("Cannot save debug image: {e}"))?;
    }

    // 4. Optionally save the rectified image.
    if let Some(ref rect_path) = args.rectified {
        eprintln!("Saving rectified image → {rect_path} …");
        save_rectified_image(&image, &col_bounds, rect_path)?;
    }

    // 5. Build the spectrogram column by column.
    eprintln!("Building spectrogram ({} columns) …", col_bounds.len());
    let spec = build_spectrogram(&image, &col_bounds)
        .map_err(|e| format!("Spectrogram build failed: {e}"))?;

    // 6. Synthesise audio.
    let options = SynthesisOptions {
        sample_rate: args.sample_rate,
        gain: args.gain,
        decode_gamma: 1.0,
        mode: AmplitudeMode::Db {
            min_db: args.min_db,
            max_db: args.max_db,
        },
    };
    eprintln!(
        "Synthesising audio ({} columns × {} samples/col = {} samples at {} Hz) …",
        spec.num_columns(),
        args.samples_per_column,
        spec.num_columns() * args.samples_per_column,
        args.sample_rate,
    );

    let num_samples = spec.num_columns() * args.samples_per_column;
    let mut samples = vec![0.0_f32; num_samples];

    // SPS must be a const generic.  Map the runtime value to supported choices.
    match args.samples_per_column {
        n if n <= 256 => spectrogram_to_audio::<_, 256>(&spec, &options, &mut samples),
        n if n <= 353 => spectrogram_to_audio::<_, 353>(&spec, &options, &mut samples),
        n if n <= 384 => spectrogram_to_audio::<_, 384>(&spec, &options, &mut samples),
        n if n <= 512 => spectrogram_to_audio::<_, 512>(&spec, &options, &mut samples),
        n if n <= 1024 => spectrogram_to_audio::<_, 1024>(&spec, &options, &mut samples),
        _ => spectrogram_to_audio::<_, 2048>(&spec, &options, &mut samples),
    }

    // 7. Write WAV.
    eprintln!("Writing WAV → {} …", args.output);
    write_wav(&args.output, &samples, args.sample_rate)
        .map_err(|e| format!("Cannot write WAV: {e}"))?;

    eprintln!("Done.");
    Ok(())
}

fn main() {
    let args = Args::parse();
    if let Err(e) = run(&args) {
        eprintln!("error: {e}");
        process::exit(1);
    }
}
