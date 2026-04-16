//! `phonopaper robust-decode` — decode a possibly distorted camera photo of a
//! `PhonoPaper` print.

use std::path::Path;

use clap::Args;
use image::DynamicImage;

use crate::cmd::dispatch_sps_synth;

/// Arguments for the `robust-decode` sub-command.
#[derive(Args)]
pub struct RobustDecodeArgs {
    /// Input `PhonoPaper` image (PNG or JPEG), possibly taken by a camera.
    #[arg(short, long, value_name = "FILE")]
    pub input: String,

    /// Output WAV file path.
    #[arg(short, long, value_name = "FILE")]
    pub output: String,

    /// How many evenly-spaced columns to sample for marker detection.
    ///
    /// More samples give better compensation for perspective and curl at the
    /// cost of more `detect_markers_at_column` calls.  Defaults to `50`.
    #[arg(long, default_value_t = 50, value_name = "N")]
    pub sample_columns: u32,

    /// Master output gain (linear).
    #[arg(long, default_value_t = 3.0, value_name = "GAIN")]
    pub gain: f32,

    /// Output audio sample rate in Hz.
    #[arg(long, default_value_t = 44_100, value_name = "HZ")]
    pub sample_rate: u32,

    /// Number of PCM samples to synthesise per image column.
    #[arg(long, default_value_t = 353, value_name = "N")]
    pub samples_per_column: usize,

    /// Lower bound of the dB window (maps to white / silence).
    #[arg(long, default_value_t = -60.0, value_name = "DB")]
    pub min_db: f32,

    /// Upper bound of the dB window (maps to black / maximum amplitude).
    #[arg(long, default_value_t = -10.0, value_name = "DB")]
    pub max_db: f32,

    /// Binary amplitude threshold (e.g. 0.85 for Android-like decoding).
    ///
    /// When set, the decoder uses [`phonopaper_rs::decode::AmplitudeMode::Linear`]
    /// and treats each pixel as either fully on (amplitude ≥ threshold) or
    /// silent (amplitude < threshold).
    ///
    /// Default: disabled (uses `--min-db` / `--max-db` dB window instead).
    #[arg(long, value_name = "T", conflicts_with = "linear")]
    pub amplitude_threshold: Option<f32>,

    /// Use linear (non-dB) amplitude mode with fractional pixel values.
    ///
    /// Selects [`phonopaper_rs::decode::AmplitudeMode::Linear`] with no
    /// threshold.  Ignored if `--amplitude-threshold` is set.
    #[arg(long)]
    pub linear: bool,

    /// Save a debug PNG with the detected `data_top`/`data_bottom` lines overlaid.
    #[arg(long, value_name = "FILE")]
    pub debug_image: Option<String>,

    /// Save a perspective-rectified PNG of the data area.
    #[arg(long, value_name = "FILE")]
    pub rectified: Option<String>,
}

/// Detect markers at evenly-spaced sample columns and interpolate
/// `data_top` / `data_bottom` for every image column.
///
/// Returns a `Vec` of length `width`, where each entry is the interpolated
/// `(data_top, data_bottom)` pixel rows for that column.
///
/// # Errors
///
/// Returns an error string if no sample column has a detectable marker pattern.
fn interpolate_bounds(
    image: &DynamicImage,
    sample_columns: u32,
) -> Result<Vec<(f32, f32)>, String> {
    use image::GenericImageView as _;
    use phonopaper_rs::decode::detect_markers_at_column;

    let (width, _) = image.dimensions();
    let n_samples = sample_columns.min(width).max(2);

    let mut sample_xs: Vec<u32> = (0..n_samples)
        .map(|i| {
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
    sample_xs.dedup();

    let mut detected: Vec<(u32, f32, f32)> = Vec::with_capacity(sample_xs.len());
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
            Err(_) => n_failed += 1,
        }
    }

    if detected.is_empty() {
        return Err("No PhonoPaper marker pattern found in any sample column.".to_string());
    }

    let fail_pct = n_failed * 100 / sample_xs.len();
    if fail_pct > 20 {
        eprintln!(
            "  Warning: marker detection failed in {n_failed}/{} sample columns ({fail_pct}%). \
             Results may be inaccurate.",
            sample_xs.len()
        );
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "column index converted to f32 for interpolation arithmetic"
    )]
    let result: Vec<(f32, f32)> = (0..width)
        .map(|x| {
            let xf = x as f32;
            let pos = detected.partition_point(|&(ax, _, _)| ax <= x);
            match pos {
                0 => (detected[0].1, detected[0].2),
                p if p >= detected.len() => {
                    let last = detected.len() - 1;
                    (detected[last].1, detected[last].2)
                }
                p => {
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

/// Build a [`phonopaper_rs::SpectrogramVec`] from the image using per-column
/// [`phonopaper_rs::decode::DataBounds`].
///
/// # Errors
///
/// Returns an error if amplitude extraction fails for any column.
fn build_spectrogram(
    image: &DynamicImage,
    col_bounds: &[(f32, f32)],
) -> Result<phonopaper_rs::SpectrogramVec, phonopaper_rs::PhonoPaperError> {
    use phonopaper_rs::decode::{DataBounds, column_amplitudes_from_image_into};
    use phonopaper_rs::format::TOTAL_BINS;

    let width = col_bounds.len();
    let mut spec = phonopaper_rs::SpectrogramVec::new(width);
    let mut amp_buf = [0.0_f32; TOTAL_BINS];

    for (col_x, &(top_f, bottom_f)) in col_bounds.iter().enumerate() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "interpolated pixel row, clamped to ≥ 0; truncation is correct for pixel indices"
        )]
        #[expect(
            clippy::cast_sign_loss,
            reason = "values are clamped to ≥ 0.0 before conversion"
        )]
        let data_top = top_f.max(0.0).round() as u32;
        #[expect(clippy::cast_possible_truncation, reason = "same as data_top")]
        #[expect(clippy::cast_sign_loss, reason = "clamped via .max(0.0)")]
        let data_bottom = bottom_f.max(0.0).round() as u32;
        let bounds = DataBounds {
            data_top,
            data_bottom,
        };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "col_x iterates over 0..width where width ≤ image.width() (u32)"
        )]
        column_amplitudes_from_image_into(image, Some(bounds), col_x as u32, &mut amp_buf)?;
        spec.column_mut(col_x).copy_from_slice(&amp_buf);
    }

    Ok(spec)
}

/// Overlay the per-column boundary lines onto the image and save as PNG.
///
/// # Errors
///
/// Returns an [`image::ImageError`] if the file cannot be saved.
fn save_debug_image(
    image: &DynamicImage,
    col_bounds: &[(f32, f32)],
    path: impl AsRef<Path>,
) -> image::ImageResult<()> {
    use image::{GenericImageView as _, Rgba};

    let mut rgba = image.to_rgba8();
    let (_, height) = image.dimensions();

    for (col_x, &(top_f, bottom_f)) in col_bounds.iter().enumerate() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "col_x ≤ image.width() which fits in u32"
        )]
        let x = col_x as u32;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "top_f is an interpolated pixel row derived from u32 image height; rounded and clamped"
        )]
        #[expect(clippy::cast_sign_loss, reason = "clamped via .max(0.0)")]
        let top_y = (top_f.max(0.0).round() as u32).min(height.saturating_sub(1));
        #[expect(clippy::cast_possible_truncation, reason = "same as top_y")]
        #[expect(clippy::cast_sign_loss, reason = "clamped via .max(0.0)")]
        let bottom_y = (bottom_f.max(0.0).round() as u32).min(height.saturating_sub(1));
        rgba.put_pixel(x, top_y, Rgba([255, 0, 0, 255]));
        rgba.put_pixel(x, bottom_y, Rgba([0, 0, 255, 255]));
    }

    image::DynamicImage::ImageRgba8(rgba).save(path)
}

/// Save a perspective-rectified PNG of the data area.
///
/// # Errors
///
/// Returns a string error if the projection is degenerate or the file cannot
/// be saved.
fn save_rectified_image(
    img: &DynamicImage,
    col_bounds: &[(f32, f32)],
    path: impl AsRef<Path>,
) -> Result<(), String> {
    use image::GenericImageView as _;
    use imageproc::geometric_transformations::{Interpolation, Projection, warp};

    if col_bounds.len() < 2 {
        return Err("Image too narrow to rectify.".to_string());
    }

    let (width, _) = img.dimensions();
    #[expect(
        clippy::cast_precision_loss,
        reason = "width - 1 is a pixel coordinate; converted to f32 for imageproc projection arithmetic"
    )]
    let w_f = (width - 1) as f32;

    let last = col_bounds.len() - 1;
    let (top_left_y, bot_left_y) = col_bounds[0];
    let (top_right_y, bot_right_y) = col_bounds[last];

    let mut heights: Vec<f32> = col_bounds.iter().map(|&(t, b)| (b - t).max(0.0)).collect();
    heights.sort_by(f32::total_cmp);
    let median_height = heights[heights.len() / 2];
    if median_height < 1.0 {
        return Err("Detected data area has zero height; cannot rectify.".to_string());
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "median_height is a pixel row count derived from u32 image dimensions"
    )]
    #[expect(clippy::cast_sign_loss, reason = "median_height is clamped to ≥ 1.0")]
    let out_h = median_height.round() as u32;
    let out_w = width;

    #[expect(
        clippy::cast_precision_loss,
        reason = "output canvas dimensions converted to f32 for imageproc projection points"
    )]
    let dst_right = (out_w - 1) as f32;
    #[expect(
        clippy::cast_precision_loss,
        reason = "output canvas height converted to f32 for imageproc projection points"
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

    let rgb_input = img.to_rgb8();
    let warped = warp(
        &rgb_input,
        &projection,
        Interpolation::Bilinear,
        image::Rgb([255u8, 255, 255]),
    );
    warped.save(path).map_err(|e| e.to_string())
}

/// Run the `robust-decode` subcommand.
///
/// # Errors
///
/// Returns a [`phonopaper_rs::PhonoPaperError`] if the image cannot be loaded,
/// marker detection finds no valid columns, spectrogram extraction fails, or
/// the output WAV file cannot be written.
pub fn run(args: &RobustDecodeArgs) -> phonopaper_rs::Result<()> {
    use image::GenericImageView as _;
    use phonopaper_rs::decode::{AmplitudeMode, SynthesisOptions, write_wav};

    eprintln!("Loading  {} …", args.input);
    let img = image::open(&args.input).map_err(phonopaper_rs::PhonoPaperError::ImageError)?;
    let (width, height) = img.dimensions();
    eprintln!("  {width} × {height} pixels");

    eprintln!(
        "Detecting markers (sampling {} columns) …",
        args.sample_columns
    );
    let col_bounds = interpolate_bounds(&img, args.sample_columns)
        .map_err(phonopaper_rs::PhonoPaperError::InvalidFormat)?;

    #[expect(
        clippy::cast_precision_loss,
        reason = "col_bounds.len() is image width (u32); f64 precision is sufficient for a mean"
    )]
    let mean_top =
        col_bounds.iter().map(|&(t, _)| f64::from(t)).sum::<f64>() / col_bounds.len() as f64;
    #[expect(clippy::cast_precision_loss, reason = "same as mean_top")]
    let mean_bot =
        col_bounds.iter().map(|&(_, b)| f64::from(b)).sum::<f64>() / col_bounds.len() as f64;
    eprintln!(
        "  mean data_top={mean_top:.1}px  mean data_bottom={mean_bot:.1}px  \
         mean height={:.1}px",
        mean_bot - mean_top
    );

    if let Some(ref debug_path) = args.debug_image {
        eprintln!("Saving debug image → {debug_path} …");
        save_debug_image(&img, &col_bounds, debug_path)
            .map_err(phonopaper_rs::PhonoPaperError::ImageError)?;
    }

    if let Some(ref rect_path) = args.rectified {
        eprintln!("Saving rectified image → {rect_path} …");
        save_rectified_image(&img, &col_bounds, rect_path)
            .map_err(phonopaper_rs::PhonoPaperError::InvalidFormat)?;
    }

    eprintln!("Building spectrogram ({} columns) …", col_bounds.len());
    let spec = build_spectrogram(&img, &col_bounds)?;

    let mode = if let Some(t) = args.amplitude_threshold {
        AmplitudeMode::Linear { threshold: Some(t) }
    } else if args.linear {
        AmplitudeMode::Linear { threshold: None }
    } else {
        AmplitudeMode::Db {
            min_db: args.min_db,
            max_db: args.max_db,
        }
    };

    let options = SynthesisOptions {
        sample_rate: args.sample_rate,
        gain: args.gain,
        decode_gamma: 1.0,
        mode,
    };
    let num_samples = spec.num_columns() * args.samples_per_column;
    eprintln!(
        "Synthesising audio ({} columns × {} samples/col = {} samples at {} Hz) …",
        spec.num_columns(),
        args.samples_per_column,
        num_samples,
        args.sample_rate,
    );
    let mut samples = vec![0.0_f32; num_samples];
    dispatch_sps_synth!(args.samples_per_column, &spec, &options, &mut samples);

    // Write mono 16-bit PCM WAV.
    eprintln!("Writing WAV → {} …", args.output);
    write_wav(&args.output, &samples, args.sample_rate)?;

    eprintln!("Done.");
    Ok(())
}
