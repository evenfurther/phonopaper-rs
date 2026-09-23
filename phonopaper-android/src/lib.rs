//! Android JNI bridge for `phonopaper-rs`.
//!
//! The Android UI stays thin and delegates all `PhonoPaper` decoding work to
//! Rust so the mobile application can share the same decoding logic as the
//! workspace library.
//!
//! # Pipeline
//!
//! 1. The neural-network detector (`phonopaper_rs::decode::nn`) tells whether
//!    the frame contains a sheet and where its four corners are, whatever its
//!    rotation and perspective.  The live preview overlay shows that
//!    quadrilateral ([`detect_pattern_corners`]).
//! 2. For decoding, the quadrilateral is **rectified**: a homography maps it to
//!    an upright rectangle that is resampled from the frame
//!    ([`rectify_pattern`]).
//! 3. The existing per-column stripe detector then reads the marker bands of
//!    the rectified image and the spectrogram is synthesised
//!    ([`decode_image_to_pcm`]).  If the network sees no pattern (e.g. a clean
//!    raster export that does not look like a photo), the stripe detector is
//!    run on the original image as before.

use std::ptr;
use std::sync::LazyLock;

use image::{DynamicImage, GrayImage, Luma};
use jni::{
    Env, EnvUnowned,
    errors::ThrowRuntimeExAndDefault,
    objects::{JByteArray, JClass},
    sys::{jfloatArray, jshortArray},
};
use phonopaper_rs::{
    SpectrogramVec,
    decode::{
        AmplitudeMode, DataBounds, SynthesisOptions, column_amplitudes_from_image_into,
        detect_markers_at_column, nn::PatternDetector, spectrogram_to_audio,
    },
    format::TOTAL_BINS,
};

const SAMPLE_COLUMNS: u32 = 50;
const SAMPLES_PER_COLUMN: usize = 353;
const SAMPLE_RATE: u32 = 44_100;
const GAIN: f32 = 0.15;
const THRESHOLD: f32 = 0.85;
const NO_MARKER_MESSAGE: &str = "No `PhonoPaper` marker pattern found in the supplied image.";
const NO_CLUSTER_MESSAGE: &str =
    "No sufficiently wide `PhonoPaper` marker cluster found in the supplied image.";

/// Presence probability above which the network's detection is trusted.
const DETECTION_THRESHOLD: f32 = 0.5;
/// Maximum deviation of the detected corners from an upright rectangle, as a
/// fraction of the image side, below which the image is read without
/// rectification.  Twice the detector's typical corner error.
const AXIS_ALIGNED_TOLERANCE: f32 = 0.03;
/// White border added around the rectified ink box, as a fraction of its
/// height (vertically) and width (horizontally).  The stripe detector needs a
/// light run before the first stripe, and the network's corners are only
/// accurate to a percent or two of the frame.
const RECTIFY_PAD_Y: f64 = 0.12;
const RECTIFY_PAD_X: f64 = 0.03;
/// Bounds of the rectified ink box, in pixels.
const RECTIFY_MIN_SIDE: f64 = 16.0;
const RECTIFY_MAX_SIDE: f64 = 4096.0;

/// The embedded detector, deserialised on first use and shared by all calls.
static DETECTOR: LazyLock<PatternDetector> = LazyLock::new(PatternDetector::new);

/// Decode a `PhonoPaper` image byte buffer into mono 16-bit PCM audio.
///
/// The neural-network detector locates the sheet first; when it finds one the
/// image is rectified before the robust per-column marker interpolation path
/// reads it, so rotated and skewed camera photos decode as well as clean
/// raster exports.  Without a network detection the original image is read
/// directly.
///
/// # Errors
///
/// Returns an error when the image bytes are invalid, the marker stripes cannot
/// be recovered, or the spectrogram cannot be synthesized.
pub fn decode_image_to_pcm(image_bytes: &[u8]) -> Result<Vec<i16>, String> {
    let image = image::load_from_memory(image_bytes).map_err(|err| err.to_string())?;
    let spectrogram = match DETECTOR.find_corners(&image, DETECTION_THRESHOLD) {
        Some(corners) if !is_axis_aligned(corners, image.width(), image.height()) => {
            let rectified = rectify_pattern(&image, corners);
            decode_spectrogram(&rectified).or_else(|_| decode_spectrogram(&image))?
        }
        _ => decode_spectrogram(&image)?,
    };
    let options = SynthesisOptions {
        sample_rate: SAMPLE_RATE,
        gain: GAIN,
        decode_gamma: 1.0,
        mode: AmplitudeMode::Linear {
            threshold: Some(THRESHOLD),
        },
    };

    let mut samples = vec![0.0_f32; spectrogram.num_columns() * SAMPLES_PER_COLUMN];
    spectrogram_to_audio::<_, SAMPLES_PER_COLUMN>(&spectrogram, &options, &mut samples);

    Ok(samples.into_iter().map(float_to_pcm16).collect())
}

/// Read an upright image with the per-column stripe detector.
fn decode_spectrogram(image: &DynamicImage) -> Result<SpectrogramVec, String> {
    let (x_start, col_bounds) = interpolate_bounds(image, SAMPLE_COLUMNS)?;
    build_spectrogram(image, x_start, &col_bounds).map_err(|err| err.to_string())
}

/// `true` when the detected quadrilateral is a rectangle aligned with the
/// image axes, up to the detector's own imprecision.
///
/// Such images (clean raster exports, carefully framed photos) are read
/// directly: the per-column stripe detector already copes with a small tilt,
/// and skipping the resampling keeps one spectrogram column per image column.
fn is_axis_aligned(corners: [[f32; 2]; 4], width: u32, height: u32) -> bool {
    #[expect(
        clippy::cast_precision_loss,
        reason = "image dimensions are far below 2^24"
    )]
    let (tol_x, tol_y) = (
        width as f32 * AXIS_ALIGNED_TOLERANCE,
        height as f32 * AXIS_ALIGNED_TOLERANCE,
    );
    let [tl, tr, br, bl] = corners;
    // A sheet turned by a quarter turn is also "aligned" with the axes but
    // must be rectified, so only the upright orientation qualifies.
    (tl[1] - tr[1]).abs() <= tol_y
        && (bl[1] - br[1]).abs() <= tol_y
        && (tl[0] - bl[0]).abs() <= tol_x
        && (tr[0] - br[0]).abs() <= tol_x
}

/// Locate a `PhonoPaper` sheet in an image with the neural-network detector.
///
/// Returns the four corners `[TL, TR, BR, BL]` (pattern orientation, clockwise
/// in the image, marker bands along `TL→TR` and `BR→BL`) as fractions of the
/// image width and height, so the caller can draw them over a preview of any
/// size.  Which of the two marker edges is the top of the sheet cannot be
/// told from the image alone; the ordering is the canonical one of
/// `phonopaper_rs::decode::nn::Detection::canonical`.
///
/// This is used by the Android preview overlay.
///
/// # Errors
///
/// Returns an error when the image bytes are invalid.  Returns `Ok(None)` when
/// the image is valid but the presence probability is below the threshold.
pub fn detect_pattern_corners(image_bytes: &[u8]) -> Result<Option<[[f32; 2]; 4]>, String> {
    let image = image::load_from_memory(image_bytes).map_err(|err| err.to_string())?;
    let detection = DETECTOR.detect(&image);
    if detection.probability < DETECTION_THRESHOLD {
        return Ok(None);
    }
    Ok(Some(detection.corners))
}

/// Resample the quadrilateral `corners` (`[TL, TR, BR, BL]` in pixels of
/// `image`, delimiting the ink box) into an upright grayscale image, with a
/// white border around it.
///
/// The output keeps the pattern's apparent size (longest edges), so no
/// resolution is lost; degenerate quadrilaterals fall back to a copy of the
/// grayscale input.
#[must_use]
pub fn rectify_pattern(image: &DynamicImage, corners: [[f32; 2]; 4]) -> DynamicImage {
    let source = image.to_luma8();
    let quad = corners.map(|[x, y]| Point::new(f64::from(x), f64::from(y)));
    let width = quad[0].distance(quad[1]).max(quad[3].distance(quad[2]));
    let height = quad[0].distance(quad[3]).max(quad[1].distance(quad[2]));
    if !(width.is_finite() && height.is_finite()) {
        return DynamicImage::ImageLuma8(source);
    }
    let width = width.round().clamp(RECTIFY_MIN_SIDE, RECTIFY_MAX_SIDE);
    let height = height.round().clamp(RECTIFY_MIN_SIDE, RECTIFY_MAX_SIDE);
    let pad_x = (width * RECTIFY_PAD_X).ceil();
    let pad_y = (height * RECTIFY_PAD_Y).ceil();
    let target = [
        Point::new(pad_x, pad_y),
        Point::new(pad_x + width, pad_y),
        Point::new(pad_x + width, pad_y + height),
        Point::new(pad_x, pad_y + height),
    ];
    let Some(to_source) = Homography::from_quads(&target, &quad) else {
        return DynamicImage::ImageLuma8(source);
    };

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "sides are rounded, positive and clamped to RECTIFY_MAX_SIDE"
    )]
    let (out_w, out_h) = ((width + 2.0 * pad_x) as u32, (height + 2.0 * pad_y) as u32);
    let output = GrayImage::from_fn(out_w, out_h, |x, y| {
        let p = to_source.apply(Point::new(f64::from(x) + 0.5, f64::from(y) + 0.5));
        Luma([sample_bilinear(&source, p)])
    });
    DynamicImage::ImageLuma8(output)
}

/// Bilinear sample of `image` at continuous coordinates (pixel centres at
/// `+0.5`); white outside the image.
fn sample_bilinear(image: &GrayImage, p: Point) -> u8 {
    let (w, h) = (f64::from(image.width()), f64::from(image.height()));
    if !(p.x.is_finite() && p.y.is_finite()) || p.x < 0.0 || p.y < 0.0 || p.x >= w || p.y >= h {
        return 255;
    }
    let fx = (p.x - 0.5).clamp(0.0, w - 1.0);
    let fy = (p.y - 0.5).clamp(0.0, h - 1.0);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to [0, side - 1], hence non-negative and within u32"
    )]
    let (x0, y0) = (fx.floor() as u32, fy.floor() as u32);
    let x1 = (x0 + 1).min(image.width() - 1);
    let y1 = (y0 + 1).min(image.height() - 1);
    let (tx, ty) = (fx - fx.floor(), fy - fy.floor());
    let at = |x, y| f64::from(image.get_pixel(x, y)[0]);
    let top = at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx;
    let bottom = at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a convex combination of u8 values rounds into [0, 255]"
    )]
    let value = (top * (1.0 - ty) + bottom * ty).round() as u8;
    value
}

// ─── Planar geometry ─────────────────────────────────────────────────────────

/// A 2-D point in pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn distance(self, other: Self) -> f64 {
        (self.x - other.x).hypot(self.y - other.y)
    }
}

/// A 3×3 projective transform, row-major.
#[derive(Debug, Clone, Copy)]
struct Homography {
    m: [f64; 9],
}

impl Homography {
    /// The homography mapping each `src[i]` onto `dst[i]`, or `None` when the
    /// points are degenerate.
    fn from_quads(src: &[Point; 4], dst: &[Point; 4]) -> Option<Self> {
        // Direct linear transform with h33 fixed to 1: 8 unknowns, 8 equations.
        let mut system = [[0.0_f64; 9]; 8];
        for i in 0..4 {
            let (sx, sy) = (src[i].x, src[i].y);
            let (dx, dy) = (dst[i].x, dst[i].y);
            system[2 * i] = [sx, sy, 1.0, 0.0, 0.0, 0.0, -dx * sx, -dx * sy, dx];
            system[2 * i + 1] = [0.0, 0.0, 0.0, sx, sy, 1.0, -dy * sx, -dy * sy, dy];
        }
        let h = solve_8x8(&mut system)?;
        Some(Self {
            m: [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0],
        })
    }

    fn apply(&self, p: Point) -> Point {
        let m = &self.m;
        let w = m[6] * p.x + m[7] * p.y + m[8];
        Point::new(
            (m[0] * p.x + m[1] * p.y + m[2]) / w,
            (m[3] * p.x + m[4] * p.y + m[5]) / w,
        )
    }
}

/// Solve the 8×8 augmented system `a·h = b` (column 8 holds `b`) by Gaussian
/// elimination with partial pivoting.
fn solve_8x8(a: &mut [[f64; 9]; 8]) -> Option<[f64; 8]> {
    for col in 0..8 {
        let pivot = (col..8).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        for row in (col + 1)..8 {
            let factor = a[row][col] / a[col][col];
            if factor != 0.0 {
                for k in col..9 {
                    a[row][k] -= factor * a[col][k];
                }
            }
        }
    }
    let mut h = [0.0_f64; 8];
    for col in (0..8).rev() {
        let mut acc = a[col][8];
        for k in (col + 1)..8 {
            acc -= a[col][k] * h[k];
        }
        h[col] = acc / a[col][col];
    }
    Some(h)
}

// ─── JNI entry points ────────────────────────────────────────────────────────

/// JNI entry point used by the Android application to decode an image into PCM.
#[must_use]
#[unsafe(export_name = "Java_com_evenfurther_phonopaper_PhonopaperNative_decodeImageToPcm")]
pub extern "system" fn java_decode_image_to_pcm(
    mut env: EnvUnowned,
    _class: JClass,
    image_bytes: JByteArray,
) -> jshortArray {
    env.with_env(|env| -> jni::errors::Result<_> {
        match decode_image_to_pcm_array(env, image_bytes) {
            Ok(array) => Ok(array),
            Err(message) => {
                let _ = env.throw(message);
                Ok(ptr::null_mut())
            }
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

/// JNI entry point used by the Android application to locate the sheet in the
/// live preview.  Returns `x0 y0 x1 y1 x2 y2 x3 y3` as fractions of the image
/// size, or `null` when no pattern is detected.
#[must_use]
#[unsafe(export_name = "Java_com_evenfurther_phonopaper_PhonopaperNative_detectPatternCorners")]
pub extern "system" fn java_detect_pattern_corners(
    mut env: EnvUnowned,
    _class: JClass,
    image_bytes: JByteArray,
) -> jfloatArray {
    env.with_env(|env| -> jni::errors::Result<_> {
        match detect_pattern_corners_array(env, image_bytes) {
            Ok(array) => Ok(array),
            Err(message) => {
                let _ = env.throw(message);
                Ok(ptr::null_mut())
            }
        }
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

fn decode_image_to_pcm_array(
    env: &mut Env<'_>,
    image_bytes: JByteArray,
) -> Result<jshortArray, String> {
    let bytes = env
        .convert_byte_array(image_bytes)
        .map_err(|err| err.to_string())?;
    let pcm = decode_image_to_pcm(&bytes)?;
    let output = env
        .new_short_array(pcm.len())
        .map_err(|err| err.to_string())?;
    output
        .set_region(env, 0, &pcm)
        .map_err(|err| err.to_string())?;

    Ok(output.into_raw())
}

fn detect_pattern_corners_array(
    env: &mut Env<'_>,
    image_bytes: JByteArray,
) -> Result<jfloatArray, String> {
    let bytes = env
        .convert_byte_array(image_bytes)
        .map_err(|err| err.to_string())?;
    let Some(corners) = detect_pattern_corners(&bytes)? else {
        return Ok(ptr::null_mut());
    };

    let flat: Vec<f32> = corners.iter().flatten().copied().collect();
    let output = env.new_float_array(8).map_err(|err| err.to_string())?;
    output
        .set_region(env, 0, &flat)
        .map_err(|err| err.to_string())?;

    Ok(output.into_raw())
}

fn interpolate_bounds(
    image: &DynamicImage,
    sample_columns: u32,
) -> Result<(u32, Vec<(f32, f32)>), String> {
    let detected = sample_detected_bounds(image, sample_columns)?;
    let (left, right) = detected_span(&detected).ok_or_else(|| NO_MARKER_MESSAGE.to_string())?;

    Ok((
        left,
        interpolate_bounds_from_detected(left, right, &detected),
    ))
}

fn sample_detected_bounds(
    image: &DynamicImage,
    sample_columns: u32,
) -> Result<Vec<(u32, f32, f32)>, String> {
    use image::GenericImageView as _;

    let (width, _) = image.dimensions();
    if width == 0 {
        return Err("The image has zero width and cannot be decoded.".to_string());
    }
    let n_samples = sample_columns.min(width).max(2);

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
            let x = (f64::from(i) / f64::from(n_samples - 1) * f64::from(width - 1)).round() as u32;
            x.min(width - 1)
        })
        .collect();
    sample_xs.dedup();

    let mut sample_results = Vec::with_capacity(sample_xs.len());
    for &col_x in &sample_xs {
        sample_results.push((col_x, detect_markers_at_column(image, col_x).ok()));
    }

    let cluster = select_consistent_cluster(&sample_results)?;
    Ok(cluster
        .iter()
        .map(|&(col_x, bounds)| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "pixel coordinates are converted to f32 for interpolation"
            )]
            (col_x, bounds.data_top as f32, bounds.data_bottom as f32)
        })
        .collect())
}

fn detected_span(detected: &[(u32, f32, f32)]) -> Option<(u32, u32)> {
    Some((detected.first()?.0, detected.last()?.0))
}

fn interpolate_bounds_from_detected(
    left: u32,
    right: u32,
    detected: &[(u32, f32, f32)],
) -> Vec<(f32, f32)> {
    #[expect(
        clippy::cast_precision_loss,
        reason = "column indices are converted to f32 for interpolation arithmetic"
    )]
    (left..=right)
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
                        reason = "anchor positions are converted to f32 for interpolation"
                    )]
                    let t = (xf - x0 as f32) / (x1 as f32 - x0 as f32);
                    (t0 + t * (t1 - t0), b0 + t * (b1 - b0))
                }
            }
        })
        .collect()
}

fn select_consistent_cluster(
    sample_results: &[(u32, Option<DataBounds>)],
) -> Result<Vec<(u32, DataBounds)>, String> {
    let min_cluster_len = sample_results.len().min(3);
    let mut best_cluster: Vec<(u32, DataBounds)> = Vec::new();
    let mut current_cluster: Vec<(u32, DataBounds)> = Vec::new();

    for &(col_x, bounds) in sample_results {
        match bounds {
            Some(bounds) => {
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
            None => {
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
        return Err(NO_CLUSTER_MESSAGE.to_string());
    }

    Ok(best_cluster)
}

fn build_spectrogram(
    image: &DynamicImage,
    x_start: u32,
    col_bounds: &[(f32, f32)],
) -> phonopaper_rs::Result<SpectrogramVec> {
    let mut spectrogram = SpectrogramVec::new(col_bounds.len());
    let mut amplitudes = [0.0_f32; TOTAL_BINS];

    for (col_x, &(top_f, bottom_f)) in col_bounds.iter().enumerate() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "interpolated row is clamped and rounded before conversion to u32"
        )]
        #[expect(
            clippy::cast_sign_loss,
            reason = "interpolated bounds are clamped to non-negative values"
        )]
        let data_top = top_f.max(0.0).round() as u32;
        #[expect(clippy::cast_possible_truncation, reason = "same as `data_top`")]
        #[expect(clippy::cast_sign_loss, reason = "same as `data_top`")]
        let data_bottom = bottom_f.max(0.0).round() as u32;
        let bounds = DataBounds {
            data_top,
            data_bottom,
        };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "spectrogram columns are offset from x_start and stay within image width"
        )]
        let image_col = x_start + col_x as u32;
        column_amplitudes_from_image_into(image, Some(bounds), image_col, &mut amplitudes)?;
        spectrogram.column_mut(col_x).copy_from_slice(&amplitudes);
    }

    Ok(spectrogram)
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

fn float_to_pcm16(sample: f32) -> i16 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "rounded sample is clamped to the 16-bit PCM output range"
    )]
    let pcm = (sample.clamp(-1.0, 1.0) * 32_767.5).round() as i16;
    pcm
}
