//! Android JNI bridge for `phonopaper-rs`.
//!
//! The Android UI stays thin and delegates all `PhonoPaper` decoding work to
//! Rust so the mobile application can share the same decoding logic as the
//! workspace library.

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    ptr,
};

use image::DynamicImage;
use jni::{
    JNIEnv,
    objects::{JByteArray, JClass},
    sys::jshortArray,
};
use phonopaper_rs::{
    SpectrogramVec,
    decode::{
        AmplitudeMode, DataBounds, SynthesisOptions, column_amplitudes_from_image_into,
        detect_markers_at_column, spectrogram_to_audio,
    },
    format::TOTAL_BINS,
};

const SAMPLE_COLUMNS: u32 = 50;
const SAMPLES_PER_COLUMN: usize = 353;
const SAMPLE_RATE: u32 = 44_100;
const GAIN: f32 = 0.15;
const THRESHOLD: f32 = 0.85;

/// Decode a `PhonoPaper` image byte buffer into mono 16-bit PCM audio.
///
/// The function uses the robust per-column marker interpolation path so camera
/// photos can be decoded in addition to clean raster exports.
///
/// # Errors
///
/// Returns an error when the image bytes are invalid, the marker stripes cannot
/// be recovered, or the spectrogram cannot be synthesized.
pub fn decode_image_to_pcm(image_bytes: &[u8]) -> Result<Vec<i16>, String> {
    let image = image::load_from_memory(image_bytes).map_err(|err| err.to_string())?;
    let (x_start, col_bounds) = interpolate_bounds(&image, SAMPLE_COLUMNS)?;
    let spectrogram =
        build_spectrogram(&image, x_start, &col_bounds).map_err(|err| err.to_string())?;
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

/// JNI entry point used by the Android application to decode an image into PCM.
#[must_use]
#[unsafe(export_name = "Java_com_evenfurther_phonopaper_PhonopaperNative_decodeImageToPcm")]
pub extern "system" fn java_decode_image_to_pcm(
    mut env: JNIEnv,
    _class: JClass,
    image_bytes: JByteArray,
) -> jshortArray {
    match catch_unwind(AssertUnwindSafe(|| {
        decode_image_to_pcm_array(&mut env, image_bytes)
    })) {
        Ok(Ok(array)) => array,
        Ok(Err(message)) => {
            let _ = env.throw_new("java/lang/RuntimeException", message);
            ptr::null_mut()
        }
        Err(_) => {
            let _ = env.throw_new(
                "java/lang/RuntimeException",
                "Rust panic while decoding image.",
            );
            ptr::null_mut()
        }
    }
}

fn decode_image_to_pcm_array(
    env: &mut JNIEnv,
    image_bytes: JByteArray,
) -> Result<jshortArray, String> {
    let bytes = env
        .convert_byte_array(image_bytes)
        .map_err(|err| err.to_string())?;
    let pcm = decode_image_to_pcm(&bytes)?;
    let len =
        i32::try_from(pcm.len()).map_err(|_| "Decoded audio is too large for JNI.".to_string())?;
    let output = env.new_short_array(len).map_err(|err| err.to_string())?;
    env.set_short_array_region(&output, 0, &pcm)
        .map_err(|err| err.to_string())?;

    Ok(output.into_raw())
}

fn interpolate_bounds(
    image: &DynamicImage,
    sample_columns: u32,
) -> Result<(u32, Vec<(f32, f32)>), String> {
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
    let left = cluster.first().map_or(0, |(x, _)| *x);
    let right = cluster.last().map_or(left, |(x, _)| *x);

    let detected: Vec<(u32, f32, f32)> = cluster
        .iter()
        .map(|&(col_x, bounds)| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "pixel coordinates are converted to f32 for interpolation"
            )]
            (col_x, bounds.data_top as f32, bounds.data_bottom as f32)
        })
        .collect();

    if detected.is_empty() {
        return Err("No `PhonoPaper` marker pattern found in the supplied image.".to_string());
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "column indices are converted to f32 for interpolation arithmetic"
    )]
    let bounds = (left..=right)
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
        .collect();

    Ok((left, bounds))
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
        return Err(
            "No sufficiently wide `PhonoPaper` marker cluster found in the supplied image."
                .to_string(),
        );
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
