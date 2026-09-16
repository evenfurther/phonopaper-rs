//! Android JNI bridge for `phonopaper-rs`.
//!
//! The Android UI stays thin and delegates all `PhonoPaper` decoding work to
//! Rust so the mobile application can share the same decoding logic as the
//! workspace library.

use std::{panic::{AssertUnwindSafe, catch_unwind}, ptr};

use image::DynamicImage;
use jni::{
    JNIEnv,
    objects::{JByteArray, JClass, JShortArray},
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
    let col_bounds = interpolate_bounds(&image, SAMPLE_COLUMNS)?;
    let spectrogram = build_spectrogram(&image, &col_bounds).map_err(|err| err.to_string())?;
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
#[unsafe(export_name = "Java_com_evenfurther_phonopaper_PhonopaperNative_decodeImageToPcm")]
pub extern "system" fn java_decode_image_to_pcm(
    mut env: JNIEnv,
    _class: JClass,
    image_bytes: JByteArray,
) -> jshortArray {
    with_runtime_exception(&mut env, |env| decode_image_to_pcm_array(env, image_bytes))
        .map_or(ptr::null_mut(), JShortArray::into_raw)
}

fn with_runtime_exception<T>(
    env: &mut JNIEnv,
    f: impl FnOnce(&mut JNIEnv) -> Result<T, String>,
) -> Option<T> {
    match catch_unwind(AssertUnwindSafe(|| f(env))) {
        Ok(Ok(value)) => Some(value),
        Ok(Err(message)) => {
            let _ = env.throw_new("java/lang/RuntimeException", message);
            None
        }
        Err(_) => {
            let _ = env.throw_new("java/lang/RuntimeException", "Rust panic while decoding image.");
            None
        }
    }
}

fn decode_image_to_pcm_array(
    env: &mut JNIEnv,
    image_bytes: JByteArray,
) -> Result<JShortArray<'static>, String> {
    let bytes = env
        .convert_byte_array(image_bytes)
        .map_err(|err| err.to_string())?;
    let pcm = decode_image_to_pcm(&bytes)?;
    let len = i32::try_from(pcm.len()).map_err(|_| "Decoded audio is too large for JNI.".to_string())?;
    let output = env.new_short_array(len).map_err(|err| err.to_string())?;
    env.set_short_array_region(&output, 0, &pcm)
        .map_err(|err| err.to_string())?;

    Ok(output)
}

fn interpolate_bounds(image: &DynamicImage, sample_columns: u32) -> Result<Vec<(f32, f32)>, String> {
    use image::GenericImageView as _;

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
                reason = ".round() on a non-negative f64 product is always non-negative"
            )]
            let x = (f64::from(i) / f64::from(n_samples - 1) * f64::from(width - 1)).round() as u32;
            x.min(width - 1)
        })
        .collect();
    sample_xs.dedup();

    let mut detected = Vec::with_capacity(sample_xs.len());
    for &col_x in &sample_xs {
        if let Ok(bounds) = detect_markers_at_column(image, col_x) {
            #[expect(
                clippy::cast_precision_loss,
                reason = "pixel coordinates are converted to f32 for interpolation"
            )]
            detected.push((col_x, bounds.data_top as f32, bounds.data_bottom as f32));
        }
    }

    if detected.is_empty() {
        return Err("No `PhonoPaper` marker pattern found in the supplied image.".to_string());
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "column indices are converted to f32 for interpolation arithmetic"
    )]
    Ok((0..width)
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
        .collect())
}

fn build_spectrogram(
    image: &DynamicImage,
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
            reason = "image columns are indexed by u32 and the width comes from the image"
        )]
        column_amplitudes_from_image_into(image, Some(bounds), col_x as u32, &mut amplitudes)?;
        spectrogram.column_mut(col_x).copy_from_slice(&amplitudes);
    }

    Ok(spectrogram)
}

fn float_to_pcm16(sample: f32) -> i16 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "rounded sample is clamped to the 16-bit PCM output range"
    )]
    (sample.clamp(-1.0, 1.0) * 32_767.5).round() as i16
}

#[cfg(test)]
mod tests {
    use phonopaper_rs::{
        SpectrogramVec,
        render::{RenderOptions, spectrogram_to_image},
    };

    use super::decode_image_to_pcm;

    #[test]
    fn decode_generated_phonopaper_image() {
        let mut spectrogram = SpectrogramVec::new(24);
        for col in 0..spectrogram.num_columns() {
            spectrogram.set(col, 120, 1.0);
            if col % 3 == 0 {
                spectrogram.set(col, 144, 0.5);
            }
        }

        let image = spectrogram_to_image(&spectrogram, &RenderOptions::default());
        let mut png = Vec::new();
        image
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .expect("write PNG test fixture");

        let pcm = decode_image_to_pcm(&png).expect("decode generated image");
        assert!(!pcm.is_empty());
        assert!(pcm.iter().any(|&sample| sample != 0));
    }

    #[test]
    fn reject_invalid_image_bytes() {
        let err = decode_image_to_pcm(b"not an image").expect_err("invalid input should fail");
        assert!(!err.is_empty());
    }
}
