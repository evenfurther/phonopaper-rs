//! End-to-end file I/O tests for:
//! - [`phonopaper_rs::encode::encode_audio_to_image`]
//! - [`phonopaper_rs::decode::decode_image_to_wav`]
//! - [`phonopaper_rs::decode::decode_image_to_wav_sps`]
//!
//! All temporary files are created via the `tempfile` crate and cleaned up
//! automatically when the test exits.

use std::f32::consts::TAU;
use std::io::Write as _;

use phonopaper_rs::decode::{SynthesisOptions, decode_image_to_wav, decode_image_to_wav_sps};
use phonopaper_rs::encode::{AnalysisOptions, encode_audio_to_image};
use phonopaper_rs::render::RenderOptions;

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Write a minimal 16-bit mono PCM WAV file containing a 440 Hz sine wave.
///
/// `num_samples` samples are written at `sample_rate` Hz.  Returns the path of
/// the temporary file (kept alive by the returned `NamedTempFile`).
fn write_sine_wav(
    sample_rate: u32,
    num_samples: usize,
) -> (tempfile::NamedTempFile, std::path::PathBuf) {
    let mut tmp = tempfile::Builder::new()
        .suffix(".wav")
        .tempfile()
        .expect("failed to create temp WAV file");

    let freq = 440.0_f32;
    #[expect(
        clippy::cast_precision_loss,
        reason = "sample_rate and sample index are small enough that f32 precision is sufficient \
                  for computing a 440 Hz sine wave"
    )]
    let samples: Vec<i16> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let sample_f = f32::sin(TAU * freq * t) * 16_000.0;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "value is scaled to [-16000, 16000] which fits in i16"
            )]
            let sample_i = sample_f as i16;
            sample_i
        })
        .collect();

    // Write a raw 16-bit mono WAV by hand so we don't need hound as a
    // dev-dependency (it is already a regular dependency, but using it
    // directly here would introduce unnecessary coupling to its API in tests).
    let num_samples_u32 = u32::try_from(num_samples).expect("num_samples fits in u32");
    let data_bytes = num_samples_u32 * 2;
    let fmt_size: u32 = 16;
    let file_size = 4 + (8 + fmt_size) + (8 + data_bytes);

    tmp.write_all(b"RIFF").unwrap();
    tmp.write_all(&file_size.to_le_bytes()).unwrap();
    tmp.write_all(b"WAVE").unwrap();
    tmp.write_all(b"fmt ").unwrap();
    tmp.write_all(&fmt_size.to_le_bytes()).unwrap();
    tmp.write_all(&1u16.to_le_bytes()).unwrap(); // PCM
    tmp.write_all(&1u16.to_le_bytes()).unwrap(); // mono
    tmp.write_all(&sample_rate.to_le_bytes()).unwrap();
    tmp.write_all(&(sample_rate * 2).to_le_bytes()).unwrap();
    tmp.write_all(&2u16.to_le_bytes()).unwrap();
    tmp.write_all(&16u16.to_le_bytes()).unwrap();
    tmp.write_all(b"data").unwrap();
    tmp.write_all(&data_bytes.to_le_bytes()).unwrap();
    for s in &samples {
        tmp.write_all(&s.to_le_bytes()).unwrap();
    }
    tmp.flush().unwrap();

    let path = tmp.path().to_path_buf();
    (tmp, path)
}

/// Read a WAV file and return all samples as `i16`.  Uses a hand-rolled parser
/// so we do not introduce a test-only dependency on hound's write API.
fn read_wav_samples(path: &std::path::Path) -> (Vec<i16>, u32) {
    let data = std::fs::read(path).expect("failed to read WAV file");

    // Skip to the "fmt " chunk to read sample_rate and num_channels.
    // Minimal parser: assumes fmt comes before data (always true for WAVs we write).
    assert_eq!(&data[0..4], b"RIFF");
    assert_eq!(&data[8..12], b"WAVE");

    let mut pos = 12usize;
    let mut sample_rate = 0u32;
    let mut num_channels = 0u16;
    let mut samples = Vec::new();

    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        pos += 8;

        if chunk_id == b"fmt " {
            num_channels = u16::from_le_bytes(data[pos + 2..pos + 4].try_into().unwrap());
            sample_rate = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap());
        } else if chunk_id == b"data" {
            let mut i = 0;
            while i + 2 <= chunk_size {
                let s = i16::from_le_bytes(data[pos + i..pos + i + 2].try_into().unwrap());
                samples.push(s);
                i += 2;
            }
        }
        pos += chunk_size;
    }

    assert_eq!(num_channels, 1, "expected mono WAV");
    (samples, sample_rate)
}

// ─── encode_audio_to_image ─────────────────────────────────────────────────────

/// `encode_audio_to_image` produces a valid `PhonoPaper` PNG: `detect_markers`
/// must succeed on the output image.
#[test]
fn encode_audio_to_image_produces_valid_markers() {
    use phonopaper_rs::decode::detect_markers;

    let (_wav_tmp, wav_path) = write_sine_wav(44_100, 44_100); // 1 second of 440 Hz

    let img_tmp = tempfile::Builder::new()
        .suffix(".png")
        .tempfile()
        .expect("failed to create temp PNG file");
    let img_path = img_tmp.path().to_path_buf();

    encode_audio_to_image(
        &wav_path,
        &img_path,
        AnalysisOptions::default(),
        RenderOptions::default(),
    )
    .expect("encode_audio_to_image should succeed");

    let img = image::open(&img_path).expect("output PNG should be openable");
    detect_markers(&img).expect("detect_markers should succeed on the encoded PhonoPaper image");
}

/// `encode_audio_to_image` with a non-existent input file returns an error.
#[test]
fn encode_audio_to_image_missing_input_is_error() {
    let img_tmp = tempfile::Builder::new()
        .suffix(".png")
        .tempfile()
        .expect("failed to create temp PNG");

    let result = encode_audio_to_image(
        "/nonexistent/does_not_exist.wav",
        img_tmp.path(),
        AnalysisOptions::default(),
        RenderOptions::default(),
    );
    assert!(result.is_err(), "missing input file should return an error");
}

// ─── decode_image_to_wav ─────────────────────────────────────────────────────

/// `decode_image_to_wav` produces a non-empty WAV with non-zero samples when
/// given a `PhonoPaper` image that was encoded from a sine wave.
#[test]
fn decode_image_to_wav_produces_audio() {
    // 1. Encode a short sine to a PhonoPaper PNG.
    let (_wav_tmp, wav_path) = write_sine_wav(44_100, 44_100);

    let img_tmp = tempfile::Builder::new()
        .suffix(".png")
        .tempfile()
        .expect("failed to create temp PNG");
    let img_path = img_tmp.path().to_path_buf();

    encode_audio_to_image(
        &wav_path,
        &img_path,
        AnalysisOptions::default(),
        RenderOptions::default(),
    )
    .expect("encode_audio_to_image should succeed");

    // 2. Decode the PNG back to a WAV.
    let out_tmp = tempfile::Builder::new()
        .suffix(".wav")
        .tempfile()
        .expect("failed to create temp output WAV");
    let out_path = out_tmp.path().to_path_buf();

    decode_image_to_wav(&img_path, &out_path, SynthesisOptions::default())
        .expect("decode_image_to_wav should succeed");

    // 3. Verify the output WAV is non-empty and contains non-zero samples.
    let (samples, _) = read_wav_samples(&out_path);
    assert!(!samples.is_empty(), "decoded WAV should not be empty");
    assert!(
        samples.iter().any(|&s| s != 0),
        "decoded WAV should contain non-zero samples"
    );
}

/// `decode_image_to_wav` with a non-existent input returns an error.
#[test]
fn decode_image_to_wav_missing_input_is_error() {
    let out_tmp = tempfile::Builder::new()
        .suffix(".wav")
        .tempfile()
        .expect("failed to create temp output WAV");

    let result = decode_image_to_wav(
        "/nonexistent/does_not_exist.png",
        out_tmp.path(),
        SynthesisOptions::default(),
    );
    assert!(result.is_err(), "missing input file should return an error");
}

// ─── decode_image_to_wav_sps ─────────────────────────────────────────────────

/// `decode_image_to_wav_sps::<512>` produces a WAV whose sample count equals
/// `num_image_columns * 512`.
#[test]
fn decode_image_to_wav_sps_sample_count() {
    // 1. Encode a sine wave to a PhonoPaper PNG.
    let (_wav_tmp, wav_path) = write_sine_wav(44_100, 44_100);

    let img_tmp = tempfile::Builder::new()
        .suffix(".png")
        .tempfile()
        .expect("failed to create temp PNG");
    let img_path = img_tmp.path().to_path_buf();

    let render_opts = RenderOptions::default();
    encode_audio_to_image(
        &wav_path,
        &img_path,
        AnalysisOptions::default(),
        render_opts,
    )
    .expect("encode_audio_to_image should succeed");

    // Count the image columns so we can predict the output length.
    let img = image::open(&img_path).expect("PNG should be openable");
    let num_columns = img.width() as usize;

    // 2. Decode with SPS = 512.
    let out_tmp = tempfile::Builder::new()
        .suffix(".wav")
        .tempfile()
        .expect("failed to create temp output WAV");
    let out_path = out_tmp.path().to_path_buf();

    decode_image_to_wav_sps::<512>(&img_path, &out_path, SynthesisOptions::default())
        .expect("decode_image_to_wav_sps should succeed");

    // 3. Verify the sample count.
    let (samples, _) = read_wav_samples(&out_path);
    assert_eq!(
        samples.len(),
        num_columns * 512,
        "WAV sample count should equal num_columns ({num_columns}) * 512"
    );
}

// ─── stereo WAV encode ────────────────────────────────────────────────────────

/// `encode_audio_to_image` with a stereo (2-channel) WAV downmixes to mono and
/// succeeds.  The output image must contain valid `PhonoPaper` markers.
#[test]
fn encode_audio_to_image_stereo_succeeds() {
    use phonopaper_rs::decode::detect_markers;

    // Build a minimal stereo 16-bit PCM WAV by hand.
    let sample_rate: u32 = 44_100;
    let num_frames: usize = 8_192; // each frame = 2 interleaved i16 samples; must be ≥ fft_window
    let channels: u16 = 2;

    let mut tmp = tempfile::Builder::new()
        .suffix(".wav")
        .tempfile()
        .expect("failed to create temp WAV file");

    let freq = 440.0_f32;
    #[expect(
        clippy::cast_precision_loss,
        reason = "sample_rate and sample index are small enough for f32 sine synthesis"
    )]
    let stereo_samples: Vec<i16> = (0..num_frames)
        .flat_map(|i| {
            let t = i as f32 / sample_rate as f32;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "value is scaled to [-16000, 16000] which fits in i16"
            )]
            let val = (f32::sin(TAU * freq * t) * 16_000.0) as i16;
            [val, val] // identical L / R channels
        })
        .collect();

    let num_data_bytes = u32::try_from(stereo_samples.len() * 2).expect("fits in u32");
    let fmt_size: u32 = 16;
    let file_size = 4 + (8 + fmt_size) + (8 + num_data_bytes);

    tmp.write_all(b"RIFF").unwrap();
    tmp.write_all(&file_size.to_le_bytes()).unwrap();
    tmp.write_all(b"WAVE").unwrap();
    tmp.write_all(b"fmt ").unwrap();
    tmp.write_all(&fmt_size.to_le_bytes()).unwrap();
    tmp.write_all(&1u16.to_le_bytes()).unwrap(); // PCM
    tmp.write_all(&channels.to_le_bytes()).unwrap(); // stereo
    tmp.write_all(&sample_rate.to_le_bytes()).unwrap();
    tmp.write_all(&(sample_rate * u32::from(channels) * 2).to_le_bytes())
        .unwrap(); // byte rate
    tmp.write_all(&(channels * 2).to_le_bytes()).unwrap(); // block align
    tmp.write_all(&16u16.to_le_bytes()).unwrap(); // bits per sample
    tmp.write_all(b"data").unwrap();
    tmp.write_all(&num_data_bytes.to_le_bytes()).unwrap();
    for s in &stereo_samples {
        tmp.write_all(&s.to_le_bytes()).unwrap();
    }
    tmp.flush().unwrap();

    let wav_path = tmp.path().to_path_buf();

    let img_tmp = tempfile::Builder::new()
        .suffix(".png")
        .tempfile()
        .expect("failed to create temp PNG");
    let img_path = img_tmp.path().to_path_buf();

    encode_audio_to_image(
        &wav_path,
        &img_path,
        AnalysisOptions::default(),
        RenderOptions::default(),
    )
    .expect("encode_audio_to_image should succeed with a stereo input");

    let img = image::open(&img_path).expect("output PNG should be openable");
    detect_markers(&img)
        .expect("detect_markers should succeed on a stereo-sourced PhonoPaper image");
}

// ─── integer-format WAV encode ────────────────────────────────────────────────

/// `encode_audio_to_image` with a 16-bit integer PCM WAV (the most common WAV
/// format) exercises the integer-to-float sample conversion path and must
/// produce a valid `PhonoPaper` image.
///
/// The helper `write_sine_wav` already writes 16-bit integer PCM, so this test
/// explicitly verifies that the integer decode path (not the float path) is
/// taken and that the image is valid.
#[test]
fn encode_audio_to_image_integer_pcm_succeeds() {
    use phonopaper_rs::decode::detect_markers;

    // write_sine_wav produces a standard 16-bit integer PCM file — exactly the
    // format handled by the `SampleFormat::Int` branch in encode.rs.
    let (_wav_tmp, wav_path) = write_sine_wav(44_100, 44_100);

    let img_tmp = tempfile::Builder::new()
        .suffix(".png")
        .tempfile()
        .expect("failed to create temp PNG");
    let img_path = img_tmp.path().to_path_buf();

    encode_audio_to_image(
        &wav_path,
        &img_path,
        AnalysisOptions::default(),
        RenderOptions::default(),
    )
    .expect("encode_audio_to_image should succeed with a 16-bit integer PCM WAV");

    let img = image::open(&img_path).expect("output PNG should be openable");
    detect_markers(&img)
        .expect("detect_markers should succeed on an integer-PCM-sourced PhonoPaper image");
}

/// `encode_audio_to_image` with a 32-bit float PCM WAV exercises the float
/// sample reading path (`SampleFormat::Float`).
#[test]
fn encode_audio_to_image_float_pcm_succeeds() {
    use phonopaper_rs::decode::detect_markers;

    // Build a minimal 32-bit float mono WAV.
    let sample_rate: u32 = 44_100;
    let num_samples: usize = 8_192; // must be ≥ default fft_window (4096)

    let mut tmp = tempfile::Builder::new()
        .suffix(".wav")
        .tempfile()
        .expect("failed to create temp WAV file");

    let freq = 440.0_f32;
    #[expect(
        clippy::cast_precision_loss,
        reason = "sample index is small relative to f32 mantissa"
    )]
    let float_samples: Vec<f32> = (0..num_samples)
        .map(|i| f32::sin(TAU * freq * i as f32 / sample_rate as f32))
        .collect();

    let num_data_bytes = u32::try_from(float_samples.len() * 4).expect("fits in u32");
    let fmt_size: u32 = 16;
    let file_size = 4 + (8 + fmt_size) + (8 + num_data_bytes);

    tmp.write_all(b"RIFF").unwrap();
    tmp.write_all(&file_size.to_le_bytes()).unwrap();
    tmp.write_all(b"WAVE").unwrap();
    tmp.write_all(b"fmt ").unwrap();
    tmp.write_all(&fmt_size.to_le_bytes()).unwrap();
    tmp.write_all(&3u16.to_le_bytes()).unwrap(); // WAVE_FORMAT_IEEE_FLOAT
    tmp.write_all(&1u16.to_le_bytes()).unwrap(); // mono
    tmp.write_all(&sample_rate.to_le_bytes()).unwrap();
    tmp.write_all(&(sample_rate * 4).to_le_bytes()).unwrap(); // byte rate
    tmp.write_all(&4u16.to_le_bytes()).unwrap(); // block align
    tmp.write_all(&32u16.to_le_bytes()).unwrap(); // bits per sample
    tmp.write_all(b"data").unwrap();
    tmp.write_all(&num_data_bytes.to_le_bytes()).unwrap();
    for s in &float_samples {
        tmp.write_all(&s.to_le_bytes()).unwrap();
    }
    tmp.flush().unwrap();

    let wav_path = tmp.path().to_path_buf();

    let img_tmp = tempfile::Builder::new()
        .suffix(".png")
        .tempfile()
        .expect("failed to create temp PNG");
    let img_path = img_tmp.path().to_path_buf();

    encode_audio_to_image(
        &wav_path,
        &img_path,
        AnalysisOptions::default(),
        RenderOptions::default(),
    )
    .expect("encode_audio_to_image should succeed with a 32-bit float PCM WAV");

    let img = image::open(&img_path).expect("output PNG should be openable");
    detect_markers(&img)
        .expect("detect_markers should succeed on a float-PCM-sourced PhonoPaper image");
}

// ─── JPEG input decode ────────────────────────────────────────────────────────

/// `decode_image_to_wav` accepts a JPEG file as input.
///
/// JPEG is a supported input format: `image::open` detects it by extension and
/// magic bytes, and the `image` crate's default feature set includes the JPEG
/// decoder.  This test verifies the full path: encode a sine wave to PNG,
/// re-save as JPEG (lossy), then decode the JPEG and confirm that non-silent
/// audio is produced.
///
/// Note: JPEG compression degrades `PhonoPaper` images — this test only checks
/// that the decode pipeline *accepts* JPEG, not that quality is preserved.
#[test]
fn decode_image_to_wav_jpeg_input() {
    // 1. Encode a 440 Hz sine to a PhonoPaper PNG.
    let (_wav_tmp, wav_path) = write_sine_wav(44_100, 44_100);

    let png_tmp = tempfile::Builder::new()
        .suffix(".png")
        .tempfile()
        .expect("failed to create temp PNG");
    let png_path = png_tmp.path().to_path_buf();

    encode_audio_to_image(
        &wav_path,
        &png_path,
        AnalysisOptions::default(),
        RenderOptions::default(),
    )
    .expect("encode_audio_to_image should succeed");

    // 2. Re-save the PNG as a JPEG via the image crate.
    let jpg_tmp = tempfile::Builder::new()
        .suffix(".jpg")
        .tempfile()
        .expect("failed to create temp JPEG");
    let jpg_path = jpg_tmp.path().to_path_buf();

    image::open(&png_path)
        .expect("PNG should open")
        .save(&jpg_path)
        .expect("saving as JPEG should succeed");

    // 3. Decode the JPEG back to a WAV and verify non-silent output.
    let out_tmp = tempfile::Builder::new()
        .suffix(".wav")
        .tempfile()
        .expect("failed to create temp output WAV");
    let out_path = out_tmp.path().to_path_buf();

    decode_image_to_wav(&jpg_path, &out_path, SynthesisOptions::default())
        .expect("decode_image_to_wav should accept a JPEG input");

    let (samples, _) = read_wav_samples(&out_path);
    assert!(
        !samples.is_empty(),
        "decoded WAV from JPEG input should not be empty"
    );
    assert!(
        samples.iter().any(|&s| s != 0),
        "decoded WAV from JPEG input should contain non-zero samples"
    );
}
