//! Integration tests for [`phonopaper_rs::audio::read_audio_file`].
//!
//! Tests cover:
//! - Reading a WAV file (happy path, round-trip sample count and sample rate).
//! - Reading a mono MP3 file (happy path, non-empty result, correct sample rate).
//! - Reading a stereo MP3 file (happy path, stereo downmixed to mono).
//! - Error on an unrecognised file extension.
//! - Error on a non-existent file.
//! - `encode_audio_to_image` called with an MP3 path produces a valid `PhonoPaper` image.

use std::io::Write as _;

use phonopaper_rs::audio::read_audio_file;
use phonopaper_rs::encode::{AnalysisOptions, encode_audio_to_image};
use phonopaper_rs::render::RenderOptions;

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Write a minimal 16-bit mono PCM WAV file containing a 440 Hz sine wave and
/// return `(temp_file_guard, path)`.
fn write_sine_wav(
    sample_rate: u32,
    num_samples: usize,
) -> (tempfile::NamedTempFile, std::path::PathBuf) {
    use std::f32::consts::TAU;

    let mut tmp = tempfile::Builder::new()
        .suffix(".wav")
        .tempfile()
        .expect("failed to create temp WAV file");

    let freq = 440.0_f32;
    #[expect(
        clippy::cast_precision_loss,
        reason = "sample index is small; f32 precision is sufficient for test sine synthesis"
    )]
    let samples: Vec<i16> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "value is in [-16000, 16000] which fits in i16"
            )]
            let s = (f32::sin(TAU * freq * t) * 16_000.0) as i16;
            s
        })
        .collect();

    let num_samples_u32 = u32::try_from(num_samples).expect("fits in u32");
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

// ─── WAV tests ────────────────────────────────────────────────────────────────

/// `read_audio_file` on a 44 100 Hz mono WAV returns the correct sample rate
/// and the expected sample count.
#[test]
fn test_read_wav_sample_rate_and_count() {
    let (_tmp, path) = write_sine_wav(44_100, 8_000);
    let (samples, sample_rate) =
        read_audio_file(&path).expect("read_audio_file should succeed on a WAV");
    assert_eq!(sample_rate, 44_100);
    assert_eq!(
        samples.len(),
        8_000,
        "mono sample count must equal the WAV frame count"
    );
}

/// `read_audio_file` on a 22 050 Hz WAV returns the correct (non-default) sample rate.
#[test]
fn test_read_wav_non_standard_sample_rate() {
    let (_tmp, path) = write_sine_wav(22_050, 4_000);
    let (_, sample_rate) = read_audio_file(&path).expect("read_audio_file should succeed");
    assert_eq!(sample_rate, 22_050);
}

/// `read_audio_file` on a WAV returns non-zero samples (the input is a sine wave,
/// not silence).
#[test]
fn test_read_wav_non_zero_samples() {
    let (_tmp, path) = write_sine_wav(44_100, 4_096);
    let (samples, _) = read_audio_file(&path).expect("read_audio_file should succeed");
    assert!(
        samples.iter().any(|&s| s != 0.0),
        "sine WAV should produce non-zero samples"
    );
}

// ─── MP3 tests ────────────────────────────────────────────────────────────────

/// Resolved path to the mono silence MP3 fixture.
fn silence_mp3() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/silence.mp3")
}

/// Resolved path to the stereo silence MP3 fixture.
fn silence_stereo_mp3() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/silence_stereo.mp3")
}

/// Resolved path to the mono 440 Hz sine MP3 fixture.
fn sine_440hz_mp3() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sine_440hz.mp3")
}

/// `read_audio_file` on the mono silence MP3 fixture returns the correct sample
/// rate and a non-empty sample vector.
#[test]
fn test_read_mp3_mono_silence() {
    let (samples, sample_rate) =
        read_audio_file(silence_mp3()).expect("read_audio_file should succeed on silence.mp3");
    assert_eq!(sample_rate, 44_100);
    assert!(
        !samples.is_empty(),
        "MP3 fixture should produce at least one sample"
    );
}

/// `read_audio_file` on the stereo silence MP3 fixture returns correct sample
/// rate and downmixes to a mono (single-plane) result.
#[test]
fn test_read_mp3_stereo_downmix() {
    let (samples, sample_rate) = read_audio_file(silence_stereo_mp3())
        .expect("read_audio_file should succeed on stereo MP3");
    assert_eq!(sample_rate, 44_100);
    assert!(
        !samples.is_empty(),
        "stereo MP3 fixture should produce samples after downmix"
    );
    // The result is mono: the sample count should be ≈ 1 × duration_seconds × sample_rate.
    // A 1-second file at 44100 Hz → around 44100 samples (with possible encoder padding/lead-in).
    assert!(
        samples.len() > 40_000 && samples.len() < 50_000,
        "stereo→mono downmix of 1-second 44100 Hz MP3 should yield ~44100 samples, got {}",
        samples.len()
    );
}

/// `read_audio_file` on the 440 Hz sine MP3 fixture returns non-zero samples.
#[test]
fn test_read_mp3_sine_non_zero() {
    let (samples, _) =
        read_audio_file(sine_440hz_mp3()).expect("read_audio_file should succeed on sine MP3");
    assert!(!samples.is_empty(), "sine MP3 fixture must produce samples");
    assert!(
        samples.iter().any(|&s| s != 0.0),
        "440 Hz sine MP3 should produce non-zero samples"
    );
}

// ─── Error cases ──────────────────────────────────────────────────────────────

/// `read_audio_file` with an unrecognised extension returns `InvalidFormat`.
#[test]
fn test_read_audio_unknown_extension_is_error() {
    let tmp = tempfile::Builder::new()
        .suffix(".ogg")
        .tempfile()
        .expect("failed to create temp file");
    let result = read_audio_file(tmp.path());
    assert!(
        result.is_err(),
        "unrecognised extension should return an error"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("ogg") || err.contains("format") || err.contains("Unsupported"),
        "error message should mention the unsupported format: {err}"
    );
}

/// `read_audio_file` with a non-existent path returns an error.
#[test]
fn test_read_audio_missing_file_is_error() {
    let result = read_audio_file("/nonexistent/does_not_exist.mp3");
    assert!(result.is_err(), "missing file should return an error");
}

// ─── encode_audio_to_image with MP3 input ───────────────────────────────────────

/// `encode_audio_to_image` with an MP3 path (using the new `read_audio_file`
/// back-end) produces a valid `PhonoPaper` image: `detect_markers` must succeed.
#[test]
fn test_encode_mp3_to_image() {
    use phonopaper_rs::decode::detect_markers;

    let img_tmp = tempfile::Builder::new()
        .suffix(".png")
        .tempfile()
        .expect("failed to create temp PNG");
    let img_path = img_tmp.path().to_path_buf();

    encode_audio_to_image(
        sine_440hz_mp3(),
        &img_path,
        AnalysisOptions::default(),
        RenderOptions::default(),
    )
    .expect("encode_audio_to_image should succeed with an MP3 input");

    let img = image::open(&img_path).expect("output PNG should be openable");
    detect_markers(&img)
        .expect("detect_markers should succeed on a PhonoPaper image encoded from MP3");
}
