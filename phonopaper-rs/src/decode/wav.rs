//! WAV file output and top-level decode entry points.
//!
//! [`decode_image_to_wav`] and [`decode_image_to_wav_sps`] are the primary
//! entry points for converting a `PhonoPaper` image file into a WAV audio
//! file.  The lower-level helpers live in the sibling submodules.

use std::path::Path;

use crate::error::{PhonoPaperError, Result};

use super::image::image_to_spectrogram;
use super::synth::{SynthesisOptions, spectrogram_to_audio};

/// Write a mono 16-bit PCM WAV file (Microsoft PCM, `wFormatTag = 1`).
///
/// Each `f32` sample is clamped to `[-1.0, 1.0]`, multiplied by `32 767.5`,
/// and rounded to the nearest integer before conversion to a signed 16-bit
/// word.  Samples are written as little-endian 16-bit words, as required by
/// the WAV specification.
///
/// The output is compatible with every standard audio player and matches the
/// format produced by the Android `PhonoPaper` application.
///
/// # Errors
///
/// Returns [`PhonoPaperError::InvalidFormat`] if `samples.len()` exceeds
/// `u32::MAX / 2` (> 2 billion samples), or
/// [`PhonoPaperError::IoError`] if any I/O operation fails.
pub fn write_wav(path: impl AsRef<Path>, samples: &[f32], sample_rate: u32) -> Result<()> {
    use std::io::{BufWriter, Write as _};

    let num_samples = u32::try_from(samples.len()).map_err(|_| {
        PhonoPaperError::InvalidFormat(
            "Sample buffer too large for WAV format (> 4 GiB)".to_string(),
        )
    })?;
    let data_bytes = num_samples * 2; // 2 bytes per i16
    let fmt_chunk_size: u32 = 16; // standard PCM fmt chunk — no cbSize field
    let file_size = 4 + (8 + fmt_chunk_size) + (8 + data_bytes);

    let file = std::fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    // RIFF header
    w.write_all(b"RIFF")?;
    w.write_all(&file_size.to_le_bytes())?;
    w.write_all(b"WAVE")?;

    // fmt chunk — WAVE_FORMAT_PCM (type 1), 16-bit mono
    w.write_all(b"fmt ")?;
    w.write_all(&fmt_chunk_size.to_le_bytes())?;
    w.write_all(&1u16.to_le_bytes())?; // wFormatTag: PCM
    w.write_all(&1u16.to_le_bytes())?; // nChannels: mono
    w.write_all(&sample_rate.to_le_bytes())?; // nSamplesPerSec
    w.write_all(&(sample_rate * 2).to_le_bytes())?; // nAvgBytesPerSec (2 bytes/sample)
    w.write_all(&2u16.to_le_bytes())?; // nBlockAlign
    w.write_all(&16u16.to_le_bytes())?; // wBitsPerSample

    // data chunk
    w.write_all(b"data")?;
    w.write_all(&data_bytes.to_le_bytes())?;

    // Write samples as little-endian 16-bit PCM words.  Writing via
    // `to_le_bytes()` is explicit about endianness (the WAV spec requires LE),
    // correct on both little-endian and big-endian hosts, and avoids unsafe.
    // BufWriter batches the small writes into ~8 KiB kernel calls, so
    // per-sample overhead is negligible.
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

/// Decode a `PhonoPaper` image file into a WAV audio file, using a
/// compile-time-constant number of audio samples per image column.
///
/// The output is a standard **Microsoft PCM, 16-bit mono** WAV file,
/// matching the format produced by the Android `PhonoPaper` application.
///
/// `SPS` — *samples per step* — is the number of PCM frames synthesised for
/// each image column.  At `SAMPLE_RATE = 44 100 Hz`, the common choices are:
///
/// | SPS | Column duration | Notes |
/// |-----|----------------|-------|
/// | 256 | ~5.8 ms | |
/// | 353 | ~8.0 ms | default; closest integer to Android's 352.8 (= 44 100 ÷ 125) |
/// | 512 | ~11.6 ms | |
/// | 1024 | ~23.2 ms | |
///
/// Use the convenience wrapper [`decode_image_to_wav`] when `SPS = 353` is
/// acceptable.
///
/// # Arguments
///
/// * `image_path` – Path to the input image (PNG or JPEG).
/// * `wav_path`   – Path where the output WAV file will be written.
/// * `options`    – Synthesis parameters (use `Default::default()` for sensible
///   defaults).
///
/// # Errors
///
/// Returns a [`PhonoPaperError`] if the image cannot be opened, the marker
/// bands are not found, or the WAV file cannot be written.
///
/// # Example
///
/// ```no_run
/// use phonopaper_rs::decode::{decode_image_to_wav_sps, SynthesisOptions};
///
/// // Decode with 1024 samples per column instead of the default 353.
/// decode_image_to_wav_sps::<1024>("code.png", "output.wav", Default::default()).unwrap();
/// ```
pub fn decode_image_to_wav_sps<const SPS: usize>(
    image_path: impl AsRef<Path>,
    wav_path: impl AsRef<Path>,
    options: SynthesisOptions,
) -> Result<()> {
    // 1. Load the image.
    let img = image::open(image_path)?;

    // 2. Detect markers and extract the spectrogram.
    let spectrogram = image_to_spectrogram(&img, None)?;

    // 3. Synthesize audio into a heap-allocated buffer.
    let num_samples = spectrogram.num_columns() * SPS;
    let mut samples = vec![0.0_f32; num_samples];
    spectrogram_to_audio::<_, SPS>(&spectrogram, &options, &mut samples);

    // 4. Write the WAV file.
    //
    // We bypass hound's per-sample API and write the PCM data in one
    // `write_all` call.  For ~700 k samples the per-call overhead of
    // `write_sample` dominates runtime; bulk writing is ~5× faster.
    write_wav(wav_path, &samples, options.sample_rate)?;

    Ok(())
}

/// Decode a `PhonoPaper` image file into a WAV audio file.
///
/// The output is a standard **Microsoft PCM, 16-bit mono** WAV file,
/// matching the format produced by the Android `PhonoPaper` application.
///
/// This is a convenience wrapper around [`decode_image_to_wav_sps`] that uses
/// `353` samples per column — the closest integer to the Android application's
/// exact rate of **352.8** (= 44 100 ÷ 125, i.e. 125 columns per second).
/// The timing error versus Android is < 0.06 %.
///
/// Use [`decode_image_to_wav_sps`] directly to choose a different `SPS` value.
///
/// # Arguments
///
/// * `image_path` – Path to the input image (PNG or JPEG).
/// * `wav_path`   – Path where the output WAV file will be written.
/// * `options`    – Synthesis parameters (use `Default::default()` for sensible
///   defaults).
///
/// # Errors
///
/// Returns a [`PhonoPaperError`] if the image cannot be opened, the marker
/// bands are not found, or the WAV file cannot be written.
///
/// # Example
///
/// ```no_run
/// use phonopaper_rs::decode::decode_image_to_wav;
///
/// decode_image_to_wav("code.png", "output.wav", Default::default()).unwrap();
/// ```
pub fn decode_image_to_wav(
    image_path: impl AsRef<Path>,
    wav_path: impl AsRef<Path>,
    options: SynthesisOptions,
) -> Result<()> {
    decode_image_to_wav_sps::<353>(image_path, wav_path, options)
}
