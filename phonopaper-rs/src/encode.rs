//! Encode audio into a `PhonoPaper` image.
//!
//! # Pipeline
//!
//! ```text
//! Audio file  ──► audio_to_spectrogram ──► spectrogram_to_image ──► image file
//! ```
//!
//! The top-level entry point is [`encode_audio_to_image`].

use std::f32::consts::PI;
use std::path::Path;

use num_complex::Complex;
use rustfft::FftPlanner;

use crate::error::{PhonoPaperError, Result};
use crate::format::{TOTAL_BINS, index_to_freq};
use crate::spectrogram::{Spectrogram, SpectrogramVec};

/// Options controlling the WAV → spectrogram analysis step.
#[derive(Debug, Clone, Copy)]
pub struct AnalysisOptions {
    /// Size of the FFT window in samples.
    ///
    /// Larger windows give better frequency resolution but coarser time
    /// resolution. Must be a power of two. Defaults to `4096`.
    pub fft_window: usize,
    /// Hop size in samples between successive FFT frames.
    ///
    /// Determines the number of image columns per second of audio.
    /// Defaults to `353` — matching Android's column rate of 352.8
    /// (= 44 100 ÷ 125, i.e. 125 columns per second).  At this setting a
    /// decoded image will have the same duration as the source audio.
    pub hop_size: usize,
    /// Lower bound of the dB range mapped to amplitude `0.0` (silence).
    ///
    /// FFT magnitudes at or below this level produce a white (silent) pixel.
    /// Defaults to `-60.0` dB.
    ///
    /// **Why not `-100.0` dB (the Web Audio API default)?**
    /// A typical microphone recording has a noise floor around −70 to −80 dB.
    /// With `min_db = -100`, even −80 dB noise gets encoded as 22 % pixel
    /// amplitude.  The decoder then synthesises all 384 bins at that level,
    /// producing a constant broadband rumble that dominates quiet passages and
    /// skews the spectral balance toward low frequencies (where more PP bins
    /// share each FFT bin).  Setting `min_db = -60` keeps the coding range in
    /// the musically meaningful region and makes bins below the noise floor
    /// produce silent (white) pixels.
    pub min_db: f32,
    /// Upper bound of the dB range mapped to amplitude `1.0` (maximum).
    ///
    /// FFT magnitudes at or above this level produce a black (maximum) pixel.
    /// Defaults to `-10.0` dB.  For a Hann-windowed FFT a full-scale sine
    /// (amplitude 1.0) produces `web_audio_mag = 0.25` (= -12 dBFS), so
    /// `-10.0` leaves a small headroom margin without saturating real-world
    /// audio.  Note: the Web Audio API `AnalyserNode` uses `-30.0` dB as its
    /// `maxDecibels` default, which saturates any bin louder than -30 dBFS and
    /// is unsuitable for accurate round-trip audio encoding.
    pub max_db: f32,
}

impl Default for AnalysisOptions {
    fn default() -> Self {
        Self {
            fft_window: 4096,
            hop_size: 353,
            min_db: -60.0,
            max_db: -10.0,
        }
    }
}

// ─── Audio → Spectrogram ─────────────────────────────────────────────────────

/// Precompute a Hann window of length `n`.
fn make_hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "FFT window sizes are at most 65536 in practice, well within f32 precision"
            )]
            {
                0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos())
            }
        })
        .collect()
}

/// Analyse a mono PCM buffer and produce a [`Spectrogram`].
///
/// The function performs an STFT on `samples` using the given `options`.  Each
/// FFT frame is mapped to the `PhonoPaper` logarithmic frequency grid by finding,
/// for every `PhonoPaper` bin, the FFT magnitude that corresponds to that bin's
/// centre frequency.
///
/// Returns a [`Spectrogram`] where the number of time columns equals the number
/// of STFT frames.
///
/// # Errors
///
/// Returns [`PhonoPaperError::InvalidFormat`] if `fft_window` or `hop_size` is
/// zero.
pub fn audio_to_spectrogram(
    samples: &[f32],
    sample_rate: u32,
    options: &AnalysisOptions,
) -> Result<SpectrogramVec> {
    let fft_size = options.fft_window;
    let hop = options.hop_size;
    let sr = f64::from(sample_rate);

    if fft_size == 0 || hop == 0 {
        return Err(PhonoPaperError::InvalidFormat(
            "fft_window and hop_size must be > 0".to_string(),
        ));
    }

    // Pre-compute the fractional FFT bin index for each PhonoPaper frequency
    // bin.  Storing f64 fractional indices allows linear interpolation between
    // the two adjacent FFT bins, which eliminates the staircase artefact that
    // would otherwise appear at high frequencies where the log-spaced PP grid
    // is denser than the FFT grid (multiple PP bins collapse onto the same FFT
    // bin when only the nearest integer bin is used).
    let phono_to_fft_f: Vec<f64> = (0..TOTAL_BINS)
        .map(|bin| {
            let freq = index_to_freq(bin);
            // freq > 0 and fft_size/sr > 0, so the product is always positive.
            // Clamp to [0, fft_size/2] (the Nyquist limit).
            #[expect(
                clippy::cast_precision_loss,
                reason = "fft_size ≤ 65536 in practice; the cast to f64 is exact for these values"
            )]
            (freq * fft_size as f64 / sr).clamp(0.0, (fft_size / 2) as f64)
        })
        .collect();

    // dB-scale normalisation matching the Web Audio API / JS reference encoder.
    //
    // The Web Audio AnalyserNode divides the raw FFT output by `fft_size` before
    // computing dB, so a full-scale sine has magnitude 1.0 / fft_size per bin.
    // We replicate this: web_audio_mag = |X[k]| / fft_size.
    // Then: dB = 20 * log10(web_audio_mag)
    //       amplitude = clamp((dB - min_db) / (max_db - min_db), 0.0, 1.0)
    //
    // fft_size ≤ 65536 so the cast to f32 is exact (well within 23-bit mantissa).
    #[expect(
        clippy::cast_precision_loss,
        reason = "fft_size ≤ 65536, which is exact in f32 (23-bit mantissa covers up to 2^24)"
    )]
    let fft_size_f32 = fft_size as f32;
    let min_db = options.min_db;
    let max_db = options.max_db;
    let db_range = max_db - min_db;

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);

    // Count frames.
    let num_frames = if samples.len() >= fft_size {
        (samples.len() - fft_size) / hop + 1
    } else {
        0
    };

    if num_frames == 0 {
        return Ok(Spectrogram::new(0));
    }

    let mut spec = Spectrogram::new(num_frames);
    let mut scratch = vec![Complex::new(0.0f32, 0.0); fft.get_outofplace_scratch_len()];

    // Preallocate reusable per-frame buffers (avoids allocation inside the loop).
    let hann_win = make_hann_window(fft_size);
    let mut window = vec![Complex::new(0.0f32, 0.0); fft_size];
    let mut out = vec![Complex::new(0.0f32, 0.0); fft_size];

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop;
        let end = (start + fft_size).min(samples.len());

        // Copy samples into the window buffer, applying the Hann window to the
        // real part immediately; zero-pad if this is the last (short) frame.
        let src = &samples[start..end];
        for (i, c) in window.iter_mut().enumerate() {
            c.re = if i < src.len() {
                src[i] * hann_win[i]
            } else {
                0.0
            };
            c.im = 0.0;
        }

        fft.process_outofplace_with_scratch(&mut window, &mut out, &mut scratch);

        // Map FFT magnitudes to PhonoPaper bins using dB-scale normalisation.
        // Linear interpolation between the two adjacent FFT bins that bracket
        // the exact fractional bin index removes the staircase artefact at
        // high frequencies where multiple PP bins share the same FFT bin.
        for (bin, &frac_idx) in phono_to_fft_f.iter().enumerate().take(TOTAL_BINS) {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "frac_idx is clamped to [0, fft_size/2] which fits in usize; floor() is non-negative"
            )]
            let lo = frac_idx.floor() as usize;
            let hi = (lo + 1).min(fft_size / 2);
            #[expect(
                clippy::cast_possible_truncation,
                reason = "frac().abs() < 1.0; the f64→f32 cast loses < 1e-7 relative error, inaudible"
            )]
            let t = frac_idx.fract() as f32;
            // Web Audio normalisation: divide raw magnitude by fft_size.
            let mag_lo = out[lo].norm() / fft_size_f32;
            let mag_hi = out[hi].norm() / fft_size_f32;
            let web_audio_mag = mag_lo * (1.0 - t) + mag_hi * t;
            // Convert to dB; clamp to a small positive value to avoid log(0).
            let db = 20.0 * web_audio_mag.max(1e-10_f32).log10();
            // Map [min_db, max_db] linearly to [0.0, 1.0] and clamp.
            let amplitude = ((db - min_db) / db_range).clamp(0.0, 1.0);
            spec.set(frame_idx, bin, amplitude);
        }
    }

    Ok(spec)
}

// ─── Top-level entry point ────────────────────────────────────────────────────

/// Encode an audio file into a `PhonoPaper` PNG image.
///
/// # Arguments
///
/// * `audio_path`  – Path to the input audio file (WAV or MP3; mono
///   or stereo; stereo will be downmixed to mono).
/// * `image_path`  – Path where the output PNG image will be written.
/// * `analysis`    – STFT analysis parameters (use `Default::default()` for
///   sensible defaults).
/// * `render`      – Image rendering parameters (use `Default::default()` for
///   sensible defaults).
///
/// # Errors
///
/// Returns a [`PhonoPaperError`] if the audio file cannot be read or the image
/// cannot be saved.
///
/// # Example
///
/// ```no_run
/// use phonopaper_rs::encode::encode_audio_to_image;
///
/// encode_audio_to_image("input.wav", "code.png", Default::default(), Default::default()).unwrap();
/// ```
pub fn encode_audio_to_image(
    audio_path: impl AsRef<Path>,
    image_path: impl AsRef<Path>,
    analysis: AnalysisOptions,
    render: crate::render::RenderOptions,
) -> Result<()> {
    // 1. Read the audio file (WAV or MP3) and downmix to mono.
    let (mono, sample_rate) = crate::audio::read_audio_file(audio_path)?;

    // 2. Analyse: audio → spectrogram.
    let spectrogram = audio_to_spectrogram(&mono, sample_rate, &analysis)?;

    // 3. Render: spectrogram → image.
    let img = crate::render::spectrogram_to_image(&spectrogram, &render);

    // 4. Save the image.
    img.save(image_path)?;

    Ok(())
}
