//! Encode audio into a `PhonoPaper` image.
//!
//! # Pipeline
//!
//! ```text
//! Audio file  ──► audio_to_spectrogram ──► spectrogram_to_image ──► image file
//! ```
//!
//! The top-level entry point is [`encode_audio_to_image`].

use std::path::Path;

use num_complex::Complex;
use rustfft::FftPlanner;

use crate::error::{PhonoPaperError, Result};
use crate::format::{TOTAL_BINS, index_to_freq};
use crate::spectrogram::{Spectrogram, SpectrogramVec};

/// Options controlling the WAV → spectrogram analysis step.
#[derive(Debug, Clone, Copy)]
pub struct AnalysisOptions {
    /// Largest FFT window in samples.  Must be a power of two.
    ///
    /// Each `PhonoPaper` bin covers 1/48 octave, so its bandwidth is about
    /// 1.4 % of its centre frequency: ≈ 0.9 Hz at the bottom of the grid
    /// (C2) and ≈ 60 Hz at C8.  The encoder therefore uses a
    /// **multi-resolution** analysis: every bin is measured with the smallest
    /// power-of-two window (at least [`MIN_FFT_WINDOW`], at most `fft_window`)
    /// whose FFT resolution is no coarser than the bin's bandwidth.  High bins
    /// keep a good time resolution while low bins get the frequency resolution
    /// they need.
    ///
    /// Defaults to `32768` (≈ 1.35 Hz resolution at 44.1 kHz), enough to
    /// separate adjacent bins down to ≈ 90 Hz and to keep low notes on pitch
    /// down to C2.  A single 4096-sample window (the previous default) is
    /// ≈ 10.8 Hz wide, which smears everything below ≈ 200 Hz across several
    /// semitones.
    pub fft_window: usize,
    /// Hop size in samples between successive image columns.
    ///
    /// Determines the number of image columns per second of audio.
    /// Defaults to `353` — matching Android's column rate of 352.8
    /// (= 44 100 ÷ 125, i.e. 125 columns per second).  At this setting a
    /// decoded image will have the same duration as the source audio.
    pub hop_size: usize,
    /// Lower bound of the dB range mapped to amplitude `0.0` (silence).
    ///
    /// Bins whose level is at or below this produce a white (silent) pixel.
    /// Defaults to `-60.0` dB.
    ///
    /// **Why not `-100.0` dB (the Web Audio API default)?**
    /// A typical microphone recording has a noise floor around −70 to −80 dB.
    /// With `min_db = -100`, even −80 dB noise gets encoded as 22 % pixel
    /// amplitude.  The decoder then synthesises all 384 bins at that level,
    /// producing a constant broadband rumble that dominates quiet passages.
    /// Setting `min_db = -60` keeps the coding range in the musically
    /// meaningful region and makes bins below the noise floor produce silent
    /// (white) pixels.
    pub min_db: f32,
    /// Upper bound of the dB range mapped to amplitude `1.0` (maximum).
    ///
    /// Bins whose level is at or above this produce a black (maximum) pixel.
    /// Defaults to `-10.0` dB.  The level scale follows the Web Audio
    /// `AnalyserNode` convention used by earlier versions of this library: a
    /// full-scale sine (amplitude 1.0) whose energy falls in one bin measures
    /// `0.25` (−12 dBFS), so `-10.0` leaves a small headroom margin.
    pub max_db: f32,
}

impl Default for AnalysisOptions {
    fn default() -> Self {
        Self {
            fft_window: 32_768,
            hop_size: 353,
            min_db: -60.0,
            max_db: -10.0,
        }
    }
}

/// Smallest FFT window used by the multi-resolution analysis (or
/// [`AnalysisOptions::fft_window`] if that is smaller).
///
/// At 44.1 kHz a 2048-sample window resolves 21.5 Hz, which is finer than the
/// bandwidth of every bin above ≈ 1.5 kHz.
pub const MIN_FFT_WINDOW: usize = 2048;

/// Level of a full-scale sine wave on the dB scale used by
/// [`AnalysisOptions::min_db`] / [`AnalysisOptions::max_db`]: `20·log10(0.25)`
/// ≈ −12 dBFS.  A decoder gain of `1 / FULL_SCALE_SINE_LEVEL = 4` therefore
/// reproduces the original level.
pub const FULL_SCALE_SINE_LEVEL: f32 = 0.25;

// ─── Audio → Spectrogram ─────────────────────────────────────────────────────

/// Precompute a periodic Hann window of length `n`.
fn make_hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "FFT window sizes are at most a few hundred thousand, exact in f64"
            )]
            let phase = 2.0 * core::f64::consts::PI * i as f64 / n as f64;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "window coefficients are in [0, 1]; f64→f32 rounding is inaudible"
            )]
            let w = (0.5 - 0.5 * phase.cos()) as f32;
            w
        })
        .collect()
}

/// Lower and upper frequency edges of `PhonoPaper` bin `bin` (± 1/96 octave
/// around its centre), so that adjacent bins tile the axis without gaps.
fn bin_edges(bin: usize) -> (f64, f64) {
    let f = index_to_freq(bin);
    let half = 2f64.powf(1.0 / 96.0);
    (f / half, f * half)
}

/// Sparse weights mapping the power spectrum of one FFT size onto the
/// `PhonoPaper` bins analysed with that size.
struct BandPlan {
    /// FFT size in samples.
    size: usize,
    /// Periodic Hann window of length `size`.
    window: Vec<f32>,
    /// `1 / (size · Σ w²)`.  A sinusoid of peak amplitude `A` whose energy
    /// lies in a band contributes `(A/2)² · size · Σ w²` to the one-sided sum
    /// of `|X[k]|²`, so `A² = 4 · power_scale · Σ|X[k]|²`.
    power_scale: f32,
    /// For each analysed `PhonoPaper` bin: `(bin, first_fft_bin, weights)`.
    bins: Vec<(usize, usize, Vec<f32>)>,
}

/// Choose, for every `PhonoPaper` bin, the FFT size used to analyse it and
/// precompute the band weights.
fn plan_bands(sample_rate: f64, max_window: usize) -> Vec<BandPlan> {
    let min_window = MIN_FFT_WINDOW.min(max_window);
    let mut sizes = Vec::new();
    let mut n = min_window;
    while n <= max_window {
        sizes.push(n);
        n *= 2;
    }

    let mut plans: Vec<BandPlan> = sizes
        .iter()
        .map(|&size| {
            let window = make_hann_window(size);
            let sum_sq: f64 = window.iter().map(|&w| f64::from(w) * f64::from(w)).sum();
            #[expect(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                reason = "size is a power of two ≤ 2^20, exact in f64; the f32 scale factor only needs ~7 digits"
            )]
            let power_scale = (1.0 / (size as f64 * sum_sq)) as f32;
            BandPlan {
                size,
                window,
                power_scale,
                bins: Vec::new(),
            }
        })
        .collect();

    for bin in 0..TOTAL_BINS {
        let (lo, hi) = bin_edges(bin);
        // Smallest window whose resolution is at least as fine as the band.
        #[expect(
            clippy::cast_precision_loss,
            reason = "window sizes are powers of two ≤ 2^20, exact in f64"
        )]
        let idx = sizes
            .iter()
            .position(|&n| sample_rate / n as f64 <= hi - lo)
            .unwrap_or(sizes.len() - 1);
        let plan = &mut plans[idx];
        #[expect(
            clippy::cast_precision_loss,
            reason = "window sizes are powers of two ≤ 2^20, exact in f64"
        )]
        let df = sample_rate / plan.size as f64;
        let nyquist_bin = plan.size / 2;
        // FFT bin k represents the frequency cell [(k − ½)·df, (k + ½)·df);
        // its weight is the fraction of that cell inside [lo, hi).
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "lo, hi > 0 and df > 0, so both indices are small non-negative values"
        )]
        let (k_first, k_last) = (
            (lo / df + 0.5).floor() as usize,
            ((hi / df + 0.5).floor() as usize).min(nyquist_bin),
        );
        if k_first > k_last {
            // The whole band lies above the Nyquist frequency.
            continue;
        }
        let weights = (k_first..=k_last)
            .map(|k| {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "k ≤ size/2 ≤ 2^19, exact in f64"
                )]
                let centre = k as f64 * df;
                let overlap = (hi.min(centre + df / 2.0) - lo.max(centre - df / 2.0)).max(0.0);
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "weights are in [0, 1]; f32 precision is ample"
                )]
                let w = (overlap / df) as f32;
                w
            })
            .collect();
        plan.bins.push((bin, k_first, weights));
    }

    plans.retain(|p| !p.bins.is_empty());
    plans
}

/// Analyse a mono PCM buffer and produce a [`Spectrogram`].
///
/// The spectrogram has one column per `hop_size` samples
/// (`ceil(samples.len() / hop_size)` columns), so a decoded image lasts as
/// long as the source audio.  Column `c` describes the audio around sample
/// `c · hop_size + hop_size / 2`.
///
/// Each `PhonoPaper` bin covers the frequency band `[f/2^(1/96), f·2^(1/96))`
/// around its centre frequency `f`.  Its value is the **energy in that band**
/// (the sum of the power spectrum over the FFT bins it overlaps), measured
/// with a window long enough to resolve the band (see
/// [`AnalysisOptions::fft_window`]).  Summing band energy instead of sampling
/// the spectrum at the bin centre keeps the spectral balance flat: sampling
/// overweights low frequencies (several bins read the same wide FFT bin) and
/// misses energy between high bins (one FFT bin spans less than one band).
///
/// The level is converted to dB (with a full-scale sine measuring
/// [`FULL_SCALE_SINE_LEVEL`]) and mapped linearly from
/// `[min_db, max_db]` to `[0.0, 1.0]`.
///
/// # Errors
///
/// Returns [`PhonoPaperError::InvalidFormat`] if `hop_size` is zero or
/// `fft_window` is not a power of two.
pub fn audio_to_spectrogram(
    samples: &[f32],
    sample_rate: u32,
    options: &AnalysisOptions,
) -> Result<SpectrogramVec> {
    let hop = options.hop_size;
    if hop == 0 {
        return Err(PhonoPaperError::InvalidFormat(
            "hop_size must be > 0".to_string(),
        ));
    }
    if !options.fft_window.is_power_of_two() || options.fft_window < 2 {
        return Err(PhonoPaperError::InvalidFormat(
            "fft_window must be a power of two ≥ 2".to_string(),
        ));
    }

    let num_columns = samples.len().div_ceil(hop);
    let mut spec = Spectrogram::new(num_columns);
    if num_columns == 0 {
        return Ok(spec);
    }

    let plans = plan_bands(f64::from(sample_rate), options.fft_window);
    let min_db = options.min_db;
    let db_range = options.max_db - min_db;
    let mut planner = FftPlanner::<f32>::new();

    for plan in &plans {
        let fft = planner.plan_fft_forward(plan.size);
        let mut buf = vec![Complex::new(0.0f32, 0.0); plan.size];
        let mut scratch = vec![Complex::new(0.0f32, 0.0); fft.get_inplace_scratch_len()];
        let half = plan.size / 2;

        // A window of `size` samples cannot resolve changes faster than about
        // `size / 8` samples, so long windows are evaluated only every
        // `stride` columns and the band levels are interpolated in between.
        // This keeps the cost of the 32 768-sample analysis reasonable without
        // changing the result audibly.
        let stride = (plan.size / (8 * hop)).max(1);
        let mut key_columns: Vec<usize> = (0..num_columns).step_by(stride).collect();
        if key_columns.last() != Some(&(num_columns - 1)) {
            key_columns.push(num_columns - 1);
        }
        let mut prev: Option<(usize, Vec<f32>)> = None;

        for &col in &key_columns {
            // Window centred on the middle of the column; samples outside the
            // signal are zero.
            let centre = col * hop + hop / 2;
            for (i, c) in buf.iter_mut().enumerate() {
                let x = (centre + i)
                    .checked_sub(half)
                    .and_then(|n| samples.get(n))
                    .copied()
                    .unwrap_or(0.0);
                *c = Complex::new(x * plan.window[i], 0.0);
            }
            fft.process_with_scratch(&mut buf, &mut scratch);

            // Peak amplitude of the sinusoids in each band, on the Web
            // Audio scale (full-scale sine = 0.25).
            let levels: Vec<f32> = plan
                .bins
                .iter()
                .map(|(_, k_first, weights)| {
                    let energy: f32 = weights
                        .iter()
                        .zip(&buf[*k_first..])
                        .map(|(&w, x)| w * x.norm_sqr())
                        .sum();
                    (4.0 * energy * plan.power_scale).sqrt() * FULL_SCALE_SINE_LEVEL
                })
                .collect();

            let from = prev.as_ref().map_or(col, |(c, _)| c + 1);
            for fill in from..=col {
                let column = spec.column_mut(fill);
                for (i, (bin, _, _)) in plan.bins.iter().enumerate() {
                    let level = match &prev {
                        Some((pc, pl)) if fill < col => {
                            #[expect(
                                clippy::cast_precision_loss,
                                reason = "column distances are below the stride (a few dozen)"
                            )]
                            let t = (fill - pc) as f32 / (col - pc) as f32;
                            pl[i] + (levels[i] - pl[i]) * t
                        }
                        _ => levels[i],
                    };
                    let db = 20.0 * level.max(1e-10_f32).log10();
                    column[*bin] = ((db - min_db) / db_range).clamp(0.0, 1.0);
                }
            }
            prev = Some((col, levels));
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
