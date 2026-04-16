//! Additive sine-wave synthesis: [`Synthesizer`] and [`spectrogram_to_audio`].
//!
//! Both the stateful [`Synthesizer`] (for real-time column-by-column playback)
//! and the batch [`spectrogram_to_audio`] function live here.  The batch
//! function is implemented as a thin wrapper over [`Synthesizer`] to avoid
//! duplicating the inner hot loops.

use core::f64::consts::PI;

use crate::format::{SAMPLE_RATE, TOTAL_BINS, index_to_freq};

// ─── Options ─────────────────────────────────────────────────────────────────

/// Determines how per-bin stored amplitudes (pixel darkness values in [0, 1])
/// are mapped to linear synthesis amplitudes before the gain is applied.
///
/// ## Which mode to use
///
/// | Image encoded with | Use |
/// |---|---|
/// | [`crate::encode::AnalysisOptions`] (default dB path) | [`AmplitudeMode::Db`] |
/// | No dB mapping (linear normalisation, e.g. the Android app) | [`AmplitudeMode::Linear`] |
#[derive(Debug, Clone, Copy)]
pub enum AmplitudeMode {
    /// Map the stored pixel amplitude directly to the synthesis amplitude.
    ///
    /// `linear_amp = stored_amp` (after any gamma correction).
    ///
    /// An optional binary threshold can be applied: when `threshold` is
    /// `Some(t)`, bins with `stored_amp >= t` are synthesised at full
    /// amplitude (`1.0`) and bins below are silenced.  When `None`, the raw
    /// fractional amplitude is used.
    ///
    /// **Use this mode for images encoded without a dB window** (i.e. when
    /// the pixel brightness was mapped linearly from amplitude, not via a dB
    /// scale).  The reference Android application appears to use a threshold
    /// near `0.85` in linear mode; see `PHONOPAPER_SPEC.md §6.2`.
    ///
    /// A `gain` around `0.15` is appropriate in this mode to avoid clipping.
    Linear {
        /// Optional binarisation threshold in `[0.0, 1.0]`.
        ///
        /// `None` → fractional amplitude; `Some(t)` → on/off at `t`.
        threshold: Option<f32>,
    },

    /// Invert the dB encoding applied by [`crate::encode::AnalysisOptions`]
    /// to recover the true linear magnitude.
    ///
    /// The encoder stored:
    /// ```text
    /// stored_amp = clamp((dB − min_db) / (max_db − min_db), 0.0, 1.0)
    /// ```
    ///
    /// This mode inverts that:
    /// ```text
    /// if stored_amp == 0.0 → linear_amp = 0.0   (silence sentinel)
    /// else                 → dB = min_db + stored_amp × (max_db − min_db)
    ///                         linear_amp = 10 ^ (dB / 20)
    /// ```
    ///
    /// A stored amplitude of `0.0` always means silence regardless of the dB
    /// window — the encoder clamped it to zero because the bin was below
    /// `min_db`.
    ///
    /// **Pixel darkness and output volume in dB mode:**
    ///
    /// - White pixel (luma 255) → stored amplitude `0.0` → silence.
    /// - Black pixel (luma 0)   → stored amplitude `1.0` → maximum loudness.
    ///
    /// The relationship is exponential (dB-linear), not linear.  For the
    /// default window (−60 to −10 dB), mid-gray (`stored_amp = 0.5`) maps
    /// to −35 dBFS, while black maps to −10 dBFS — a 25 dB (18×) difference.
    ///
    /// `min_db` and `max_db` must match the values used during encoding.
    /// A `gain` around `3.0` is appropriate for this mode.
    Db {
        /// Lower bound of the dB window. Must be negative. Defaults to `−60.0`.
        min_db: f32,
        /// Upper bound of the dB window. Must be negative and > `min_db`. Defaults to `−10.0`.
        max_db: f32,
    },
}

impl Default for AmplitudeMode {
    fn default() -> Self {
        Self::Db {
            min_db: -60.0,
            max_db: -10.0,
        }
    }
}

/// Options for audio synthesis.
#[derive(Debug, Clone, Copy)]
pub struct SynthesisOptions {
    /// Output sample rate in Hz. Defaults to [`SAMPLE_RATE`] (44 100 Hz).
    pub sample_rate: u32,
    /// Master gain applied to the output before clipping.
    ///
    /// Scales the synthesized waveform before writing to WAV.  After dB
    /// inversion, per-bin amplitudes equal `web_audio_mag = |X[k]| / fft_size`.
    /// For a Hann-windowed FFT, a sine of amplitude `A` produces
    /// `web_audio_mag = A / 4`, so `gain = 4.0` would give unity amplitude
    /// for a perfectly-aligned single FFT bin.  However, the `PhonoPaper`
    /// log-frequency grid has ≈1.5 PP bins per FFT bin (multiple PP bins read
    /// the same FFT magnitude), boosting the output by ≈√1.5 ≈ 1.22×.
    /// The default `3.0` (= 4 / 1.33) compensates and avoids clipping on
    /// typical music content.  Lower this if you still hear distortion.
    ///
    /// Use `AmplitudeMode::Linear` with `gain ≈ 0.15` for images encoded
    /// without a dB window.
    ///
    /// ## Pixel darkness and output volume
    ///
    /// In the `PhonoPaper` format, **darker pixels = louder sound**:
    ///
    /// - White pixel (luma 255) → stored amplitude `0.0` → silence.
    /// - Black pixel (luma 0)   → stored amplitude `1.0` → maximum loudness.
    ///
    /// With dB inversion active (the default), the stored amplitude maps to a
    /// *dB value* before synthesis, so the relationship between pixel darkness
    /// and output volume is exponential (dB-linear), not linear.  Specifically:
    ///
    /// ```text
    /// stored_amp = (dB − min_db) / (max_db − min_db)
    /// linear_amp = 10 ^ (dB / 20)
    /// ```
    ///
    /// This means halving the pixel darkness (going from black to mid-gray)
    /// does *not* halve the output amplitude — it reduces it by `db_range / 2`
    /// decibels.  For the default window of −60 to −10 dB, mid-gray (stored
    /// amp 0.5) corresponds to −35 dBFS, while black corresponds to −10 dBFS —
    /// a 25 dB difference, or about 18× in linear amplitude.
    pub gain: f32,
    /// Gamma correction exponent for decoding pixel brightness to amplitude.
    ///
    /// The pixel-to-amplitude mapping is:
    ///
    /// ```text
    /// amplitude = (1 − pixel / 255) ^ (1 / gamma)
    /// ```
    ///
    /// This is the exact inverse of the encoding step in [`crate::render::RenderOptions::gamma`],
    /// which applies `pixel = round((1 − amplitude^gamma) × 255)`.  To decode
    /// an image correctly you must use the same gamma value that was used to
    /// encode it.
    ///
    /// Defaults to `1.0` (linear, no correction).  Set to the encoder's gamma
    /// (e.g. `0.15`) when decoding images that were encoded with gamma
    /// compression.
    pub decode_gamma: f32,
    /// Amplitude mapping mode — controls how stored pixel amplitudes are
    /// converted to linear synthesis amplitudes.
    ///
    /// See [`AmplitudeMode`] for the available variants and their semantics.
    /// Defaults to [`AmplitudeMode::Db`] with the standard `−60` to `−10` dB
    /// window, matching the default [`crate::encode::AnalysisOptions`].
    pub mode: AmplitudeMode,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            sample_rate: SAMPLE_RATE,
            gain: 3.0,
            decode_gamma: 1.0,
            mode: AmplitudeMode::default(),
        }
    }
}

// ─── Stateful synthesizer ─────────────────────────────────────────────────────

/// Stateful additive-synthesis engine for **real-time, column-by-column**
/// audio playback.
///
/// The const generic parameter `SPS` is the number of audio samples produced
/// per image column (samples-per-step).  It defaults to `353`, the closest
/// integer to the Android app's exact rate of 352.8 (= 44 100 ÷ 125,
/// i.e. 125 columns per second); the timing error is < 0.06 %.  All internal
/// scratch buffers
/// are stack-allocated `[f32; SPS]` arrays, so no heap allocator is required.
///
/// Unlike [`spectrogram_to_audio`], which processes a complete [`Spectrogram`]
/// in one shot, `Synthesizer` owns the phasor state between calls and exposes a
/// [`synthesize_column`](Synthesizer::synthesize_column) method that accepts one
/// column of amplitude values at a time.  This is the correct abstraction for
/// the **sliding-camera** use case:
///
/// 1. Load the image and call [`super::detect_markers`] **once** to get a
///    [`super::DataBounds`] value.
/// 2. Create a `Synthesizer` with your [`SynthesisOptions`].
/// 3. In your camera / playhead loop, call
///    [`super::column_amplitudes_from_image`] for the current x position, then
///    feed the result to [`synthesize_column`](Synthesizer::synthesize_column).
/// 4. The method writes the PCM burst into the `output` slice you supply.
///
/// Because the phasor state is preserved across calls, there are **no clicks or
/// phase discontinuities** at column boundaries even when columns are requested
/// at arbitrary intervals.
///
/// ## Example
///
/// ```no_run
/// use phonopaper_rs::decode::{
///     Synthesizer, SynthesisOptions, column_amplitudes_from_image, detect_markers,
/// };
///
/// let image = image::open("code.png").unwrap();
/// let bounds = detect_markers(&image).unwrap();
/// // SPS = 353 samples per column (the default, matching Android).
/// let mut synth = Synthesizer::<353>::new(SynthesisOptions::default());
/// let mut pcm = [0.0_f32; 353];
///
/// // Simulate the camera sliding across column 42.
/// let amps = column_amplitudes_from_image(&image, Some(bounds), 42).unwrap();
/// synth.synthesize_column(&amps, &mut pcm);
/// // Send `pcm` to the audio output device…
/// ```
#[derive(Debug, Clone)]
pub struct Synthesizer<const SPS: usize = 353> {
    /// Synthesis parameters (sample rate and gain).
    options: SynthesisOptions,
    /// Per-bin 1-sample step rotor — real part: `cos(ω)`.
    rot_c: [f32; TOTAL_BINS],
    /// Per-bin 1-sample step rotor — imaginary part: `sin(ω)`.
    rot_s: [f32; TOTAL_BINS],
    /// Per-bin 4-sample step rotor — real part: `cos(4ω)`.
    step4_c: [f32; TOTAL_BINS],
    /// Per-bin 4-sample step rotor — imaginary part: `sin(4ω)`.
    step4_s: [f32; TOTAL_BINS],
    /// Per-bin full-column skip rotor — real part: `cos(ω · SPS)`.
    skip_c: [f32; TOTAL_BINS],
    /// Per-bin full-column skip rotor — imaginary part: `sin(ω · SPS)`.
    skip_s: [f32; TOTAL_BINS],
    /// Persistent phasor real parts across columns; initialised to `1.0`.
    ph_c: [f32; TOTAL_BINS],
    /// Persistent phasor imaginary parts across columns; initialised to `0.0`.
    ph_s: [f32; TOTAL_BINS],
    /// Internal scratch buffer — accumulates per-column amplitude sum.
    buf: [f32; SPS],
    /// Internal scratch buffer — precomputed phasor imaginary parts.
    ph_im: [f32; SPS],
}

impl<const SPS: usize> Synthesizer<SPS> {
    /// Create a new `Synthesizer` with the given [`SynthesisOptions`].
    ///
    /// All phasors are initialised to phase zero.  The rotor tables are
    /// precomputed once here and reused for every subsequent
    /// [`synthesize_column`](Self::synthesize_column) call.
    #[must_use]
    pub fn new(options: SynthesisOptions) -> Self {
        let sr = f64::from(options.sample_rate);

        let mut rot_c = [0.0_f32; TOTAL_BINS];
        let mut rot_s = [0.0_f32; TOTAL_BINS];
        let mut step4_c = [0.0_f32; TOTAL_BINS];
        let mut step4_s = [0.0_f32; TOTAL_BINS];
        let mut skip_c = [0.0_f32; TOTAL_BINS];
        let mut skip_s = [0.0_f32; TOTAL_BINS];

        #[expect(
            clippy::cast_precision_loss,
            reason = "SPS is typically 512; values this small are exact in f64"
        )]
        let sps_f64 = SPS as f64;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "all angles are bounded f64; f64→f32 narrowing loses < 1e-7 relative error, inaudible"
        )]
        for bin in 0..TOTAL_BINS {
            let omega = 2.0 * PI * index_to_freq(bin) / sr;
            rot_c[bin] = omega.cos() as f32;
            rot_s[bin] = omega.sin() as f32;
            step4_c[bin] = (4.0 * omega).cos() as f32;
            step4_s[bin] = (4.0 * omega).sin() as f32;
            skip_c[bin] = (omega * sps_f64).cos() as f32;
            skip_s[bin] = (omega * sps_f64).sin() as f32;
        }

        Self {
            options,
            rot_c,
            rot_s,
            step4_c,
            step4_s,
            skip_c,
            skip_s,
            ph_c: [1.0_f32; TOTAL_BINS],
            ph_s: [0.0_f32; TOTAL_BINS],
            buf: [0.0_f32; SPS],
            ph_im: [0.0_f32; SPS],
        }
    }

    /// Synthesize one column of audio from a slice of per-bin amplitudes.
    ///
    /// `amplitudes` must have exactly [`TOTAL_BINS`] elements (one per
    /// frequency bin, index 0 = highest frequency).  The values should be in
    /// `[0.0, 1.0]`; values outside that range are not clamped on input but
    /// the output is clamped to `[-1.0, 1.0]` after gain is applied.
    ///
    /// The amplitude mapping is controlled by [`SynthesisOptions::mode`]:
    /// in [`AmplitudeMode::Linear`] mode an optional threshold binarises each
    /// bin; in [`AmplitudeMode::Db`] mode the stored amplitude is inverted
    /// through the dB window to recover the true linear magnitude.
    ///
    /// The `SPS` mono PCM samples in `[-1.0, 1.0]` are written into `output`.
    /// The internal phasor state is updated so that the next call continues
    /// without a phase discontinuity.
    ///
    /// # Panics
    ///
    /// Panics if `amplitudes.len() != TOTAL_BINS`.
    pub fn synthesize_column(&mut self, amplitudes: &[f32], output: &mut [f32; SPS]) {
        assert_eq!(
            amplitudes.len(),
            TOTAL_BINS,
            "amplitudes slice must have exactly TOTAL_BINS ({TOTAL_BINS}) elements, got {}",
            amplitudes.len(),
        );

        let gain = self.options.gain;
        let decode_gamma = self.options.decode_gamma;
        let mode = self.options.mode;

        self.buf.fill(0.0);

        for (bin, &amp_raw) in amplitudes.iter().enumerate() {
            // Step 1 — map stored amplitude to linear synthesis amplitude.
            let amp_linear = match mode {
                AmplitudeMode::Db { min_db, max_db } => {
                    if amp_raw == 0.0 {
                        0.0
                    } else {
                        let db_range = max_db - min_db;
                        let db = min_db + amp_raw * db_range;
                        10.0_f32.powf(db / 20.0)
                    }
                }
                AmplitudeMode::Linear { .. } => amp_raw,
            };

            // Step 2 — apply inverse gamma correction: amp = amp ^ (1/decode_gamma).
            // This reverses the encoder's `pixel = (1 - amp^gamma) * 255` step.
            // `powf(1.0)` is a no-op in IEEE 754, so no fast-path is needed.
            let amp_gamma = amp_linear.powf(1.0 / decode_gamma);

            // Step 3 — apply optional binary threshold (only in Linear mode).
            let amp = match mode {
                AmplitudeMode::Linear { threshold: Some(t) } => {
                    if amp_gamma >= t {
                        1.0_f32
                    } else {
                        0.0_f32
                    }
                }
                _ => amp_gamma,
            };

            if amp == 0.0 {
                // Silent bin: O(1) skip via the full-column skip rotor.
                let sc = self.skip_c[bin];
                let ss = self.skip_s[bin];
                let pc = self.ph_c[bin];
                let ps = self.ph_s[bin];
                self.ph_c[bin] = pc * sc - ps * ss;
                self.ph_s[bin] = pc * ss + ps * sc;
            } else {
                // Active bin: 4-lane polyphase precompute + vectorised accumulate.
                fill_phasor_im(
                    &mut self.ph_im,
                    self.ph_c[bin],
                    self.ph_s[bin],
                    self.rot_c[bin],
                    self.rot_s[bin],
                    self.step4_c[bin],
                    self.step4_s[bin],
                );

                // Advance persistent phasor via the skip rotor.
                let sc = self.skip_c[bin];
                let ss = self.skip_s[bin];
                let pc = self.ph_c[bin];
                let ps = self.ph_s[bin];
                self.ph_c[bin] = pc * sc - ps * ss;
                self.ph_s[bin] = pc * ss + ps * sc;

                for (b, &im) in self.buf.iter_mut().zip(self.ph_im.iter()) {
                    *b += amp * im;
                }
            }
        }

        for (o, &b) in output.iter_mut().zip(self.buf.iter()) {
            *o = (b * gain).clamp(-1.0, 1.0);
        }
    }

    /// Reset all phasors to phase zero.
    ///
    /// Call this when starting a new playback session or after a large
    /// discontinuous jump in the playhead position where phase continuity is
    /// no longer meaningful.
    pub fn reset(&mut self) {
        self.ph_c.fill(1.0);
        self.ph_s.fill(0.0);
    }

    /// Renormalise all phasors to unit magnitude.
    ///
    /// Floating-point rounding causes each complex multiply to introduce a tiny
    /// magnitude error (≈ 1 × 10⁻⁷ per step for `f32`).  After many thousand
    /// columns the phasor magnitudes can drift from 1.0, changing the output
    /// amplitude.  Periodic renormalisation prevents this drift.
    ///
    /// A phasor at zero magnitude is reset to phase zero (`(1.0, 0.0)`) to
    /// avoid a division-by-zero.
    pub fn renormalize(&mut self) {
        for bin in 0..TOTAL_BINS {
            let norm = (self.ph_c[bin] * self.ph_c[bin] + self.ph_s[bin] * self.ph_s[bin]).sqrt();
            if norm > 0.0 {
                self.ph_c[bin] /= norm;
                self.ph_s[bin] /= norm;
            } else {
                self.ph_c[bin] = 1.0;
                self.ph_s[bin] = 0.0;
            }
        }
    }

    /// Return a reference to the [`SynthesisOptions`] this synthesizer was
    /// created with.
    #[must_use]
    pub fn options(&self) -> &SynthesisOptions {
        &self.options
    }
}

// ─── Phasor helpers ───────────────────────────────────────────────────────────

/// Fill `ph_im` with the imaginary parts of the phasor sequence starting at
/// `(ph_re_init, ph_im_init)` using a 4-lane polyphase recurrence.
///
/// The 4 independent lanes (each advanced by `z⁴`) give the CPU's out-of-order
/// engine 4-way ILP on the serial multiply chain, cutting effective latency from
/// ~4 cycles/step to ~1 cycle/step.
///
/// `rot_c / rot_s` are `cos(ω) / sin(ω)` (1-step rotor);
/// `step4_c / step4_s` are `cos(4ω) / sin(4ω)` (4-step rotor).
#[expect(
    clippy::inline_always,
    reason = "inlining is required so that ph_im is a local scratch buffer visible \
              to the caller's accumulate loop; without it LLVM may re-fuse the two \
              passes and revert to a scalar reduction"
)]
#[inline(always)]
fn fill_phasor_im(
    ph_im: &mut [f32],
    ph_re_init: f32,
    ph_im_init: f32,
    rot_c: f32,
    rot_s: f32,
    step4_c: f32,
    step4_s: f32,
) {
    // Initialise the 4 lane phasors at offsets z^0 … z^3.
    let (c0, s0) = (ph_re_init, ph_im_init);
    let c1 = c0 * rot_c - s0 * rot_s;
    let s1 = c0 * rot_s + s0 * rot_c;
    let c2 = c1 * rot_c - s1 * rot_s;
    let s2 = c1 * rot_s + s1 * rot_c;
    let c3 = c2 * rot_c - s2 * rot_s;
    let s3 = c2 * rot_s + s2 * rot_c;

    let mut lc0 = c0;
    let mut ls0 = s0;
    let mut lc1 = c1;
    let mut ls1 = s1;
    let mut lc2 = c2;
    let mut ls2 = s2;
    let mut lc3 = c3;
    let mut ls3 = s3;

    // Main loop: groups of 4 samples — all 4 lanes step by z^4 independently.
    let chunks = ph_im.chunks_exact_mut(4);
    for chunk in chunks {
        chunk[0] = ls0;
        chunk[1] = ls1;
        chunk[2] = ls2;
        chunk[3] = ls3;
        let nc0 = lc0 * step4_c - ls0 * step4_s;
        ls0 = lc0 * step4_s + ls0 * step4_c;
        lc0 = nc0;
        let nc1 = lc1 * step4_c - ls1 * step4_s;
        ls1 = lc1 * step4_s + ls1 * step4_c;
        lc1 = nc1;
        let nc2 = lc2 * step4_c - ls2 * step4_s;
        ls2 = lc2 * step4_s + ls2 * step4_c;
        lc2 = nc2;
        let nc3 = lc3 * step4_c - ls3 * step4_s;
        ls3 = lc3 * step4_s + ls3 * step4_c;
        lc3 = nc3;
    }
    // Scalar tail for SPS values not divisible by 4 (e.g. SPS=353).
    // With SPS=353, this handles the final 1 sample (353 = 88*4 + 1).
    for im in ph_im.chunks_exact_mut(4).into_remainder() {
        *im = ls0;
        let nc = lc0 * rot_c - ls0 * rot_s;
        ls0 = lc0 * rot_s + ls0 * rot_c;
        lc0 = nc;
    }
}

// ─── Batch synthesis ──────────────────────────────────────────────────────────

/// Number of columns between phasor renormalisation steps.
///
/// `f32` accumulates ≈ 1.2 × 10⁻⁷ relative magnitude error per complex
/// multiply.  After 128 columns of `SPS = 353` steps per column, the worst-case
/// drift is `(1 + 1.2e-7)^(128 × 353) ≈ 1.005` — about 0.5 %.  Renormalising
/// every 128 columns keeps the error below 0.5 % at the cost of one pass over
/// `TOTAL_BINS` (= 384) `f32` values — negligible compared to synthesis.
const RENORM_INTERVAL: usize = 128;

/// Synthesize audio from a complete [`Spectrogram`], writing into a
/// caller-supplied output buffer.
///
/// `output` must have length `spec.num_columns() * SPS`.
///
/// For each time column the active frequency bins are identified.  Their
/// corresponding sine-wave oscillators are stepped forward by `SPS` samples.
/// The phases of all oscillators are maintained continuously across columns to
/// avoid clicks at column boundaries.
///
/// ## Implementation — three-table phasor synthesis with 4-lane polyphase
///
/// This function is a thin wrapper over [`Synthesizer`]: it constructs one
/// synthesizer from `options`, then calls
/// [`Synthesizer::synthesize_column`] for each column.  All performance
/// properties of `Synthesizer` therefore apply here too:
///
/// ### Rotor tables
///
/// Three precomputed tables are built once before the main loop:
///
/// | Table | Value | Purpose |
/// |---|---|---|
/// | `rot_c / rot_s` | `cos(ω)`, `sin(ω)` | per-sample phasor step |
/// | `step4_c / step4_s` | `cos(4ω)`, `sin(4ω)` | 4-lane polyphase advance |
/// | `skip_c / skip_s` | `cos(ω·SPS)`, `sin(ω·SPS)` | silent-bin O(1) skip |
///
/// ### Column-skip rotor (silent bins)
///
/// For a **silent** bin the phasor must still advance by exactly `SPS` samples
/// so that phase continuity is preserved when that bin becomes active again.
/// The skip rotor reduces this from O(`SPS`) serial multiplies to a single
/// complex multiply.  For the realistic benchmark (every 8th bin active,
/// 336 silent bins per column) this alone eliminates ~241 M serial operations.
///
/// ### Active bins — 4-lane polyphase phasor precompute
///
/// #### Why a single serial recurrence is slow
///
/// The phasor recurrence `p_{n+1} = p_n · z` has a **loop-carried dependence**:
/// every step depends on the previous one.  The FPU multiply latency on modern
/// x86 is ~4 cycles, so `SPS = 512` steps cost ≈ 2048 cycles — a hard serial
/// bottleneck that even LLVM cannot remove.
///
/// #### The 4-lane solution
///
/// Instead of advancing one phasor, advance **4 independent phasors** in
/// parallel, each starting at a different offset within the same oscillator:
///
/// ```text
/// lane 0: p₀,  p₄,  p₈,  …   (samples 0, 4, 8, …)
/// lane 1: p₁,  p₅,  p₉,  …   (samples 1, 5, 9, …)
/// lane 2: p₂,  p₆,  p₁₀, …   (samples 2, 6, 10, …)
/// lane 3: p₃,  p₇,  p₁₁, …   (samples 3, 7, 11, …)
/// ```
///
/// All four lanes advance by `z⁴` per iteration, so their recurrences are
/// **completely independent**.  The CPU's out-of-order engine pipelines the
/// four multiply chains, turning a 4-cycle-latency serial loop into an
/// effectively 1-cycle-per-group throughput loop (4× ILP improvement).
///
/// #### Accumulate pass
///
/// `buf[k] += amp * ph_im[k]` has no loop-carried dependence and is
/// auto-vectorised by LLVM as 8-wide AVX2 FMA instructions.
///
/// # Panics
///
/// Panics if `output.len() != spec.num_columns() * SPS`.
pub fn spectrogram_to_audio<S: AsRef<[f32]>, const SPS: usize>(
    spec: &crate::spectrogram::Spectrogram<S>,
    options: &SynthesisOptions,
    output: &mut [f32],
) {
    let num_columns = spec.num_columns();
    assert_eq!(
        output.len(),
        num_columns * SPS,
        "output.len() must equal spec.num_columns() * SPS ({} * {} = {}), got {}",
        num_columns,
        SPS,
        num_columns * SPS,
        output.len(),
    );

    let mut synth = Synthesizer::<SPS>::new(*options);

    for col in 0..num_columns {
        // Periodically renormalise phasors to prevent f32 magnitude drift from
        // accumulating over long spectrograms (see RENORM_INTERVAL).
        if col > 0 && col % RENORM_INTERVAL == 0 {
            synth.renormalize();
        }

        let chunk = &mut output[col * SPS..(col + 1) * SPS];
        // SAFETY: chunk has exactly SPS elements by the bounds above.
        let chunk_arr: &mut [f32; SPS] = chunk
            .try_into()
            .expect("slice length is SPS by construction");
        // SAFETY: col is in 0..num_columns by the loop bounds above.
        synth.synthesize_column(spec.column_or_panic(col), chunk_arr);
    }
}
