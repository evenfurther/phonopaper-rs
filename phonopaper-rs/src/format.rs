//! `PhonoPaper` format constants and frequency-index mappings.
//!
//! The `PhonoPaper` format encodes audio as a grayscale spectrogram image:
//! - **X axis (left → right):** time
//! - **Y axis (top → bottom):** frequency, from high (bin 0 ≈ 16 744 Hz,
//!   above hearing) down to low (bin 383 ≈ 66 Hz, C2)
//! - **Pixel brightness:** white = silence, black = maximum amplitude
//!
//! The frequency axis uses a logarithmic (musical) scale with 384 bins.  The
//! musically useful range is approximately **C2 (≈ 65 Hz) to C8 (≈ 4186 Hz)**
//! — 6 audible octaves.  The top 96 bins (rows 0–179 of the data area) fall
//! above C8 and are essentially inaudible; the reference Android application
//! synthesises only bins 96–336 (C3–C8).  The formula uses 4 subdivisions per
//! semitone, giving 48 bins per octave and 384 bins total.

/// Number of frequency subdivisions per semitone.
pub const MULTITONES: usize = 4;

/// Number of semitones in an octave.
pub const SEMITONES_PER_OCTAVE: usize = 12;

/// Number of octaves covered by the format.
pub const OCTAVES: usize = 8;

/// Total number of frequency bins (384 = 8 octaves × 12 semitones × 4 subdivisions).
pub const TOTAL_BINS: usize = OCTAVES * SEMITONES_PER_OCTAVE * MULTITONES;

/// Number of frequency bins per octave (96 = 12 semitones × 4 subdivisions).
pub const BINS_PER_OCTAVE: usize = SEMITONES_PER_OCTAVE * MULTITONES;

/// Lowest frequency in the format — the frequency at bin index `TOTAL_BINS - 1`
/// (≈ 66.36 Hz, C2).
pub const LOW_FREQ: f64 = 66.357_748_918_998_77;

/// Highest frequency in the format — the frequency at bin index `0`
/// (≈ 16 744 Hz, above the human hearing range).
///
/// The highest *audible* bin is index 96, which corresponds to C8 (≈ 4186 Hz).
pub const HIGH_FREQ: f64 = 16_744.036_179_619_156;

/// Standard audio sample rate used for synthesis and analysis.
pub const SAMPLE_RATE: u32 = 44_100;

/// Number of audio samples synthesised per image column by the reference
/// Android `PhonoPaper` application.
///
/// Measured from native Android recordings: the app plays back at exactly
/// **125 columns per second** (`44 100 ÷ 125 = 352.8` samples/column).
/// Because `SPS` must be a compile-time `usize` constant, round to `353`
/// where an integer is required; the timing error is < 0.06 %.
///
/// This library defaults to `353` samples/column — the nearest integer to this
/// value, with < 0.06 % timing error.  Use `512` for higher synthesis
/// resolution at the cost of slower (1.45×) playback.
pub const ANDROID_SPS: f64 = SAMPLE_RATE as f64 / 125.0; // = 352.8

// Compile-time constants used in the frequency formulas, pre-cast to f64 to
// avoid repeated usize→f64 casts in hot paths.
// Values are 48, 252 and 384 — all representable exactly as f64.
#[expect(clippy::cast_precision_loss, reason = "value is 48, exact in f64")]
const BINS_F64: f64 = (SEMITONES_PER_OCTAVE * MULTITONES) as f64; // = 48.0
#[expect(clippy::cast_precision_loss, reason = "value is 252, exact in f64")]
const PIVOT_F64: f64 = (63 * MULTITONES) as f64; // = 252.0
#[expect(clippy::cast_precision_loss, reason = "value is 384, exact in f64")]
const TOTAL_BINS_F64: f64 = TOTAL_BINS as f64; // = 384.0

/// Convert a frequency bin index to its corresponding frequency in Hz.
///
/// - Index `0` → highest frequency (≈ 16 744 Hz, above hearing; top of image)
/// - Index `96` → C8 ≈ 4186 Hz (top of the audible range)
/// - Index `252` → A4 = 440 Hz
/// - Index `TOTAL_BINS - 1` → lowest frequency (≈ 66.36 Hz, C2; bottom of image)
///
/// The formula matches the reference Android `PhonoPaper` application:
/// `freq = 2^((63 × MULTITONES − index) / (12 × MULTITONES)) × 440`
///      `= 2^((252 − index) / 48) × 440`
#[must_use]
pub fn index_to_freq(index: usize) -> f64 {
    // index is at most TOTAL_BINS-1 = 383; the cast to f64 is exact.
    #[expect(clippy::cast_precision_loss, reason = "index ≤ 383, exact in f64")]
    let diff = PIVOT_F64 - index as f64;
    2f64.powf(diff / BINS_F64) * 440.0
}

/// Convert a frequency in Hz to the nearest frequency bin index.
///
/// Returns the bin index whose frequency is closest to the given value.
/// The result is clamped to `[0, TOTAL_BINS - 1]`.
#[must_use]
pub fn freq_to_index(freq: f64) -> usize {
    let diff = (freq / 440.0).log2() * BINS_F64;
    let index = PIVOT_F64 - diff;
    // round() then clamp guarantees the value is in [0, 383] before the cast.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "value is clamped to [0.0, 383.0] before cast, so it is non-negative and fits in usize"
    )]
    let i = index.round().clamp(0.0, TOTAL_BINS_F64 - 1.0) as usize;
    i
}
