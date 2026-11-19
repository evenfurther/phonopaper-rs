//! `PhonoPaper` format constants and frequency-index mappings.
//!
//! The `PhonoPaper` format encodes audio as a grayscale spectrogram image:
//! - **X axis (left → right):** time
//! - **Y axis (top → bottom):** frequency, from high (bin 0 ≈ 16 504 Hz,
//!   above hearing) down to low (bin 383 = 65.406 Hz, C2)
//! - **Pixel brightness:** white = silence, black = maximum amplitude
//!
//! The frequency axis uses a logarithmic (musical) scale with 384 bins.  The
//! grid starts at **C2 = 65.406 Hz** at the bottom of the image (the value of
//! `PHONOPAPER_LOWEST_FREQ` in the official `PhonoPaper` / Virtual ANS
//! programs, and "from 65.4 Hz" in the official specification image) and
//! spans 8 octaves upwards.  Bin 95 is C8 (4186 Hz).  The formula uses 4
//! subdivisions per semitone, giving 48 bins per octave and 384 bins total.

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
/// (C2 = 65.406 Hz).
///
/// This is `PHONOPAPER_LOWEST_FREQ` in the official `PhonoPaper` and Virtual
/// ANS programs (`65.406_36` as an `f32`), and the official specification
/// image states "low freqs from 65.4Hz".
pub const LOW_FREQ: f64 = 65.406_391_325_149_66;

/// Highest frequency in the format — the frequency at bin index `0`
/// (≈ 16 504 Hz, `LOW_FREQ × 2^(383/48)`).
///
/// Index 95 corresponds to C8 (≈ 4186 Hz).
pub const HIGH_FREQ: f64 = 16_503.980_258_575_066;

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
// Values are 48, 251 and 384 — all representable exactly as f64.
#[expect(clippy::cast_precision_loss, reason = "value is 48, exact in f64")]
const BINS_F64: f64 = (SEMITONES_PER_OCTAVE * MULTITONES) as f64; // = 48.0
/// Bin index of A4 = 440 Hz.  Bin 383 is then `440 × 2^(−132/48)` = C2,
/// i.e. exactly 11 semitones (44 bins) below A4 plus two octaves.
#[expect(clippy::cast_precision_loss, reason = "value is 251, exact in f64")]
const PIVOT_F64: f64 = (TOTAL_BINS - 1 - 33 * MULTITONES) as f64; // = 251.0
#[expect(clippy::cast_precision_loss, reason = "value is 384, exact in f64")]
const TOTAL_BINS_F64: f64 = TOTAL_BINS as f64; // = 384.0

/// Convert a frequency bin index to its corresponding frequency in Hz.
///
/// - Index `0` → highest frequency (≈ 16 504 Hz, above hearing; top of image)
/// - Index `95` → C8 ≈ 4186 Hz
/// - Index `251` → A4 = 440 Hz
/// - Index `TOTAL_BINS - 1` → lowest frequency (C2 = 65.406 Hz; bottom of image)
///
/// The official application computes `LOWEST_FREQ × 2^(row_from_bottom / 48)`
/// with `LOWEST_FREQ = 65.406 Hz`, which is equivalent to:
/// `freq = 2^((251 − index) / 48) × 440`
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
