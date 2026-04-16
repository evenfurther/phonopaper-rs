//! Shared test utilities used across multiple integration test files.

use core::f64::consts::PI;

/// Compute the normalised DFT power at frequency `freq_hz` for `samples`.
///
/// Returns `|Σ sample[t] · exp(-i 2π f t / sr)| / N`, which is maximised
/// when the signal contains a sinusoidal component at exactly `freq_hz`.
///
/// This is a direct correlation (Goertzel-style) and has O(N) cost per
/// frequency.  It is suitable for short test buffers (a few thousand samples)
/// but not for production use.
pub fn dft_power_at(samples: &[f32], freq_hz: f64, sample_rate: u32) -> f64 {
    let sr = f64::from(sample_rate);
    let n = samples.len();
    let mut re = 0.0f64;
    let mut im = 0.0f64;
    for (t, &s) in samples.iter().enumerate() {
        #[expect(
            clippy::cast_precision_loss,
            reason = "t is a sample index; test buffers are at most a few seconds \
                      at 44_100 Hz (< 2^17), well within f64's 52-bit mantissa"
        )]
        let angle = 2.0 * PI * freq_hz * t as f64 / sr;
        re += f64::from(s) * angle.cos();
        im += f64::from(s) * angle.sin();
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "n is a test buffer length bounded by SR*seconds (< 2^17); exact in f64"
    )]
    let norm = n as f64;
    (re * re + im * im).sqrt() / norm
}
