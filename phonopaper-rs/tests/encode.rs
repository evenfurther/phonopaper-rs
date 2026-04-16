//! Tests for [`phonopaper_rs::encode::audio_to_spectrogram`].
//!
//! Strategy: generate synthetic PCM buffers (pure sines, dual tones, silence)
//! and verify that the recovered [`Spectrogram`] amplitudes match expectations.

use std::f32::consts::PI;

use phonopaper_rs::encode::{AnalysisOptions, audio_to_spectrogram};
use phonopaper_rs::format::{SAMPLE_RATE, freq_to_index};

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Generate `num_samples` of a pure sine at `freq_hz` with peak amplitude 1.0.
fn sine_wave(freq_hz: f32, sample_rate: u32, num_samples: usize) -> Vec<f32> {
    #[expect(
        clippy::cast_precision_loss,
        reason = "sample_rate = 44_100, exact in f32 (< 2^24)"
    )]
    let sr = sample_rate as f32;
    (0..num_samples)
        .map(|t| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "t < num_samples ≤ 44_100; exact in f32 (< 2^17)"
            )]
            (2.0 * PI * freq_hz * t as f32 / sr).sin()
        })
        .collect()
}

/// `AnalysisOptions` with a large window for good frequency resolution.
fn test_options() -> AnalysisOptions {
    AnalysisOptions {
        fft_window: 4096,
        hop_size: 512,
        min_db: -100.0,
        max_db: -10.0,
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

/// Encoding a pure sine at A4 (440 Hz) produces a spectrogram whose bin for
/// 440 Hz has clearly higher amplitude than a distant bin.
#[test]
fn pure_tone_lands_on_correct_bin() {
    let freq = 440.0_f32;
    let sr = SAMPLE_RATE;
    // Generate enough audio for several FFT frames.
    let samples = sine_wave(freq, sr, sr as usize); // 1 second

    let opts = test_options();
    let spec = audio_to_spectrogram(&samples, sr, &opts).unwrap();
    assert!(spec.num_columns() > 0, "expected at least one frame");

    let target_bin = freq_to_index(f64::from(freq));
    // A bin well away from 440 Hz — use 200 Hz (bin for ~200 Hz).
    let other_bin = freq_to_index(200.0);

    // Average amplitude across all frames for each bin.
    let avg = |bin: usize| {
        let sum: f32 = (0..spec.num_columns()).map(|c| spec.get(c, bin)).sum();
        #[expect(
            clippy::cast_precision_loss,
            reason = "num_columns is a frame count bounded by test input length; \
                      values used here are at most a few hundred, exact in f32"
        )]
        {
            sum / spec.num_columns() as f32
        }
    };
    let amp_target = avg(target_bin);
    let amp_other = avg(other_bin);

    assert!(
        amp_target > amp_other * 5.0,
        "bin {target_bin} (440 Hz) avg amplitude {amp_target:.4} should be much larger \
         than bin {other_bin} (200 Hz) avg amplitude {amp_other:.4}"
    );
    assert!(
        amp_target > 0.5,
        "expected strong amplitude at target bin, got {amp_target:.4}"
    );
}

/// A full-scale sine wave produces amplitude close to 1.0 at the target bin.
#[test]
fn pure_tone_amplitude_near_one() {
    let freq = 440.0_f32;
    let sr = SAMPLE_RATE;
    let samples = sine_wave(freq, sr, sr as usize);

    let opts = test_options();
    let spec = audio_to_spectrogram(&samples, sr, &opts).unwrap();

    let target_bin = freq_to_index(f64::from(freq));
    // Take a middle frame to avoid edge effects.
    let mid = spec.num_columns() / 2;
    let amp = spec.get(mid, target_bin);

    assert!(
        amp > 0.8,
        "expected amplitude near 1.0 for full-scale sine at target bin, got {amp:.4}"
    );
}

/// Encoding a silent buffer produces a spectrogram with all amplitudes near zero.
#[test]
fn silence_produces_near_zero_spectrogram() {
    let sr = SAMPLE_RATE;
    let samples = vec![0.0f32; sr as usize];

    let opts = test_options();
    let spec = audio_to_spectrogram(&samples, sr, &opts).unwrap();

    let max_amp = (0..spec.num_columns())
        .flat_map(|c| (0..phonopaper_rs::format::TOTAL_BINS).map(move |b| (c, b)))
        .map(|(c, b)| spec.get(c, b))
        .fold(0.0f32, f32::max);

    assert!(
        max_amp < 0.01,
        "silent audio should produce near-zero spectrogram, max amplitude = {max_amp:.6}"
    );
}

/// Encoding a sum of two sines produces a spectrogram where both target bins
/// are active and a distant silent bin is near zero.
#[test]
fn dual_tone_both_bins_active() {
    let freq_a = 440.0_f32; // A4
    let freq_b = 880.0_f32; // A5
    let sr = SAMPLE_RATE;
    let n = sr as usize;

    // Sum of two unit-amplitude sines, normalised to stay in [-1, 1].
    let samples: Vec<f32> = (0..n)
        .map(|t| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "t < 44_100; exact in f32 (< 2^17)"
            )]
            let ta = t as f32;
            #[expect(
                clippy::cast_precision_loss,
                reason = "sr = SAMPLE_RATE = 44_100; exact in f32 (< 2^24)"
            )]
            let sr_f = sr as f32;
            0.5 * (2.0 * PI * freq_a * ta / sr_f).sin()
                + 0.5 * (2.0 * PI * freq_b * ta / sr_f).sin()
        })
        .collect();

    let opts = test_options();
    let spec = audio_to_spectrogram(&samples, sr, &opts).unwrap();

    let bin_a = freq_to_index(f64::from(freq_a));
    let bin_b = freq_to_index(f64::from(freq_b));
    // A bin far from both tones (around 100 Hz).
    let bin_silent = freq_to_index(100.0);

    let avg = |bin: usize| {
        let sum: f32 = (0..spec.num_columns()).map(|c| spec.get(c, bin)).sum();
        #[expect(
            clippy::cast_precision_loss,
            reason = "num_columns is a frame count bounded by test input; \
                      values here are at most a few hundred, exact in f32"
        )]
        {
            sum / spec.num_columns() as f32
        }
    };

    let amp_a = avg(bin_a);
    let amp_b = avg(bin_b);
    let amp_silent = avg(bin_silent);

    assert!(
        amp_a > amp_silent * 5.0,
        "bin {bin_a} (440 Hz) avg {amp_a:.4} should dominate silent bin {bin_silent} ({amp_silent:.4})"
    );
    assert!(
        amp_b > amp_silent * 5.0,
        "bin {bin_b} (880 Hz) avg {amp_b:.4} should dominate silent bin {bin_silent} ({amp_silent:.4})"
    );
}

/// `audio_to_spectrogram` returns an error when `fft_window` is zero.
#[test]
fn zero_fft_window_is_error() {
    let opts = AnalysisOptions {
        fft_window: 0,
        hop_size: 512,
        min_db: -100.0,
        max_db: -10.0,
    };
    let result = audio_to_spectrogram(&[0.0f32; 1024], SAMPLE_RATE, &opts);
    assert!(result.is_err(), "expected error for fft_window = 0");
}

/// `audio_to_spectrogram` returns an error when `hop_size` is zero.
#[test]
fn zero_hop_size_is_error() {
    let opts = AnalysisOptions {
        fft_window: 1024,
        hop_size: 0,
        min_db: -100.0,
        max_db: -10.0,
    };
    let result = audio_to_spectrogram(&[0.0f32; 1024], SAMPLE_RATE, &opts);
    assert!(result.is_err(), "expected error for hop_size = 0");
}

/// A sample buffer shorter than `fft_window` yields zero FFT frames, so
/// `audio_to_spectrogram` returns an empty (0-column) spectrogram rather than
/// an error.
#[test]
fn too_short_input_returns_empty_spectrogram() {
    let opts = AnalysisOptions {
        fft_window: 1024,
        hop_size: 512,
        min_db: -100.0,
        max_db: -10.0,
    };
    // 512 samples < fft_window (1024) → 0 frames.
    let spec = audio_to_spectrogram(&[0.0f32; 512], SAMPLE_RATE, &opts)
        .expect("short input should not be an error");
    assert_eq!(
        spec.num_columns(),
        0,
        "a sample buffer shorter than fft_window should produce a 0-column spectrogram"
    );
}

/// `audio_to_spectrogram` with a buffer of exactly `fft_window` samples
/// produces exactly 1 FFT frame.
#[test]
fn exactly_one_fft_window_gives_one_frame() {
    let opts = AnalysisOptions {
        fft_window: 512,
        hop_size: 256,
        min_db: -100.0,
        max_db: -10.0,
    };
    let samples = vec![0.0f32; 512]; // exactly fft_window
    let spec = audio_to_spectrogram(&samples, SAMPLE_RATE, &opts)
        .expect("buffer of exactly fft_window should produce one frame");
    assert_eq!(
        spec.num_columns(),
        1,
        "exactly one window yields exactly one frame"
    );
}
