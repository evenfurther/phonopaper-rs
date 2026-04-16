//! Tests for [`phonopaper_rs::decode::spectrogram_to_audio`],
//! [`phonopaper_rs::decode::Synthesizer`], and
//! [`phonopaper_rs::decode::column_amplitudes_from_image`].
//!
//! Strategy: build a [`Spectrogram`] programmatically, synthesize audio, then
//! verify the frequency content of the output using manual DFT correlation
//! (no external FFT dependency needed for these small checks).

#[path = "helpers/mod.rs"]
mod helpers;
use helpers::dft_power_at;

use phonopaper_rs::decode::{AmplitudeMode, SynthesisOptions, Synthesizer, spectrogram_to_audio};
use phonopaper_rs::format::{SAMPLE_RATE, TOTAL_BINS, freq_to_index, index_to_freq};
use phonopaper_rs::spectrogram::Spectrogram;

// ─── helpers ─────────────────────────────────────────────────────────────────

/// `SynthesisOptions` tuned for tests: unit gain so amplitudes are predictable,
/// and enough samples per column for clean frequency resolution.
fn test_options() -> SynthesisOptions {
    SynthesisOptions {
        sample_rate: SAMPLE_RATE,
        gain: 1.0,
        decode_gamma: 1.0,
        mode: AmplitudeMode::Linear { threshold: None },
    }
}

// ─── spectrogram_to_audio tests ───────────────────────────────────────────────

/// A spectrogram with a single active bin synthesizes audio that contains
/// predominantly the corresponding frequency.
#[test]
fn pure_tone_single_bin() {
    const NUM_COLS: usize = 4;
    const SPS: usize = 1024;

    let target_freq = 440.0_f64; // A4
    let bin = freq_to_index(target_freq);
    let actual_freq = index_to_freq(bin); // exact frequency for this bin

    let mut buf = [0.0f32; NUM_COLS * TOTAL_BINS];
    let mut spec = Spectrogram::from_storage(NUM_COLS, &mut buf[..]).unwrap();
    for col in 0..NUM_COLS {
        spec.set(col, bin, 1.0);
    }

    let opts = test_options();
    let mut samples = [0.0_f32; NUM_COLS * SPS];
    spectrogram_to_audio::<_, SPS>(&spec, &opts, &mut samples);

    // Power at the exact bin frequency should be much larger than at a
    // clearly different frequency (660 Hz = a minor third above A4).
    let power_at_target = dft_power_at(&samples, actual_freq, opts.sample_rate);
    let power_at_other = dft_power_at(&samples, 660.0, opts.sample_rate);

    assert!(
        power_at_target > power_at_other * 5.0,
        "expected strong component at {actual_freq:.1} Hz (power {power_at_target:.4}) \
         but got similar power at 660 Hz ({power_at_other:.4})"
    );
}

/// A silent spectrogram (all amplitudes zero) produces all-zero audio output.
#[test]
fn silence_produces_zero_audio() {
    const NUM_COLS: usize = 8;
    const SPS: usize = 1024;

    let buf = [0.0f32; NUM_COLS * TOTAL_BINS];
    let spec = Spectrogram::from_storage(NUM_COLS, &buf[..]).unwrap();
    let opts = test_options();
    #[expect(
        clippy::large_stack_arrays,
        reason = "test output buffer; deliberately stack-allocated to verify no_std compatibility"
    )]
    let mut samples = [0.0_f32; NUM_COLS * SPS];
    spectrogram_to_audio::<_, SPS>(&spec, &opts, &mut samples);

    assert!(
        samples.iter().all(|&s| s == 0.0),
        "silent spectrogram should produce zero audio"
    );
    assert_eq!(
        samples.len(),
        NUM_COLS * SPS,
        "output length should be num_columns * SPS"
    );
}

/// A spectrogram with two active bins synthesizes audio containing both
/// corresponding frequencies, each stronger than an unrelated frequency.
#[test]
fn dual_tone_two_bins() {
    const NUM_COLS: usize = 4;
    const SPS: usize = 1024;

    let freq_a = 440.0_f64; // A4
    let freq_b = 880.0_f64; // A5

    let bin_a = freq_to_index(freq_a);
    let bin_b = freq_to_index(freq_b);
    let actual_a = index_to_freq(bin_a);
    let actual_b = index_to_freq(bin_b);

    let mut buf = [0.0f32; NUM_COLS * TOTAL_BINS];
    let mut spec = Spectrogram::from_storage(NUM_COLS, &mut buf[..]).unwrap();
    for col in 0..NUM_COLS {
        spec.set(col, bin_a, 1.0);
        spec.set(col, bin_b, 1.0);
    }

    let opts = test_options();
    let mut samples = [0.0_f32; NUM_COLS * SPS];
    spectrogram_to_audio::<_, SPS>(&spec, &opts, &mut samples);

    let power_a = dft_power_at(&samples, actual_a, opts.sample_rate);
    let power_b = dft_power_at(&samples, actual_b, opts.sample_rate);
    // 550 Hz sits between the two tones and should be much weaker.
    let power_mid = dft_power_at(&samples, 550.0, opts.sample_rate);

    assert!(
        power_a > power_mid * 3.0,
        "expected strong A4 component ({power_a:.4}) vs 550 Hz ({power_mid:.4})"
    );
    assert!(
        power_b > power_mid * 3.0,
        "expected strong A5 component ({power_b:.4}) vs 550 Hz ({power_mid:.4})"
    );
}

/// Halving the amplitude in the spectrogram roughly halves the peak signal
/// magnitude (tested with gain=1.0 and a single bin).
#[test]
fn amplitude_scales_output() {
    const NUM_COLS: usize = 4;
    const SPS: usize = 1024;

    let bin = freq_to_index(440.0);
    let opts = test_options();

    let rms = |v: &[f32]| {
        let sum_sq: f64 = v.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
        #[expect(
            clippy::cast_precision_loss,
            reason = "v.len() is a sample count bounded by test buffer size; exact in f64"
        )]
        (sum_sq / v.len() as f64).sqrt()
    };

    let mut buf_full = [0.0f32; NUM_COLS * TOTAL_BINS];
    let mut spec_full = Spectrogram::from_storage(NUM_COLS, &mut buf_full[..]).unwrap();
    for col in 0..NUM_COLS {
        spec_full.set(col, bin, 1.0);
    }
    let mut samples_full = [0.0_f32; NUM_COLS * SPS];
    spectrogram_to_audio::<_, SPS>(&spec_full, &opts, &mut samples_full);

    let mut buf_half = [0.0f32; NUM_COLS * TOTAL_BINS];
    let mut spec_half = Spectrogram::from_storage(NUM_COLS, &mut buf_half[..]).unwrap();
    for col in 0..NUM_COLS {
        spec_half.set(col, bin, 0.5);
    }
    let mut samples_half = [0.0_f32; NUM_COLS * SPS];
    spectrogram_to_audio::<_, SPS>(&spec_half, &opts, &mut samples_half);

    let rms_full = rms(&samples_full);
    let rms_half = rms(&samples_half);

    // RMS of the half-amplitude signal should be within 5% of half the full.
    let ratio = rms_half / rms_full;
    assert!(
        (ratio - 0.5).abs() < 0.05,
        "expected RMS ratio ≈ 0.5, got {ratio:.4}"
    );
}

/// Output length equals `num_columns * SPS` for a non-trivial spectrogram.
#[test]
fn output_length_is_correct() {
    const SPS: usize = 256;
    const NUM_COLS: usize = 7;

    let bin = freq_to_index(440.0);
    let mut buf = [0.0f32; NUM_COLS * TOTAL_BINS];
    let mut spec = Spectrogram::from_storage(NUM_COLS, &mut buf[..]).unwrap();
    spec.set(0, bin, 1.0);

    let opts = SynthesisOptions {
        sample_rate: SAMPLE_RATE,
        gain: 1.0,
        decode_gamma: 1.0,
        mode: AmplitudeMode::Linear { threshold: None },
    };
    let mut samples = [0.0_f32; NUM_COLS * SPS];
    spectrogram_to_audio::<_, SPS>(&spec, &opts, &mut samples);
    assert_eq!(samples.len(), NUM_COLS * SPS);
}

// ─── Synthesizer tests ────────────────────────────────────────────────────────

/// Synthesizing one column with `Synthesizer` produces the same PCM as the
/// first `SPS` samples from `spectrogram_to_audio` on a 1-column spectrogram.
///
/// Both start at phase zero, so the outputs must be bit-for-bit identical.
#[test]
fn synthesizer_single_column_matches_full() {
    const SPS: usize = 512;

    let bin = freq_to_index(440.0);
    let mut amps = [0.0_f32; TOTAL_BINS];
    amps[bin] = 0.8;

    let opts = SynthesisOptions {
        sample_rate: SAMPLE_RATE,
        gain: 1.0,
        decode_gamma: 1.0,
        mode: AmplitudeMode::Linear { threshold: None },
    };

    // Reference: spectrogram_to_audio on a 1-column spectrogram.
    let mut spec_buf = [0.0f32; TOTAL_BINS];
    let mut spec = Spectrogram::from_storage(1, &mut spec_buf[..]).unwrap();
    spec.column_mut(0).copy_from_slice(&amps);
    let mut reference = [0.0_f32; SPS];
    spectrogram_to_audio::<_, SPS>(&spec, &opts, &mut reference);

    // Under test: Synthesizer::synthesize_column.
    let mut synth = Synthesizer::<SPS>::new(opts);
    let mut result = [0.0_f32; SPS];
    synth.synthesize_column(&amps, &mut result);

    for (i, (&r, &e)) in result.iter().zip(reference.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-6,
            "sample {i}: synthesizer={r} vs spectrogram_to_audio={e} (diff={})",
            (r - e).abs()
        );
    }
}

/// Consecutive `synthesize_column` calls are phase-continuous: the last sample
/// of one burst and the first sample of the next must not form a discontinuity
/// larger than the per-sample change within each burst.
#[test]
fn synthesizer_phase_continuous() {
    const SPS: usize = 512;

    let bin = freq_to_index(440.0);
    let mut amps = [0.0_f32; TOTAL_BINS];
    amps[bin] = 1.0;

    let opts = SynthesisOptions {
        sample_rate: SAMPLE_RATE,
        gain: 1.0,
        decode_gamma: 1.0,
        mode: AmplitudeMode::Linear { threshold: None },
    };

    let mut synth = Synthesizer::<SPS>::new(opts);
    let mut col0 = [0.0_f32; SPS];
    let mut col1 = [0.0_f32; SPS];
    synth.synthesize_column(&amps, &mut col0);
    synth.synthesize_column(&amps, &mut col1);

    // The jump at the column boundary (last sample of col0 → first of col1)
    // should be no larger than the largest within-column step.
    let boundary_jump = (col1[0] - col0[SPS - 1]).abs();

    let max_within_step = col0
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        boundary_jump <= max_within_step * 1.5,
        "phase discontinuity at column boundary: jump={boundary_jump:.6} \
         but max within-column step={max_within_step:.6}"
    );
}

/// `Synthesizer::reset` returns the phasor to phase zero so the next call
/// produces the same output as a freshly created synthesizer would.
#[test]
fn synthesizer_reset_restores_initial_state() {
    const SPS: usize = 256;

    let bin = freq_to_index(880.0);
    let mut amps = [0.0_f32; TOTAL_BINS];
    amps[bin] = 0.5;

    let opts = SynthesisOptions {
        sample_rate: SAMPLE_RATE,
        gain: 1.0,
        decode_gamma: 1.0,
        mode: AmplitudeMode::Linear { threshold: None },
    };

    // Fresh synthesizer: first column output.
    let mut synth = Synthesizer::<SPS>::new(opts);
    let mut first_run = [0.0_f32; SPS];
    synth.synthesize_column(&amps, &mut first_run);

    // Advance by several more columns to move the phasor away from zero.
    let mut tmp = [0.0_f32; SPS];
    for _ in 0..5 {
        synth.synthesize_column(&amps, &mut tmp);
    }

    // After reset the phasor is back at zero → same output as the first run.
    synth.reset();
    let mut after_reset = [0.0_f32; SPS];
    synth.synthesize_column(&amps, &mut after_reset);

    for (i, (&r, &e)) in after_reset.iter().zip(first_run.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-6,
            "sample {i}: after_reset={r} vs first_run={e} (diff={})",
            (r - e).abs()
        );
    }
}

/// `Synthesizer::renormalize` preserves output amplitude and does not corrupt
/// phase continuity.
///
/// We synthesise two batches of columns: the first batch advances the phasor
/// by enough steps to accumulate measurable f32 magnitude drift, then call
/// `renormalize()`, then synthesise more columns.  We verify that:
/// 1. Renormalization does not change the output amplitude by more than a
///    small tolerance.
/// 2. Phase continuity is maintained across the renormalization boundary
///    (the boundary jump is within the normal within-column step size).
#[test]
fn synthesizer_renormalize_preserves_output() {
    const SPS: usize = 256;
    // Use a bin near the middle of the frequency range.
    let bin = freq_to_index(440.0);
    let mut amps = [0.0_f32; TOTAL_BINS];
    amps[bin] = 0.8;

    let opts = SynthesisOptions {
        sample_rate: SAMPLE_RATE,
        gain: 1.0,
        decode_gamma: 1.0,
        mode: AmplitudeMode::Linear { threshold: None },
    };

    let mut synth = Synthesizer::<SPS>::new(opts);
    let mut before = [0.0_f32; SPS];
    let mut after = [0.0_f32; SPS];
    let mut tmp = [0.0_f32; SPS];

    // Advance the phasor for 100 columns to accumulate magnitude drift.
    for _ in 0..100 {
        synth.synthesize_column(&amps, &mut tmp);
    }
    // Capture output just before renormalizing.
    synth.synthesize_column(&amps, &mut before);

    // Renormalize and immediately synthesise the next column.
    synth.renormalize();
    synth.synthesize_column(&amps, &mut after);

    // The RMS amplitude of `after` should be close to that of `before`
    // (within 1 %, which is well above the typical f32 drift of ~0.01 %).
    let rms = |buf: &[f32]| -> f32 {
        let sum_sq: f32 = buf.iter().map(|&x| x * x).sum();
        #[expect(
            clippy::cast_precision_loss,
            reason = "buf.len() is SPS = 256; exact in f32"
        )]
        let len_f = buf.len() as f32;
        (sum_sq / len_f).sqrt()
    };
    let rms_before = rms(&before);
    let rms_after = rms(&after);
    assert!(
        (rms_before - rms_after).abs() / rms_before.max(1e-6) < 0.02,
        "renormalize changed RMS amplitude by more than 2 %: before={rms_before:.6} after={rms_after:.6}"
    );
}

/// A synthetic image with all-black pixels in the data area should decode to
/// amplitude 1.0 for all bins; all-white should decode to 0.0.
#[test]
fn column_amplitudes_from_image_black_and_white() {
    use image::{GrayImage, Luma};
    use phonopaper_rs::decode::{DataBounds, column_amplitudes_from_image};

    // TOTAL_BINS = 384, always fits in u32.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "TOTAL_BINS = 384, always fits in u32"
    )]
    let data_height = TOTAL_BINS as u32; // 1:1 pixel-per-bin for simplicity
    let width = 4_u32;
    // A 4×384 black image (pixel value 0 → amplitude 1.0).
    let black_img = GrayImage::from_pixel(width, data_height, Luma([0u8]));
    let black_dyn = image::DynamicImage::ImageLuma8(black_img);

    // Provide bounds manually so we don't need marker stripes.
    let bounds = DataBounds {
        data_top: 0,
        data_bottom: data_height,
    };

    let amps = column_amplitudes_from_image(&black_dyn, Some(bounds), 0).unwrap();
    assert_eq!(amps.len(), TOTAL_BINS);
    for (i, &a) in amps.iter().enumerate() {
        assert!(
            (a - 1.0).abs() < 1.0 / 255.0,
            "bin {i}: expected amplitude ≈ 1.0 for black pixel, got {a}"
        );
    }

    // A 4×384 white image (pixel value 255 → amplitude 0.0).
    let white_img = GrayImage::from_pixel(width, data_height, Luma([255u8]));
    let white_dyn = image::DynamicImage::ImageLuma8(white_img);
    let amps_white = column_amplitudes_from_image(&white_dyn, Some(bounds), 0).unwrap();
    for (i, &a) in amps_white.iter().enumerate() {
        assert!(
            a.abs() < 1.0 / 255.0,
            "bin {i}: expected amplitude ≈ 0.0 for white pixel, got {a}"
        );
    }
}

/// Out-of-range column index returns an error rather than panicking.
#[test]
fn column_amplitudes_from_image_out_of_range() {
    use image::{GrayImage, Luma};
    use phonopaper_rs::decode::{DataBounds, column_amplitudes_from_image};

    #[expect(
        clippy::cast_possible_truncation,
        reason = "TOTAL_BINS = 384, always fits in u32"
    )]
    let height = TOTAL_BINS as u32;
    let img = GrayImage::from_pixel(10, height, Luma([128u8]));
    let dyn_img = image::DynamicImage::ImageLuma8(img);
    let bounds = DataBounds {
        data_top: 0,
        data_bottom: height,
    };

    let result = column_amplitudes_from_image(&dyn_img, Some(bounds), 99);
    assert!(
        result.is_err(),
        "expected error for out-of-range col_x, got Ok"
    );
}

/// `column_amplitudes_from_image_into` produces the same results as
/// `column_amplitudes_from_image` but writes into a caller-supplied buffer
/// without any heap allocation.
#[test]
fn column_amplitudes_from_image_into_matches_vec_version() {
    use image::{GrayImage, Luma};
    use phonopaper_rs::decode::{
        DataBounds, column_amplitudes_from_image, column_amplitudes_from_image_into,
    };
    use phonopaper_rs::format::TOTAL_BINS;

    #[expect(
        clippy::cast_possible_truncation,
        reason = "TOTAL_BINS = 384, always fits in u32"
    )]
    let data_height = TOTAL_BINS as u32;
    let width = 4_u32;

    // Mid-gray image so not all amplitudes are identical (boring test).
    let img = GrayImage::from_fn(width, data_height, |x, y| {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "x + y ≤ 3 + 383 = 386; wrapping truncation is intentional here"
        )]
        Luma([(x + y) as u8])
    });
    let dyn_img = image::DynamicImage::ImageLuma8(img);
    let bounds = DataBounds {
        data_top: 0,
        data_bottom: data_height,
    };

    for col in 0..width {
        let vec_result = column_amplitudes_from_image(&dyn_img, Some(bounds), col).unwrap();
        let mut buf = [0.0_f32; TOTAL_BINS];
        column_amplitudes_from_image_into(&dyn_img, Some(bounds), col, &mut buf).unwrap();

        for (bin, (&v, &b)) in vec_result.iter().zip(buf.iter()).enumerate() {
            assert!(
                (v - b).abs() < 1e-6,
                "col {col} bin {bin}: vec={v} into={b}"
            );
        }
    }
}

// ─── AmplitudeMode::Linear threshold tests ────────────────────────────────────

/// With a binary threshold, a bin whose decoded amplitude is above the threshold
/// should produce the same output as if the amplitude were exactly 1.0 (fully on).
/// A bin below the threshold should produce silence.
#[test]
fn amplitude_threshold_binarises_signal() {
    const SPS: usize = 512;
    const NUM_COLS: usize = 2;

    let bin = freq_to_index(440.0);

    // Two spectrograms: one with amplitude 0.9 (above 0.85 threshold),
    // one with amplitude 0.5 (below 0.85 threshold).
    let make_spec = |amp: f32| {
        let mut buf = vec![0.0_f32; NUM_COLS * TOTAL_BINS];
        let mut spec = Spectrogram::from_storage(NUM_COLS, &mut buf[..]).unwrap();
        for col in 0..NUM_COLS {
            spec.set(col, bin, amp);
        }
        buf
    };

    let opts_thresh = SynthesisOptions {
        sample_rate: SAMPLE_RATE,
        gain: 1.0,
        decode_gamma: 1.0,
        mode: AmplitudeMode::Linear {
            threshold: Some(0.85),
        },
    };
    let opts_frac = SynthesisOptions {
        sample_rate: SAMPLE_RATE,
        gain: 1.0,
        decode_gamma: 1.0,
        mode: AmplitudeMode::Linear { threshold: None },
    };

    // amp=0.9 with threshold=0.85 should behave like amp=1.0 fractional.
    let buf_090 = make_spec(0.9);
    let spec_090 = Spectrogram::from_storage(NUM_COLS, &buf_090[..]).unwrap();
    let mut out_090_thresh = vec![0.0_f32; NUM_COLS * SPS];
    spectrogram_to_audio::<_, SPS>(&spec_090, &opts_thresh, &mut out_090_thresh);

    let buf_100 = make_spec(1.0);
    let spec_100 = Spectrogram::from_storage(NUM_COLS, &buf_100[..]).unwrap();
    let mut out_100_frac = vec![0.0_f32; NUM_COLS * SPS];
    spectrogram_to_audio::<_, SPS>(&spec_100, &opts_frac, &mut out_100_frac);

    for (i, (&a, &b)) in out_090_thresh.iter().zip(out_100_frac.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "sample {i}: threshold(0.9)={a} vs fractional(1.0)={b} — should be identical"
        );
    }

    // amp=0.5 with threshold=0.85 should produce silence (all zeros).
    let buf_050 = make_spec(0.5);
    let spec_050 = Spectrogram::from_storage(NUM_COLS, &buf_050[..]).unwrap();
    let mut out_050_thresh = vec![0.0_f32; NUM_COLS * SPS];
    spectrogram_to_audio::<_, SPS>(&spec_050, &opts_thresh, &mut out_050_thresh);

    assert!(
        out_050_thresh.iter().all(|&s| s == 0.0),
        "amplitude 0.5 below threshold 0.85 should produce silence"
    );
}

/// Without a threshold (fractional mode), amplitude 0.5 should produce roughly
/// half the RMS of amplitude 1.0 — confirming the threshold is actually `None`.
#[test]
fn no_threshold_uses_fractional_amplitude() {
    const SPS: usize = 512;
    const NUM_COLS: usize = 2;

    let bin = freq_to_index(440.0);
    let opts = SynthesisOptions {
        sample_rate: SAMPLE_RATE,
        gain: 1.0,
        decode_gamma: 1.0,
        mode: AmplitudeMode::Linear { threshold: None },
    };

    let rms = |v: &[f32]| {
        let sum_sq: f64 = v.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
        #[expect(
            clippy::cast_precision_loss,
            reason = "v.len() is a small test buffer length, exact in f64"
        )]
        (sum_sq / v.len() as f64).sqrt()
    };

    let make_and_synth = |amp: f32| {
        let mut buf = vec![0.0_f32; NUM_COLS * TOTAL_BINS];
        let mut spec = Spectrogram::from_storage(NUM_COLS, &mut buf[..]).unwrap();
        for col in 0..NUM_COLS {
            spec.set(col, bin, amp);
        }
        let mut out = vec![0.0_f32; NUM_COLS * SPS];
        spectrogram_to_audio::<_, SPS>(&spec, &opts, &mut out);
        rms(&out)
    };

    let rms_full = make_and_synth(1.0);
    let rms_half = make_and_synth(0.5);

    let ratio = rms_half / rms_full;
    assert!(
        (ratio - 0.5).abs() < 0.05,
        "fractional mode: expected RMS ratio ≈ 0.5, got {ratio:.4}"
    );
}

// ─── Synthesizer::options ────────────────────────────────────────────────────

/// `Synthesizer::options()` returns a reference to the `SynthesisOptions`
/// passed to `Synthesizer::new`.
#[test]
fn synthesizer_options_returns_construction_options() {
    let opts = SynthesisOptions {
        sample_rate: 22_050,
        gain: 2.5,
        decode_gamma: 0.5,
        mode: AmplitudeMode::Linear {
            threshold: Some(0.7),
        },
    };
    let synth = Synthesizer::<256>::new(opts);
    let returned = synth.options();

    assert_eq!(returned.sample_rate, opts.sample_rate);
    assert!((returned.gain - opts.gain).abs() < 1e-6);
    assert!((returned.decode_gamma - opts.decode_gamma).abs() < 1e-6);
    // Check mode matches.
    match (&returned.mode, &opts.mode) {
        (
            AmplitudeMode::Linear {
                threshold: Some(rt),
            },
            AmplitudeMode::Linear {
                threshold: Some(ot),
            },
        ) => assert!((rt - ot).abs() < 1e-6),
        _ => panic!("mode did not round-trip correctly"),
    }
}

// ─── SynthesisOptions::default ───────────────────────────────────────────────

/// `SynthesisOptions::default()` produces the documented defaults:
/// `sample_rate = SAMPLE_RATE`, `gain = 3.0`, `decode_gamma = 1.0`, and
/// `mode = AmplitudeMode::Db { min_db: -60.0, max_db: -10.0 }`.
#[test]
fn synthesis_options_default_values() {
    let opts = SynthesisOptions::default();

    assert_eq!(
        opts.sample_rate, SAMPLE_RATE,
        "default sample_rate should be SAMPLE_RATE"
    );
    assert!((opts.gain - 3.0).abs() < 1e-6, "default gain should be 3.0");
    assert!(
        (opts.decode_gamma - 1.0).abs() < 1e-6,
        "default decode_gamma should be 1.0"
    );
    match opts.mode {
        AmplitudeMode::Db { min_db, max_db } => {
            assert!(
                (min_db - (-60.0)).abs() < 1e-6,
                "default min_db should be -60.0"
            );
            assert!(
                (max_db - (-10.0)).abs() < 1e-6,
                "default max_db should be -10.0"
            );
        }
        AmplitudeMode::Linear { .. } => panic!("default AmplitudeMode should be Db"),
    }
}

// ─── image_to_spectrogram ────────────────────────────────────────────────────

/// `image_to_spectrogram` called with `None` bounds runs `detect_markers`
/// internally.  Round-tripping a rendered spectrogram recovers the correct
/// number of columns and a known amplitude.
#[test]
fn image_to_spectrogram_round_trip() {
    use image::DynamicImage;
    use phonopaper_rs::decode::image_to_spectrogram;
    use phonopaper_rs::format::freq_to_index;
    use phonopaper_rs::render::{RenderOptions, spectrogram_to_image};
    use phonopaper_rs::spectrogram::SpectrogramVec;

    let num_cols = 6;
    let bin = freq_to_index(440.0); // A4

    let mut spec_in = SpectrogramVec::new(num_cols);
    for col in 0..num_cols {
        spec_in.set(col, bin, 1.0);
    }

    let opts = RenderOptions {
        draw_octave_lines: false,
        ..RenderOptions::default()
    };
    let rgb = spectrogram_to_image(&spec_in, &opts);
    let dyn_img = DynamicImage::ImageRgb8(rgb);

    // Pass None so detect_markers runs automatically.
    let spec_out = image_to_spectrogram(&dyn_img, None)
        .expect("image_to_spectrogram should succeed on a valid PhonoPaper image");

    assert_eq!(
        spec_out.num_columns(),
        num_cols,
        "recovered spectrogram should have the same number of columns"
    );

    // The active bin should decode to a high amplitude.
    #[expect(
        clippy::cast_precision_loss,
        reason = "num_cols = 6; sum over 6 f32 values, exact in f32"
    )]
    let mean_amp = (0..num_cols).map(|c| spec_out.get(c, bin)).sum::<f32>() / num_cols as f32;
    assert!(
        mean_amp > 0.5,
        "active bin {bin} should decode to amplitude > 0.5, got {mean_amp:.3}"
    );
}

// ─── AmplitudeMode::Db synthesis ─────────────────────────────────────────────

/// Synthesizing with `AmplitudeMode::Db` exercises the dB-to-linear conversion
/// path: a non-zero amplitude in a single bin should produce audible output
/// with the correct frequency content.
#[test]
fn db_mode_synthesis_produces_frequency_content() {
    const SPS: usize = 1024;
    const NUM_COLS: usize = 4;

    let freq = 440.0_f64;
    let bin = freq_to_index(freq);
    let actual_freq = index_to_freq(bin);

    let opts = SynthesisOptions {
        sample_rate: SAMPLE_RATE,
        gain: 1.0,
        decode_gamma: 1.0,
        mode: AmplitudeMode::Db {
            min_db: -60.0,
            max_db: -10.0,
        },
    };

    let mut buf = [0.0f32; NUM_COLS * TOTAL_BINS];
    let mut spec = Spectrogram::from_storage(NUM_COLS, &mut buf[..]).unwrap();
    for col in 0..NUM_COLS {
        spec.set(col, bin, 1.0); // max amplitude → dB level = max_db
    }

    let mut samples = [0.0_f32; NUM_COLS * SPS];
    spectrogram_to_audio::<_, SPS>(&spec, &opts, &mut samples);

    // The signal should not be silent.
    let max_abs = samples.iter().map(|&s| s.abs()).fold(0.0_f32, f32::max);
    assert!(
        max_abs > 1e-4,
        "Db-mode synthesis should produce a non-trivial signal, max_abs = {max_abs:.6}"
    );

    // The dominant frequency should be at the target bin.
    let power_target = dft_power_at(&samples, actual_freq, opts.sample_rate);
    let power_other = dft_power_at(&samples, 200.0, opts.sample_rate);
    assert!(
        power_target > power_other * 3.0,
        "Db-mode: bin {bin} ({actual_freq:.1} Hz) power {power_target:.4} \
         should dominate 200 Hz ({power_other:.4})"
    );
}

/// In `AmplitudeMode::Db` mode a bin with amplitude 0.0 must produce silence
/// (the `amp_raw == 0.0` fast-path that returns 0.0 immediately).
#[test]
fn db_mode_zero_amplitude_is_silent() {
    const SPS: usize = 512;
    const NUM_COLS: usize = 2;

    let opts = SynthesisOptions {
        sample_rate: SAMPLE_RATE,
        gain: 1.0,
        decode_gamma: 1.0,
        mode: AmplitudeMode::Db {
            min_db: -60.0,
            max_db: -10.0,
        },
    };

    let buf = [0.0f32; NUM_COLS * TOTAL_BINS]; // all amplitudes = 0
    let spec = Spectrogram::from_storage(NUM_COLS, &buf[..]).unwrap();
    let mut samples = [0.0_f32; NUM_COLS * SPS];
    spectrogram_to_audio::<_, SPS>(&spec, &opts, &mut samples);

    assert!(
        samples.iter().all(|&s| s == 0.0),
        "Db-mode: all-zero spectrogram should produce silent output"
    );
}

// ─── decode_gamma != 1.0 ─────────────────────────────────────────────────────

/// `decode_gamma != 1.0` exercises the gamma-correction (`powf`) branch inside
/// `synthesize_column`.  With gamma = 0.5, the inverse correction raises the
/// amplitude to the power 2.0, which reduces output for sub-unit amplitudes.
#[test]
fn synthesis_with_decode_gamma_is_different_from_gamma_one() {
    const SPS: usize = 512;
    const NUM_COLS: usize = 2;

    let bin = freq_to_index(440.0);
    let amp = 0.5_f32; // non-trivial sub-unit amplitude

    let make_samples = |decode_gamma: f32| {
        let opts = SynthesisOptions {
            sample_rate: SAMPLE_RATE,
            gain: 1.0,
            decode_gamma,
            mode: AmplitudeMode::Linear { threshold: None },
        };
        let mut buf = [0.0f32; NUM_COLS * TOTAL_BINS];
        let mut spec = Spectrogram::from_storage(NUM_COLS, &mut buf[..]).unwrap();
        for col in 0..NUM_COLS {
            spec.set(col, bin, amp);
        }
        let mut out = [0.0_f32; NUM_COLS * SPS];
        spectrogram_to_audio::<_, SPS>(&spec, &opts, &mut out);
        out
    };

    let samples_gamma1 = make_samples(1.0);
    let samples_gamma2 = make_samples(2.0); // powf(0.5) → amplitude = 0.5^0.5 ≈ 0.707

    // gamma=1.0 and gamma=2.0 should produce different RMS values.
    let rms = |v: &[f32]| {
        let sum_sq: f64 = v.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
        #[expect(
            clippy::cast_precision_loss,
            reason = "small test buffer length, exact in f64"
        )]
        (sum_sq / v.len() as f64).sqrt()
    };

    let rms1 = rms(&samples_gamma1);
    let rms2 = rms(&samples_gamma2);

    assert!(
        (rms1 - rms2).abs() > 1e-4,
        "decode_gamma=2.0 should produce different RMS than decode_gamma=1.0; \
         rms1={rms1:.6} rms2={rms2:.6}"
    );

    // gamma=2.0 amplifies: 0.5^(1/2) ≈ 0.707 > 0.5, so rms2 > rms1.
    assert!(
        rms2 > rms1,
        "decode_gamma=2.0 applied to amp=0.5 should give higher RMS than gamma=1.0; \
         rms2={rms2:.6} rms1={rms1:.6}"
    );
}

// ─── Spectrogram::get out-of-bounds ──────────────────────────────────────────

/// `Spectrogram::get` with an out-of-bounds column or bin index returns 0.0
/// rather than panicking.
#[test]
fn spectrogram_get_out_of_bounds_returns_zero() {
    let mut buf = [0.0f32; 2 * TOTAL_BINS];
    let mut spec = Spectrogram::from_storage(2, &mut buf[..]).unwrap();
    spec.set(0, 0, 0.7);
    spec.set(1, TOTAL_BINS - 1, 0.3);

    // Spectrogram::get returns the literal 0.0f32 for out-of-bounds indices —
    // not a computed value — so bitwise equality is appropriate here.
    #[expect(
        clippy::float_cmp,
        reason = "Spectrogram::get returns the literal 0.0f32 for out-of-bounds; \
                  exact bitwise equality is correct and intentional"
    )]
    {
        // col out of range
        assert_eq!(
            spec.get(2, 0),
            0.0,
            "get with col >= num_columns should return 0.0"
        );
        // bin out of range
        assert_eq!(
            spec.get(0, TOTAL_BINS),
            0.0,
            "get with bin >= TOTAL_BINS should return 0.0"
        );
        // both out of range
        assert_eq!(
            spec.get(99, 999),
            0.0,
            "get with both col and bin out of range should return 0.0"
        );
    }
}

// ─── image_to_spectrogram: zero-data-area error ───────────────────────────────

/// `image_to_spectrogram` with manually supplied `DataBounds` where
/// `data_top == data_bottom` (zero height) must return an error rather than
/// producing a degenerate spectrogram.
#[test]
fn image_to_spectrogram_zero_height_bounds_is_error() {
    use image::DynamicImage;
    use phonopaper_rs::decode::{DataBounds, image_to_spectrogram};

    #[expect(
        clippy::cast_possible_truncation,
        reason = "TOTAL_BINS = 384, fits in u32"
    )]
    let img_height = TOTAL_BINS as u32;
    let img = image::GrayImage::from_pixel(4, img_height, image::Luma([128u8]));
    let dyn_img = DynamicImage::ImageLuma8(img);

    // zero-height bounds: data_top == data_bottom
    let bounds = DataBounds {
        data_top: 10,
        data_bottom: 10,
    };

    let result = image_to_spectrogram(&dyn_img, Some(bounds));
    assert!(
        result.is_err(),
        "image_to_spectrogram with zero data height should return an error"
    );
}

// ─── column_amplitudes_from_image_into: zero-height error ────────────────────

/// `column_amplitudes_from_image_into` with `DataBounds` where
/// `data_top == data_bottom` (zero height) must return an error.
#[test]
fn column_amplitudes_from_image_into_zero_height_is_error() {
    use image::DynamicImage;
    use phonopaper_rs::decode::{DataBounds, column_amplitudes_from_image_into};

    #[expect(
        clippy::cast_possible_truncation,
        reason = "TOTAL_BINS = 384, fits in u32"
    )]
    let img_height = TOTAL_BINS as u32;
    let img = image::GrayImage::from_pixel(4, img_height, image::Luma([128u8]));
    let dyn_img = DynamicImage::ImageLuma8(img);

    let bounds = DataBounds {
        data_top: 10,
        data_bottom: 10,
    };

    let mut out = [0.0f32; TOTAL_BINS];
    let result = column_amplitudes_from_image_into(&dyn_img, Some(bounds), 0, &mut out);
    assert!(
        result.is_err(),
        "column_amplitudes_from_image_into with zero data height should return an error"
    );
}

/// `column_amplitudes_from_image_into` with `bounds = None` runs
/// `detect_markers` internally.  This exercises the `None => detect_markers`
/// code path inside the function.
#[test]
fn column_amplitudes_from_image_into_auto_detect_markers() {
    use image::DynamicImage;
    use phonopaper_rs::decode::column_amplitudes_from_image_into;
    use phonopaper_rs::render::{RenderOptions, spectrogram_to_image};
    use phonopaper_rs::spectrogram::SpectrogramVec;

    // Build a valid PhonoPaper image so detect_markers can succeed.
    let opts = RenderOptions {
        draw_octave_lines: false,
        ..RenderOptions::default()
    };
    let spec = SpectrogramVec::new(4);
    let rgb = spectrogram_to_image(&spec, &opts);
    let img = DynamicImage::ImageRgb8(rgb);

    // Pass None so detect_markers runs internally.
    let mut out = [0.0f32; TOTAL_BINS];
    column_amplitudes_from_image_into(&img, None, 0, &mut out).expect(
        "column_amplitudes_from_image_into with None bounds should succeed on a valid image",
    );

    // The data area is all-white (silence) so all amplitudes should be near 0.
    let max_amp = out.iter().copied().fold(0.0_f32, f32::max);
    assert!(
        max_amp < 0.1,
        "white data area should decode to near-zero amplitudes, max={max_amp:.4}"
    );
}
