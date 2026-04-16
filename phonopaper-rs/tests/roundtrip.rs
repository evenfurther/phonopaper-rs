//! Round-trip tests: spectrogram → audio → spectrogram.
//!
//! These tests verify that the encode and decode halves of the pipeline are
//! mutually consistent.  A [`Spectrogram`] with a known frequency content is
//! synthesized into audio by [`spectrogram_to_audio`], then the audio is
//! re-analysed by [`audio_to_spectrogram`].  The recovered spectrogram should
//! show a dominant peak at the same bin(s) as the original.
//!
//! A second group of tests exercises the **full dB encode/decode path**
//! (WAV → STFT dB spectrogram → synthesis) to guard against noise-floor
//! accumulation and spectral-balance distortion.

use std::f32::consts::PI;

#[path = "helpers/mod.rs"]
mod helpers;
use helpers::dft_power_at;

use phonopaper_rs::decode::{AmplitudeMode, SynthesisOptions, spectrogram_to_audio};
use phonopaper_rs::encode::{AnalysisOptions, audio_to_spectrogram};
use phonopaper_rs::format::{SAMPLE_RATE, TOTAL_BINS, freq_to_index, index_to_freq};
use phonopaper_rs::spectrogram::SpectrogramVec;

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Return the bin index with the highest average amplitude across all columns.
fn dominant_bin(spec: &SpectrogramVec) -> usize {
    let ncols = spec.num_columns();
    let nbins = phonopaper_rs::format::TOTAL_BINS;
    (0..nbins)
        .max_by(|&a, &b| {
            let avg = |bin: usize| -> f32 {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "ncols is a frame count bounded by test inputs (≤ 100); exact in f32"
                )]
                {
                    (0..ncols).map(|c| spec.get(c, bin)).sum::<f32>() / ncols as f32
                }
            };
            avg(a).partial_cmp(&avg(b)).unwrap()
        })
        .unwrap()
}

/// Synthesis options with unit gain so the audio amplitude is predictable.
fn synth_opts() -> SynthesisOptions {
    SynthesisOptions {
        sample_rate: SAMPLE_RATE,
        gain: 1.0,
        decode_gamma: 1.0,
        mode: AmplitudeMode::Linear { threshold: None },
    }
}

/// Analysis options with a large window for good frequency resolution.
fn analysis_opts() -> AnalysisOptions {
    AnalysisOptions {
        fft_window: 4096,
        hop_size: 512,
        min_db: -100.0,
        max_db: -10.0,
    }
}

/// Analysis + synthesis options using the **dB encode/decode path** with
/// matched min/max dB.  Use these for tests that exercise the full pipeline.
fn db_analysis_opts() -> AnalysisOptions {
    AnalysisOptions {
        fft_window: 4096,
        hop_size: 512,
        min_db: -60.0,
        max_db: -10.0,
    }
}

fn db_synth_opts() -> SynthesisOptions {
    SynthesisOptions {
        sample_rate: SAMPLE_RATE,
        gain: 3.0,
        decode_gamma: 1.0,
        mode: AmplitudeMode::Db {
            min_db: -60.0,
            max_db: -10.0,
        },
    }
}

/// Generate `num_samples` of a pure sine at `freq_hz` with peak amplitude `amp`.
fn sine_wave(freq_hz: f32, amp: f32, num_samples: usize) -> Vec<f32> {
    #[expect(
        clippy::cast_precision_loss,
        reason = "SAMPLE_RATE = 44_100, exact in f32 (< 2^24)"
    )]
    let sr = SAMPLE_RATE as f32;
    (0..num_samples)
        .map(|t| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "t < num_samples ≤ 44_100 in tests; exact in f32 (< 2^17)"
            )]
            {
                amp * (2.0 * PI * freq_hz * t as f32 / sr).sin()
            }
        })
        .collect()
}

// ─── existing tests ───────────────────────────────────────────────────────────

/// A spectrogram with a single active bin round-trips back to the same bin.
///
/// Spectrogram (bin N active) → audio → re-analyse → recovered spectrogram
/// should have bin N as its dominant frequency.
#[test]
fn single_bin_round_trip() {
    let target_freq = 440.0_f64; // A4
    let bin = freq_to_index(target_freq);

    // Build a spectrogram long enough to give the STFT several frames.
    // With hop_size=512, 100 columns × 512 samples/col = 51_200 samples ≈ 100 frames.
    let num_cols = 100;
    let mut spec = SpectrogramVec::new(num_cols);
    for col in 0..num_cols {
        spec.set(col, bin, 1.0);
    }

    // Synthesize audio.
    let mut samples = vec![0.0_f32; num_cols * 512];
    spectrogram_to_audio::<_, 512>(&spec, &synth_opts(), &mut samples);

    // Re-analyse.
    let recovered = audio_to_spectrogram(&samples, SAMPLE_RATE, &analysis_opts()).unwrap();
    assert!(recovered.num_columns() > 0, "expected at least one frame");

    // The dominant bin in the recovered spectrogram should be close to the
    // original.  Multiple PhonoPaper bins can share the same FFT bin, and
    // spectral leakage means the peak may land on an adjacent bin — accept
    // within ±4 bins (< one semitone) of the target.
    let recovered_bin = dominant_bin(&recovered);
    let recovered_freq = index_to_freq(recovered_bin);
    let original_freq = index_to_freq(bin);

    assert!(
        recovered_bin.abs_diff(bin) <= 4,
        "dominant bin after round-trip is {recovered_bin} ({recovered_freq:.1} Hz), \
         expected {bin} ({original_freq:.1} Hz) ± 4 bins"
    );
}

/// A spectrogram with two active bins round-trips so both bins are among the
/// strongest in the recovered spectrogram.
#[test]
fn dual_bin_round_trip() {
    let freq_a = 440.0_f64; // A4
    let freq_b = 880.0_f64; // A5

    let bin_a = freq_to_index(freq_a);
    let bin_b = freq_to_index(freq_b);

    let num_cols = 100;
    let mut spec = SpectrogramVec::new(num_cols);
    for col in 0..num_cols {
        // Equal amplitude for both tones.
        spec.set(col, bin_a, 0.5);
        spec.set(col, bin_b, 0.5);
    }

    // Synthesize and re-analyse.
    let mut samples = vec![0.0_f32; num_cols * 512];
    spectrogram_to_audio::<_, 512>(&spec, &synth_opts(), &mut samples);
    let recovered = audio_to_spectrogram(&samples, SAMPLE_RATE, &analysis_opts()).unwrap();

    let ncols = recovered.num_columns();
    let avg = |bin: usize| -> f32 {
        #[expect(
            clippy::cast_precision_loss,
            reason = "ncols is a frame count bounded by test inputs (≤ 100); exact in f32"
        )]
        {
            (0..ncols).map(|c| recovered.get(c, bin)).sum::<f32>() / ncols as f32
        }
    };

    let amp_a = avg(bin_a);
    let amp_b = avg(bin_b);
    // A clearly silent bin far from both tones.
    let silent_bin = freq_to_index(100.0);
    let amp_silent = avg(silent_bin);

    assert!(
        amp_a > amp_silent * 5.0,
        "bin {bin_a} ({freq_a:.0} Hz) avg amplitude {amp_a:.4} should dominate \
         silent bin {silent_bin} ({amp_silent:.4}) after round-trip"
    );
    assert!(
        amp_b > amp_silent * 5.0,
        "bin {bin_b} ({freq_b:.0} Hz) avg amplitude {amp_b:.4} should dominate \
         silent bin {silent_bin} ({amp_silent:.4}) after round-trip"
    );
}

/// A silent spectrogram round-trips to an empty/near-zero audio and back to a
/// near-zero spectrogram.
#[test]
fn silence_round_trip() {
    let spec = SpectrogramVec::new(50);
    let mut samples = vec![0.0_f32; 50 * 512];
    spectrogram_to_audio::<_, 512>(&spec, &synth_opts(), &mut samples);
    assert!(
        samples.iter().all(|&s| s == 0.0),
        "silent spectrogram should produce zero audio"
    );

    let recovered = audio_to_spectrogram(&samples, SAMPLE_RATE, &analysis_opts()).unwrap();
    let max_amp = (0..recovered.num_columns())
        .flat_map(|c| (0..phonopaper_rs::format::TOTAL_BINS).map(move |b| (c, b)))
        .map(|(c, b)| recovered.get(c, b))
        .fold(0.0f32, f32::max);

    assert!(
        max_amp < 0.01,
        "silent round-trip should recover near-zero spectrogram, max = {max_amp:.6}"
    );
}

// ─── full dB pipeline tests ───────────────────────────────────────────────────

/// Encoding and decoding digital silence (all-zero PCM) through the full dB
/// pipeline must produce near-zero audio output.
///
/// This guards against the noise-floor accumulation bug where floating-point
/// rounding artefacts in the FFT (well below the quantisation noise of 16-bit
/// audio) get encoded as small but non-zero pixel amplitudes, which the decoder
/// then synthesises across all 384 bins, producing a perceptible constant tone.
#[test]
fn silence_through_db_pipeline_is_quiet() {
    const SPS: usize = 512;

    // 1 second of exact digital silence.
    let silence = vec![0.0_f32; SAMPLE_RATE as usize];

    let spec = audio_to_spectrogram(&silence, SAMPLE_RATE, &db_analysis_opts()).unwrap();

    // All spectrogram bins should be at amplitude 0 (FFT of zeros → 0 magnitude
    // → below min_db → stored amplitude 0).
    let max_stored = (0..spec.num_columns())
        .flat_map(|c| (0..TOTAL_BINS).map(move |b| (c, b)))
        .map(|(c, b)| spec.get(c, b))
        .fold(0.0f32, f32::max);
    assert!(
        max_stored < 1e-6,
        "digital silence should produce zero-amplitude spectrogram, max stored = {max_stored:.2e}"
    );

    // Synthesise and check the audio RMS is negligible.
    let ncols = spec.num_columns();
    let mut audio_out = vec![0.0_f32; ncols * SPS];
    spectrogram_to_audio::<_, SPS>(&spec, &db_synth_opts(), &mut audio_out);

    let rms: f64 = {
        let sum_sq: f64 = audio_out.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
        #[expect(
            clippy::cast_precision_loss,
            reason = "audio_out.len() is a small test buffer; exact in f64"
        )]
        (sum_sq / audio_out.len() as f64).sqrt()
    };
    assert!(
        rms < 1e-6,
        "decoding a silence spectrogram should produce near-zero audio, RMS = {rms:.2e}"
    );
}

/// The dominant frequency band in the **output** of a full dB encode → decode
/// cycle matches the dominant frequency band in the **input**.
///
/// This test guards against spectral-balance distortion caused by noise-floor
/// encoding: if bins below the true signal level carry non-zero amplitude they
/// are synthesised as real sines, potentially shifting the perceived centre of
/// gravity to lower (or other) frequencies.
///
/// Strategy: encode a mix of a loud mid-range tone (880 Hz, −6 dBFS) and a
/// quieter high tone (3520 Hz, −18 dBFS).  Both are well above the noise floor
/// of digital silence, so both should survive the round-trip.  Verify that the
/// 880 Hz component dominates the output, not some noise artefact at another
/// frequency.
#[test]
fn full_db_roundtrip_preserves_spectral_balance() {
    const SPS: usize = 512;

    // Two tones: loud mid (880 Hz at 0.5 amplitude) and quieter high (3520 Hz
    // at 0.125 amplitude).  880 Hz is about 12 dB louder; the output should
    // also have 880 Hz dominating.
    let n = SAMPLE_RATE as usize;
    let mut input: Vec<f32> = sine_wave(880.0, 0.5, n);
    let high = sine_wave(3520.0, 0.125, n);
    for (s, h) in input.iter_mut().zip(high.iter()) {
        *s += h;
    }

    // Encode.
    let spec = audio_to_spectrogram(&input, SAMPLE_RATE, &db_analysis_opts()).unwrap();

    // Decode.
    let ncols = spec.num_columns();
    let mut output = vec![0.0_f32; ncols * SPS];
    spectrogram_to_audio::<_, SPS>(&spec, &db_synth_opts(), &mut output);

    // Measure DFT power at both target frequencies and at a control frequency
    // (110 Hz — far from both tones and below the mix energy).
    let p880_in = dft_power_at(&input, 880.0, SAMPLE_RATE);
    let p3520_in = dft_power_at(&input, 3520.0, SAMPLE_RATE);
    let p880_out = dft_power_at(&output, 880.0, SAMPLE_RATE);
    let p3520_out = dft_power_at(&output, 3520.0, SAMPLE_RATE);
    let p110_out = dft_power_at(&output, 110.0, SAMPLE_RATE);

    // Both signal components should survive the round-trip above the noise.
    assert!(
        p880_out > p110_out * 10.0,
        "880 Hz power after round-trip ({p880_out:.4}) should dominate \
         110 Hz control ({p110_out:.4}) by 10×"
    );
    assert!(
        p3520_out > p110_out * 5.0,
        "3520 Hz power after round-trip ({p3520_out:.4}) should dominate \
         110 Hz control ({p110_out:.4}) by 5×"
    );

    // The 880 Hz component should still be louder than the 3520 Hz component
    // (it was 12 dB louder in the input).  Accept it being at least 3× louder
    // (≈ 10 dB) to allow some rounding/overlap loss.
    assert!(
        p880_out > p3520_out * 3.0,
        "880 Hz ({p880_out:.4}) should remain louder than 3520 Hz ({p3520_out:.4}) \
         by at least 3× after round-trip; input ratio was {:.2}×",
        p880_in / p3520_in
    );
}

/// After a full dB encode → decode cycle, the amplitude ratio between two tones
/// is preserved within a factor of 2.
///
/// Encode two sines (440 Hz at amplitude 0.4, 880 Hz at amplitude 0.2 — a 2:1
/// ratio, or 6 dB).  In the decoded audio the 440 Hz component must be at least
/// 1.5× and at most 4× the 880 Hz component (allowing ±3 dB slack on top of
/// the expected 6 dB difference).  This catches any systematic spectral
/// colouring introduced by the codec.
#[test]
fn full_db_roundtrip_preserves_amplitude_ratio() {
    const SPS: usize = 512;

    let n = SAMPLE_RATE as usize;
    let mut input: Vec<f32> = sine_wave(440.0, 0.4, n);
    let high = sine_wave(880.0, 0.2, n);
    for (s, h) in input.iter_mut().zip(high.iter()) {
        *s += h;
    }

    let spec = audio_to_spectrogram(&input, SAMPLE_RATE, &db_analysis_opts()).unwrap();

    let mut output = vec![0.0_f32; spec.num_columns() * SPS];
    spectrogram_to_audio::<_, SPS>(&spec, &db_synth_opts(), &mut output);

    let p440 = dft_power_at(&output, 440.0, SAMPLE_RATE);
    let p880 = dft_power_at(&output, 880.0, SAMPLE_RATE);

    assert!(
        p440 > 0.0 && p880 > 0.0,
        "both tones must survive the round-trip (440 Hz power={p440:.4}, 880 Hz power={p880:.4})"
    );

    let ratio = p440 / p880;
    // Input ratio was 0.4/0.2 = 2.0 in amplitude → 4.0 in power.
    // Allow ±50% slack (factor of 2) on the power ratio: expect 2.0 – 8.0.
    assert!(
        (2.0..=8.0).contains(&ratio),
        "amplitude ratio 440 Hz / 880 Hz in output should be 2–8× (reflecting 6 dB \
         input difference), got {ratio:.3}×"
    );
}

// ─── pixel brightness ↔ output volume tests ───────────────────────────────────

/// **Darker pixels produce louder output.** White (luma = 255) is silence;
/// black (luma = 0) is maximum amplitude.  This is the fundamental
/// `PhonoPaper` convention.
///
/// In fractional (non-threshold) mode the decoded amplitude is
/// `stored_amp = 1 − luma / 255`, which after dB inversion gives a
/// *louder* output for darker pixels.  This test makes that invariant
/// explicit and machine-checked.
///
/// We encode two sines at the *same frequency* but different amplitudes so
/// that one lands near the bottom of the dB window (quiet → nearly white
/// pixel) and one near the top (loud → nearly black pixel).  After decoding,
/// the louder source must produce louder output.
#[test]
fn darker_pixel_produces_louder_output() {
    const SPS: usize = 512;

    let n = SAMPLE_RATE as usize;

    // Loud sine: amplitude 0.4 → web_audio_mag ≈ 0.1 → −20 dBFS
    //   stored_amp = (−20 − (−60)) / 50 = 0.80  →  pixel ≈ 51  (fairly dark)
    // Quiet sine: amplitude 0.04 → web_audio_mag ≈ 0.01 → −40 dBFS
    //   stored_amp = (−40 − (−60)) / 50 = 0.40  →  pixel ≈ 153 (fairly light)
    let loud_input = sine_wave(440.0, 0.4, n);
    let quiet_input = sine_wave(440.0, 0.04, n);

    let spec_loud = audio_to_spectrogram(&loud_input, SAMPLE_RATE, &db_analysis_opts()).unwrap();
    let spec_quiet = audio_to_spectrogram(&quiet_input, SAMPLE_RATE, &db_analysis_opts()).unwrap();

    // Verify that the darker (louder) spectrogram has a higher stored amplitude
    // at the 440 Hz bin.
    let bin_440 = freq_to_index(440.0);
    let mid = spec_loud.num_columns() / 2;
    let stored_loud = spec_loud.get(mid, bin_440);
    let stored_quiet = spec_quiet.get(mid, bin_440);
    // Darker pixel = higher stored amplitude = lower luma.
    assert!(
        stored_loud > stored_quiet,
        "louder input should produce higher stored amplitude (darker pixel): \
         loud={stored_loud:.4} quiet={stored_quiet:.4}"
    );

    // Now decode both and verify the loud input still produces louder output.
    let decode = |spec: &SpectrogramVec| {
        let mut out = vec![0.0_f32; spec.num_columns() * SPS];
        spectrogram_to_audio::<_, SPS>(spec, &db_synth_opts(), &mut out);
        let sum_sq: f64 = out.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
        #[expect(
            clippy::cast_precision_loss,
            reason = "out.len() is a small test buffer; exact in f64"
        )]
        (sum_sq / out.len() as f64).sqrt() // RMS
    };

    let rms_loud = decode(&spec_loud);
    let rms_quiet = decode(&spec_quiet);

    assert!(
        rms_loud > rms_quiet,
        "decoding the louder (darker) spectrogram should give higher RMS output: \
         rms_loud={rms_loud:.4} rms_quiet={rms_quiet:.4}"
    );

    // The loud input was 10× the quiet input (20 dB).  After the round-trip
    // through a 50 dB window the dB distance is preserved, so the amplitude
    // ratio should still be significant (at least 5×).
    assert!(
        rms_loud > rms_quiet * 5.0,
        "loud source (10× amplitude) should decode to at least 5× RMS of quiet source; \
         got loud={rms_loud:.4} quiet={rms_quiet:.4} ratio={:.2}×",
        rms_loud / rms_quiet
    );
}
