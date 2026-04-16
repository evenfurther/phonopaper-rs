//! Generate the `PhonoPaper` image for *Angine de Poitrine* — a short
//! original microtonal riff composed for this demonstration.
//!
//! "Angine de poitrine" is French for *angina pectoris* (chest pain / cardiac
//! constriction).  The riff reflects this image: a chromatic phrase that
//! squeezes around E3, uses a quarter-tone step upward, and resolves downward
//! through a sinking bass note — like pressure being released.
//!
//! **Notes chosen:**
//! ```text
//! E3  (164.81 Hz) — 0.35 s  — first note, tense
//! E3+ (167.21 Hz) — 0.20 s  — E3 + quarter-tone (microtonal squeeze up)
//! F3  (174.61 Hz) — 0.25 s  — half-step up
//! E3+ (167.21 Hz) — 0.20 s  — squeeze back down
//! E3  (164.81 Hz) — 0.45 s  — return to E3, longer
//! Eb3 (155.56 Hz) — 0.30 s  — sinking
//! D3  (146.83 Hz) — 0.30 s  — sinking further
//! B2  (123.47 Hz) — 0.80 s  — release: low bass note
//! ```
//!
//! A quarter-tone is 25 cents = half a semitone.
//! `E3 + quarter-tone ≈ 164.81 × 2^(0.25/12) ≈ 167.21 Hz`.
//!
//! Run with:
//! ```bash
//! cargo run --example angine
//! ```

use std::f32::consts::PI;

use phonopaper_rs::encode::{AnalysisOptions, audio_to_spectrogram};
use phonopaper_rs::format::SAMPLE_RATE;
use phonopaper_rs::render::{RenderOptions, spectrogram_to_image};

/// Synthesize a pure sine tone and append it to `out`.
fn append_sine(out: &mut Vec<f32>, freq_hz: f32, duration_s: f32, sample_rate: u32) {
    // SAMPLE_RATE = 44_100: fits exactly in f32 (< 2^24).
    #[expect(
        clippy::cast_precision_loss,
        reason = "SAMPLE_RATE = 44_100, exact in f32 (< 2^24)"
    )]
    let sr_f = sample_rate as f32;
    // duration_s * sr_f is at most ~0.8 * 44100 ≈ 35280, positive and ≤ usize::MAX.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "value is duration_s * sample_rate ≤ ~35280, positive and far below usize::MAX"
    )]
    let n = (duration_s * sr_f).round() as usize;
    let omega = 2.0 * PI * freq_hz / sr_f;
    // Start phase at 0; amplitude 0.8 to leave headroom.
    let amp = 0.8_f32;
    for i in 0..n {
        #[expect(
            clippy::cast_precision_loss,
            reason = "i < ~40000 per note, well within f32's exact range (< 2^24)"
        )]
        out.push(amp * (omega * i as f32).sin());
    }
}

/// Apply a short linear fade-in and fade-out to reduce clicks between notes.
fn apply_fade(samples: &mut [f32], fade_samples: usize) {
    let n = samples.len();
    let fade = fade_samples.min(n / 2);
    for i in 0..fade {
        #[expect(
            clippy::cast_precision_loss,
            reason = "fade ≤ ~220 samples, exact in f32"
        )]
        let t = i as f32 / fade as f32;
        samples[i] *= t;
        samples[n - 1 - i] *= t;
    }
}

fn main() -> phonopaper_rs::Result<()> {
    let sample_rate = SAMPLE_RATE;

    // Quarter-tone interval ratio: 2^(0.5/12) (half a semitone).
    // E3 × quarter_tone ≈ 164.81 × 1.0293 ≈ 169.64 Hz
    // But the traditional quarter-tone above E3 is E3 + 25 cents:
    //   E3 × 2^(25/1200) = 164.81 × 1.01457 ≈ 167.21 Hz
    // (25 cents = 1/4 of a semitone = 1200/48 cents = 2 PhonoPaper bins)

    // The riff: (frequency_hz, duration_s)
    let notes: &[(f32, f32)] = &[
        (164.81, 0.35), // E3           — tense opening
        (167.21, 0.20), // E3 + quarter-tone — microtonal squeeze up
        (174.61, 0.25), // F3           — half-step up, constriction peak
        (167.21, 0.20), // E3 + quarter-tone — constriction persists
        (164.81, 0.45), // E3           — return, longer → slight fatigue
        (155.56, 0.30), // Eb3          — sinking
        (146.83, 0.30), // D3           — sinking further
        (123.47, 0.80), // B2           — release: low, resonant resolution
    ];

    let mut pcm: Vec<f32> = Vec::new();
    let fade_samples = {
        // 0.010 * 44100 = 441, positive and well within usize::MAX.
        #[expect(
            clippy::cast_precision_loss,
            reason = "SAMPLE_RATE = 44_100, exact in f32"
        )]
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "10 ms at 44100 Hz = 441 samples, positive and far below usize::MAX"
        )]
        let v = (0.010_f32 * sample_rate as f32).round() as usize;
        v
    }; // 10 ms fade

    for &(freq, dur) in notes {
        let start = pcm.len();
        append_sine(&mut pcm, freq, dur, sample_rate);
        let end = pcm.len();
        apply_fade(&mut pcm[start..end], fade_samples);
    }

    // Encode to a spectrogram, then render to a `PhonoPaper` image.
    let analysis = AnalysisOptions {
        fft_window: 4096,
        hop_size: 353, // Android's 125 col/s rate
        ..AnalysisOptions::default()
    };
    let render = RenderOptions {
        draw_octave_lines: false,
        ..RenderOptions::default()
    };

    let spec = audio_to_spectrogram(&pcm, sample_rate, &analysis)?;
    let img = spectrogram_to_image(&spec, &render);

    let out_path = "/tmp/angine.png";
    img.save(out_path)
        .map_err(|e| phonopaper_rs::PhonoPaperError::IoError(std::io::Error::other(e)))?;

    println!(
        "Saved {out_path}  ({}×{} px, {} columns)",
        img.width(),
        img.height(),
        spec.num_columns()
    );
    println!();
    println!("\"Angine de Poitrine\" — an original microtonal riff");
    println!("(\"angine de poitrine\" = French for angina pectoris)");
    println!();
    println!("Notes:");
    for &(freq, dur) in notes {
        println!("  {freq:.2} Hz  ×  {dur:.2}s");
    }
    println!();
    println!("  E3  = 164.81 Hz   (tense opening)");
    println!("  E3+ = 167.21 Hz   (E3 + quarter-tone, microtonal squeeze)");
    println!("  F3  = 174.61 Hz   (constriction peak)");
    println!("  Eb3 = 155.56 Hz   (sinking)");
    println!("  D3  = 146.83 Hz   (sinking further)");
    println!("  B2  = 123.47 Hz   (release, low resonant resolution)");

    Ok(())
}
