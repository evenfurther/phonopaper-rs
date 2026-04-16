//! `compare_android` — compare library decode output against the Android reference WAV.
//!
//! Usage:
//!   `cargo run --release --example compare_android`
//!   `cargo run --release --example compare_android -- /tmp/mozart2.jpg /tmp/mozart2.wav`
//!
//! Computes three complementary quality metrics for each configuration:
//!
//!   `img_semi`  — correlation between the image's decoded pixel-amplitude columns
//!                 and the reference WAV's STFT columns (same metric as the earlier
//!                 threshold sweep; robust, ~0.93 range for good configs)
//!   `col_semi`  — column-by-column correlation of the synthesised-audio STFT vs
//!                 the reference STFT (lower because synthesis spreads energy)
//!   `glob_semi` — correlation of time-averaged frequency energy profiles

use std::f32::consts::PI;

use image::DynamicImage;
use num_complex::Complex;
use rustfft::FftPlanner;

use phonopaper_rs::decode::{
    AmplitudeMode, DataBounds, SynthesisOptions, column_amplitudes_from_image_into,
    decode_image_to_wav_sps, detect_markers,
};
use phonopaper_rs::format::{SAMPLE_RATE, TOTAL_BINS, index_to_freq};

// ─── WAV I/O ──────────────────────────────────────────────────────────────────

fn read_wav_mono(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("cannot open WAV");
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.expect("read")).collect(),
        hound::SampleFormat::Int => {
            #[expect(
                clippy::cast_precision_loss,
                reason = "audio full-scale; ≤24-bit integer is exact in f32"
            )]
            let max = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| {
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "i32→f32 for audio; minor LSB rounding inaudible"
                    )]
                    let sf = s.expect("read") as f32;
                    sf / max
                })
                .collect()
        }
    };
    if channels == 1 {
        samples
    } else {
        #[expect(clippy::cast_precision_loss, reason = "channels ≤ 65535, exact in f32")]
        let ch = channels as f32;
        samples
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / ch)
            .collect()
    }
}

// ─── `PhonoPaper`-bin STFT ──────────────────────────────────────────────────────

/// Compute a `PhonoPaper` amplitude spectrogram via STFT with `hop = sps`.
///
/// Returns a flat buffer of `(n_cols * TOTAL_BINS)` magnitudes in column-major
/// order: `buf[col * TOTAL_BINS + bin]`.
fn phonopaper_stft(samples: &[f32], sps: usize) -> Vec<f32> {
    const FFT_SIZE: usize = 4096;
    #[expect(
        clippy::cast_precision_loss,
        reason = "SAMPLE_RATE=44100, exact in f32"
    )]
    let sr = SAMPLE_RATE as f32;

    let phono_to_fft: Vec<usize> = (0..TOTAL_BINS)
        .map(|b| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "FFT_SIZE=4096, index_to_freq returns ≤4186; product fits in f32"
            )]
            let fft_sz_f = FFT_SIZE as f32;
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "freq*FFT_SIZE/sr is positive and bounded by FFT_SIZE/2"
            )]
            let k = (index_to_freq(b) as f32 * fft_sz_f / sr).round() as usize;
            k.min(FFT_SIZE / 2)
        })
        .collect();

    let hann: Vec<f32> = (0..FFT_SIZE)
        .map(|i| {
            #[expect(clippy::cast_precision_loss, reason = "FFT_SIZE=4096, exact in f32")]
            {
                0.5 * (1.0 - (2.0 * PI * i as f32 / (FFT_SIZE - 1) as f32).cos())
            }
        })
        .collect();

    let num_frames = if samples.len() >= FFT_SIZE {
        (samples.len() - FFT_SIZE) / sps + 1
    } else {
        0
    };

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let mut scratch = vec![Complex::new(0.0f32, 0.0); fft.get_outofplace_scratch_len()];
    let mut win_buf = vec![Complex::new(0.0f32, 0.0); FFT_SIZE];
    let mut out_buf = vec![Complex::new(0.0f32, 0.0); FFT_SIZE];

    let mut result = vec![0.0f32; num_frames * TOTAL_BINS];

    for frame in 0..num_frames {
        let start = frame * sps;
        let end = (start + FFT_SIZE).min(samples.len());
        for (i, c) in win_buf.iter_mut().enumerate() {
            c.re = if i < end - start {
                samples[start + i] * hann[i]
            } else {
                0.0
            };
            c.im = 0.0;
        }
        fft.process_outofplace_with_scratch(&mut win_buf, &mut out_buf, &mut scratch);
        for (bin, &fft_bin) in phono_to_fft.iter().enumerate() {
            result[frame * TOTAL_BINS + bin] = out_buf[fft_bin].norm();
        }
    }
    result
}

fn stft_num_cols(stft: &[f32]) -> usize {
    stft.len() / TOTAL_BINS
}

fn stft_col(stft: &[f32], c: usize) -> &[f32] {
    &stft[c * TOTAL_BINS..(c + 1) * TOTAL_BINS]
}

// ─── Image amplitude spectrogram ─────────────────────────────────────────────

/// Extract the image's per-column pixel amplitudes as a flat
/// `(n_cols * TOTAL_BINS)` buffer, applying an optional binary threshold.
fn image_amp_spectrogram(
    image: &DynamicImage,
    bounds: DataBounds,
    threshold: Option<f32>,
) -> Vec<f32> {
    let width = image.width() as usize;
    let mut out = vec![0.0f32; width * TOTAL_BINS];
    let mut col_buf = [0.0f32; TOTAL_BINS];
    for col in 0..width {
        #[expect(clippy::cast_possible_truncation, reason = "col < width which is u32")]
        column_amplitudes_from_image_into(image, Some(bounds), col as u32, &mut col_buf)
            .expect("col read");
        if let Some(t) = threshold {
            for v in &mut col_buf {
                *v = if *v >= t { 1.0 } else { 0.0 };
            }
        }
        out[col * TOTAL_BINS..(col + 1) * TOTAL_BINS].copy_from_slice(&col_buf);
    }
    out
}

// ─── Correlation helpers ──────────────────────────────────────────────────────

fn pearson(a: &[f32], b: &[f32]) -> Option<f32> {
    #[expect(
        clippy::cast_precision_loss,
        reason = "len ≤ TOTAL_BINS=384, exact in f32"
    )]
    let n = a.len() as f32;
    let ma = a.iter().sum::<f32>() / n;
    let mb = b.iter().sum::<f32>() / n;
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let da: f32 = a.iter().map(|x| (x - ma).powi(2)).sum::<f32>().sqrt();
    let db: f32 = b.iter().map(|y| (y - mb).powi(2)).sum::<f32>().sqrt();
    let den = da * db;
    if den < 1e-9 { None } else { Some(num / den) }
}

fn to_semitones(col_slice: &[f32]) -> Vec<f32> {
    const SEMITONES: usize = 96;
    const BSEMI: usize = TOTAL_BINS / SEMITONES;
    (0..SEMITONES)
        .map(|s| col_slice[s * BSEMI..(s + 1) * BSEMI].iter().sum())
        .collect()
}

/// Mean per-column Pearson correlation (semitone-aggregated), trimmed to
/// the shorter of the two spectrograms.
fn mean_semi_corr(ref_stft: &[f32], dec_stft: &[f32]) -> f32 {
    let n = stft_num_cols(ref_stft).min(stft_num_cols(dec_stft));
    let vals: Vec<f32> = (0..n)
        .filter_map(|c| {
            let rs = to_semitones(stft_col(ref_stft, c));
            let ds = to_semitones(stft_col(dec_stft, c));
            pearson(&rs, &ds)
        })
        .collect();
    if vals.is_empty() {
        0.0
    } else {
        #[expect(clippy::cast_precision_loss, reason = "len ≤ n_cols, exact in f32")]
        {
            vals.iter().sum::<f32>() / vals.len() as f32
        }
    }
}

/// Correlation of the time-averaged per-bin RMS energy profiles.
fn global_semi_corr(ref_stft: &[f32], dec_stft: &[f32]) -> f32 {
    let mean_rms = |stft: &[f32]| -> Vec<f32> {
        let nc = stft_num_cols(stft);
        if nc == 0 {
            return vec![0.0; TOTAL_BINS];
        }
        let mut e = vec![0.0f32; TOTAL_BINS];
        for c in 0..nc {
            for (ev, &v) in e.iter_mut().zip(stft_col(stft, c)) {
                *ev += v * v;
            }
        }
        #[expect(clippy::cast_precision_loss, reason = "nc ≤ ~1400, exact in f32")]
        let nc_f = nc as f32;
        for ev in &mut e {
            *ev = (*ev / nc_f).sqrt();
        }
        e
    };
    let re = mean_rms(ref_stft);
    let de = mean_rms(dec_stft);
    let rs = to_semitones(&re);
    let ds = to_semitones(&de);
    pearson(&rs, &ds).unwrap_or(0.0)
}

// ─── Decode helpers ───────────────────────────────────────────────────────────

fn decode_sps<const SPS: usize>(img: &str, threshold: Option<f32>, gain: f32) -> Vec<f32> {
    let thr_str = threshold.map_or_else(|| "none".to_string(), |t| format!("{t:.2}"));
    let tmp = std::env::temp_dir().join(format!("pp_cmp_sps{SPS}_thr{thr_str}.wav"));
    let opts = SynthesisOptions {
        sample_rate: SAMPLE_RATE,
        gain,
        decode_gamma: 1.0,
        mode: AmplitudeMode::Linear { threshold },
    };
    decode_image_to_wav_sps::<SPS>(img, &tmp, opts).expect("decode");
    let s = read_wav_mono(tmp.to_str().expect("temp path is valid UTF-8"));
    let _ = std::fs::remove_file(&tmp);
    s
}

// ─── Configurations ───────────────────────────────────────────────────────────

// (label, sps, threshold, gain)
type Cfg = (&'static str, u32, Option<f32>, f32);

const CONFIGS: &[Cfg] = &[
    ("fractional  sps=512  [previous default]", 512, None, 0.15),
    ("fractional  sps=353", 353, None, 0.15),
    ("thresh=0.70 sps=353", 353, Some(0.70), 0.15),
    ("thresh=0.80 sps=353", 353, Some(0.80), 0.15),
    ("thresh=0.85 sps=353  [Android-like]", 353, Some(0.85), 0.15),
    ("thresh=0.90 sps=353", 353, Some(0.90), 0.15),
    ("thresh=0.95 sps=353", 353, Some(0.95), 0.15),
    ("thresh=0.85 sps=512", 512, Some(0.85), 0.15),
];

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let img_path = args.get(1).map_or("/tmp/mozart2.jpg", String::as_str);
    let ref_wav = args.get(2).map_or("/tmp/mozart2.wav", String::as_str);

    println!("Image   : {img_path}");
    println!("Ref WAV : {ref_wav}");
    println!();

    // ── Reference audio ───────────────────────────────────────────────────────
    println!("Reading reference WAV…");
    let ref_samples = read_wav_mono(ref_wav);
    #[expect(
        clippy::cast_precision_loss,
        reason = "sample count fits in f32 for ≤5 min audio"
    )]
    let ref_dur = ref_samples.len() as f32 / SAMPLE_RATE as f32;
    println!("  {} samples  ({ref_dur:.2} s)", ref_samples.len());

    println!("Computing reference audio STFTs…");
    let ref_stft_353 = phonopaper_stft(&ref_samples, 353);
    let ref_stft_512 = phonopaper_stft(&ref_samples, 512);
    println!(
        "  sps=353: {} cols   sps=512: {} cols",
        stft_num_cols(&ref_stft_353),
        stft_num_cols(&ref_stft_512)
    );
    println!();

    // ── Reference image ───────────────────────────────────────────────────────
    println!("Loading image and detecting markers…");
    let img = image::open(img_path).expect("open image");
    let bounds = detect_markers(&img).expect("detect markers");
    println!(
        "  data_top={} data_bottom={} height={}",
        bounds.data_top,
        bounds.data_bottom,
        bounds.height()
    );
    println!();

    // ── Results table ─────────────────────────────────────────────────────────
    println!(
        "{:<48}  {:>9}  {:>9}  {:>9}  {:>8}",
        "Configuration", "img_semi", "col_semi", "glob_semi", "dur (s)"
    );
    println!("{}", "-".repeat(90));

    // img_semi: image pixel amplitudes vs reference audio STFT
    // col_semi: synthesised audio STFT vs reference audio STFT (column-aligned)
    // glob_semi: time-averaged energy correlation
    let mut results: Vec<(&str, f32, f32, f32)> = Vec::new();

    for &(label, sps, threshold, gain) in CONFIGS {
        print!(
            "  [{sps:3} sps  thr={:4}] … ",
            threshold.map_or_else(|| "none".to_string(), |t| format!("{t:.2}"))
        );
        let _ = std::io::Write::flush(&mut std::io::stdout());

        // Metric 1: image pixel amplitudes vs reference STFT
        // (uses the same metric as the prior threshold sweep — no synthesis involved)
        let img_stft = image_amp_spectrogram(&img, bounds, threshold);
        let ref_stft_img = if sps == 353 {
            &ref_stft_353
        } else {
            &ref_stft_512
        };
        let img_s = mean_semi_corr(ref_stft_img, &img_stft);

        // Metrics 2 & 3: synthesised audio vs reference audio
        let decoded = match sps {
            353 => decode_sps::<353>(img_path, threshold, gain),
            512 => decode_sps::<512>(img_path, threshold, gain),
            _ => unreachable!(),
        };

        #[expect(clippy::cast_precision_loss, reason = "sample count fits in f32")]
        let dur = decoded.len() as f32 / SAMPLE_RATE as f32;
        let dec_stft = phonopaper_stft(&decoded, sps as usize);

        let col_s = mean_semi_corr(ref_stft_img, &dec_stft);
        let glob_s = global_semi_corr(ref_stft_img, &dec_stft);

        println!("done ({dur:.1} s)");
        println!("  {label:<48}  {img_s:>9.4}  {col_s:>9.4}  {glob_s:>9.4}  {dur:>8.2}");

        results.push((label, img_s, col_s, glob_s));
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    println!();
    println!("── Summary (ranked by img_semi — pixel vs audio reference) ──────────────");
    let mut by_img = results.clone();
    by_img.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!(
        "  {:<48}  {:>9}  {:>9}  {:>9}",
        "Configuration", "img_semi", "col_semi", "glob_semi"
    );
    for (label, img_s, col_s, glob_s) in &by_img {
        println!("  {label:<48}  {img_s:>9.4}  {col_s:>9.4}  {glob_s:>9.4}");
    }

    let old = results
        .iter()
        .find(|r| r.0.contains("previous default"))
        .unwrap();
    let winner = &by_img[0];
    println!();
    println!(
        "Old default  img_semi: {:.4}  col_semi: {:.4}  glob_semi: {:.4}",
        old.1, old.2, old.3
    );
    println!(
        "Best config  img_semi: {:.4}  col_semi: {:.4}  glob_semi: {:.4}  [{}]",
        winner.1, winner.2, winner.3, winner.0
    );
    println!(
        "Improvement in img_semi: {:+.1}%",
        (winner.1 - old.1) / old.1.abs() * 100.0
    );
}
