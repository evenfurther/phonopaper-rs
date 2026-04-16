//! IAI-Callgrind benchmark for [`phonopaper_rs::encode::audio_to_spectrogram`].
//!
//! Mirrors the Criterion benchmark in `encode.rs` but uses instruction-count
//! measurements via Callgrind for deterministic, noise-free results.

use std::f32::consts::PI;

use std::hint::black_box;

use iai_callgrind::{library_benchmark, library_benchmark_group, main as iai_main};
use phonopaper_rs::encode::{AnalysisOptions, audio_to_spectrogram};
use phonopaper_rs::format::SAMPLE_RATE;

fn make_sine(freq_hz: f32, num_samples: usize) -> Vec<f32> {
    #[expect(
        clippy::cast_precision_loss,
        reason = "SAMPLE_RATE = 44_100 and num_samples ≤ 3 * 44_100 = 132_300 < 2^17; both fit exactly in f32"
    )]
    let sr = SAMPLE_RATE as f32;
    (0..num_samples)
        .map(|t| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "t < 132_300 < 2^17; exact in f32"
            )]
            (2.0 * PI * freq_hz * t as f32 / sr).sin()
        })
        .collect()
}

fn setup_3s_sine() -> Vec<f32> {
    make_sine(440.0, SAMPLE_RATE as usize * 3)
}

#[library_benchmark]
#[bench::sine_3s(setup_3s_sine())]
fn bench_audio_to_spectrogram(samples: Vec<f32>) {
    let opts = AnalysisOptions::default();
    black_box(audio_to_spectrogram(black_box(&samples), SAMPLE_RATE, &opts).unwrap());
}

library_benchmark_group!(
    name = encode_benches;
    benchmarks = bench_audio_to_spectrogram
);

iai_main!(library_benchmark_groups = encode_benches);
