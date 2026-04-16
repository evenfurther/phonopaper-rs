//! Benchmark for [`phonopaper_rs::encode::audio_to_spectrogram`].

use std::f32::consts::PI;

use criterion::{Criterion, criterion_group, criterion_main};
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

fn bench_audio_to_spectrogram(c: &mut Criterion) {
    // 3 seconds of audio at 44.1 kHz — typical short clip.
    let samples = make_sine(440.0, SAMPLE_RATE as usize * 3);
    let opts = AnalysisOptions::default(); // fft_window=4096, hop_size=353

    c.bench_function("audio_to_spectrogram 3s", |b| {
        b.iter(|| audio_to_spectrogram(&samples, SAMPLE_RATE, &opts).unwrap());
    });
}

criterion_group!(benches, bench_audio_to_spectrogram);
criterion_main!(benches);
