//! Benchmarks for the decode pipeline.
//!
//! Covers three public functions:
//!
//! * [`spectrogram_to_audio`] — batch synthesis of a full spectrogram.
//! * [`Synthesizer::synthesize_column`] — per-column real-time synthesis.
//! * [`column_amplitudes_from_image`] — reading one column of pixels from
//!   a `DynamicImage` into a 384-element amplitude slice.
//!
//! ## Synthetic inputs
//!
//! All benchmarks use programmatically generated data so they run without
//! external files and reproduce identically on every machine.
//!
//! * **Realistic** density: every 8th bin active at amplitude 0.8.
//!   Represents a typical musical image (~12.5 % bin occupancy).
//! * **All-bins** density: every bin active at amplitude 1.0.
//!   Worst-case for the synthesis loop.
//! * **Standard image** geometry: 1400 × 720 px data area, matching the
//!   pixel dimensions of a typical `PhonoPaper` JPEG printed at A4 size.

use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main, measurement::WallTime,
};
use image::{DynamicImage, GrayImage, Luma};
use phonopaper_rs::decode::{
    AmplitudeMode, DataBounds, SynthesisOptions, Synthesizer, column_amplitudes_from_image,
    spectrogram_to_audio,
};
use phonopaper_rs::format::{SAMPLE_RATE, TOTAL_BINS};
use phonopaper_rs::spectrogram::SpectrogramVec;

// ─── Shared helpers ───────────────────────────────────────────────────────────

fn realistic_opts() -> SynthesisOptions {
    SynthesisOptions {
        sample_rate: SAMPLE_RATE,
        gain: 0.15,
        decode_gamma: 1.0,
        mode: AmplitudeMode::Linear { threshold: None },
    }
}

/// Build a spectrogram with every `step`-th bin active at `amp`.
fn make_spectrogram(num_columns: usize, step: usize, amp: f32) -> SpectrogramVec {
    let mut spec = SpectrogramVec::new(num_columns);
    for col in 0..num_columns {
        for bin in (0..TOTAL_BINS).step_by(step) {
            spec.set(col, bin, amp);
        }
    }
    spec
}

/// Build a single-column amplitude slice with every `step`-th bin active.
fn make_amplitudes(step: usize, amp: f32) -> Vec<f32> {
    let mut amps = vec![0.0_f32; TOTAL_BINS];
    for bin in (0..TOTAL_BINS).step_by(step) {
        amps[bin] = amp;
    }
    amps
}

/// Build a synthetic `GrayImage` whose data area is `data_width × data_height`
/// pixels, with each pixel set to `luma`.  Returns the image as a
/// `DynamicImage` together with the corresponding [`DataBounds`] (so the
/// caller can pass `Some(bounds)` to [`column_amplitudes_from_image`] and
/// skip marker detection).
fn make_gray_image(data_width: u32, data_height: u32, luma: u8) -> (DynamicImage, DataBounds) {
    let img = GrayImage::from_pixel(data_width, data_height, Luma([luma]));
    let bounds = DataBounds {
        data_top: 0,
        data_bottom: data_height,
    };
    (DynamicImage::ImageLuma8(img), bounds)
}

// ─── spectrogram_to_audio ─────────────────────────────────────────────────────

fn bench_spectrogram_to_audio(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectrogram_to_audio");

    // Realistic density: every 8th bin active (48/384 = 12.5 %).
    let spec_realistic = make_spectrogram(1400, 8, 0.8);
    let mut out_realistic = vec![0.0_f32; 1400 * 512];
    group.bench_function("realistic_1400col", |b| {
        b.iter(|| {
            spectrogram_to_audio::<_, 512>(&spec_realistic, &realistic_opts(), &mut out_realistic);
        });
    });

    // Worst case: all 384 bins active.
    let spec_dense = make_spectrogram(1400, 1, 1.0);
    let mut out_dense = vec![0.0_f32; 1400 * 512];
    group.bench_function("all_bins_1400col", |b| {
        b.iter(|| spectrogram_to_audio::<_, 512>(&spec_dense, &realistic_opts(), &mut out_dense));
    });

    group.finish();
}

// ─── Synthesizer::synthesize_column ──────────────────────────────────────────

/// Benchmark `Synthesizer::synthesize_column` for a single column.
///
/// This is the metric that matters most for real-time applications: the
/// latency of one call must fit inside the audio device's callback window
/// (typically 512 samples / 44 100 Hz ≈ 11.6 ms).
fn bench_synthesizer_column(c: &mut Criterion) {
    let mut group: BenchmarkGroup<WallTime> = c.benchmark_group("synthesizer_column");

    bench_synthesizer_density(&mut group, "realistic", 8, 0.8);
    bench_synthesizer_density(&mut group, "all_bins", 1, 1.0);

    group.finish();
}

fn bench_synthesizer_density(
    group: &mut BenchmarkGroup<WallTime>,
    label: &str,
    step: usize,
    amp: f32,
) {
    let amps = make_amplitudes(step, amp);
    let opts = realistic_opts();

    // Warm up the phasor state for a few columns so the first measured
    // iteration is not an outlier (cold phasors).
    let mut synth = Synthesizer::<512>::new(opts);
    let mut pcm = [0.0_f32; 512];
    for _ in 0..10 {
        synth.synthesize_column(&amps, &mut pcm);
    }

    group.bench_with_input(BenchmarkId::new("single_col", label), label, |b, _| {
        b.iter(|| synth.synthesize_column(&amps, &mut pcm));
    });
}

// ─── column_amplitudes_from_image ─────────────────────────────────────────────

/// Benchmark `column_amplitudes_from_image` for a standard-size image.
///
/// A standard `PhonoPaper` JPEG has a data area of 720 px tall (8 octaves ×
/// 90 px/octave) and ~1400 px wide.  The benchmark reads column 700 (the
/// middle) with pre-computed [`DataBounds`] so marker detection is not
/// included in the measurement.
fn bench_column_amplitudes_from_image(c: &mut Criterion) {
    // Standard PhonoPaper data area: 1400 × 720 px, mid-grey pixels.
    let (image, bounds) = make_gray_image(1400, 720, 128);

    c.bench_function("column_amplitudes_from_image/720px", |b| {
        b.iter(|| column_amplitudes_from_image(&image, Some(bounds), 700).unwrap());
    });
}

// ─── criterion wiring ─────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_spectrogram_to_audio,
    bench_synthesizer_column,
    bench_column_amplitudes_from_image,
);
criterion_main!(benches);
