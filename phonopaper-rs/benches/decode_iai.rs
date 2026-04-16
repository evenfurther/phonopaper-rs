//! IAI-Callgrind benchmarks for the decode pipeline.
//!
//! Mirrors the Criterion benchmarks in `decode.rs` but uses instruction-count
//! measurements via Callgrind for deterministic, noise-free results.
//!
//! Three public functions are covered:
//!
//! * [`spectrogram_to_audio`] — batch synthesis of a full spectrogram.
//! * [`Synthesizer::synthesize_column`] — per-column synthesis.
//! * [`column_amplitudes_from_image`] — reading one amplitude column from an
//!   image.

use std::hint::black_box;

use iai_callgrind::{library_benchmark, library_benchmark_group, main as iai_main};
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

fn make_spectrogram(num_columns: usize, step: usize, amp: f32) -> SpectrogramVec {
    let mut spec = SpectrogramVec::new(num_columns);
    for col in 0..num_columns {
        for bin in (0..TOTAL_BINS).step_by(step) {
            spec.set(col, bin, amp);
        }
    }
    spec
}

fn make_amplitudes(step: usize, amp: f32) -> Vec<f32> {
    let mut amps = vec![0.0_f32; TOTAL_BINS];
    for bin in (0..TOTAL_BINS).step_by(step) {
        amps[bin] = amp;
    }
    amps
}

fn make_gray_image(data_width: u32, data_height: u32, luma: u8) -> (DynamicImage, DataBounds) {
    let img = GrayImage::from_pixel(data_width, data_height, Luma([luma]));
    let bounds = DataBounds {
        data_top: 0,
        data_bottom: data_height,
    };
    (DynamicImage::ImageLuma8(img), bounds)
}

// ─── spectrogram_to_audio ─────────────────────────────────────────────────────

fn setup_realistic_spectrogram() -> (SpectrogramVec, Vec<f32>) {
    let spec = make_spectrogram(1400, 8, 0.8);
    let out = vec![0.0_f32; 1400 * 512];
    (spec, out)
}

fn setup_dense_spectrogram() -> (SpectrogramVec, Vec<f32>) {
    let spec = make_spectrogram(1400, 1, 1.0);
    let out = vec![0.0_f32; 1400 * 512];
    (spec, out)
}

#[library_benchmark]
#[bench::realistic(setup_realistic_spectrogram())]
#[bench::all_bins(setup_dense_spectrogram())]
fn bench_spectrogram_to_audio((spec, mut out): (SpectrogramVec, Vec<f32>)) {
    spectrogram_to_audio::<_, 512>(black_box(&spec), black_box(&realistic_opts()), &mut out);
}

// ─── Synthesizer::synthesize_column ──────────────────────────────────────────

fn setup_synth_realistic() -> (Synthesizer<512>, Vec<f32>, Vec<f32>) {
    let amps = make_amplitudes(8, 0.8);
    let opts = realistic_opts();
    let mut synth = Synthesizer::<512>::new(opts);
    let mut pcm = [0.0_f32; 512];
    // Warm up so phasors are in a settled state.
    for _ in 0..10 {
        synth.synthesize_column(&amps, &mut pcm);
    }
    (synth, amps, pcm.to_vec())
}

fn setup_synth_all_bins() -> (Synthesizer<512>, Vec<f32>, Vec<f32>) {
    let amps = make_amplitudes(1, 1.0);
    let opts = realistic_opts();
    let mut synth = Synthesizer::<512>::new(opts);
    let mut pcm = [0.0_f32; 512];
    for _ in 0..10 {
        synth.synthesize_column(&amps, &mut pcm);
    }
    (synth, amps, pcm.to_vec())
}

#[library_benchmark]
#[bench::realistic(setup_synth_realistic())]
#[bench::all_bins(setup_synth_all_bins())]
fn bench_synthesizer_column((mut synth, amps, mut pcm): (Synthesizer<512>, Vec<f32>, Vec<f32>)) {
    let mut pcm_arr = [0.0_f32; 512];
    pcm_arr.copy_from_slice(&pcm[..512]);
    synth.synthesize_column(black_box(&amps), black_box(&mut pcm_arr));
    pcm[..512].copy_from_slice(&pcm_arr);
}

// ─── column_amplitudes_from_image ─────────────────────────────────────────────

fn setup_gray_image() -> (DynamicImage, DataBounds) {
    make_gray_image(1400, 720, 128)
}

#[library_benchmark]
#[bench::px720(setup_gray_image())]
fn bench_column_amplitudes_from_image((image, bounds): (DynamicImage, DataBounds)) {
    black_box(
        column_amplitudes_from_image(black_box(&image), black_box(Some(bounds)), 700).unwrap(),
    );
}

// ─── iai wiring ───────────────────────────────────────────────────────────────

library_benchmark_group!(
    name = decode_benches;
    benchmarks =
        bench_spectrogram_to_audio,
        bench_synthesizer_column,
        bench_column_amplitudes_from_image
);

iai_main!(library_benchmark_groups = decode_benches);
