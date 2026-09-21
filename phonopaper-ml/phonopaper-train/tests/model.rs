//! Shape and loss sanity checks on the CPU backend.

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::prelude::*;
use phonopaper_train::model::{DetectorConfig, OUTPUT_SIZE, decode_output};
use phonopaper_train::training::detection_loss;

type B = NdArray;

#[test]
fn forward_produces_nine_outputs_per_image() {
    let device = NdArrayDevice::Cpu;
    let model = DetectorConfig::new().init::<B>(&device);
    let input = Tensor::<B, 4>::zeros([3, 1, 128, 128], &device);
    let out = model.forward(input);
    assert_eq!(out.dims(), [3, OUTPUT_SIZE]);
}

#[test]
fn input_size_can_be_changed() {
    let device = NdArrayDevice::Cpu;
    let model = DetectorConfig::new().with_input_size(64).init::<B>(&device);
    assert_eq!(model.input_size(), 64);
    let out = model.forward(Tensor::<B, 4>::zeros([1, 1, 64, 64], &device));
    assert_eq!(out.dims(), [1, OUTPUT_SIZE]);
}

#[test]
#[should_panic(expected = "multiple of 32")]
fn invalid_input_size_panics() {
    let device = NdArrayDevice::Cpu;
    let _ = DetectorConfig::new()
        .with_input_size(100)
        .init::<B>(&device);
}

#[test]
fn detect_returns_probability_in_unit_interval() {
    let device = NdArrayDevice::Cpu;
    let model = DetectorConfig::new().with_input_size(32).init::<B>(&device);
    let det = model.detect(&vec![128u8; 32 * 32], &device);
    assert!((0.0..=1.0).contains(&det.probability));
    let px = det.corners_in_pixels(640.0, 480.0);
    assert!((px[0][0] - det.corners[0][0] * 640.0).abs() < 1e-4);
}

#[test]
fn decode_output_applies_sigmoid_and_splits_corners() {
    let device = NdArrayDevice::Cpu;
    let raw = Tensor::<B, 2>::from_floats([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], &device);
    let det = decode_output(&raw);
    assert_eq!(det.len(), 1);
    assert!((det[0].probability - 0.5).abs() < 1e-6);
    assert!((det[0].corners[3][1] - 0.8).abs() < 1e-6);
}

#[test]
fn perfect_prediction_has_near_zero_loss() {
    let device = NdArrayDevice::Cpu;
    let targets = Tensor::<B, 2>::from_floats(
        [
            [1.0, 0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        &device,
    );
    // Very confident logits, exact corners.
    let output = Tensor::<B, 2>::from_floats(
        [
            [20.0, 0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9],
            [-20.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        ],
        &device,
    );
    let loss: f32 = detection_loss(output, targets).into_scalar();
    assert!(loss < 1e-4, "loss = {loss}");
}

#[test]
fn corner_errors_on_negatives_do_not_count_but_positives_do() {
    let device = NdArrayDevice::Cpu;
    let targets =
        Tensor::<B, 2>::from_floats([[1.0, 0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9]], &device);
    let good =
        Tensor::<B, 2>::from_floats([[20.0, 0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9]], &device);
    let bad =
        Tensor::<B, 2>::from_floats([[20.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]], &device);
    let l_good: f32 = detection_loss(good, targets.clone()).into_scalar();
    let l_bad: f32 = detection_loss(bad, targets).into_scalar();
    assert!(l_bad > l_good + 0.1, "good = {l_good}, bad = {l_bad}");

    let neg_targets = Tensor::<B, 2>::from_floats([[0.0; 9]], &device);
    let neg_a =
        Tensor::<B, 2>::from_floats([[-20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], &device);
    let neg_b =
        Tensor::<B, 2>::from_floats([[-20.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]], &device);
    let la: f32 = detection_loss(neg_a, neg_targets.clone()).into_scalar();
    let lb: f32 = detection_loss(neg_b, neg_targets).into_scalar();
    assert!(
        (la - lb).abs() < 1e-6,
        "negatives must not be penalised on corners"
    );
}
