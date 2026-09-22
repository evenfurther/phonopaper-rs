//! Shape and loss sanity checks on the CPU backend.

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::prelude::*;
use phonopaper_train::model::{Detection, DetectorConfig, OUTPUT_SIZE, decode_output, soft_argmax};
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

#[test]
fn loss_is_invariant_under_a_half_turn_of_the_sheet() {
    // TL, TR, BR, BL of an upright rectangle …
    let upright = [1.0, 0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9];
    // … and the same rectangle labelled from the opposite end.
    let turned = [1.0, 0.9, 0.9, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1];
    let device = NdArrayDevice::Cpu;
    let output =
        Tensor::<B, 2>::from_floats([[20.0, 0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9]], &device);
    let l_upright: f32 = detection_loss(
        output.clone(),
        Tensor::<B, 2>::from_floats([upright], &device),
    )
    .into_scalar();
    let l_turned: f32 =
        detection_loss(output, Tensor::<B, 2>::from_floats([turned], &device)).into_scalar();
    assert!(l_upright < 1e-4, "exact prediction: {l_upright}");
    assert!(
        (l_upright - l_turned).abs() < 1e-6,
        "both labellings must score the same: {l_upright} vs {l_turned}"
    );
    // A 90° relabelling (bands on the wrong edges) is NOT equivalent.
    let quarter = [1.0, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.1, 0.1];
    let out2 =
        Tensor::<B, 2>::from_floats([[20.0, 0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9]], &device);
    let l_quarter: f32 =
        detection_loss(out2, Tensor::<B, 2>::from_floats([quarter], &device)).into_scalar();
    assert!(
        l_quarter > 0.1,
        "quarter turn must be penalised: {l_quarter}"
    );
}

#[test]
fn soft_argmax_recovers_a_peaked_cell() {
    let device = NdArrayDevice::Cpu;
    // 4 corners × 4×4 grid; put a very sharp peak for corner 0 at (col 3,
    // row 0), for corner 1 at (col 0, row 3), flat elsewhere.
    let mut data = vec![0.0_f32; 4 * 16];
    data[3] = 100.0; // corner 0, row 0, col 3
    data[16 + 12] = 100.0; // corner 1, row 3, col 0
    let heat = Tensor::<B, 4>::from_floats(TensorData::new(data, [1, 4, 4, 4]), &device);
    let coords: Vec<f32> = soft_argmax(heat).into_data().to_vec().unwrap();
    // Grid spans [-0.1, 1.1]; cell centres at -0.1 + 1.2 * (i + 0.5) / 4.
    let centre = |i: f32| -0.1 + 1.2 * (i + 0.5) / 4.0;
    assert!((coords[0] - centre(3.0)).abs() < 1e-3, "x0 = {}", coords[0]);
    assert!((coords[1] - centre(0.0)).abs() < 1e-3, "y0 = {}", coords[1]);
    assert!((coords[2] - centre(0.0)).abs() < 1e-3, "x1 = {}", coords[2]);
    assert!((coords[3] - centre(3.0)).abs() < 1e-3, "y1 = {}", coords[3]);
    // A flat heat-map yields the grid centre.
    assert!((coords[4] - 0.5).abs() < 1e-5 && (coords[5] - 0.5).abs() < 1e-5);
}

#[test]
fn heatmap_size_is_stride_8() {
    let device = NdArrayDevice::Cpu;
    let model = DetectorConfig::new().init::<B>(&device);
    assert_eq!(model.heatmap_size(), 16);
}

#[test]
fn canonical_detection_starts_with_the_higher_corner() {
    let det = Detection {
        probability: 0.9,
        corners: [[0.9, 0.9], [0.1, 0.9], [0.1, 0.1], [0.9, 0.1]],
    };
    let canon = det.canonical();
    assert_eq!(canon.corners[0], [0.1, 0.1]);
    assert_eq!(canon.corners[1], [0.9, 0.1]);
    assert_eq!(canon.canonical(), canon, "canonical is idempotent");
    assert_eq!(det.rotated_180().rotated_180(), det);
}
