//! Shape and loss sanity checks on the CPU backend.

use burn::prelude::*;
use burn::tensor::Device;
use phonopaper_train::model::{Detection, DetectorConfig, OUTPUT_SIZE, decode_output, soft_argmax};
use phonopaper_train::training::heatmap_loss;

#[test]
fn forward_produces_nine_outputs_per_image() {
    let device = Device::flex();
    let model = DetectorConfig::new().init(&device);
    let input = Tensor::<4>::zeros([3, 1, 128, 128], &device);
    let out = model.forward(input);
    assert_eq!(out.dims(), [3, OUTPUT_SIZE]);
}

#[test]
fn input_size_can_be_changed() {
    let device = Device::flex();
    let model = DetectorConfig::new().with_input_size(64).init(&device);
    assert_eq!(model.input_size(), 64);
    let out = model.forward(Tensor::<4>::zeros([1, 1, 64, 64], &device));
    assert_eq!(out.dims(), [1, OUTPUT_SIZE]);
}

#[test]
#[should_panic(expected = "multiple of 32")]
fn invalid_input_size_panics() {
    let device = Device::flex();
    let _ = DetectorConfig::new().with_input_size(100).init(&device);
}

#[test]
fn detect_returns_probability_in_unit_interval() {
    let device = Device::flex();
    let model = DetectorConfig::new().with_input_size(32).init(&device);
    let det = model.detect(&vec![128u8; 32 * 32], &device);
    assert!((0.0..=1.0).contains(&det.probability));
    let px = det.corners_in_pixels(640.0, 480.0);
    assert!((px[0][0] - det.corners[0][0] * 640.0).abs() < 1e-4);
}

#[test]
fn decode_output_applies_sigmoid_and_splits_corners() {
    let device = Device::flex();
    let raw = Tensor::<2>::from_floats([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], &device);
    let det = decode_output(&raw);
    assert_eq!(det.len(), 1);
    assert!((det[0].probability - 0.5).abs() < 1e-6);
    assert!((det[0].corners[3][1] - 0.8).abs() < 1e-6);
}

fn corner_targets(device: &Device) -> Tensor<2> {
    Tensor::<2>::from_floats(
        [[
            1.0, 0.0875, 0.0875, 0.9125, 0.0875, 0.9125, 0.9125, 0.0875, 0.9125,
        ]],
        device,
    )
}

fn corner_logits(sharp: bool, rotate_channels: bool, device: &Device) -> Tensor<4> {
    let side = 16;
    let cells = side * side;
    let positions = [(2, 2), (13, 2), (13, 13), (2, 13)];
    let mut values = vec![0.0_f32; 4 * cells];
    for channel in 0..4 {
        let source = if rotate_channels {
            (channel + 2) % 4
        } else {
            channel
        };
        let (col, row) = positions[source];
        values[channel * cells + row * side + col] = if sharp { 12.0 } else { 2.0 };
    }
    Tensor::<1>::from_floats(values.as_slice(), device).reshape([1, 4, side, side])
}

#[test]
fn sharp_heatmaps_beat_diffuse_heatmaps() {
    let device = Device::flex();
    let targets = corner_targets(&device);
    let sharp: f32 =
        heatmap_loss(corner_logits(true, false, &device), targets.clone()).into_scalar();
    let diffuse: f32 = heatmap_loss(corner_logits(false, false, &device), targets).into_scalar();
    assert!(sharp < diffuse, "sharp={sharp}, diffuse={diffuse}");
}

#[test]
fn heatmap_loss_is_invariant_to_half_turn_channel_rotation() {
    let device = Device::flex();
    let targets = corner_targets(&device);
    let direct: f32 =
        heatmap_loss(corner_logits(true, false, &device), targets.clone()).into_scalar();
    let rotated: f32 = heatmap_loss(corner_logits(true, true, &device), targets).into_scalar();
    assert!((direct - rotated).abs() < 1e-5, "{direct} vs {rotated}");
}

#[test]
fn negatives_mask_heatmaps_and_all_negative_batch_is_safe() {
    let device = Device::flex();
    let targets = Tensor::<2>::zeros([2, 9], &device);
    let loss: f32 =
        heatmap_loss(Tensor::<4>::zeros([2, 4, 16, 16], &device), targets).into_scalar();
    assert!(loss.is_finite());
    assert_eq!(loss, 0.0);
}

#[test]
fn soft_argmax_recovers_a_peaked_cell() {
    let device = Device::flex();
    // 4 corners × 4×4 grid; put a very sharp peak for corner 0 at (col 3,
    // row 0), for corner 1 at (col 0, row 3), flat elsewhere.
    let mut data = vec![0.0_f32; 4 * 16];
    data[3] = 100.0; // corner 0, row 0, col 3
    data[16 + 12] = 100.0; // corner 1, row 3, col 0
    let heat = Tensor::<4>::from_floats(TensorData::new(data, [1, 4, 4, 4]), &device);
    let coords: Vec<f32> = soft_argmax(heat).into_data().try_into_vec::<f32>().unwrap();
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
fn heatmaps_are_genuine_stride_2() {
    let device = Device::flex();
    let model = DetectorConfig::new().init(&device);
    assert_eq!(model.heatmap_size(), 64);
    let prediction = model.forward_with_heatmaps(Tensor::<4>::zeros([2, 1, 128, 128], &device));
    assert_eq!(prediction.output.dims(), [2, OUTPUT_SIZE]);
    assert_eq!(prediction.heatmaps.dims(), [2, 4, 64, 64]);
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
