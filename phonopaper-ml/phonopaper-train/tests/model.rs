//! Shape and loss sanity checks on the CPU backend.

use burn::prelude::*;
use burn::tensor::Device;
use phonopaper_train::model::{
    CORNER_FUSION_STAGES, CORNER_HEAD_CHANNELS, Detection, DetectorConfig, DetectorOutput,
    OUTPUT_SIZE, decode_corners, decode_output, soft_argmax,
};
use phonopaper_train::training::{
    assign_corner_targets, corner_cell_targets, detection_loss_breakdown, edge_vector_loss,
    heatmap_loss, relative_size_loss, rotate_180,
};

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

fn corners_tensor(corners: [f32; 8], device: &Device) -> Tensor<2> {
    Tensor::<1>::from_floats(corners.as_slice(), device).reshape([1, 8])
}

fn present(device: &Device) -> Tensor<2> {
    Tensor::<2>::ones([1, 1], device)
}

fn scalar(tensor: Tensor<1>) -> f32 {
    tensor.into_scalar()
}

#[test]
fn shared_assignment_selects_rotated_equivalent_target() {
    let device = Device::flex();
    let truth = corner_targets(&device).narrow(1, 1, 8);
    let rotated = rotate_180(truth.clone());
    let assigned: Vec<f32> = assign_corner_targets(rotated.clone(), truth)
        .into_data()
        .try_into_vec()
        .unwrap();
    let expected: Vec<f32> = rotated.into_data().try_into_vec().unwrap();
    assert_eq!(assigned, expected);
}

#[test]
fn corner_cell_targets_encode_nearest_cell_and_exact_offset() {
    let device = Device::flex();
    let corners = corners_tensor([0.23, 0.37, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8], &device);
    let cells = corner_cell_targets(corners, 4, 4);
    let indices: Vec<i32> = cells.indices.into_data().try_into_vec().unwrap();
    let xs: Vec<f32> = cells.x_offsets.into_data().try_into_vec().unwrap();
    let ys: Vec<f32> = cells.y_offsets.into_data().try_into_vec().unwrap();
    assert_eq!(indices[0], 5);
    assert!((xs[0] + 0.12).abs() < 1e-6, "x offset = {}", xs[0]);
    assert!((ys[0] - 0.02).abs() < 1e-6, "y offset = {}", ys[0]);
}

#[test]
fn edge_loss_is_translation_invariant_and_rejects_contraction() {
    let device = Device::flex();
    let truth = corners_tensor([0.1, 0.2, 0.9, 0.2, 0.9, 0.8, 0.1, 0.8], &device);
    let translated = corners_tensor([0.2, 0.1, 1.0, 0.1, 1.0, 0.7, 0.2, 0.7], &device);
    let contracted = corners_tensor([0.2, 0.275, 0.8, 0.275, 0.8, 0.725, 0.2, 0.725], &device);
    let translated_loss = scalar(edge_vector_loss(
        translated,
        truth.clone(),
        present(&device),
    ));
    let contracted_loss = scalar(edge_vector_loss(contracted, truth, present(&device)));
    assert!(translated_loss < 1e-6, "translated={translated_loss}");
    assert!(contracted_loss > 0.05, "contracted={contracted_loss}");
}

#[test]
fn relative_size_loss_favors_exact_over_contracted_geometry() {
    let device = Device::flex();
    let truth = corners_tensor([0.1, 0.2, 0.9, 0.2, 0.9, 0.8, 0.1, 0.8], &device);
    let contracted = corners_tensor([0.2, 0.275, 0.8, 0.275, 0.8, 0.725, 0.2, 0.725], &device);
    let exact = scalar(relative_size_loss(
        truth.clone(),
        truth.clone(),
        present(&device),
    ));
    let contracted = scalar(relative_size_loss(contracted, truth, present(&device)));
    assert!(exact < 1e-6);
    assert!(contracted > 0.2, "contracted={contracted}");
}

fn prediction_for(corners: [f32; 8], target: [f32; 8], device: &Device) -> DetectorOutput {
    let side = 4;
    let cells = corner_cell_targets(corners_tensor(target, device), side, side);
    let indices: Vec<i32> = cells.indices.clone().into_data().try_into_vec().unwrap();
    let target_x: Vec<f32> = cells.x_offsets.into_data().try_into_vec().unwrap();
    let target_y: Vec<f32> = cells.y_offsets.into_data().try_into_vec().unwrap();
    let mut heatmaps = vec![-8.0_f32; 4 * side * side];
    let mut x_offsets = vec![0.0_f32; 4 * side * side];
    let mut y_offsets = vec![0.0_f32; 4 * side * side];
    let half_cell = 0.15_f32;
    for corner in 0..4 {
        let index = usize::try_from(indices[corner]).unwrap();
        heatmaps[corner * side * side + index] = 8.0;
        x_offsets[corner * side * side + index] = (target_x[corner] / half_cell).atanh();
        y_offsets[corner * side * side + index] = (target_y[corner] / half_cell).atanh();
    }
    let heatmaps =
        Tensor::<1>::from_floats(heatmaps.as_slice(), device).reshape([1, 4, side, side]);
    let x_offsets =
        Tensor::<1>::from_floats(x_offsets.as_slice(), device).reshape([1, 4, side, side]);
    let y_offsets =
        Tensor::<1>::from_floats(y_offsets.as_slice(), device).reshape([1, 4, side, side]);
    let coords = corners_tensor(corners, device);
    let output = Tensor::cat(vec![Tensor::<2>::from_floats([[8.0]], device), coords], 1);
    let corner_raw = Tensor::cat(
        vec![heatmaps.clone(), x_offsets.clone(), y_offsets.clone()],
        1,
    );
    DetectorOutput {
        output,
        heatmaps,
        x_offsets,
        y_offsets,
        corner_raw,
    }
}

#[test]
fn total_loss_favors_correct_geometry_over_contraction() {
    let device = Device::flex();
    let target = [0.12, 0.22, 0.88, 0.22, 0.88, 0.78, 0.12, 0.78];
    let contracted = [0.22, 0.29, 0.78, 0.29, 0.78, 0.71, 0.22, 0.71];
    let targets = Tensor::<1>::from_floats(
        [1.0, 0.12, 0.22, 0.88, 0.22, 0.88, 0.78, 0.12, 0.78].as_slice(),
        &device,
    )
    .reshape([1, 9]);
    let exact = detection_loss_breakdown(prediction_for(target, target, &device), targets.clone());
    let wrong = detection_loss_breakdown(prediction_for(contracted, target, &device), targets);
    assert!(scalar(exact.total) < scalar(wrong.total));
}

#[test]
fn all_negative_breakdown_has_finite_zero_spatial_terms() {
    let device = Device::flex();
    let prediction = prediction_for([0.0; 8], [0.0; 8], &device);
    let losses = detection_loss_breakdown(prediction, Tensor::<2>::zeros([1, 9], &device));
    for loss in [
        losses.heatmap,
        losses.offset,
        losses.coordinate,
        losses.edge,
        losses.size,
    ] {
        let value = scalar(loss);
        assert!(value.is_finite());
        assert_eq!(value, 0.0);
    }
    assert!(scalar(losses.total).is_finite());
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
fn corner_head_fuses_all_stages_and_exposes_raw_maps() {
    let device = Device::flex();
    let model = DetectorConfig::new().init(&device);
    assert_eq!(CORNER_FUSION_STAGES, 5);
    assert_eq!(CORNER_HEAD_CHANNELS, 12);
    assert_eq!(model.heatmap_size(), 64);
    let prediction = model.forward_with_heatmaps(Tensor::<4>::zeros([2, 1, 128, 128], &device));
    assert_eq!(prediction.output.dims(), [2, OUTPUT_SIZE]);
    assert_eq!(prediction.corner_raw.dims(), [2, 12, 64, 64]);
    assert_eq!(prediction.heatmaps.dims(), [2, 4, 64, 64]);
    assert_eq!(prediction.x_offsets.dims(), [2, 4, 64, 64]);
    assert_eq!(prediction.y_offsets.dims(), [2, 4, 64, 64]);
}

fn raw_corner_maps(
    side: usize,
    peaks: &[(usize, usize, usize, f32)],
    offsets: &[(usize, usize, usize, f32, f32)],
    device: &Device,
) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
    let cells = side * side;
    let mut logits = vec![0.0; 4 * cells];
    let mut xs = vec![0.0; 4 * cells];
    let mut ys = vec![0.0; 4 * cells];
    for &(corner, row, col, value) in peaks {
        logits[corner * cells + row * side + col] = value;
    }
    for &(corner, row, col, x, y) in offsets {
        let index = corner * cells + row * side + col;
        xs[index] = x;
        ys[index] = y;
    }
    (
        Tensor::<1>::from_floats(logits.as_slice(), device).reshape([1, 4, side, side]),
        Tensor::<1>::from_floats(xs.as_slice(), device).reshape([1, 4, side, side]),
        Tensor::<1>::from_floats(ys.as_slice(), device).reshape([1, 4, side, side]),
    )
}

#[expect(
    clippy::cast_precision_loss,
    reason = "test grids contain only two or four cells"
)]
fn grid_centre(index: usize, side: usize) -> f32 {
    -0.1 + 1.2 * (index as f32 + 0.5) / side as f32
}

#[test]
fn hard_decoder_returns_selected_cell_centres() {
    let device = Device::flex();
    let peaks = [
        (0, 0, 3, 10.0),
        (1, 3, 0, 10.0),
        (2, 1, 2, 10.0),
        (3, 2, 1, 10.0),
    ];
    let (logits, xs, ys) = raw_corner_maps(4, &peaks, &[], &device);
    let coords: Vec<f32> = decode_corners(logits, xs, ys)
        .into_data()
        .try_into_vec()
        .unwrap();
    let expected = [
        grid_centre(3, 4),
        grid_centre(0, 4),
        grid_centre(0, 4),
        grid_centre(3, 4),
        grid_centre(2, 4),
        grid_centre(1, 4),
        grid_centre(1, 4),
        grid_centre(2, 4),
    ];
    for (actual, expected) in coords.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
    }
}

#[test]
fn hard_decoder_applies_bounded_selected_cell_offsets() {
    let device = Device::flex();
    let peaks = [(0, 1, 2, 10.0)];
    let offsets = [(0, 1, 2, 1.0, -1.0)];
    let (logits, xs, ys) = raw_corner_maps(4, &peaks, &offsets, &device);
    let coords: Vec<f32> = decode_corners(logits, xs, ys)
        .into_data()
        .try_into_vec()
        .unwrap();
    let displacement = 0.15 * 1.0_f32.tanh();
    assert!((coords[0] - (grid_centre(2, 4) + displacement)).abs() < 1e-6);
    assert!((coords[1] - (grid_centre(1, 4) - displacement)).abs() < 1e-6);
}

#[test]
fn hard_decoder_ignores_diffuse_secondary_mass() {
    let device = Device::flex();
    let mut peaks = vec![(0, 0, 0, 10.0)];
    for row in 0..4 {
        for col in 0..4 {
            if row != 0 || col != 0 {
                peaks.push((0, row, col, 9.9));
            }
        }
    }
    let (logits, xs, ys) = raw_corner_maps(4, &peaks, &[], &device);
    let coords: Vec<f32> = decode_corners(logits, xs, ys)
        .into_data()
        .try_into_vec()
        .unwrap();
    assert!((coords[0] - grid_centre(0, 4)).abs() < 1e-6);
    assert!((coords[1] - grid_centre(0, 4)).abs() < 1e-6);
}

#[test]
fn offset_channels_correspond_to_the_same_corner_and_cell() {
    let device = Device::flex();
    let peaks = [
        (0, 0, 0, 10.0),
        (1, 0, 1, 10.0),
        (2, 1, 0, 10.0),
        (3, 1, 1, 10.0),
    ];
    let offsets = [
        (0, 0, 0, -2.0, 2.0),
        (1, 0, 1, -1.0, 1.0),
        (2, 1, 0, 1.0, -1.0),
        (3, 1, 1, 2.0, -2.0),
        (0, 1, 1, 100.0, 100.0),
    ];
    let (logits, xs, ys) = raw_corner_maps(2, &peaks, &offsets, &device);
    let coords: Vec<f32> = decode_corners(logits, xs, ys)
        .into_data()
        .try_into_vec()
        .unwrap();
    let half_cell = 0.3;
    let expected = [
        grid_centre(0, 2) - half_cell * 2.0_f32.tanh(),
        grid_centre(0, 2) + half_cell * 2.0_f32.tanh(),
        grid_centre(1, 2) - half_cell * 1.0_f32.tanh(),
        grid_centre(0, 2) + half_cell * 1.0_f32.tanh(),
        grid_centre(0, 2) + half_cell * 1.0_f32.tanh(),
        grid_centre(1, 2) - half_cell * 1.0_f32.tanh(),
        grid_centre(1, 2) + half_cell * 2.0_f32.tanh(),
        grid_centre(1, 2) - half_cell * 2.0_f32.tanh(),
    ];
    for (actual, expected) in coords.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
    }
}

#[test]
fn raw_corner_maps_follow_variable_input_size() {
    let device = Device::flex();
    for input_size in [64, 128] {
        let model = DetectorConfig::new()
            .with_input_size(input_size)
            .init(&device);
        let prediction = model
            .forward_with_heatmaps(Tensor::<4>::zeros([1, 1, input_size, input_size], &device));
        assert_eq!(
            prediction.corner_raw.dims(),
            [1, 12, input_size / 2, input_size / 2]
        );
    }
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
