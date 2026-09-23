//! Integration tests for the neural-network pattern detector
//! (`phonopaper_rs::decode::nn`, `nn-detector` feature).
#![cfg(feature = "nn-detector")]

use image::{DynamicImage, GrayImage, Luma};
use imageproc::geometric_transformations::{Border, Interpolation, Projection, warp_into};
use phonopaper_rs::SpectrogramVec;
use phonopaper_rs::decode::nn::{Detection, PatternDetector, embedded_config, load, prepare_image};
use phonopaper_rs::render::{RenderOptions, spectrogram_to_image};

/// Render a `PhonoPaper` pattern with a few notes as an 8-bit grayscale image,
/// cropped to its **ink box** (the white margins above and below the marker
/// bands removed) — that is what the detector's corners delimit.
fn render_pattern(num_columns: usize) -> GrayImage {
    let mut spec = SpectrogramVec::new(num_columns);
    for col in 0..num_columns {
        // A slow glide plus a steady chord, so the data area is not blank.
        let glide = 100 + (col * 150) / num_columns;
        spec.set(col, glide, 1.0);
        spec.set(col, glide + 48, 0.6);
        if (col / 40) % 2 == 0 {
            spec.set(col, 250, 0.8);
        }
    }
    // Fewer pixels per octave than the print default keeps the stripes a
    // reasonable fraction of the pattern once it is shrunk into the frame.
    let opts = RenderOptions {
        px_per_octave: 40,
        ..RenderOptions::default()
    };
    let full = DynamicImage::ImageRgb8(spectrogram_to_image(&spec, &opts)).into_luma8();
    image::imageops::crop_imm(
        &full,
        0,
        opts.margin,
        full.width(),
        full.height() - 2 * opts.margin,
    )
    .to_image()
}

/// Paste `pattern` onto a `width × height` light-gray background so that its
/// corners land on `quad` (`[TL, TR, BR, BL]` in pattern orientation).
fn scene_with_pattern(
    pattern: &GrayImage,
    width: u32,
    height: u32,
    quad: [(f32, f32); 4],
) -> DynamicImage {
    #[expect(clippy::cast_precision_loss, reason = "test image dimensions are tiny")]
    let (pw, ph) = (pattern.width() as f32, pattern.height() as f32);
    let from = [(0.0, 0.0), (pw, 0.0), (pw, ph), (0.0, ph)];
    let projection = Projection::from_control_points(from, quad).expect("non-degenerate quad");
    let mut canvas = GrayImage::from_pixel(width, height, Luma([205]));
    warp_into(
        pattern,
        projection,
        Interpolation::Bilinear,
        Border::Constant(Luma([205])),
        &mut canvas,
    );
    DynamicImage::ImageLuma8(canvas)
}

/// Mean distance between detected and expected corners, taking the 180°
/// ambiguity into account (the pattern looks the same upside down).
fn mean_corner_error(detected: [[f32; 2]; 4], expected: [(f32, f32); 4]) -> f32 {
    let error = |shift: usize| {
        (0..4)
            .map(|i| {
                let (ex, ey) = expected[(i + shift) % 4];
                let [dx, dy] = detected[i];
                (dx - ex).hypot(dy - ey)
            })
            .sum::<f32>()
            / 4.0
    };
    error(0).min(error(2))
}

#[test]
fn embedded_config_matches_model() {
    let config = embedded_config();
    assert_eq!(config.input_size, 128);
    let model = load();
    assert_eq!(model.input_size(), 128);
    assert_eq!(model.heatmap_size(), 32);
    let detector = PatternDetector::new();
    assert_eq!(detector.input_size(), 128);
    assert_eq!(detector.model().input_size(), 128);
}

#[test]
fn prepare_image_stretches_to_square_gray() {
    let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(300, 100, Luma([42])));
    let pixels = prepare_image(&img, 128);
    assert_eq!(pixels.len(), 128 * 128);
    assert!(pixels.iter().all(|&p| p == 42));
}

#[test]
fn detects_upright_pattern() {
    let pattern = render_pattern(600);
    let quad = [(60.0, 60.0), (260.0, 60.0), (260.0, 180.0), (60.0, 180.0)];
    let scene = scene_with_pattern(&pattern, 320, 240, quad);

    let detector = PatternDetector::new();
    let detection = detector.detect(&scene);
    assert!(
        detection.probability > 0.9,
        "expected a pattern, got probability {}",
        detection.probability
    );
    let corners = detection.corners_in_pixels(320.0, 240.0);
    let error = mean_corner_error(corners, quad);
    assert!(
        error < 6.0,
        "mean corner error {error} px, corners {corners:?}"
    );

    // `find_corners` agrees with `detect`.
    let found = detector.find_corners(&scene, 0.5).expect("pattern present");
    assert_eq!(found, corners);
    assert!(detector.find_corners(&scene, 1.01).is_none());
}

#[test]
fn detects_rotated_pattern() {
    let pattern = render_pattern(600);
    // Rotated by roughly 25° with a little perspective.
    let quad = [(70.0, 15.0), (290.0, 75.0), (250.0, 225.0), (30.0, 165.0)];
    let scene = scene_with_pattern(&pattern, 320, 240, quad);

    let detector = PatternDetector::new();
    let detection = detector.detect(&scene);
    assert!(
        detection.probability > 0.5,
        "expected a pattern, got probability {}",
        detection.probability
    );
    let corners = detection.corners_in_pixels(320.0, 240.0);
    let error = mean_corner_error(corners, quad);
    assert!(
        error < 12.0,
        "mean corner error {error} px, corners {corners:?}"
    );
}

#[test]
fn rejects_background_without_pattern() {
    let detector = PatternDetector::new();

    // Flat background.
    let flat = DynamicImage::ImageLuma8(GrayImage::from_pixel(320, 240, Luma([180])));
    let detection = detector.detect(&flat);
    assert!(
        detection.probability < 0.5,
        "flat image: probability {}",
        detection.probability
    );
    assert!(detector.find_corners(&flat, 0.5).is_none());

    // Background with a few dark rectangles (no stripe topology).
    let mut boxes = GrayImage::from_pixel(320, 240, Luma([220]));
    for (x0, y0, x1, y1) in [(20, 20, 120, 90), (160, 40, 300, 60), (50, 150, 250, 230)] {
        for y in y0..y1 {
            for x in x0..x1 {
                boxes.put_pixel(x, y, Luma([40]));
            }
        }
    }
    let detection = detector.detect(&DynamicImage::ImageLuma8(boxes));
    assert!(
        detection.probability < 0.5,
        "rectangles: probability {}",
        detection.probability
    );
}

#[test]
fn detect_prepared_matches_detect() {
    let pattern = render_pattern(600);
    let quad = [(20.0, 30.0), (300.0, 30.0), (300.0, 220.0), (20.0, 220.0)];
    let scene = scene_with_pattern(&pattern, 320, 240, quad);
    let detector = PatternDetector::new();
    let prepared = prepare_image(&scene, detector.input_size());
    assert_eq!(detector.detect_prepared(&prepared), detector.detect(&scene));
}

#[test]
fn detection_orderings() {
    let d = Detection {
        probability: 0.9,
        corners: [[0.6, 0.8], [0.2, 0.8], [0.2, 0.1], [0.6, 0.1]],
    };
    let rotated = d.rotated_180();
    assert_eq!(rotated.probability, 0.9);
    assert_eq!(
        rotated.corners,
        [[0.2, 0.1], [0.6, 0.1], [0.6, 0.8], [0.2, 0.8]]
    );
    // Canonical form starts with the higher corner and is idempotent.
    assert_eq!(d.canonical(), rotated);
    assert_eq!(rotated.canonical(), rotated);
    let [px, py] = d.corners_in_pixels(100.0, 10.0)[0];
    assert!((px - 60.0).abs() < 1e-4 && (py - 8.0).abs() < 1e-4);

    // Ties on `y` are broken by `x`.
    let tie = Detection {
        probability: 0.5,
        corners: [[0.7, 0.5], [0.9, 0.5], [0.3, 0.5], [0.1, 0.5]],
    };
    assert_eq!(tie.canonical().corners[0], [0.3, 0.5]);
    let tie_first = Detection {
        probability: 0.5,
        corners: [[0.1, 0.5], [0.3, 0.5], [0.7, 0.5], [0.9, 0.5]],
    };
    assert_eq!(tie_first.canonical(), tie_first);
}
