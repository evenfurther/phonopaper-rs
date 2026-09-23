use image::{DynamicImage, GrayImage, Luma};
use imageproc::geometric_transformations::{Border, Interpolation, Projection, warp_into};
use phonopaper_android::{decode_image_to_pcm, detect_pattern_corners, rectify_pattern};
use phonopaper_rs::{
    SpectrogramVec,
    render::{RenderOptions, spectrogram_to_image},
};

fn encode_png(image: &image::DynamicImage) -> Vec<u8> {
    let mut png = Vec::new();
    image
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("write PNG test fixture");
    png
}

fn deterministic_noise(width: u32, height: u32, seed: u64) -> image::DynamicImage {
    let mut state = seed;
    let img = image::GrayImage::from_fn(width, height, |_x, _y| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let value = (state >> 56) as u8;
        image::Luma([value])
    });
    image::DynamicImage::ImageLuma8(img)
}

#[test]
fn decode_generated_phonopaper_image() {
    let mut spectrogram = SpectrogramVec::new(24);
    for col in 0..spectrogram.num_columns() {
        spectrogram.set(col, 120, 1.0);
        if col % 3 == 0 {
            spectrogram.set(col, 144, 0.5);
        }
    }

    let image = spectrogram_to_image(&spectrogram, &RenderOptions::default());
    let png = encode_png(&image::DynamicImage::ImageRgb8(image));
    let pcm = decode_image_to_pcm(&png).expect("decode generated image");
    assert_ne!(pcm.len(), 0, "decoded audio must not be empty");
    assert!(pcm.iter().any(|&sample| sample != 0));
}

#[test]
fn reject_invalid_image_bytes() {
    let err = decode_image_to_pcm(b"not an image").expect_err("invalid input should fail");
    assert_ne!(err.len(), 0, "error message must not be empty");
}

#[test]
fn reject_noise_image() {
    let png = encode_png(&deterministic_noise(80, 484, 17));
    let err = decode_image_to_pcm(&png).expect_err("noise image should not decode as PhonoPaper");
    assert!(err.contains("marker"), "unexpected error: {err}");
}

#[test]
fn decode_only_embedded_pattern_columns() {
    let pattern_width = 16usize;
    let left_padding = 12u32;
    let right_padding = 12u32;
    let pattern_width_u32 = u32::try_from(pattern_width).expect("pattern width fits in u32");
    let total_width = left_padding + pattern_width_u32 + right_padding;

    let mut spectrogram = SpectrogramVec::new(pattern_width);
    for col in 0..pattern_width {
        spectrogram.set(col, 120, 1.0);
        if col % 4 == 0 {
            spectrogram.set(col, 156, 0.75);
        }
    }

    let pattern = spectrogram_to_image(&spectrogram, &RenderOptions::default());
    let mut composite =
        image::RgbImage::from_pixel(total_width, pattern.height(), image::Rgb([255u8, 255, 255]));
    for x in 0..pattern.width() {
        for y in 0..pattern.height() {
            composite.put_pixel(left_padding + x, y, *pattern.get_pixel(x, y));
        }
    }

    let png = encode_png(&image::DynamicImage::ImageRgb8(composite));
    let pcm = decode_image_to_pcm(&png).expect("embedded pattern should decode");
    assert_eq!(pcm.len(), pattern_width * 353);
    assert!(pcm.iter().any(|&sample| sample != 0));
}

/// A `PhonoPaper` pattern with a steady note, cropped to its ink box (the
/// white margins above and below the marker bands removed — that is what the
/// detector's corners delimit).
fn ink_box_pattern(num_columns: usize) -> GrayImage {
    let mut spectrogram = SpectrogramVec::new(num_columns);
    for col in 0..num_columns {
        spectrogram.set(col, 120, 1.0);
        spectrogram.set(col, 168, 0.7);
    }
    let opts = RenderOptions {
        px_per_octave: 40,
        ..RenderOptions::default()
    };
    let full = DynamicImage::ImageRgb8(spectrogram_to_image(&spectrogram, &opts)).into_luma8();
    image::imageops::crop_imm(
        &full,
        0,
        opts.margin,
        full.width(),
        full.height() - 2 * opts.margin,
    )
    .to_image()
}

/// Warp `pattern` so that its ink box lands on `quad` (`[TL, TR, BR, BL]`)
/// over a light-gray `width × height` background.
fn camera_scene(
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

/// Rotated by roughly 25° with some perspective.
const CAMERA_QUAD: [(f32, f32); 4] = [(70.0, 15.0), (290.0, 75.0), (250.0, 225.0), (30.0, 165.0)];

#[test]
fn detect_pattern_corners_in_camera_like_scene() {
    let scene = camera_scene(&ink_box_pattern(600), 320, 240, CAMERA_QUAD);
    let png = encode_png(&scene);

    let corners = detect_pattern_corners(&png)
        .expect("scene should be analyzable")
        .expect("scene should contain a pattern");
    // Corners come back as fractions of the image size, in canonical order;
    // the sheet looks the same upside down so accept both orderings.
    let error = |shift: usize| {
        (0..4)
            .map(|i| {
                let (ex, ey) = CAMERA_QUAD[(i + shift) % 4];
                let [fx, fy] = corners[i];
                (fx * 320.0 - ex).hypot(fy * 240.0 - ey)
            })
            .sum::<f32>()
            / 4.0
    };
    let error = error(0).min(error(2));
    assert!(
        error < 12.0,
        "mean corner error {error} px, corners {corners:?}"
    );
}

#[test]
fn detect_pattern_corners_blank_image_returns_none() {
    let blank = GrayImage::from_pixel(64, 48, Luma([255]));
    let png = encode_png(&DynamicImage::ImageLuma8(blank));

    let corners = detect_pattern_corners(&png).expect("blank PNG is still a valid image");
    assert_eq!(corners, None);
}

#[test]
fn detect_pattern_corners_rejects_invalid_bytes() {
    let err = detect_pattern_corners(b"not an image").expect_err("invalid input should fail");
    assert_ne!(err.len(), 0);
}

#[test]
fn decode_rotated_camera_like_scene() {
    // The stripe detector alone cannot read a rotated sheet; the network +
    // rectification path must.
    let scene = camera_scene(&ink_box_pattern(600), 320, 240, CAMERA_QUAD);
    let png = encode_png(&scene);

    let pcm = decode_image_to_pcm(&png).expect("rotated pattern should decode");
    assert!(pcm.len() >= 100 * 353, "only {} samples decoded", pcm.len());
    assert!(pcm.iter().any(|&sample| sample != 0));
}

#[test]
fn rectify_pattern_produces_upright_ink_box() {
    let scene = camera_scene(&ink_box_pattern(600), 320, 240, CAMERA_QUAD);
    let corners = CAMERA_QUAD.map(|(x, y)| [x, y]);

    let rectified = rectify_pattern(&scene, corners).into_luma8();
    // Longest edges: top ≈ 228 px, left ≈ 155 px, plus the white border.
    assert!(rectified.width() > 228 && rectified.width() < 260);
    assert!(rectified.height() > 155 && rectified.height() < 200);
    // Border is white; the centre column crosses dark marker stripes.
    assert_eq!(rectified.get_pixel(0, 0)[0], 255);
    let centre_x = rectified.width() / 2;
    let dark_rows = (0..rectified.height())
        .filter(|&y| rectified.get_pixel(centre_x, y)[0] < 128)
        .count();
    assert!(
        dark_rows >= 8,
        "only {dark_rows} dark rows in the centre column"
    );

    // Degenerate corners (all identical) fall back to the grayscale input.
    let fallback = rectify_pattern(&scene, [[10.0, 10.0]; 4]).into_luma8();
    assert_eq!((fallback.width(), fallback.height()), (320, 240));
}
