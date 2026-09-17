use image::{GrayImage, Luma};
use phonopaper_android::{decode_image_to_pcm, detect_preview_bounds};
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
    assert!(!pcm.is_empty());
    assert!(pcm.iter().any(|&sample| sample != 0));
}

#[test]
fn reject_invalid_image_bytes() {
    let err = decode_image_to_pcm(b"not an image").expect_err("invalid input should fail");
    assert!(!err.is_empty());
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

#[test]
fn detect_preview_bounds_generated_phonopaper_image() {
    let mut spectrogram = SpectrogramVec::new(24);
    for col in 0..spectrogram.num_columns() {
        spectrogram.set(col, 120, 1.0);
        if col % 2 == 0 {
            spectrogram.set(col, 144, 0.5);
        }
    }

    let image = spectrogram_to_image(&spectrogram, &RenderOptions::default());
    let png = encode_png(&image::DynamicImage::ImageRgb8(image));

    let bounds = detect_preview_bounds(&png)
        .expect("generated image should be analyzable")
        .expect("generated image should contain preview bounds");
    assert!(bounds.0 < bounds.1);
}

#[test]
fn detect_preview_bounds_blank_image_returns_none() {
    let blank = GrayImage::from_pixel(32, 32, Luma([255]));
    let png = encode_png(&image::DynamicImage::ImageLuma8(blank));

    let bounds = detect_preview_bounds(&png).expect("blank PNG is still a valid image");
    assert_eq!(bounds, None);
}
