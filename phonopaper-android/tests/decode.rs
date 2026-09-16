use phonopaper_android::decode_image_to_pcm;
use phonopaper_rs::{
    SpectrogramVec,
    render::{RenderOptions, spectrogram_to_image},
};

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
    let mut png = Vec::new();
    image
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("write PNG test fixture");

    let pcm = decode_image_to_pcm(&png).expect("decode generated image");
    assert!(!pcm.is_empty());
    assert!(pcm.iter().any(|&sample| sample != 0));
}

#[test]
fn reject_invalid_image_bytes() {
    let err = decode_image_to_pcm(b"not an image").expect_err("invalid input should fail");
    assert!(!err.is_empty());
}
