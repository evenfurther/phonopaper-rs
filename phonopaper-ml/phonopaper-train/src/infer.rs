//! Running a trained detector on arbitrary image files.

use std::path::Path;

use burn::prelude::*;
use image::imageops::FilterType;
use serde::Serialize;

use crate::model::Detector;

/// Result of [`infer_file`], serialisable as JSON.
#[derive(Debug, Clone, Serialize)]
pub struct FileDetection {
    /// Path of the analysed image.
    pub file: String,
    /// Original image dimensions in pixels.
    pub width: u32,
    /// Original image dimensions in pixels.
    pub height: u32,
    /// Probability that a pattern is present.
    pub probability: f32,
    /// `true` when `probability ≥ threshold`.
    pub present: bool,
    /// Corners `[TL, TR, BR, BL]` as `[x, y]` in **original image pixels**.
    pub corners: [[f32; 2]; 4],
}

/// Prepare an image for the network: grayscale, stretched to `size × size`.
///
/// The aspect ratio is **not** preserved (the training data covers arbitrary
/// shear and anisotropic scaling); the predicted normalised corners are
/// mapped back by multiplying with the original width and height.
#[must_use]
pub fn prepare_image(img: &image::DynamicImage, size: u32) -> Vec<u8> {
    img.resize_exact(size, size, FilterType::Triangle)
        .into_luma8()
        .into_raw()
}

/// Run the detector on an image file.
///
/// # Errors
///
/// Returns a message when the image cannot be opened.
pub fn infer_file<B: Backend>(
    model: &Detector<B>,
    path: &Path,
    threshold: f32,
    device: &B::Device,
) -> Result<FileDetection, String> {
    let img = image::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let size = u32::try_from(model.input_size()).map_err(|e| e.to_string())?;
    let pixels = prepare_image(&img, size);
    let det = model.detect(&pixels, device);
    #[expect(
        clippy::cast_precision_loss,
        reason = "image dimensions are far below 2^24"
    )]
    let corners = det.corners_in_pixels(img.width() as f32, img.height() as f32);
    Ok(FileDetection {
        file: path.display().to_string(),
        width: img.width(),
        height: img.height(),
        probability: det.probability,
        present: det.probability >= threshold,
        corners,
    })
}
