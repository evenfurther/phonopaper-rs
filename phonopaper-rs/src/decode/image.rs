//! Image → [`Spectrogram`] decode and column-amplitude extraction.

use image::{DynamicImage, GenericImageView};

use crate::error::{PhonoPaperError, Result};
use crate::format::TOTAL_BINS;
use crate::spectrogram::{Spectrogram, SpectrogramVec};

use super::markers::{DataBounds, detect_markers, pixel_luma};

// ─── Pixel ↔ Spectrogram free functions ──────────────────────────────────────

/// Fill a spectrogram from a raw pixel buffer (grayscale, one byte per pixel).
///
/// `pixels[row * width + col]` is the grayscale value of row `row`, column `col`
/// within the data area. Row 0 is the top (highest frequency), row `height - 1`
/// is the bottom (lowest frequency).
///
/// `height` need not equal `TOTAL_BINS`; the function performs nearest-neighbour
/// resampling by mapping each frequency bin to the **centre of its pixel range**.
///
/// ## Row-mapping formula
///
/// The image encoder assigns pixel row `r` to bin `⌊r · TOTAL_BINS / height⌋`.
/// The inverse — given bin `b`, which row to read — is the midpoint of the
/// contiguous run of rows that all map to bin `b`:
///
/// ```text
/// src_row = ⌊(2·b + 1) · height / (2 · TOTAL_BINS)⌋
/// ```
///
/// Using the centre (rather than the floor `⌊b · height / TOTAL_BINS⌋`) removes
/// a systematic one-bin upward error that affects 87.5 % of bins when
/// `height ≠ TOTAL_BINS`.
///
/// Pixel value: `0` (black) → amplitude `1.0`; `255` (white) → amplitude `0.0`.
///
/// # Panics
///
/// Panics if `spec.num_columns() != width` or if `pixels.len() != width * height`.
pub fn fill_spectrogram_from_pixels<S: AsRef<[f32]> + AsMut<[f32]>>(
    spec: &mut Spectrogram<S>,
    pixels: &[u8],
    width: usize,
    height: usize,
) {
    assert_eq!(spec.num_columns(), width, "width must match num_columns");
    assert_eq!(
        pixels.len(),
        width * height,
        "pixels.len() must equal width * height"
    );
    for col in 0..width {
        for bin in 0..TOTAL_BINS {
            let src_row = ((2 * bin + 1) * height / (2 * TOTAL_BINS)).min(height - 1);
            let pixel = pixels[src_row * width + col];
            let amplitude = 1.0 - f32::from(pixel) / 255.0;
            spec.set(col, bin, amplitude);
        }
    }
}

/// Build a [`Spectrogram`] from a raw pixel buffer (grayscale, one byte per pixel).
///
/// This is a convenience wrapper around [`fill_spectrogram_from_pixels`] that
/// allocates the backing buffer.  Prefer [`fill_spectrogram_from_pixels`] when
/// an allocator is not available.
///
/// Pixel value: `0` (black) → amplitude `1.0`; `255` (white) → amplitude `0.0`.
#[must_use]
pub fn spectrogram_from_pixels(pixels: &[u8], width: usize, height: usize) -> SpectrogramVec {
    let mut spec = SpectrogramVec::new(width);
    fill_spectrogram_from_pixels(&mut spec, pixels, width, height);
    spec
}

// ─── Image → Spectrogram ─────────────────────────────────────────────────────

/// Extract the audio data from a `PhonoPaper` image as a [`Spectrogram`].
///
/// 1. Calls [`detect_markers`] to find the data area bounds.
/// 2. Crops the image to the data area.
/// 3. Converts each pixel's brightness to an amplitude value.
///
/// If `bounds` is `None`, marker detection is performed automatically.
///
/// # Errors
///
/// Returns a [`PhonoPaperError`] if marker detection fails or the data area
/// has zero size.
pub fn image_to_spectrogram(
    image: &DynamicImage,
    bounds: Option<DataBounds>,
) -> Result<SpectrogramVec> {
    let bounds = match bounds {
        Some(b) => b,
        None => detect_markers(image)?,
    };

    let (img_width, _) = image.dimensions();
    let data_height = bounds.height() as usize;
    let data_width = img_width as usize;

    if data_height == 0 || data_width == 0 {
        return Err(PhonoPaperError::InvalidFormat(
            "Data area has zero size after marker detection.".to_string(),
        ));
    }

    // Extract the grayscale pixels for the data area.
    let mut pixels = vec![0u8; data_width * data_height];
    for row in 0..data_height {
        // data_height came from bounds.height() which is u32, so row fits in u32.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "row iterates 0..data_height, and data_height is derived from a u32"
        )]
        let img_row = bounds.data_top + row as u32;
        for col in 0..data_width {
            // data_width came from img_width which is u32, so col fits in u32.
            #[expect(
                clippy::cast_possible_truncation,
                reason = "col iterates 0..data_width, and data_width is derived from a u32"
            )]
            let luma = pixel_luma(image.get_pixel(col as u32, img_row));
            pixels[row * data_width + col] = luma;
        }
    }

    Ok(spectrogram_from_pixels(&pixels, data_width, data_height))
}

// ─── Column amplitude readers ─────────────────────────────────────────────────

/// Read the amplitude values for a single vertical pixel column of a
/// `PhonoPaper` image into a caller-supplied buffer, **without allocating**.
///
/// This is the allocation-free variant of [`column_amplitudes_from_image`],
/// intended for **real-time / sliding-camera** playback where avoiding per-call
/// heap allocation matters.  The API is otherwise identical.
///
/// `out` must be exactly `TOTAL_BINS` elements long; the amplitudes are written
/// directly into it.  Index 0 is the highest-frequency bin; index
/// `TOTAL_BINS - 1` is the lowest.  Each value is in `[0.0, 1.0]`:
/// `0.0` = silence (white pixel), `1.0` = maximum amplitude (black pixel).
///
/// `col_x` is the **image-wide** x-coordinate (0 = left edge of the full image).
///
/// # Errors
///
/// Returns [`PhonoPaperError::InvalidFormat`] if `col_x` is out of range or the
/// data area has zero height, and [`PhonoPaperError::MarkerNotFound`] if
/// `bounds` is `None` and automatic marker detection fails.
pub fn column_amplitudes_from_image_into(
    image: &DynamicImage,
    bounds: Option<DataBounds>,
    col_x: u32,
    out: &mut [f32; TOTAL_BINS],
) -> Result<()> {
    let bounds = match bounds {
        Some(b) => b,
        None => detect_markers(image)?,
    };

    let (img_width, _) = image.dimensions();
    if col_x >= img_width {
        return Err(PhonoPaperError::InvalidFormat(
            "col_x is out of range for image width".to_string(),
        ));
    }

    let data_height = bounds.height() as usize;
    if data_height == 0 {
        return Err(PhonoPaperError::InvalidFormat(
            "Data area has zero height after marker detection.".to_string(),
        ));
    }

    // For each bin, sample the pixel at the centre row of that bin's range
    // directly from the image — no intermediate pixel-column buffer needed.
    for (bin, slot) in out.iter_mut().enumerate() {
        let src_row = ((2 * bin + 1) * data_height / (2 * TOTAL_BINS)).min(data_height - 1);
        // data_height is derived from a u32, so the cast is safe.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "src_row < data_height ≤ bounds.height() which is u32; fits in u32"
        )]
        let img_row = bounds.data_top + src_row as u32;
        let luma = pixel_luma(image.get_pixel(col_x, img_row));
        *slot = 1.0 - f32::from(luma) / 255.0;
    }

    Ok(())
}

/// Read the amplitude values for a single vertical pixel column of a
/// `PhonoPaper` image.
///
/// This is the building block for **real-time / sliding-camera** playback: the
/// caller loads the image once, calls [`detect_markers`] once to get the
/// [`DataBounds`], then calls this function in a tight loop as the camera
/// (or playhead) moves across the image, feeding each result to
/// [`super::synth::Synthesizer::synthesize_column`].
///
/// For performance-sensitive real-time use prefer
/// [`column_amplitudes_from_image_into`], which writes into a caller-supplied
/// `[f32; TOTAL_BINS]` array and performs no heap allocation.
///
/// `col_x` is the **image-wide** x-coordinate of the column to read (0 = left
/// edge of the full image, including any margin).  It must be less than the
/// image width.
///
/// Returns a `Vec<f32>` of length [`TOTAL_BINS`] where index 0 is the highest
/// frequency bin and index `TOTAL_BINS - 1` is the lowest.  Each value is in
/// `[0.0, 1.0]`: `0.0` = silence (white pixel), `1.0` = maximum amplitude
/// (black pixel).
///
/// # Errors
///
/// Returns [`PhonoPaperError::InvalidFormat`] if the column index is out of
/// range or the data area derived from `bounds` has zero height, and
/// [`PhonoPaperError::MarkerNotFound`] if `bounds` is `None` and automatic
/// marker detection fails.
pub fn column_amplitudes_from_image(
    image: &DynamicImage,
    bounds: Option<DataBounds>,
    col_x: u32,
) -> Result<Vec<f32>> {
    let mut out = [0.0_f32; TOTAL_BINS];
    column_amplitudes_from_image_into(image, bounds, col_x, &mut out)?;
    Ok(out.to_vec())
}
