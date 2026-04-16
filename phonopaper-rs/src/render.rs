//! Rendering helpers for converting a [`Spectrogram`] into a `PhonoPaper` pixel image.
//!
//! The low-level, allocation-free entry point is [`spectrogram_to_image_buf`],
//! which writes into a caller-supplied grayscale buffer.  The higher-level
//! [`spectrogram_to_image`] convenience function allocates and returns an
//! [`image::RgbImage`] directly.

use image::{ImageBuffer, RgbImage};

use crate::format::{OCTAVES, TOTAL_BINS};
use crate::spectrogram::Spectrogram;

// ─── Options ─────────────────────────────────────────────────────────────────

/// Options controlling the spectrogram → image rendering step.
#[derive(Debug, Clone, Copy)]
pub struct RenderOptions {
    /// Height (in pixels) of the data area per octave.
    ///
    /// The total data-area height will be `px_per_octave * OCTAVES`.
    /// Defaults to `90`.
    pub px_per_octave: u32,
    /// Height (in pixels) of each thin stripe in the marker bands.
    ///
    /// Defaults to `9`.
    pub thin_stripe: u32,
    /// Height (in pixels) of the thick stripe in the marker bands.
    ///
    /// Should be ≥ 3× `thin_stripe`. Defaults to `39`.
    pub thick_stripe: u32,
    /// Height (in pixels) of the white gap between marker stripes.
    ///
    /// Defaults to `10`.
    pub marker_gap: u32,
    /// Height (in pixels) of the white margin above/below the marker bands.
    ///
    /// Defaults to `88`.
    pub margin: u32,
    /// Whether to draw light-gray octave separator lines. Defaults to `false`.
    ///
    /// > ⚠️ **Warning:** When enabled, octave separator lines are drawn at
    /// > luminance 200/255, which decoders interpret as amplitude ≈ 0.22.
    /// > This produces a constant audible hum at each of the 7 octave-boundary
    /// > frequencies during playback.  The reference PhonoPaper application
    /// > does not draw these lines.  Enable only for visual inspection of the
    /// > image, not for images intended to be decoded as audio.
    pub draw_octave_lines: bool,
    /// Gamma correction exponent applied to amplitudes before writing pixels.
    ///
    /// The encoding formula is `pixel = round((1 − amplitude^gamma) × 255)`.
    ///
    /// - `gamma = 1.0` (default): linear — amplitude maps directly to pixel
    ///   darkness.
    /// - `gamma < 1.0` (e.g. `0.5`): expands quiet amplitudes — a bin at 0.5
    ///   encodes as `0.5^0.5 ≈ 0.71`, producing a darker pixel and louder
    ///   playback.  Use this to make quiet content more visible in the printed
    ///   image.
    /// - `gamma > 1.0` (e.g. `2.0`): compresses mid-level amplitudes toward
    ///   silence — a bin at 0.5 encodes as `0.5^2.0 = 0.25`, producing a
    ///   lighter pixel and quieter playback.
    ///
    /// To decode an image encoded with a non-unity gamma, set
    /// [`crate::decode::SynthesisOptions::decode_gamma`] to the same value.
    pub gamma: f32,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            px_per_octave: 90,
            thin_stripe: 9,
            thick_stripe: 39,
            marker_gap: 10,
            margin: 88,
            draw_octave_lines: false,
            gamma: 1.0,
        }
    }
}

impl RenderOptions {
    /// Height in pixels of one marker band (top or bottom), including its
    /// outer margin.
    ///
    /// The band layout (top-to-bottom) is:
    /// `margin | thin | gap | thin | gap | thick | gap | thin`
    #[must_use]
    pub fn marker_band_height(&self) -> u32 {
        self.margin
            + self.thin_stripe
            + self.marker_gap
            + self.thin_stripe
            + self.marker_gap
            + self.thick_stripe
            + self.marker_gap
            + self.thin_stripe
    }

    /// Total image height in pixels for a spectrogram rendered with these
    /// options.
    #[must_use]
    pub fn image_height(&self) -> u32 {
        // OCTAVES = 8, safely fits in u32.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "OCTAVES is the compile-time constant 8, which fits trivially in u32"
        )]
        let data_height = self.px_per_octave * OCTAVES as u32;
        2 * self.marker_band_height() + data_height
    }
}

// ─── Marker band ─────────────────────────────────────────────────────────────

/// The 8 stripes of a `PhonoPaper` marker band, as `(luma, height_px)` pairs.
///
/// Pattern (top of image working downward):
/// `margin | thin | gap | thin | gap | thick | gap | thin`
type MarkerRows = [(u8, u32); 8];

fn marker_rows(opts: &RenderOptions) -> MarkerRows {
    [
        (255u8, opts.margin),     // white margin
        (0u8, opts.thin_stripe),  // thin black stripe
        (255u8, opts.marker_gap), // white gap
        (0u8, opts.thin_stripe),  // thin black stripe
        (255u8, opts.marker_gap), // white gap
        (0u8, opts.thick_stripe), // THICK black stripe
        (255u8, opts.marker_gap), // white gap
        (0u8, opts.thin_stripe),  // thin black stripe
    ]
}

// ─── Amplitude → luma ────────────────────────────────────────────────────────

/// Convert an amplitude in `[0, 1]` to a grayscale pixel value, applying an
/// optional gamma correction.
///
/// `amplitude = 0.0` → luma `255` (white, silence).
/// `amplitude = 1.0` → luma `0` (black, maximum loudness).
fn amplitude_to_luma(amplitude: f32, gamma: f32) -> u8 {
    // `powf(1.0)` is a no-op in IEEE 754, so no fast-path is needed here.
    // Removing the branch avoids a fragile float-equality check that would
    // silently degrade accuracy for computed gamma values that are
    // mathematically 1.0 but not exactly representable.
    let a = amplitude.powf(gamma);
    // `a` is in [0.0, 1.0] because `set` clamps on write; `(1-a)*255 ∈ [0,255]`.
    let luma_f = ((1.0 - a) * 255.0).round();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a is in [0.0, 1.0], so (1-a)*255 is in [0.0, 255.0] before the cast"
    )]
    let luma = luma_f as u8;
    luma
}

// ─── spectrogram_to_image_buf ─────────────────────────────────────────────────

/// Render a `PhonoPaper` image into a caller-supplied grayscale pixel buffer.
///
/// The output buffer must have length `image_width * opts.image_height()` and
/// `image_width` must equal `spec.num_columns()`.  Use [`image_buf_size`] to
/// compute the required length.
///
/// The buffer is filled in **row-major order**: `out[row * width + col]` is the
/// pixel at `(col, row)`.  Row 0 is the top of the image (the top marker band's
/// outer white margin).  Pixel value `255` = white (silence), `0` = black
/// (maximum amplitude).
///
/// The rendered image includes:
/// - Top marker band (margin → thin → gap → thin → gap → thick → gap → thin, top-to-bottom)
/// - Data area (one image column per spectrogram time step)
/// - Bottom marker band (top band mirrored: thin → gap → thick → gap → thin → gap → thin → margin, top-to-bottom)
/// - Optional light-gray octave separator lines (see [`RenderOptions::draw_octave_lines`])
///
/// # Panics
///
/// Panics if `out.len() != spec.num_columns() * opts.image_height() as usize`.
pub fn spectrogram_to_image_buf<S: AsRef<[f32]>>(
    spec: &Spectrogram<S>,
    opts: &RenderOptions,
    out: &mut [u8],
) {
    let width = spec.num_columns();
    // OCTAVES = 8, safely fits in u32.
    let data_height = opts.px_per_octave as usize * OCTAVES;
    let img_height = opts.image_height() as usize;

    assert_eq!(
        out.len(),
        width * img_height,
        "out.len() must equal spec.num_columns() * opts.image_height() ({} * {} = {}), got {}",
        width,
        img_height,
        width * img_height,
        out.len(),
    );

    // Fill everything white.
    out.fill(255u8);

    // Helper: fill a horizontal run of rows with a constant luma.
    let fill_band = |buf: &mut [u8], y_start: usize, height: usize, luma: u8| {
        let start = y_start * width;
        let end = (y_start + height) * width;
        buf[start..end].fill(luma);
    };

    // Draw top marker band.
    let top_rows = marker_rows(opts);
    let mut y = 0usize;
    for (luma, h) in top_rows {
        let h = h as usize;
        fill_band(out, y, h, luma);
        y += h;
    }
    let data_y_start = y;

    // Draw bottom marker band — mirrored so the thin stripe is adjacent to the
    // data area (matching the spec: the innermost stripe is always thin).
    let mut by = data_y_start + data_height;
    for (luma, h) in top_rows.iter().rev() {
        let h = *h as usize;
        fill_band(out, by, h, *luma);
        by += h;
    }

    // Render the data area.
    //
    // Bin 0 → top row (highest frequency); bin TOTAL_BINS-1 → bottom row.
    //
    // We use the same centre-of-range formula as the decoder
    // (`row = floor((2·bin + 1) · height / (2 · TOTAL_BINS))`), so that
    // encode → decode round-trips have no systematic per-bin row offset.
    for col in 0..width {
        for bin in 0..TOTAL_BINS {
            let row = ((2 * bin + 1) * data_height / (2 * TOTAL_BINS)).min(data_height - 1);
            let amplitude = spec.get(col, bin);
            let luma = amplitude_to_luma(amplitude, opts.gamma);
            out[(data_y_start + row) * width + col] = luma;
        }
    }

    // Draw optional octave separator lines (light grey, luma = 200).
    if opts.draw_octave_lines {
        for octave in 1..OCTAVES {
            let sep_row = data_y_start + octave * opts.px_per_octave as usize;
            let start = sep_row * width;
            out[start..start + width].fill(200u8);
        }
    }
}

/// Compute the output buffer size (in bytes) required by [`spectrogram_to_image_buf`]
/// for a spectrogram with `num_columns` time steps and the given render options.
///
/// The returned value equals `num_columns * opts.image_height()`.
#[must_use]
pub fn image_buf_size(num_columns: usize, opts: &RenderOptions) -> usize {
    num_columns * opts.image_height() as usize
}

// ─── spectrogram_to_image ─────────────────────────────────────────────────────

/// Render a [`Spectrogram`] into a `PhonoPaper` image, returning an [`image::RgbImage`].
///
/// The rendered image includes top and bottom marker bands, the data area (one
/// column per spectrogram time step), and optional octave separator lines
/// (see [`RenderOptions::draw_octave_lines`]).
///
/// This is a convenience wrapper around [`spectrogram_to_image_buf`] that
/// allocates the grayscale pixel buffer and converts it into an
/// [`image::RgbImage`] suitable for saving as PNG or JPEG.
///
/// For a zero-allocation alternative when the output buffer is managed by the
/// caller, use [`spectrogram_to_image_buf`] directly.
#[must_use]
pub fn spectrogram_to_image<S: AsRef<[f32]>>(
    spec: &Spectrogram<S>,
    opts: &RenderOptions,
) -> RgbImage {
    let width = spec.num_columns();
    let height = opts.image_height() as usize;
    let mut gray_buf = vec![0u8; width * height];
    spectrogram_to_image_buf(spec, opts, &mut gray_buf);

    // Convert the flat grayscale buffer to an RgbImage.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "width and height are pixel dimensions derived from u32 values; they fit in u32"
    )]
    ImageBuffer::from_fn(width as u32, height as u32, |x, y| {
        let luma = gray_buf[y as usize * width + x as usize];
        image::Rgb([luma, luma, luma])
    })
}
