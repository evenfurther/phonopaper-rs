//! Vector output for `PhonoPaper` images: SVG and PDF.
//!
//! This module provides two functions that render a [`Spectrogram`] into a
//! vector format suitable for high-quality printing:
//!
//! - [`spectrogram_to_svg`] — produces a self-contained SVG string.  The
//!   full `PhonoPaper` image (marker bands + spectrogram data) is embedded as
//!   a base64-encoded PNG `<image>` element covering the entire canvas, with
//!   crisp vector `<rect>` elements drawn on top for the marker bands.
//! - [`spectrogram_to_pdf`] — produces a PDF byte vector.  The full
//!   `PhonoPaper` image is embedded as a deflate-compressed grayscale image
//!   `XObject` (`FlateDecode` filter); the marker bands are additionally
//!   painted as sharp vector rectangles on top.
//!
//! Both functions accept a [`RenderOptions`] that controls marker band geometry
//! and gamma correction, exactly as [`crate::render::spectrogram_to_image`]
//! does.  The marker bands appear at infinite resolution regardless of zoom
//! because they are also drawn as vector elements.
//!
//! ## Decoding
//!
//! SVG and PDF files produced by these functions can be decoded back to audio
//! because the full raster image is embedded verbatim.  Use
//! [`image_from_svg`] or [`image_from_pdf`] to extract the embedded image, then
//! pass it to [`crate::decode::image_to_spectrogram`] as usual.
//!
//! # Example
//!
//! ```
//! use phonopaper_rs::render::RenderOptions;
//! use phonopaper_rs::SpectrogramVec;
//! use phonopaper_rs::vector::{spectrogram_to_svg, spectrogram_to_pdf};
//!
//! // A 2-column silent spectrogram.
//! let spec = SpectrogramVec::new(2);
//! let opts = RenderOptions::default();
//!
//! let svg: String = spectrogram_to_svg(&spec, &opts);
//! assert!(svg.starts_with("<svg "));
//!
//! let pdf: Vec<u8> = spectrogram_to_pdf(&spec, &opts, Default::default());
//! assert!(pdf.starts_with(b"%PDF-"));
//! ```

use std::fmt::Write as _;

use image::DynamicImage;
use miniz_oxide::deflate::compress_to_vec_zlib;
use miniz_oxide::inflate::decompress_to_vec_zlib;
use pdf_writer::{Content, Filter, Finish, Name, Pdf, Rect, Ref};

use crate::error::{PhonoPaperError, Result};
use crate::format::OCTAVES;
use crate::render::{RenderOptions, spectrogram_to_image_buf};
use crate::spectrogram::Spectrogram;

// ─── PDF page layout ──────────────────────────────────────────────────────────

/// Controls the page size and placement of the `PhonoPaper` image in a PDF
/// produced by [`spectrogram_to_pdf`].
///
/// ## Which variant to use
///
/// | Goal | Variant |
/// |------|---------|
/// | Software pipeline (further processing, embedding) | [`PdfPageLayout::PixelPerfect`] |
/// | Human-readable print layout on standard paper | [`PdfPageLayout::FitToPage`] |
#[derive(Debug, Clone, Copy, Default)]
pub enum PdfPageLayout {
    /// Set the PDF page dimensions to the image's pixel dimensions (1 px = 1 pt).
    ///
    /// The image fills the page exactly with no margins.  At 72 dpi this
    /// produces a very large page (e.g. ≈ 49 × 38 cm for a 1400-column image);
    /// print drivers or PDF viewers are expected to scale it to physical paper.
    ///
    /// This is the default and is best for software-to-software pipelines where
    /// preserving pixel-perfect fidelity matters more than print layout.
    #[default]
    PixelPerfect,

    /// Centre the image on a fixed-size page with equal margins on all four sides.
    ///
    /// The image is scaled uniformly (aspect-ratio preserved) to fit within the
    /// page minus the margins.  If the image aspect ratio differs from the page
    /// aspect ratio, the image is centred in the remaining space on both axes.
    ///
    /// ## Common page sizes in points (1 pt = 1/72 inch)
    ///
    /// | Paper | Portrait (w × h) | Landscape (w × h) |
    /// |-------|-----------------|-------------------|
    /// | A4    | 595 × 842       | 842 × 595         |
    /// | A3    | 842 × 1190      | 1190 × 842        |
    /// | Letter| 612 × 792       | 792 × 612         |
    ///
    /// `margin_pt` is the minimum gap on **each** side of the image; the actual
    /// gap on the shorter axis will be larger (centring).
    FitToPage {
        /// PDF page width in points.
        page_width_pt: f32,
        /// PDF page height in points.
        page_height_pt: f32,
        /// Minimum margin on each side in points.  Defaults to `28.3` (10 mm).
        margin_pt: f32,
    },
}

/// Well-known PDF page sizes in points (1 pt = 1/72 inch).
///
/// These can be passed directly to [`PdfPageLayout::FitToPage`]:
///
/// ```
/// use phonopaper_rs::vector::PdfPageLayout;
/// use phonopaper_rs::vector::page_size;
///
/// let layout = PdfPageLayout::FitToPage {
///     page_width_pt:  page_size::A4_LANDSCAPE.0,
///     page_height_pt: page_size::A4_LANDSCAPE.1,
///     margin_pt: 28.35, // 10 mm
/// };
/// ```
pub mod page_size {
    /// A4 portrait (210 × 297 mm).
    pub const A4_PORTRAIT: (f32, f32) = (595.28, 841.89);
    /// A4 landscape (297 × 210 mm).
    pub const A4_LANDSCAPE: (f32, f32) = (841.89, 595.28);
    /// A3 portrait (297 × 420 mm).
    pub const A3_PORTRAIT: (f32, f32) = (841.89, 1190.55);
    /// A3 landscape (420 × 297 mm).
    pub const A3_LANDSCAPE: (f32, f32) = (1190.55, 841.89);
    /// US Letter portrait (8.5 × 11 in).
    pub const LETTER_PORTRAIT: (f32, f32) = (612.0, 792.0);
    /// US Letter landscape (11 × 8.5 in).
    pub const LETTER_LANDSCAPE: (f32, f32) = (792.0, 612.0);
}

/// Encode `data` as a base64 string (standard alphabet, no line breaks).
fn base64_encode(data: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(data)
}

/// Decode a base64 string (standard alphabet, handles padding).
///
/// # Errors
///
/// Returns [`PhonoPaperError::InvalidFormat`] if the input is not valid
/// standard-alphabet base64.
fn base64_decode(s: &str) -> Result<Vec<u8>> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD
        .decode(s)
        .map_err(|e| PhonoPaperError::InvalidFormat(format!("invalid base64: {e}")))
}

/// Render the full `PhonoPaper` image into a grayscale byte buffer and return
/// it together with the image dimensions.
pub(crate) fn render_gray<S: AsRef<[f32]>>(
    spec: &Spectrogram<S>,
    opts: &RenderOptions,
) -> (Vec<u8>, usize, usize) {
    let width = spec.num_columns();
    let height = opts.image_height() as usize;
    let mut gray = vec![0u8; width * height];
    if width > 0 {
        spectrogram_to_image_buf(spec, opts, &mut gray);
    }
    (gray, width, height)
}

/// Encode a flat grayscale byte buffer as PNG bytes.
///
/// `pixels[row * width + col]` is the pixel at `(col, row)`.
///
/// # Panics
///
/// Panics if `pixels.len() != width * height`.
fn gray_to_png(pixels: &[u8], width: usize, height: usize) -> Vec<u8> {
    use image::{GrayImage, ImageEncoder, codecs::png::PngEncoder};
    assert_eq!(pixels.len(), width * height);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "width and height are pixel dimensions from u32 render opts; they fit in u32"
    )]
    let img = GrayImage::from_raw(width as u32, height as u32, pixels.to_vec())
        .expect("buffer size matches dimensions");
    let mut png_bytes: Vec<u8> = Vec::new();
    PngEncoder::new(&mut png_bytes)
        .write_image(
            img.as_raw(),
            img.width(),
            img.height(),
            image::ColorType::L8.into(),
        )
        .expect("in-memory PNG encoding cannot fail");
    png_bytes
}

/// Marker stripe layout for a `PhonoPaper` image: `(is_black, height_px)` pairs.
///
/// Returns the 8-element array describing the top marker band from the outer
/// edge inward: `(false/true = white/black, height in pixels)`.
///
/// Pattern (outer → inner, i.e. top-to-bottom for the top band):
/// `margin | thin | gap | thin | gap | thick | gap | thin`
///
/// The bottom marker band is this array **reversed** (inner → outer, so
/// that the thin stripe is still innermost on both sides of the data area).
///
/// This function is the single source of truth for the marker stripe layout
/// shared between the SVG/PDF renderers and the CLI blank-template generator.
#[must_use]
pub fn marker_stripe_layout(opts: &RenderOptions) -> [(bool, u32); 8] {
    [
        (false, opts.margin),
        (true, opts.thin_stripe),
        (false, opts.marker_gap),
        (true, opts.thin_stripe),
        (false, opts.marker_gap),
        (true, opts.thick_stripe),
        (false, opts.marker_gap),
        (true, opts.thin_stripe),
    ]
}

// ─── spectrogram_to_svg ───────────────────────────────────────────────────────

/// Render a [`Spectrogram`] to a self-contained SVG string.
///
/// The SVG dimensions are in pixels, matching the raster image produced by
/// [`crate::render::spectrogram_to_image`]:
/// - Width  = `spec.num_columns()`
/// - Height = `opts.image_height()`
///
/// The **full** `PhonoPaper` raster (including marker bands and data area) is
/// embedded as a base64-encoded PNG `<image>` element covering the whole
/// canvas.  The marker bands are additionally drawn as crisp vector `<rect>`
/// elements on top for infinite sharpness at any zoom level.  Optional octave
/// separator lines from [`RenderOptions::draw_octave_lines`] are included as
/// `<line>` elements.
///
/// Because the full raster is embedded, the SVG can be round-tripped: use
/// [`image_from_svg`] to extract the embedded image and decode it with
/// [`crate::decode::image_to_spectrogram`].
///
/// The returned string starts with `<svg ` and is valid SVG 1.1.
///
/// # Panics
///
/// Panics if `spec.num_columns() > 0` and the internal pixel buffer cannot be
/// allocated (out of memory).
#[must_use]
pub fn spectrogram_to_svg<S: AsRef<[f32]>>(spec: &Spectrogram<S>, opts: &RenderOptions) -> String {
    let width = spec.num_columns();
    let img_h = opts.image_height() as usize;
    let band_h = opts.marker_band_height() as usize;
    let data_h = img_h - 2 * band_h;

    // Render the full raster image and embed it as a PNG.
    // Only attempt PNG encoding when there is at least one column to render;
    // `gray_to_png` panics on zero-width images.
    let full_b64 = if width > 0 {
        let (gray_full, _, _) = render_gray(spec, opts);
        let full_png = gray_to_png(&gray_full, width, img_h);
        base64_encode(&full_png)
    } else {
        String::new()
    };

    let stripes = marker_stripe_layout(opts);

    let mut svg = String::new();

    // SVG root element.
    let _ = writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" \
             xmlns:xlink=\"http://www.w3.org/1999/xlink\" \
             width=\"{width}\" height=\"{img_h}\" \
             viewBox=\"0 0 {width} {img_h}\">"
    );

    // Embed the full raster as a background PNG image.
    if width > 0 && img_h > 0 {
        let _ = writeln!(
            svg,
            "<image x=\"0\" y=\"0\" width=\"{width}\" height=\"{img_h}\" \
                 preserveAspectRatio=\"none\" \
                 href=\"data:image/png;base64,{full_b64}\"/>"
        );
    }

    // Vector marker bands drawn on top for crisp printing.
    // Top marker band (outer edge → data area, top-to-bottom).
    {
        let mut y = 0usize;
        for (is_black, h) in stripes {
            let h = h as usize;
            if is_black {
                let _ = writeln!(
                    svg,
                    "<rect x=\"0\" y=\"{y}\" width=\"{width}\" height=\"{h}\" fill=\"black\"/>"
                );
            }
            y += h;
        }
    }

    // Bottom marker band (mirror: innermost thin stripe closest to data).
    {
        let mut y = band_h + data_h;
        for (is_black, h) in stripes.iter().rev() {
            let h = *h as usize;
            if *is_black {
                let _ = writeln!(
                    svg,
                    "<rect x=\"0\" y=\"{y}\" width=\"{width}\" height=\"{h}\" fill=\"black\"/>"
                );
            }
            y += h;
        }
    }

    // Optional octave separator lines.
    if opts.draw_octave_lines {
        let px_per_octave = opts.px_per_octave as usize;
        for octave in 1..OCTAVES {
            let y = band_h + octave * px_per_octave;
            let _ = writeln!(
                svg,
                "<line x1=\"0\" y1=\"{y}\" x2=\"{width}\" y2=\"{y}\" \
                     stroke=\"#c8c8c8\" stroke-width=\"1\"/>"
            );
        }
    }

    svg.push_str("</svg>\n");
    svg
}

// ─── image_from_svg ───────────────────────────────────────────────────────────

/// Extract the embedded raster image from a `PhonoPaper` SVG string.
///
/// This function finds the first `<image ... href="data:image/png;base64,..."/>`
/// element, base64-decodes the PNG data, and returns it as a [`DynamicImage`].
///
/// # Errors
///
/// Returns [`PhonoPaperError::InvalidFormat`] if:
/// - the SVG does not contain a `data:image/png;base64,` element,
/// - the base64 data is malformed, or
/// - the embedded PNG cannot be decoded.
pub fn image_from_svg(svg: &str) -> Result<DynamicImage> {
    // Find the data URI prefix.
    const PREFIX: &str = "data:image/png;base64,";
    let start = svg.find(PREFIX).ok_or_else(|| {
        PhonoPaperError::InvalidFormat(
            "SVG does not contain a data:image/png;base64, image element".to_string(),
        )
    })? + PREFIX.len();

    // The base64 data ends at the first `"` after the prefix.
    let end = svg[start..].find('"').ok_or_else(|| {
        PhonoPaperError::InvalidFormat("Malformed SVG: unterminated base64 data URI".to_string())
    })? + start;

    let png_bytes = base64_decode(&svg[start..end])?;
    image::load_from_memory(&png_bytes).map_err(PhonoPaperError::ImageError)
}

// ─── PDF helpers ─────────────────────────────────────────────────────────────

/// Write a deflate-compressed grayscale image `XObject` into `pdf`.
///
/// `compressed` must already be zlib-framed (as produced by
/// [`compress_to_vec_zlib`]).  `width` and `height` are the pixel dimensions.
///
/// # Panics
///
/// Panics if `width == 0` or `height == 0` (caller must guard against this).
fn write_image_xobject(
    pdf: &mut Pdf,
    image_id: Ref,
    compressed: &[u8],
    width: usize,
    height: usize,
) {
    // Pixel dimensions come from u32 render opts; they fit in i32 (≤ ~65535).
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "width and height are pixel dimensions from u32 render opts; they never exceed i32::MAX in practice"
    )]
    {
        let mut image = pdf.image_xobject(image_id, compressed);
        image.filter(Filter::FlateDecode);
        image.width(width as i32);
        image.height(height as i32);
        image.color_space().device_gray();
        image.bits_per_component(8);
        image.finish();
    }
}

/// Image placement geometry for [`build_content_stream`].
///
/// All values are in PDF points.
struct Placement {
    /// X offset of the image's left edge from the page's left edge.
    x: f32,
    /// Y offset of the image's bottom edge from the page's bottom edge.
    y: f32,
    /// Rendered width of the image on the page.
    width: f32,
    /// Rendered height of the image on the page.
    height: f32,
}

/// Build the PDF content stream for a `PhonoPaper` page.
///
/// Places the image `XObject` at `placement.{x,y}` with size
/// `placement.{width} × placement.{height}` and draws the vector marker bands
/// on top using the same offset.  `band_h` and `data_h` are the pixel heights
/// of one marker band and the data area.  `has_image` controls whether the
/// `XObject` placement command is emitted (it must be `false` for a zero-width
/// or zero-height image).
fn build_content_stream(
    opts: &RenderOptions,
    image_name: Name<'_>,
    placement: &Placement,
    band_h: usize,
    data_h: usize,
    has_image: bool,
) -> Content {
    let mut content = Content::new();

    // Unpack placement geometry into short locals used throughout this function.
    let x_off = placement.x;
    let y_off = placement.y;
    let pt_w = placement.width;
    let pt_h = placement.height;

    // Place the raster image XObject.
    // The transform [a b c d e f] maps the unit square to:
    //   bottom-left  → (e,       f)      = (x_off, y_off)
    //   bottom-right → (e+a,     f)      = (x_off + pt_w, y_off)
    //   top-right    → (e+a, f+d)        = (x_off + pt_w, y_off + pt_h)
    // PDF image row 0 maps to the TOP of the unit square (y=1 before transform),
    // which lands at PDF y = y_off + pt_h — the top of the image box. ✓
    if has_image {
        content.save_state();
        content.transform([pt_w, 0.0, 0.0, pt_h, x_off, y_off]);
        content.x_object(image_name);
        content.restore_state();
    }

    // Vector marker bands drawn on top for crisp printing.
    let stripes = marker_stripe_layout(opts);

    // Stripe heights come from u32 opts; exact in f32.
    #[expect(
        clippy::cast_precision_loss,
        reason = "marker stripe heights are u32 render option fields; exact in f32 for practical values (≤ ~512 px)"
    )]
    let stripes_f: [(bool, f32); 8] = stripes.map(|(b, h)| (b, h as f32));

    // Helper closure: fill a rect with a gray value.
    // `y` is the PDF y-coordinate of the rectangle's bottom edge.
    let draw_rect = |c: &mut Content, gray: f32, x: f32, y: f32, w: f32, h: f32| {
        c.set_fill_gray(gray);
        c.rect(x, y, w, h);
        c.fill_nonzero();
    };

    // band_h and data_h are from u32 opts; exact in f32.
    #[expect(
        clippy::cast_precision_loss,
        reason = "band_h and data_h are image pixel dimensions from u32 render opts; exact in f32 for practical image sizes"
    )]
    let (band_h_f, data_h_f) = (band_h as f32, data_h as f32);

    // Scale factor: image pixels → PDF points.
    // In PixelPerfect mode (pt_h == image height in px) this is exactly 1.0.
    let scale = pt_h / (band_h_f * 2.0 + data_h_f);

    // Top marker band: iterate stripes top-to-bottom in image coords.
    // Image row 0 → PDF y = y_off + pt_h (top of image box).
    {
        let mut img_y = 0.0_f32;
        for (is_black, h_px) in stripes_f {
            let h_pt = h_px * scale;
            // pdf_y_bottom: PDF y-coordinate of the stripe's lower edge.
            let pdf_y_bottom = y_off + pt_h - img_y * scale - h_pt;
            if is_black {
                draw_rect(&mut content, 0.0, x_off, pdf_y_bottom, pt_w, h_pt);
            }
            img_y += h_px;
        }
    }

    // Bottom marker band: mirror (innermost thin stripe adjacent to data).
    {
        let mut img_y = band_h_f + data_h_f;
        for (is_black, h_px) in stripes_f.iter().rev() {
            let h_pt = h_px * scale;
            let pdf_y_bottom = y_off + pt_h - img_y * scale - h_pt;
            if *is_black {
                draw_rect(&mut content, 0.0, x_off, pdf_y_bottom, pt_w, h_pt);
            }
            img_y += h_px;
        }
    }

    // Optional octave separator lines (light gray, 1 pt stroke).
    if opts.draw_octave_lines {
        content.set_stroke_gray(200.0 / 255.0);
        content.set_line_width(1.0);
        // px_per_octave is a u32; exact in f32.
        #[expect(
            clippy::cast_precision_loss,
            reason = "px_per_octave is a u32 render option field; exact in f32 for practical values (≤ ~512 px)"
        )]
        let px_per_octave_f = opts.px_per_octave as f32;
        for octave in 1..OCTAVES {
            // octave ≤ 7; exact in f32.
            #[expect(
                clippy::cast_precision_loss,
                reason = "octave iterates 1..OCTAVES = 1..8; exact in f32"
            )]
            let img_y = band_h_f + octave as f32 * px_per_octave_f;
            let pdf_y = y_off + pt_h - img_y * scale;
            content.move_to(x_off, pdf_y);
            content.line_to(x_off + pt_w, pdf_y);
            content.stroke();
        }
    }

    content
}

// ─── spectrogram_to_pdf ───────────────────────────────────────────────────────

/// Render a [`Spectrogram`] to a PDF byte vector.
///
/// `layout` controls the page size and placement of the image:
///
/// - [`PdfPageLayout::PixelPerfect`] (default) — the PDF page dimensions equal
///   the image pixel dimensions (1 px = 1 pt).  The image fills the page with
///   no margins.  At 72 dpi this produces a large page (≈ 49 × 38 cm for a
///   1400-column image); print drivers or PDF viewers scale it to paper.
///   Best for software-to-software pipelines.
///
/// - [`PdfPageLayout::FitToPage`] — the image is scaled to fit inside the
///   specified page size minus `margin_pt` on each side, and centred both
///   horizontally and vertically.  Use this for human-readable print output on
///   A4, Letter, or other standard page sizes.
///
/// The **full** `PhonoPaper` raster (including marker bands and data area) is
/// embedded as a deflate-compressed grayscale image `XObject` (`FlateDecode`
/// filter, `DeviceGray` colour space, 8 bits per component).  The marker bands
/// are additionally painted as sharp vector rectangles on top.
///
/// Because the full raster is embedded, the PDF can be round-tripped: use
/// [`image_from_pdf`] to extract the embedded image and decode it with
/// [`crate::decode::image_to_spectrogram`].  (Round-trip extraction is
/// supported regardless of which `layout` variant is used.)
///
/// The returned `Vec<u8>` starts with `%PDF-` and is a valid, self-contained
/// PDF 1.7 document.
///
/// # Panics
///
/// Panics if `spec.num_columns() > 0` and the internal pixel buffer cannot be
/// allocated (out of memory).
#[must_use]
pub fn spectrogram_to_pdf<S: AsRef<[f32]>>(
    spec: &Spectrogram<S>,
    opts: &RenderOptions,
    layout: PdfPageLayout,
) -> Vec<u8> {
    // ── Image pixel dimensions ───────────────────────────────────────────────
    let width = spec.num_columns();
    let img_h = opts.image_height() as usize;
    let band_h = opts.marker_band_height() as usize;
    let data_h = img_h - 2 * band_h;
    let has_image = width > 0 && img_h > 0;

    // width and img_h are derived from u32 render opts; both fit in f32.
    #[expect(
        clippy::cast_precision_loss,
        reason = "width and img_h are image pixel dimensions from u32 render opts; exact in f32 for practical image sizes (≤ 65535)"
    )]
    let (pix_w, pix_h) = (width as f32, img_h as f32);

    // ── Page size and image placement in PDF points ──────────────────────────
    //
    // For PixelPerfect: page = image, no offset, scale = 1 px/pt.
    // For FitToPage: compute a uniform scale that fits the image within the
    // page minus margins, then centre it.
    // ── Page size and image placement ────────────────────────────────────────
    //
    // For PixelPerfect: page = image, no offset, 1 px = 1 pt.
    // For FitToPage: uniform scale to fit within the margin-reduced page,
    // then centre on both axes.
    let (pw, ph, placement) = match layout {
        PdfPageLayout::PixelPerfect => (
            pix_w,
            pix_h,
            Placement {
                x: 0.0,
                y: 0.0,
                width: pix_w,
                height: pix_h,
            },
        ),
        PdfPageLayout::FitToPage {
            page_width_pt,
            page_height_pt,
            margin_pt,
        } => {
            let avail_w = (page_width_pt - 2.0 * margin_pt).max(0.0);
            let avail_h = (page_height_pt - 2.0 * margin_pt).max(0.0);

            // Uniform scale — limited by whichever axis is tighter.
            // Guard against zero image dimensions to avoid division by zero.
            let scale = if pix_w > 0.0 && pix_h > 0.0 {
                (avail_w / pix_w).min(avail_h / pix_h)
            } else {
                1.0
            };

            let placed_w = pix_w * scale;
            let placed_h = pix_h * scale;

            // Centre within the page.
            let x_off = (page_width_pt - placed_w) / 2.0;
            let y_off = (page_height_pt - placed_h) / 2.0;

            (
                page_width_pt,
                page_height_pt,
                Placement {
                    x: x_off,
                    y: y_off,
                    width: placed_w,
                    height: placed_h,
                },
            )
        }
    };

    // ── Render the full image and compress ───────────────────────────────────
    let (gray_full, _, _) = render_gray(spec, opts);
    let compressed = compress_to_vec_zlib(&gray_full, 6);

    // ── PDF object IDs ───────────────────────────────────────────────────────
    let catalog_id = Ref::new(1);
    let page_tree_id = Ref::new(2);
    let page_id = Ref::new(3);
    let image_id = Ref::new(4);
    let content_id = Ref::new(5);
    let image_name = Name(b"Im1");

    let mut pdf = Pdf::new();

    // Catalog + page tree.
    pdf.catalog(catalog_id).pages(page_tree_id);
    pdf.pages(page_tree_id).kids([page_id]).count(1);

    // Page dictionary.
    {
        let mut page = pdf.page(page_id);
        page.media_box(Rect::new(0.0, 0.0, pw, ph));
        page.parent(page_tree_id);
        page.contents(content_id);
        if has_image {
            page.resources().x_objects().pair(image_name, image_id);
        }
        page.finish();
    }

    // Grayscale image XObject.
    if has_image {
        write_image_xobject(&mut pdf, image_id, &compressed, width, img_h);
    }

    // Content stream: placed image + vector marker bands.
    let content = build_content_stream(opts, image_name, &placement, band_h, data_h, has_image);
    pdf.stream(content_id, &content.finish());

    pdf.finish()
}

// ─── image_from_pdf ───────────────────────────────────────────────────────────

/// Extract the embedded raster image from a `PhonoPaper` PDF produced by
/// [`spectrogram_to_pdf`].
///
/// **Round-trip only:** this function is designed to work exclusively with PDFs
/// produced by [`spectrogram_to_pdf`] in this library.  It uses a simple
/// byte-pattern scan for `/FlateDecode`, `/Width`, and `/Height` in the raw PDF
/// byte stream, which is fragile against:
/// - PDFs with compressed cross-reference streams (PDF 1.5+ `XRef` streams)
/// - Non-standard whitespace or line endings in the image `XObject` dictionary
/// - PDFs produced by any tool other than [`spectrogram_to_pdf`]
///
/// If you pass in an arbitrary PDF and the extraction fails, you will receive
/// an `InvalidFormat("PDF round-trip only: …")` error — not a guarantee of
/// correct extraction.
///
/// This function finds the first `FlateDecode` image `XObject` in the PDF byte
/// stream, decompresses it (zlib), and reconstructs the grayscale image using
/// the width and height recorded in the `XObject` dictionary.
///
/// # Errors
///
/// Returns [`PhonoPaperError::InvalidFormat`] with the prefix
/// `"PDF round-trip only: …"` if:
/// - no `FlateDecode` image `XObject` can be located,
/// - the pixel dimensions cannot be parsed,
/// - the compressed data cannot be decompressed, or
/// - the decompressed length does not match `width × height`.
pub fn image_from_pdf(pdf: &[u8]) -> Result<DynamicImage> {
    use image::GrayImage;
    // Locate things by searching for ASCII byte-pattern landmarks.
    // The PDF dictionary is ASCII text; only the compressed pixel data is binary.

    /// Find the byte offset of the first occurrence of `needle` in `haystack`.
    fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack.windows(needle.len()).position(|w| w == needle)
    }

    // Find `/FlateDecode` — the image XObject dictionary uses it.
    let flat_pos = find_bytes(pdf, b"/FlateDecode").ok_or_else(|| {
        PhonoPaperError::InvalidFormat(
            "PDF round-trip only: no FlateDecode stream found (not a phonopaper-rs PDF?)"
                .to_string(),
        )
    })?;

    // Walk backwards from flat_pos to find the dictionary start `<<`.
    let dict_start = pdf[..flat_pos]
        .windows(2)
        .rposition(|w| w == b"<<")
        .ok_or_else(|| {
            PhonoPaperError::InvalidFormat(
                "PDF round-trip only: cannot locate image XObject dictionary".to_string(),
            )
        })?;

    // Walk forward from flat_pos to find the closing `>>`.
    let dict_end = find_bytes(&pdf[flat_pos..], b">>").ok_or_else(|| {
        PhonoPaperError::InvalidFormat(
            "PDF round-trip only: cannot locate end of image XObject dictionary".to_string(),
        )
    })? + flat_pos;

    // The dictionary slice is pure ASCII — decode it for text parsing.
    let dict = std::str::from_utf8(&pdf[dict_start..=dict_end]).map_err(|_| {
        PhonoPaperError::InvalidFormat(
            "PDF round-trip only: image XObject dictionary contains non-UTF-8 bytes".to_string(),
        )
    })?;

    // Extract `/Width` and `/Height` from the dictionary text.
    let parse_dim = |key: &str| -> Result<usize> {
        let pos = dict.find(key).ok_or_else(|| {
            PhonoPaperError::InvalidFormat(format!(
                "PDF round-trip only: image XObject missing {key}"
            ))
        })? + key.len();
        let rest = dict[pos..].trim_start();
        rest.split_whitespace()
            .next()
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or_else(|| {
                PhonoPaperError::InvalidFormat(format!(
                    "PDF round-trip only: cannot parse {key} value"
                ))
            })
    };

    let img_w = parse_dim("/Width")?;
    let img_h = parse_dim("/Height")?;

    // Find the `stream` keyword after the dictionary.
    let stream_marker = b"stream";
    let stream_offset = find_bytes(&pdf[dict_end..], stream_marker).ok_or_else(|| {
        PhonoPaperError::InvalidFormat(
            "PDF round-trip only: cannot locate stream body in image XObject".to_string(),
        )
    })? + dict_end
        + stream_marker.len();

    // Skip the line-ending after `stream` (LF or CRLF).
    let stream_data_start = if pdf.get(stream_offset) == Some(&b'\r') {
        stream_offset + 2 // CRLF
    } else {
        stream_offset + 1 // LF
    };

    // Find the `endstream` marker to delimit the compressed payload.
    let endstream_marker = b"endstream";
    let stream_data_end =
        find_bytes(&pdf[stream_data_start..], endstream_marker).ok_or_else(|| {
            PhonoPaperError::InvalidFormat(
                "PDF round-trip only: cannot locate endstream marker".to_string(),
            )
        })? + stream_data_start;

    // Strip any trailing whitespace before `endstream`.
    let stream_bytes = pdf[stream_data_start..stream_data_end].trim_ascii_end();

    // Decompress the zlib payload.
    let pixels = decompress_to_vec_zlib(stream_bytes).map_err(|e| {
        PhonoPaperError::InvalidFormat(format!("Failed to decompress PDF image XObject: {e:?}"))
    })?;

    if pixels.len() != img_w * img_h {
        return Err(PhonoPaperError::InvalidFormat(format!(
            "Decompressed PDF image size {} does not match {}×{} = {}",
            pixels.len(),
            img_w,
            img_h,
            img_w * img_h
        )));
    }

    // Reconstruct the grayscale image.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "img_w and img_h come from the PDF dictionary, parsed as usize; they fit in u32 for any practical image"
    )]
    let gray = GrayImage::from_raw(img_w as u32, img_h as u32, pixels).ok_or_else(|| {
        PhonoPaperError::InvalidFormat(
            "Failed to construct GrayImage from decompressed PDF pixels".to_string(),
        )
    })?;

    Ok(DynamicImage::ImageLuma8(gray))
}
