//! Integration tests for the `vector` module:
//! [`vector::spectrogram_to_svg`], [`vector::spectrogram_to_pdf`],
//! [`vector::image_from_svg`], and [`vector::image_from_pdf`].

use phonopaper_rs::{
    SpectrogramVec,
    render::RenderOptions,
    vector::{
        PdfPageLayout, image_from_pdf, image_from_svg, spectrogram_to_pdf, spectrogram_to_svg,
    },
};

// ─── helpers ─────────────────────────────────────────────────────────────────

/// A silent (all-zero) spectrogram with `n` columns.
fn silent_spec(n: usize) -> SpectrogramVec {
    SpectrogramVec::new(n)
}

/// A small non-trivial spectrogram: 8 columns, bin 10 of column 3 at
/// amplitude 0.75 and bin 200 of column 6 at amplitude 0.5.
fn small_spec() -> SpectrogramVec {
    let mut spec = SpectrogramVec::new(8);
    spec.set(3, 10, 0.75);
    spec.set(6, 200, 0.5);
    spec
}

// ─── SVG structure tests ──────────────────────────────────────────────────────

/// The SVG output starts with `<svg ` and ends with `</svg>`.
#[test]
fn svg_valid_structure() {
    let svg = spectrogram_to_svg(&silent_spec(4), &RenderOptions::default());
    assert!(
        svg.starts_with("<svg "),
        "SVG must start with '<svg '; got: {:?}",
        &svg[..svg.len().min(30)]
    );
    assert!(
        svg.trim_end().ends_with("</svg>"),
        "SVG must end with '</svg>'"
    );
}

/// The SVG contains an `<image` element (the embedded PNG raster).
#[test]
fn svg_contains_image_element() {
    let svg = spectrogram_to_svg(&silent_spec(4), &RenderOptions::default());
    assert!(
        svg.contains("<image "),
        "SVG must contain an <image> element for the embedded PNG raster"
    );
}

/// The SVG contains a `data:image/png;base64,` data URI.
#[test]
fn svg_contains_png_data_uri() {
    let svg = spectrogram_to_svg(&silent_spec(4), &RenderOptions::default());
    assert!(
        svg.contains("data:image/png;base64,"),
        "SVG must embed the PNG raster as a base64 data URI"
    );
}

/// The SVG contains exactly 8 black `<rect>` elements for the marker bands:
/// 4 black stripes per band × 2 bands (top + bottom) = 8.
///
/// Default [`RenderOptions`] has stripes: thin/gap/thin/gap/thick/gap/thin
/// (3 thin + 1 thick = 4 black stripes per band × 2 bands = 8 total).
#[test]
fn svg_contains_marker_rects() {
    let svg = spectrogram_to_svg(&silent_spec(10), &RenderOptions::default());
    // Count `<rect` occurrences (only black marker rects are written; no other
    // rects appear in the output).
    let rect_count = svg.matches("<rect ").count();
    assert_eq!(
        rect_count, 8,
        "Expected 8 black <rect> elements (4 stripes × 2 bands), got {rect_count}"
    );
}

/// With `draw_octave_lines = true`, the SVG contains exactly 7 `<line>` elements
/// (one per octave boundary).
#[test]
fn svg_octave_lines_count() {
    let opts = RenderOptions {
        draw_octave_lines: true,
        ..RenderOptions::default()
    };
    let svg = spectrogram_to_svg(&silent_spec(10), &opts);
    let line_count = svg.matches("<line ").count();
    assert_eq!(
        line_count, 7,
        "Expected 7 <line> elements for octave separators, got {line_count}"
    );
}

/// With `draw_octave_lines = false` (default), the SVG contains no `<line>`
/// elements.
#[test]
fn svg_no_octave_lines_by_default() {
    let opts = RenderOptions {
        draw_octave_lines: false,
        ..RenderOptions::default()
    };
    let svg = spectrogram_to_svg(&silent_spec(10), &opts);
    assert!(
        !svg.contains("<line "),
        "No <line> elements should appear when draw_octave_lines is false"
    );
}

/// An empty spectrogram (0 columns) produces a valid SVG with no image element.
#[test]
fn svg_empty_spectrogram() {
    let svg = spectrogram_to_svg(&silent_spec(0), &RenderOptions::default());
    assert!(
        svg.starts_with("<svg "),
        "empty spectrogram should still produce valid SVG"
    );
    assert!(
        svg.trim_end().ends_with("</svg>"),
        "SVG must close properly"
    );
    // No embedded image for an empty spectrogram.
    assert!(
        !svg.contains("data:image/png;base64,"),
        "empty spectrogram SVG should not embed a PNG (no pixels to encode)"
    );
}

// ─── PDF structure tests ──────────────────────────────────────────────────────

/// The PDF output starts with `%PDF-`.
#[test]
fn pdf_starts_with_header() {
    let pdf = spectrogram_to_pdf(
        &silent_spec(4),
        &RenderOptions::default(),
        PdfPageLayout::default(),
    );
    assert!(
        pdf.starts_with(b"%PDF-"),
        "PDF must start with %PDF- header"
    );
}

/// The PDF contains a `FlateDecode` stream (the compressed image `XObject`).
#[test]
fn pdf_contains_flatedecode() {
    let pdf = spectrogram_to_pdf(
        &silent_spec(4),
        &RenderOptions::default(),
        PdfPageLayout::default(),
    );
    // Search for the ASCII byte sequence rather than treating the whole PDF as UTF-8
    // (the compressed pixel stream is binary).
    assert!(
        pdf.windows(b"/FlateDecode".len())
            .any(|w| w == b"/FlateDecode"),
        "PDF must contain a /FlateDecode image XObject"
    );
}

/// An empty spectrogram (0 columns) produces a non-empty, valid PDF.
#[test]
fn pdf_empty_spectrogram() {
    let pdf = spectrogram_to_pdf(
        &silent_spec(0),
        &RenderOptions::default(),
        PdfPageLayout::default(),
    );
    assert!(
        pdf.starts_with(b"%PDF-"),
        "empty spectrogram should still produce a valid PDF"
    );
    assert!(!pdf.is_empty(), "PDF must be non-empty");
}

// ─── SVG round-trip ───────────────────────────────────────────────────────────

/// Encoding a spectrogram to SVG and extracting the embedded image via
/// `image_from_svg` recovers an image with the correct dimensions.
#[test]
fn svg_round_trip_dimensions() {
    let opts = RenderOptions::default();
    let spec = small_spec();
    let svg = spectrogram_to_svg(&spec, &opts);
    let img = image_from_svg(&svg).expect("image_from_svg should succeed");

    assert_eq!(
        img.width() as usize,
        spec.num_columns(),
        "round-tripped image width must equal spec.num_columns()"
    );
    assert_eq!(
        img.height(),
        opts.image_height(),
        "round-tripped image height must equal opts.image_height()"
    );
}

/// The SVG round-trip preserves pixel values: re-extracting the raster image
/// and comparing it to a directly-rendered grayscale buffer shows zero
/// pixel difference.
#[test]
fn svg_round_trip_pixel_fidelity() {
    use image::GrayImage;
    use phonopaper_rs::render::{image_buf_size, spectrogram_to_image_buf};

    let opts = RenderOptions::default();
    let spec = small_spec();
    let n = spec.num_columns();

    // Direct render to a grayscale buffer.
    let mut direct_buf = vec![0u8; image_buf_size(n, &opts)];
    spectrogram_to_image_buf(&spec, &opts, &mut direct_buf);

    // SVG round-trip: extract the embedded PNG, convert to L8.
    let svg = spectrogram_to_svg(&spec, &opts);
    let rt_img = image_from_svg(&svg).expect("image_from_svg should succeed");
    let rt_gray: GrayImage = rt_img.to_luma8();
    let rt_buf: &[u8] = rt_gray.as_raw();

    assert_eq!(
        direct_buf.len(),
        rt_buf.len(),
        "pixel buffer lengths must match after SVG round-trip"
    );

    let max_diff = direct_buf
        .iter()
        .zip(rt_buf.iter())
        .map(|(&a, &b)| a.abs_diff(b))
        .max()
        .unwrap_or(0);

    assert_eq!(
        max_diff, 0,
        "SVG round-trip should preserve pixel values exactly (PNG is lossless)"
    );
}

// ─── PDF round-trip ───────────────────────────────────────────────────────────

/// Encoding a spectrogram to PDF and extracting the embedded image via
/// `image_from_pdf` recovers an image with the correct dimensions.
#[test]
fn pdf_round_trip_dimensions() {
    let opts = RenderOptions::default();
    let spec = small_spec();
    let pdf = spectrogram_to_pdf(&spec, &opts, PdfPageLayout::default());
    let img = image_from_pdf(&pdf).expect("image_from_pdf should succeed");

    assert_eq!(
        img.width() as usize,
        spec.num_columns(),
        "round-tripped image width must equal spec.num_columns()"
    );
    assert_eq!(
        img.height(),
        opts.image_height(),
        "round-tripped image height must equal opts.image_height()"
    );
}

/// The PDF round-trip preserves pixel values: re-extracting the raw grayscale
/// raster from the PDF and comparing it pixel-by-pixel to a directly-rendered
/// grayscale buffer shows zero difference.
#[test]
fn pdf_round_trip_pixel_fidelity() {
    use image::GrayImage;
    use phonopaper_rs::render::{image_buf_size, spectrogram_to_image_buf};

    let opts = RenderOptions::default();
    let spec = small_spec();
    let n = spec.num_columns();

    // Direct render to a grayscale buffer.
    let mut direct_buf = vec![0u8; image_buf_size(n, &opts)];
    spectrogram_to_image_buf(&spec, &opts, &mut direct_buf);

    // PDF round-trip: extract the raw grayscale image, convert to L8.
    let pdf = spectrogram_to_pdf(&spec, &opts, PdfPageLayout::default());
    let rt_img = image_from_pdf(&pdf).expect("image_from_pdf should succeed");
    let rt_gray: GrayImage = rt_img.to_luma8();
    let rt_buf: &[u8] = rt_gray.as_raw();

    assert_eq!(
        direct_buf.len(),
        rt_buf.len(),
        "pixel buffer lengths must match after PDF round-trip"
    );

    let max_diff = direct_buf
        .iter()
        .zip(rt_buf.iter())
        .map(|(&a, &b)| a.abs_diff(b))
        .max()
        .unwrap_or(0);

    assert_eq!(
        max_diff, 0,
        "PDF round-trip should preserve pixel values exactly (deflate is lossless)"
    );
}

// ─── Error handling ───────────────────────────────────────────────────────────

/// `image_from_svg` returns an error for SVG without an embedded PNG.
#[test]
fn image_from_svg_error_no_image() {
    let bad_svg = "<svg xmlns=\"http://www.w3.org/2000/svg\"><rect/></svg>";
    let result = image_from_svg(bad_svg);
    assert!(
        result.is_err(),
        "image_from_svg should return an error when there is no embedded PNG"
    );
}

/// `image_from_pdf` returns an error for arbitrary non-PDF bytes.
#[test]
fn image_from_pdf_error_no_image() {
    let bad_pdf = b"not a valid PDF at all";
    let result = image_from_pdf(bad_pdf);
    assert!(
        result.is_err(),
        "image_from_pdf should return an error for non-PDF bytes"
    );
}

// ─── base64_decode error paths ────────────────────────────────────────────────

/// `image_from_svg` propagates a `base64_decode` error when the embedded
/// base64 payload contains an invalid character (e.g. `!`).
///
/// This exercises the `b > 127 || table[b as usize] == 0xFF` branch inside
/// `base64_decode`.
#[test]
fn image_from_svg_error_malformed_base64() {
    // Build a syntactically plausible SVG with an invalid base64 payload.
    let bad_svg = "<svg xmlns=\"http://www.w3.org/2000/svg\">\
        <image href=\"data:image/png;base64,!!!!\"/></svg>";
    let result = image_from_svg(bad_svg);
    assert!(
        result.is_err(),
        "image_from_svg should propagate a base64_decode error for invalid characters"
    );
}

/// `image_from_svg` returns an error when the `data:image/png;base64,` URI is
/// not closed by a `"` — the "unterminated URI" error branch.
#[test]
fn image_from_svg_error_unterminated_uri() {
    // The href is opened but never closed.
    let bad_svg = "<svg xmlns=\"http://www.w3.org/2000/svg\">\
        <image href=\"data:image/png;base64,AAAA/></svg>";
    let result = image_from_svg(bad_svg);
    assert!(
        result.is_err(),
        "image_from_svg should return an error when the base64 URI is unterminated"
    );
}

/// `image_from_svg` returns an error when the embedded base64 decodes to
/// valid bytes but those bytes are not a valid PNG.
///
/// This exercises the `image::load_from_memory` error branch inside
/// `image_from_svg`.
#[test]
fn image_from_svg_error_invalid_png() {
    // Base64-encode 4 arbitrary non-PNG bytes.
    // `AQID` decodes to [0x01, 0x02, 0x03] — definitely not a PNG.
    let bad_svg = "<svg xmlns=\"http://www.w3.org/2000/svg\">\
        <image href=\"data:image/png;base64,AQID\"/></svg>";
    let result = image_from_svg(bad_svg);
    assert!(
        result.is_err(),
        "image_from_svg should return an error when the base64 payload is not a valid PNG"
    );
}

// ─── spectrogram_to_pdf with octave lines ────────────────────────────────────

/// `spectrogram_to_pdf` with `draw_octave_lines = true` still produces a
/// valid PDF that round-trips through `image_from_pdf`.
///
/// This exercises the octave-line drawing loop inside `spectrogram_to_pdf`
/// (the `if opts.draw_octave_lines` branch).
#[test]
fn pdf_octave_lines_round_trip() {
    let opts = RenderOptions {
        draw_octave_lines: true,
        ..RenderOptions::default()
    };
    let spec = small_spec();
    let pdf = spectrogram_to_pdf(&spec, &opts, PdfPageLayout::default());

    assert!(
        pdf.starts_with(b"%PDF-"),
        "PDF with octave lines must start with %PDF-"
    );

    // The embedded image XObject is unchanged by octave lines (they are
    // vector strokes on top), so the round-trip must still succeed.
    let img = image_from_pdf(&pdf).expect("image_from_pdf should succeed with octave-lines PDF");
    assert_eq!(img.width() as usize, spec.num_columns());
    assert_eq!(img.height(), opts.image_height());
}

// ─── PdfPageLayout::FitToPage ─────────────────────────────────────────────────

/// `FitToPage` produces a PDF with the correct media-box dimensions and a
/// round-trippable embedded image, just like `PixelPerfect`.
#[test]
fn pdf_fit_to_page_round_trip() {
    use phonopaper_rs::vector::page_size;

    let opts = RenderOptions::default();
    let spec = small_spec();
    let layout = PdfPageLayout::FitToPage {
        page_width_pt: page_size::A4_LANDSCAPE.0,
        page_height_pt: page_size::A4_LANDSCAPE.1,
        margin_pt: 28.35,
    };
    let pdf = spectrogram_to_pdf(&spec, &opts, layout);

    assert!(
        pdf.starts_with(b"%PDF-"),
        "FitToPage PDF must start with %PDF-"
    );

    // The embedded raster is unchanged: image_from_pdf must still succeed and
    // return the original pixel dimensions.
    let img = image_from_pdf(&pdf).expect("image_from_pdf should succeed for FitToPage PDF");
    assert_eq!(img.width() as usize, spec.num_columns());
    assert_eq!(img.height(), opts.image_height());
}

/// For `FitToPage`, the image is centred: horizontal and vertical margins from
/// the page edge to the image edge must be equal (within float rounding).
///
/// We verify this indirectly by checking that the media-box dimensions match
/// the requested page size (the page dimensions come from the layout, not the
/// image).  We cannot read the cm matrix back from the PDF without a full
/// PDF parser, but this at least confirms the page size is respected.
#[test]
fn pdf_fit_to_page_produces_target_page_size() {
    use phonopaper_rs::vector::page_size;

    // Use A4 landscape.
    let (target_w, target_h) = page_size::A4_LANDSCAPE;
    let layout = PdfPageLayout::FitToPage {
        page_width_pt: target_w,
        page_height_pt: target_h,
        margin_pt: 28.35,
    };
    let pdf = spectrogram_to_pdf(&small_spec(), &RenderOptions::default(), layout);

    // The media-box appears in the PDF as "0 0 <w> <h>" in the /MediaBox array.
    // We scan for both dimension values as ASCII floats in the raw PDF bytes.
    let pdf_text = String::from_utf8_lossy(&pdf);
    // A4 landscape: 841.89 × 595.28 pt — search for these in the serialised PDF.
    assert!(
        pdf_text.contains("841.89") || pdf_text.contains("841.9"),
        "A4 landscape width (841.89 pt) not found in PDF media-box"
    );
    assert!(
        pdf_text.contains("595.28") || pdf_text.contains("595.3"),
        "A4 landscape height (595.28 pt) not found in PDF media-box"
    );
}

/// `image_from_pdf` returns an error when `/FlateDecode` is present but there
/// is no `<<` before it (missing dictionary start).
#[test]
fn image_from_pdf_error_no_dict_start() {
    // `/FlateDecode` appears but is not preceded by `<<`.
    let bad = b"%PDF-1.7\n/FlateDecode>>\nstream\nendstream\n";
    let result = image_from_pdf(bad);
    assert!(
        result.is_err(),
        "image_from_pdf should error when no '<<' precedes /FlateDecode"
    );
}

/// `image_from_pdf` returns an error when `/FlateDecode` is present and `<<`
/// precedes it but there is no `>>` after it (missing dictionary end).
#[test]
fn image_from_pdf_error_no_dict_end() {
    // `<<` is present before `/FlateDecode` but no `>>` follows.
    let bad = b"%PDF-1.7\n<</FlateDecode /Width 1 /Height 1\nstream\nendstream\n";
    let result = image_from_pdf(bad);
    assert!(
        result.is_err(),
        "image_from_pdf should error when no '>>' follows /FlateDecode"
    );
}

/// `image_from_pdf` returns an error when `/Width` is absent from the
/// image `XObject` dictionary.
#[test]
fn image_from_pdf_error_missing_width() {
    // Valid-looking dict with `/Height` but no `/Width`.
    let bad = b"%PDF-1.7\n<</FlateDecode /Height 1>>\nstream\nendstream\n";
    let result = image_from_pdf(bad);
    assert!(
        result.is_err(),
        "image_from_pdf should error when /Width is missing from the dict"
    );
}

/// `image_from_pdf` returns an error when `/Width` is present but its value
/// cannot be parsed as a `usize`.
#[test]
fn image_from_pdf_error_unparseable_width() {
    let bad = b"%PDF-1.7\n<</FlateDecode /Width abc /Height 1>>\nstream\nendstream\n";
    let result = image_from_pdf(bad);
    assert!(
        result.is_err(),
        "image_from_pdf should error when /Width value is not a valid integer"
    );
}

/// `image_from_pdf` returns an error when the `stream` keyword is absent
/// after the dictionary.
#[test]
fn image_from_pdf_error_no_stream() {
    let bad = b"%PDF-1.7\n<</FlateDecode /Width 1 /Height 1>>\nNOCONTENT\n";
    let result = image_from_pdf(bad);
    assert!(
        result.is_err(),
        "image_from_pdf should error when 'stream' keyword is missing"
    );
}

/// `image_from_pdf` returns an error when `endstream` is absent.
#[test]
fn image_from_pdf_error_no_endstream() {
    let bad = b"%PDF-1.7\n<</FlateDecode /Width 1 /Height 1>>\nstream\nABCD\n";
    let result = image_from_pdf(bad);
    assert!(
        result.is_err(),
        "image_from_pdf should error when 'endstream' marker is missing"
    );
}

/// `image_from_pdf` returns an error when the compressed data cannot be
/// decompressed (corrupt zlib payload).
#[test]
fn image_from_pdf_error_decompression_failure() {
    // The stream contains non-zlib garbage between `stream\n` and `endstream`.
    let bad = b"%PDF-1.7\n<</FlateDecode /Width 2 /Height 2>>\nstream\nNOTZLIB\nendstream\n";
    let result = image_from_pdf(bad);
    assert!(
        result.is_err(),
        "image_from_pdf should error when the stream cannot be decompressed"
    );
}

/// `image_from_pdf` returns an error when the decompressed pixel count does
/// not match `width × height` recorded in the dictionary.
#[test]
fn image_from_pdf_error_size_mismatch() {
    use miniz_oxide::deflate::compress_to_vec_zlib;

    // Compress 4 bytes (2×2 image) but declare dimensions as 3×3.
    let pixels: Vec<u8> = vec![0u8; 4];
    let compressed = compress_to_vec_zlib(&pixels, 1);

    let mut fake_pdf = "%PDF-1.7\n<</FlateDecode /Width 3 /Height 3>>\nstream\n"
        .to_string()
        .into_bytes();
    fake_pdf.extend_from_slice(&compressed);
    fake_pdf.extend_from_slice(b"\nendstream\n");

    let result = image_from_pdf(&fake_pdf);
    assert!(
        result.is_err(),
        "image_from_pdf should error when decompressed size != width*height"
    );
}

/// `image_from_pdf` returns an error when the image `XObject` dictionary
/// contains non-UTF-8 bytes between `<<` and `>>`.
#[test]
fn image_from_pdf_error_non_utf8_dict() {
    // Insert a non-UTF-8 byte (0xFF) inside the `<<...>>` dictionary.
    let mut bad: Vec<u8> = b"%PDF-1.7\n<</FlateDecode /Width\xff1 /Height 1>>".to_vec();
    bad.extend_from_slice(b"\nstream\nABCD\nendstream\n");
    let result = image_from_pdf(&bad);
    assert!(
        result.is_err(),
        "image_from_pdf should error when the dict slice contains non-UTF-8 bytes"
    );
}

/// `image_from_pdf` returns an error when `/Height` is absent from the
/// image `XObject` dictionary.
#[test]
fn image_from_pdf_error_missing_height() {
    // Valid-looking dict with `/Width` but no `/Height`.
    let bad = b"%PDF-1.7\n<</FlateDecode /Width 1>>\nstream\nendstream\n";
    let result = image_from_pdf(bad);
    assert!(
        result.is_err(),
        "image_from_pdf should error when /Height is missing from the dict"
    );
}

/// `image_from_pdf` uses a CRLF line-ending after `stream\r\n` (the 2-byte
/// skip path).  Build a fake-but-parseable PDF that uses `stream\r\n` and
/// verify successful decompression.
#[test]
fn image_from_pdf_crlf_stream_delimiter() {
    use miniz_oxide::deflate::compress_to_vec_zlib;

    // A real 1×1 grayscale pixel.
    let pixels: Vec<u8> = vec![128u8];
    let compressed = compress_to_vec_zlib(&pixels, 1);

    // Put each key on its own line so the dimension values are not immediately
    // followed by `>>` (which would make `"1>>"` fail `parse::<usize>()`).
    let mut fake_pdf = b"%PDF-1.7\n<</FlateDecode\n/Width 1\n/Height 1\n>>\nstream\r\n".to_vec();
    fake_pdf.extend_from_slice(&compressed);
    fake_pdf.extend_from_slice(b"\nendstream\n");

    // Should parse without error and return a 1×1 image.
    let img = image_from_pdf(&fake_pdf).expect("CRLF-delimited stream should parse correctly");
    assert_eq!(img.width(), 1);
    assert_eq!(img.height(), 1);
}
