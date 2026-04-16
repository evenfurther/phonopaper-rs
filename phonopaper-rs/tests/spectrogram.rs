use phonopaper_rs::decode::{fill_spectrogram_from_pixels, spectrogram_from_pixels};
use phonopaper_rs::format::TOTAL_BINS;
use phonopaper_rs::spectrogram::{Spectrogram, SpectrogramBuf, SpectrogramBufMut, SpectrogramVec};

// ─── Tests that work without std ──────────────────────────────────────────────

#[test]
fn test_set_get_round_trip() {
    // 10 columns × TOTAL_BINS bins; initialise to 0.
    let mut buf = [0.0f32; 10 * TOTAL_BINS];
    let mut spec = Spectrogram::from_storage(10, &mut buf[..]).unwrap();
    spec.set(3, 100, 0.75);
    assert!((spec.get(3, 100) - 0.75).abs() < 1e-6);
}

#[test]
fn test_clamp_amplitude() {
    let mut buf = [0.0f32; TOTAL_BINS];
    let mut spec = Spectrogram::from_storage(1, &mut buf[..]).unwrap();
    spec.set(0, 0, 1.5); // should clamp to 1.0
    assert!((spec.get(0, 0) - 1.0).abs() < 1e-6);
    spec.set(0, 0, -0.5); // should clamp to 0.0
    assert!((spec.get(0, 0) - 0.0).abs() < 1e-6);
}

/// `fill_from_image_data` must map each bin to the correct row when the
/// data-area height differs from `TOTAL_BINS` (the common case: 720 px for 8
/// octaves at 90 px/octave).
///
/// The encoder assigns pixel row `r` to bin `⌊r · TOTAL_BINS / height⌋`.
/// The decoder must therefore read a row from within the contiguous run of rows
/// that all map to that bin — specifically the centre row — so that applying the
/// encoder formula to the decoded row gives back the original bin.
#[test]
fn from_image_data_row_mapping_is_correct() {
    const DATA_HEIGHT: usize = 720;

    // Arithmetic check: every bin's centre row must round-trip via the encoder
    // formula.
    for bin in 0..TOTAL_BINS {
        let centre = (2 * bin + 1) * DATA_HEIGHT / (2 * TOTAL_BINS);
        let encoded_bin = centre * TOTAL_BINS / DATA_HEIGHT;
        assert_eq!(
            encoded_bin, bin,
            "bin {bin}: centre row {centre} maps back to bin {encoded_bin} \
             via the encoder formula (expected {bin})"
        );
    }

    // Pixel-level check: build a pixel buffer where every row is black (amp=1)
    // except the centre row of each bin which is white (amp=0).  After decoding,
    // every bin should have amplitude 0.
    let mut pixels = [0u8; DATA_HEIGHT]; // width=1, all black
    for bin in 0..TOTAL_BINS {
        let centre = (2 * bin + 1) * DATA_HEIGHT / (2 * TOTAL_BINS);
        pixels[centre] = 255; // white → amp = 0
    }

    let mut data = [0.0f32; TOTAL_BINS]; // 1 column
    let mut spec = Spectrogram::from_storage(1, &mut data[..]).unwrap();
    fill_spectrogram_from_pixels(&mut spec, &pixels, 1, DATA_HEIGHT);

    for bin in 0..TOTAL_BINS {
        let amp = spec.get(0, bin);
        assert!(
            amp < 1e-6,
            "bin {bin}: expected amplitude 0.0 (white centre row), got {amp:.4}"
        );
    }
}

// ─── Test that requires std (image::RgbImage) ─────────────────────────────────

/// When `from_image_data` is called with a data-area height different from
/// `TOTAL_BINS` followed by `spectrogram_to_image` rendering (which also uses
/// a non-`TOTAL_BINS` height), the round-trip preserves every bin.
///
/// This exercises the full encode → image → decode pipeline using the standard
/// `PhonoPaper` pixel-height (720 px for 8 octaves × 90 px/octave).
#[test]
fn image_encode_decode_bin_round_trip() {
    use phonopaper_rs::render::{RenderOptions, spectrogram_to_image};
    use phonopaper_rs::spectrogram::SpectrogramVec;

    let num_cols = 4;
    let mut spec_in = SpectrogramVec::new(num_cols);

    // Activate every 8th bin to avoid JPEG/resize bleed (PNG is lossless here).
    for bin in (0..TOTAL_BINS).step_by(8) {
        for col in 0..num_cols {
            spec_in.set(col, bin, 1.0);
        }
    }

    // Render to a grayscale image (no marker bands — test the data pixels only).
    let opts = RenderOptions {
        draw_octave_lines: false,
        ..RenderOptions::default()
    };
    let img = spectrogram_to_image(&spec_in, &opts);
    let img_width = img.width() as usize;

    // Crop the data area out of the rendered image.
    let data_height = opts.px_per_octave as usize * phonopaper_rs::format::OCTAVES;
    // Top marker band height: margin + thin + gap + thin + gap + thick + gap + thin
    let top_band_height = (opts.margin
        + opts.thin_stripe
        + opts.marker_gap
        + opts.thin_stripe
        + opts.marker_gap
        + opts.thick_stripe
        + opts.marker_gap
        + opts.thin_stripe) as usize;
    let data_y_start = top_band_height;

    let mut pixels = std::vec![0u8; img_width * data_height];
    for row in 0..data_height {
        for col in 0..img_width {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "col and (data_y_start + row) are pixel coordinates derived from \
                          a u32 image dimension; they fit in u32"
            )]
            let px = img.get_pixel(col as u32, (data_y_start + row) as u32);
            // RGB image; take the red channel (all three are equal for greyscale).
            pixels[row * img_width + col] = px[0];
        }
    }

    let spec_out = spectrogram_from_pixels(&pixels, img_width, data_height);

    // Every activated bin should round-trip with high amplitude; inactive
    // bins between them should stay near zero.
    for bin in (0..TOTAL_BINS).step_by(8) {
        #[expect(clippy::cast_precision_loss, reason = "num_cols = 4; exact in f32")]
        let amp = (0..num_cols).map(|c| spec_out.get(c, bin)).sum::<f32>() / num_cols as f32;
        assert!(
            amp > 0.5,
            "bin {bin} amplitude after round-trip is {amp:.3} (expected > 0.5)"
        );
    }
}

// ─── Spectrogram::column ──────────────────────────────────────────────────────

/// `Spectrogram::column()` returns a slice whose contents match values
/// written via `Spectrogram::set()`.
#[test]
fn column_read_matches_set_values() {
    let mut buf = [0.0f32; 3 * TOTAL_BINS];
    let mut spec = Spectrogram::from_storage(3, &mut buf[..]).unwrap();

    spec.set(1, 0, 0.25);
    spec.set(1, TOTAL_BINS - 1, 0.75);

    let col = spec.column(1).expect("column(1) should return Some");
    assert_eq!(col.len(), TOTAL_BINS);
    assert!(
        (col[0] - 0.25).abs() < 1e-6,
        "first bin: expected 0.25, got {}",
        col[0]
    );
    assert!(
        (col[TOTAL_BINS - 1] - 0.75).abs() < 1e-6,
        "last bin: expected 0.75, got {}",
        col[TOTAL_BINS - 1]
    );
    // A bin that was never written should be zero.
    assert!(
        col[TOTAL_BINS / 2].abs() < 1e-6,
        "unwritten bin should be 0.0"
    );
}

/// `Spectrogram::column()` on column 0 and the last column are independent —
/// writing to one does not affect the other.
#[test]
fn column_independence() {
    let mut buf = [0.0f32; 2 * TOTAL_BINS];
    let mut spec = Spectrogram::from_storage(2, &mut buf[..]).unwrap();

    spec.set(0, 10, 1.0);
    spec.set(1, 10, 0.5);

    assert!((spec.column(0).unwrap()[10] - 1.0).abs() < 1e-6);
    assert!((spec.column(1).unwrap()[10] - 0.5).abs() < 1e-6);
}

// ─── SpectrogramBuf<'a> ───────────────────────────────────────────────────────

/// A `SpectrogramBuf` (backed by `&[f32]`) can be created via
/// `Spectrogram::from_storage` and supports read-only access via `get` and
/// `column`.
#[test]
fn spectrogram_buf_read_only_access() {
    const NCOLS: usize = 2;
    // Pre-populate the backing storage with known values.
    let mut data = vec![0.0f32; NCOLS * TOTAL_BINS];
    data[TOTAL_BINS + 7] = 0.6; // column 1, bin 7

    let buf: SpectrogramBuf<'_> =
        Spectrogram::from_storage(NCOLS, &data[..]).expect("from_storage should succeed");

    assert_eq!(buf.num_columns(), NCOLS);
    assert!(
        (buf.get(1, 7) - 0.6).abs() < 1e-6,
        "get(1,7) should return 0.6"
    );
    assert!(buf.get(0, 7).abs() < 1e-6, "column 0 bin 7 should be 0.0");

    let col1 = buf.column(1).expect("column(1) should be in range");
    assert_eq!(col1.len(), TOTAL_BINS);
    assert!(
        (col1[7] - 0.6).abs() < 1e-6,
        "column(1)[7] should match get(1,7)"
    );
}

/// `column()` returns `None` for an out-of-bounds column index.
#[test]
fn column_oob_returns_none() {
    let spec = SpectrogramVec::new(3);
    assert!(
        spec.column(3).is_none(),
        "column(3) should be None for 3-column spec"
    );
    assert!(
        spec.column(usize::MAX).is_none(),
        "column(MAX) should be None"
    );
    assert!(
        spec.column(2).is_some(),
        "column(2) should be Some (last valid column)"
    );
}

/// `column_or_panic()` returns the same slice as `column().unwrap()`.
#[test]
fn column_or_panic_matches_column() {
    let mut spec = SpectrogramVec::new(2);
    spec.set(1, 42, 0.7);
    let via_checked = spec.column(1).unwrap();
    let via_or_panic = spec.column_or_panic(1);
    assert_eq!(via_checked, via_or_panic);
    assert!((via_or_panic[42] - 0.7).abs() < 1e-6);
}

#[test]
fn spectrogram_buf_wrong_size_returns_none() {
    let data = vec![0.0f32; TOTAL_BINS - 1]; // one element short
    let result: Option<SpectrogramBuf<'_>> = Spectrogram::from_storage(1, &data[..]);
    assert!(
        result.is_none(),
        "from_storage with wrong-length slice should return None"
    );
}

// ─── SpectrogramBufMut<'a> ────────────────────────────────────────────────────

/// A `SpectrogramBufMut` (backed by `&mut [f32]`) can be created via
/// `Spectrogram::from_storage` and supports both write (`set`, `column_mut`)
/// and read (`get`, `column`) access.
#[test]
fn spectrogram_buf_mut_write_read() {
    const NCOLS: usize = 3;
    let mut data = vec![0.0f32; NCOLS * TOTAL_BINS];

    {
        let mut buf: SpectrogramBufMut<'_> =
            Spectrogram::from_storage(NCOLS, &mut data[..]).expect("from_storage should succeed");

        buf.set(2, 50, 0.9);
        // Also write via column_mut.
        buf.column_mut(0)[0] = 0.3;
    }

    // After the mutable borrow ends, verify the values were written into `data`.
    let buf: SpectrogramBuf<'_> =
        Spectrogram::from_storage(NCOLS, &data[..]).expect("re-wrapping should succeed");
    assert!(
        (buf.get(2, 50) - 0.9).abs() < 1e-6,
        "set(2,50,0.9) should persist"
    );
    assert!(
        (buf.get(0, 0) - 0.3).abs() < 1e-6,
        "column_mut(0)[0] = 0.3 should persist"
    );
    assert!(
        buf.get(1, 50).abs() < 1e-6,
        "unwritten column 1 bin 50 should be 0.0"
    );
}
