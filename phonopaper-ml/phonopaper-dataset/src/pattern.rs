//! Rendering of "sheets of paper": genuine `PhonoPaper` prints and look-alike
//! decoys used as hard negatives.

use phonopaper_rs::format::TOTAL_BINS;
use phonopaper_rs::render::{RenderOptions, image_buf_size, spectrogram_to_image_buf};
use phonopaper_rs::spectrogram::SpectrogramVec;

use crate::canvas::Canvas;
use crate::geometry::{Quad, rect_quad};
use crate::num::{f32_of, index_of, px};
use crate::rng::Rng;

/// A rendered sheet of paper and the corners of the region of interest on it.
#[derive(Debug, Clone)]
pub struct Sheet {
    /// The paper raster (`0` = black ink, `255` = white paper, before
    /// colour grading).
    pub canvas: Canvas,
    /// Corners (TL, TR, BR, BL) of the printed `PhonoPaper` ink box in sheet
    /// pixel coordinates: from the outer edge of the top marker band's first
    /// stripe to the outer edge of the bottom band's last stripe, spanning
    /// every data column.  White margins are **not** included.
    pub ink_box: Quad,
}

/// Fill a spectrogram with random musical-looking content.
fn random_spectrogram(rng: &mut Rng, columns: usize) -> SpectrogramVec {
    let mut spec = SpectrogramVec::new(columns);
    let style = rng.range_u32(0, 9);
    match style {
        // Blank sheet (e.g. a printed template).
        0 => {}
        // Dense random texture.
        1 => {
            let max_amp = rng.range_f64(0.2, 0.9);
            for col in 0..columns {
                for bin in 0..TOTAL_BINS {
                    if rng.chance(0.3) {
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "amplitude is in [0, 1]"
                        )]
                        let amp = rng.range_f64(0.0, max_amp) as f32;
                        spec.set(col, bin, amp);
                    }
                }
            }
        }
        // Notes, chords and glides.
        _ => {
            let notes = rng.range_usize(1, 60);
            for _ in 0..notes {
                let start = rng.range_usize(0, columns - 1);
                let duration = rng.range_usize(2, 120.min(columns));
                let end = (start + duration).min(columns);
                let bin0 = rng.range_usize(60, TOTAL_BINS - 1);
                // Glide: bins per column, mostly zero.
                let glide = if rng.chance(0.3) {
                    rng.range_f64(-0.5, 0.5)
                } else {
                    0.0
                };
                let amp_max = rng.range_f64(0.3, 1.0);
                let thickness = rng.range_usize(1, 3);
                let harmonics = rng.range_usize(0, 3);
                for col in start..end {
                    let drift = glide * px(col - start);
                    let bin = index_of(px(bin0) + drift, TOTAL_BINS - 1);
                    for h in 0..=harmonics {
                        // One octave = 48 bins; harmonics go up (lower index).
                        let Some(hbin) = bin.checked_sub(48 * h) else {
                            break;
                        };
                        let amp = f32_of(amp_max / (1.0 + px(h)));
                        for t in 0..thickness {
                            spec.set(col, hbin + t, amp);
                        }
                    }
                }
            }
            // Sprinkle a little "printer dust".
            if rng.chance(0.4) {
                let dots = rng.range_usize(0, columns);
                for _ in 0..dots {
                    let col = rng.range_usize(0, columns - 1);
                    let bin = rng.range_usize(0, TOTAL_BINS - 1);
                    let amp = f32_of(rng.range_f64(0.0, 0.4));
                    spec.set(col, bin, amp);
                }
            }
        }
    }
    spec
}

/// Random but valid marker geometry, following the constraints enforced by
/// `phonopaper_rs::decode::detect_markers` (thick ≥ 3 × thin, gap within
/// `[thin/2, 4·thin]`).
fn random_render_options(rng: &mut Rng) -> RenderOptions {
    let thin = rng.range_u32(4, 14);
    let thick = thin * 3 + rng.range_u32(0, thin * 2);
    let gap = rng.range_u32((thin / 2).max(2), thin * 2);
    RenderOptions {
        px_per_octave: rng.range_u32(30, 110),
        thin_stripe: thin,
        thick_stripe: thick,
        marker_gap: gap,
        margin: rng.range_u32(8, 90),
        draw_octave_lines: rng.chance(0.15),
        gamma: 1.0,
    }
}

/// Render a genuine `PhonoPaper` sheet with random geometry and content.
#[must_use]
pub fn phonopaper_sheet(rng: &mut Rng) -> Sheet {
    let opts = random_render_options(rng);
    // Real prints range from a short jingle to a whole A4 landscape page.
    let columns = rng.range_usize(60, 1600);
    let spec = random_spectrogram(rng, columns);
    let mut buf = vec![0u8; image_buf_size(columns, &opts)];
    spectrogram_to_image_buf(&spec, &opts, &mut buf);
    let pattern_h = opts.image_height() as usize;

    let left = rng.range_usize(6, 90);
    let right = rng.range_usize(6, 90);
    let sheet_w = left + columns + right;
    let mut canvas = Canvas::filled(sheet_w, pattern_h, 255.0);
    for y in 0..pattern_h {
        for x in 0..columns {
            canvas.set(left + x, y, f32::from(buf[y * columns + x]));
        }
    }
    let margin = f64::from(opts.margin);
    #[expect(
        clippy::cast_precision_loss,
        reason = "sheet dimensions are small integers"
    )]
    let ink_box = rect_quad(
        left as f64,
        margin,
        (left + columns) as f64,
        pattern_h as f64 - margin,
    );
    Sheet { canvas, ink_box }
}

/// Render a sheet that superficially resembles `PhonoPaper` — horizontal
/// stripes, barcodes, text-like blocks — but is **not** a valid pattern.
///
/// Used as a hard negative so the network learns the marker topology rather
/// than "dark horizontal bars on white".
#[must_use]
pub fn decoy_sheet(rng: &mut Rng) -> Sheet {
    let w = rng.range_usize(150, 800);
    let h = rng.range_usize(150, 800);
    let mut canvas = Canvas::filled(w, h, 255.0);
    let kind = rng.range_u32(0, 3);
    match kind {
        // Random horizontal stripes of random widths (barcode-like, rotated).
        0 => {
            let mut y = rng.range_usize(0, 40);
            while y < h {
                let sh = rng.range_usize(2, 60);
                let dark = rng.chance(0.5);
                if dark {
                    let x0 = rng.range_usize(0, w / 4);
                    let x1 = rng.range_usize(w * 3 / 4, w);
                    #[expect(clippy::cast_possible_wrap, reason = "sheet dimensions are small")]
                    canvas.fill_rect(x0 as i64, y as i64, (x1 - x0) as i64, sh as i64, 0.0);
                }
                y += sh;
            }
        }
        // Text-like rows of small blocks.
        1 => {
            let line_h = rng.range_usize(6, 20);
            let mut y = rng.range_usize(5, 30);
            while y + line_h < h {
                let mut x = rng.range_usize(5, 30);
                while x < w {
                    let bw = rng.range_usize(3, 30);
                    if rng.chance(0.7) {
                        #[expect(
                            clippy::cast_possible_wrap,
                            reason = "sheet dimensions are small"
                        )]
                        canvas.fill_rect(
                            x as i64,
                            y as i64,
                            bw as i64,
                            (line_h * 2 / 3) as i64,
                            0.0,
                        );
                    }
                    x += bw + rng.range_usize(2, 10);
                }
                y += line_h + rng.range_usize(2, 8);
            }
        }
        // A "broken" PhonoPaper: only one marker band, or thick stripe missing.
        2 => {
            let opts = random_render_options(rng);
            let columns = rng.range_usize(60, 500);
            let spec = random_spectrogram(rng, columns);
            let mut buf = vec![0u8; image_buf_size(columns, &opts)];
            spectrogram_to_image_buf(&spec, &opts, &mut buf);
            let ph = opts.image_height() as usize;
            let band = opts.marker_band_height() as usize;
            canvas = Canvas::filled(columns, ph, 255.0);
            let erase_top = rng.chance(0.5);
            for y in 0..ph {
                let in_top_band = y < band;
                let in_bottom_band = y >= ph - band;
                let erased = (erase_top && in_top_band) || (!erase_top && in_bottom_band);
                if erased {
                    continue;
                }
                for x in 0..columns {
                    canvas.set(x, y, f32::from(buf[y * columns + x]));
                }
            }
        }
        // Grid / table.
        _ => {
            let step_x = rng.range_usize(15, 80);
            let step_y = rng.range_usize(15, 80);
            let thick = rng.range_usize(1, 6);
            let mut x = 0;
            while x < w {
                #[expect(clippy::cast_possible_wrap, reason = "sheet dimensions are small")]
                canvas.fill_rect(x as i64, 0, thick as i64, h as i64, 0.0);
                x += step_x;
            }
            let mut y = 0;
            while y < h {
                #[expect(clippy::cast_possible_wrap, reason = "sheet dimensions are small")]
                canvas.fill_rect(0, y as i64, w as i64, thick as i64, 0.0);
                y += step_y;
            }
        }
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "sheet dimensions are small integers"
    )]
    let ink_box = rect_quad(0.0, 0.0, canvas.width() as f64, canvas.height() as f64);
    Sheet { canvas, ink_box }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phonopaper_sheet_is_detectable_by_reference_decoder() {
        // The freshly rendered (undistorted) sheet must be a valid PhonoPaper
        // according to the library's own detector; otherwise the labels would
        // teach the network a wrong pattern.
        for seed in 0..20 {
            let mut rng = Rng::from_seed(seed);
            let sheet = phonopaper_sheet(&mut rng);
            let img = image::DynamicImage::ImageLuma8(sheet.canvas.to_image());
            let bounds = phonopaper_rs::decode::detect_markers(&img)
                .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
            // Data area lies strictly inside the ink box.
            assert!(f64::from(bounds.data_top) > sheet.ink_box[0].y);
            assert!(f64::from(bounds.data_bottom) < sheet.ink_box[2].y);
        }
    }

    #[test]
    fn decoy_sheets_have_full_size_ink_box() {
        let mut rng = Rng::from_seed(5);
        let sheet = decoy_sheet(&mut rng);
        assert_eq!(sheet.ink_box[2].x, px(sheet.canvas.width()));
        assert_eq!(sheet.ink_box[2].y, px(sheet.canvas.height()));
    }
}
