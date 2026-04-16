//! `phonopaper blank` — generate a blank, ready-to-print `PhonoPaper` template PDF.

use clap::Args;

/// Arguments for the `blank` sub-command.
#[derive(Args)]
pub struct BlankArgs {
    /// Output PDF file path.
    #[arg(
        short,
        long,
        default_value = "blank_phonopaper.pdf",
        value_name = "FILE"
    )]
    pub output: String,

    /// Paper format (a4, a3, letter, legal).
    #[arg(long, default_value = "a4", value_name = "FORMAT")]
    pub paper: PaperFormat,

    /// Use portrait orientation (default is landscape).
    #[arg(long)]
    pub portrait: bool,
}

/// Supported paper formats.
#[derive(Clone, Debug, clap::ValueEnum)]
pub enum PaperFormat {
    /// ISO A4 (210 × 297 mm)
    A4,
    /// ISO A3 (297 × 420 mm)
    A3,
    /// US Letter (215.9 × 279.4 mm)
    Letter,
    /// US Legal (215.9 × 355.6 mm)
    Legal,
}

/// Returns the paper dimensions in millimetres as `(short_side, long_side)`.
fn paper_mm(format: &PaperFormat) -> (f32, f32) {
    match format {
        PaperFormat::A4 => (210.0, 297.0),
        PaperFormat::A3 => (297.0, 420.0),
        PaperFormat::Letter => (215.9, 279.4),
        PaperFormat::Legal => (215.9, 355.6),
    }
}

/// Convert millimetres to PDF points (1 pt = 1/72 inch; 1 mm ≈ 2.834 645 pt).
fn mm_to_pt(mm: f32) -> f32 {
    mm * 2.834_645_5
}

#[expect(
    clippy::too_many_lines,
    reason = "single-page PDF builder with multiple distinct rendering phases; \
              splitting would obscure the shared geometry variables"
)]
#[expect(
    clippy::similar_names,
    reason = "page_w_mm/page_h_mm and data_h_pt/data_w_pt are the natural geometry variable names; \
              renaming them would obscure the coordinate arithmetic"
)]
fn generate_blank_pdf(args: &BlankArgs) -> Vec<u8> {
    use pdf_writer::{Content, Finish, Name, Pdf, Rect, Ref, Str};
    use phonopaper_rs::format::OCTAVES;
    use phonopaper_rs::render::RenderOptions;

    let opts = RenderOptions::default();

    // ── Paper dimensions ─────────────────────────────────────────────────────
    let (short_mm, long_mm) = paper_mm(&args.paper);
    let (page_w_mm, page_h_mm) = if args.portrait {
        (short_mm, long_mm)
    } else {
        (long_mm, short_mm)
    };
    let page_w = mm_to_pt(page_w_mm);
    let page_h = mm_to_pt(page_h_mm);

    // ── Layout constants (all in pt) ─────────────────────────────────────────
    let label_margin_pt = mm_to_pt(8.0);
    let label_gap_pt = mm_to_pt(1.5);
    let outer_pad_pt = mm_to_pt(5.0);

    let available_h_pt = page_h - 2.0 * outer_pad_pt;

    #[expect(
        clippy::cast_precision_loss,
        reason = "image_height() returns a u32 pixel count ≤ ~65535, exact in f32"
    )]
    let image_h_px = opts.image_height() as f32;
    let px_per_pt = image_h_px / available_h_pt;

    #[expect(
        clippy::cast_precision_loss,
        reason = "marker_band_height() is a u32 pixel count ≤ ~65535, exact in f32"
    )]
    let marker_band_pt = opts.marker_band_height() as f32 / px_per_pt;

    #[expect(
        clippy::cast_precision_loss,
        reason = "px_per_octave * OCTAVES ≤ 65535; OCTAVES = 8; both exact in f32"
    )]
    let data_h_pt = opts.px_per_octave as f32 * OCTAVES as f32 / px_per_pt;

    let data_x_start = label_margin_pt + label_gap_pt;
    let data_w_pt = page_w - data_x_start - outer_pad_pt;

    // PDF y-up: y=0 at bottom, increases upward.
    let img_y_bottom = outer_pad_pt;
    let data_y_bottom = img_y_bottom + marker_band_pt;
    let img_y_top = data_y_bottom + data_h_pt + marker_band_pt;

    #[expect(clippy::cast_precision_loss, reason = "OCTAVES = 8, exact in f32")]
    let zone_h = data_h_pt / OCTAVES as f32;

    // ── PDF object IDs ────────────────────────────────────────────────────────
    let catalog_id = Ref::new(1);
    let page_tree_id = Ref::new(2);
    let page_id = Ref::new(3);
    let font_id = Ref::new(4);
    let content_id = Ref::new(5);
    let font_name = Name(b"F1");

    let mut pdf = Pdf::new();
    pdf.catalog(catalog_id).pages(page_tree_id);
    pdf.pages(page_tree_id).kids([page_id]).count(1);

    // Page dictionary with font resource.
    {
        let mut page = pdf.page(page_id);
        page.media_box(Rect::new(0.0, 0.0, page_w, page_h));
        page.parent(page_tree_id);
        page.contents(content_id);
        page.resources().fonts().pair(font_name, font_id);
        page.finish();
    }

    // Helvetica Type1 font (standard PDF font; no embedding required).
    pdf.type1_font(font_id).base_font(Name(b"Helvetica"));

    // ── Build content stream ──────────────────────────────────────────────────
    let mut c = Content::new();

    // Marker stripe layout from the library (outer edge → data area).
    // This is the single source of truth shared with the SVG/PDF renderers.
    let stripes = phonopaper_rs::vector::marker_stripe_layout(&opts);

    // Helper: draw a filled black rectangle (PDF y-up coords).
    let draw_black_rect = |c: &mut Content, x: f32, y: f32, w: f32, h: f32| {
        c.set_fill_gray(0.0);
        c.rect(x, y, w, h);
        c.fill_nonzero();
    };

    // Convert stripe heights to points once.
    #[expect(
        clippy::cast_precision_loss,
        reason = "stripe pixel values are u32 ≤ ~512; exact in f32"
    )]
    let stripe_pt: Vec<f32> = stripes
        .iter()
        .map(|&(_, px)| px as f32 / px_per_pt)
        .collect();

    // Top marker band: drawn from img_y_top downward (stripes listed outer→inner).
    {
        let mut y = img_y_top;
        for (i, &h) in stripe_pt.iter().enumerate() {
            y -= h;
            if stripes[i].0 {
                draw_black_rect(&mut c, data_x_start, y, data_w_pt, h);
            }
        }
    }

    // Bottom marker band: mirror (listed outer→inner, drawn upward from bottom).
    {
        let mut y = img_y_bottom;
        for (i, &h) in stripe_pt.iter().enumerate() {
            if stripes[i].0 {
                draw_black_rect(&mut c, data_x_start, y, data_w_pt, h);
            }
            y += h;
        }
    }

    // Octave separator lines (light gray).
    c.set_stroke_gray(0.65);
    c.set_line_width(0.3);
    for octave in 1..OCTAVES {
        #[expect(clippy::cast_precision_loss, reason = "octave ≤ 7, exact in f32")]
        let y = data_y_bottom + octave as f32 * zone_h;
        c.move_to(data_x_start, y);
        c.line_to(data_x_start + data_w_pt, y);
        c.stroke();
    }

    // Border around the data area.
    c.set_stroke_gray(0.0);
    c.set_line_width(0.5);
    c.rect(data_x_start, data_y_bottom, data_w_pt, data_h_pt);
    c.stroke();

    // Octave tick marks on the label side.
    c.set_stroke_gray(0.0);
    c.set_line_width(0.4);
    let tick_len = mm_to_pt(1.5);
    for octave in 0..=OCTAVES {
        #[expect(clippy::cast_precision_loss, reason = "octave ≤ 8, exact in f32")]
        let y = data_y_bottom + octave as f32 * zone_h;
        c.move_to(label_margin_pt, y);
        c.line_to(label_margin_pt + tick_len, y);
        c.stroke();
    }

    // Octave labels C2–C8 (zone 0 = C2 at the bottom, zone 7 = C8 at the top).
    let font_size_pt = (zone_h * 0.3).clamp(5.0, 9.0);
    let label_x = label_margin_pt - mm_to_pt(1.0);

    for octave in 0..OCTAVES {
        let note_name = format!("C{}", octave + 2);
        #[expect(clippy::cast_precision_loss, reason = "octave ≤ 7, exact in f32")]
        let y_zone_bottom = data_y_bottom + octave as f32 * zone_h;
        let baseline_y = y_zone_bottom + font_size_pt * 0.15;

        c.begin_text();
        c.set_font(font_name, font_size_pt);
        c.set_fill_gray(0.0);
        c.next_line(label_x, baseline_y);
        c.show(Str(note_name.as_bytes()));
        c.end_text();
    }

    // "Inaudible" annotation for the top zone (above C8, > 4186 Hz).
    let top_zone_center_y = data_y_bottom + 7.5 * zone_h;
    c.begin_text();
    c.set_font(font_name, 5.0);
    c.set_fill_gray(0.55);
    c.next_line(data_x_start + mm_to_pt(1.0), top_zone_center_y);
    c.show(Str(b"(inaudible > 4186 Hz)"));
    c.end_text();

    // Footer note.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "value is data_w_pt*96/72/125 ≤ ~20 seconds; positive and far below u32::MAX"
    )]
    let approx_secs = (data_w_pt * 96.0 / 72.0 / 125.0).round() as u32;
    let footer =
        format!("PhonoPaper blank template — data area ≈ {approx_secs}s at 125 col/s (96 dpi)");
    c.begin_text();
    c.set_font(font_name, 6.0);
    c.set_fill_gray(0.45);
    c.next_line(data_x_start, outer_pad_pt * 0.45);
    c.show(Str(footer.as_bytes()));
    c.end_text();

    pdf.stream(content_id, &c.finish());
    pdf.finish()
}

/// Run the `blank` subcommand.
///
/// # Errors
///
/// Returns a [`phonopaper_rs::PhonoPaperError`] if the output PDF file cannot
/// be written.
pub fn run(args: &BlankArgs) -> phonopaper_rs::Result<()> {
    let bytes = generate_blank_pdf(args);

    std::fs::write(&args.output, &bytes).map_err(phonopaper_rs::PhonoPaperError::IoError)?;

    eprintln!("Wrote {} ({} bytes)", args.output, bytes.len());
    eprintln!(
        "Paper: {:?} | Orientation: {}",
        args.paper,
        if args.portrait {
            "portrait"
        } else {
            "landscape"
        }
    );
    Ok(())
}
