//! `phonopaper encode` — convert a WAV or MP3 audio file to a `PhonoPaper` image.

use clap::Args;
use phonopaper_rs::{
    encode::{AnalysisOptions, audio_to_spectrogram, encode_audio_to_image},
    render::RenderOptions,
    vector::{PdfPageLayout, page_size, spectrogram_to_pdf, spectrogram_to_svg},
};

use crate::{default_output, output_format};

/// Arguments for the `encode` sub-command.
#[derive(Args)]
pub struct EncodeArgs {
    /// Input audio file (WAV or MP3; any sample rate; mono or stereo).
    ///
    /// Stereo files are downmixed to mono by averaging both channels.
    pub input: String,

    /// Output image file path (PNG, JPEG, SVG, or PDF).
    ///
    /// The output format is inferred from the file extension:
    /// `.png` → PNG image, `.jpg`/`.jpeg` → JPEG image (lossy — may degrade
    /// decode quality; prefer PNG for software round-trips), `.svg` → SVG
    /// vector, `.pdf` → PDF vector.
    /// Defaults to the input filename with its extension replaced by `.png`.
    pub output: Option<String>,

    /// FFT window size in samples (must be a power of two).
    ///
    /// Larger windows give better frequency resolution but coarser time
    /// resolution. 4096 is a good default.
    #[arg(long, default_value_t = 4096, value_name = "N")]
    pub fft_window: usize,

    /// Hop size in samples between successive FFT frames.
    ///
    /// Determines how many image columns are produced per second of audio.
    /// Smaller values yield wider (more detailed) images.
    /// Defaults to `353` — matching Android's column rate of 352.8
    /// (= 44 100 ÷ 125).  At this rate a 10-second clip produces ~1250 columns
    /// and decodes back to 10 seconds at the default `samples-per-column = 353`.
    #[arg(long, default_value_t = 353, value_name = "N")]
    pub hop_size: usize,

    /// Pixels per octave in the data area (controls image height).
    ///
    /// Total data-area height = `px_per_octave` × 8. Default: 90 → 720 px.
    #[arg(long, default_value_t = 90, value_name = "PX")]
    pub px_per_octave: u32,

    /// Gamma correction exponent applied to amplitudes before writing pixels.
    ///
    /// The encoding formula is `pixel = round((1 − amplitude^gamma) × 255)`.
    /// Use `gamma < 1.0` (e.g. `0.5`) to expand quiet amplitudes toward
    /// black, making quiet content more visible and louder on decode.
    /// Use `gamma > 1.0` to compress mid-level amplitudes toward white/silence.
    /// `1.0` (the default) means no correction.
    ///
    /// Must be paired with the same value passed to `--gamma` during decode.
    #[arg(long, default_value_t = 1.0, value_name = "G")]
    pub gamma: f32,

    /// Lower bound of the dB window (maps to amplitude 0.0 / white pixel).
    ///
    /// Defaults to `-60.0` dB.  Bins quieter than this produce a white (silent)
    /// pixel and are not synthesised during decode.  Use a more negative value
    /// (e.g. `-100.0`) only if you need to capture content below −60 dBFS.
    #[arg(long, default_value_t = -60.0, value_name = "DB")]
    pub min_db: f32,

    /// Upper bound of the dB window (maps to amplitude 1.0 / black pixel).
    ///
    /// Defaults to `-10.0` dB.  For a Hann-windowed FFT a full-scale sine
    /// (amplitude 1.0) produces `web_audio_mag = 0.25` (= −12 dBFS), so
    /// `−10.0` leaves 2 dB headroom.
    #[arg(long, default_value_t = -10.0, value_name = "DB")]
    pub max_db: f32,

    /// Draw light-gray octave separator lines in the data area.
    ///
    /// ⚠️  These lines are decoded as amplitude ≈ 0.22, producing a faint hum
    /// at each of the 7 octave-boundary frequencies during playback.  Useful
    /// for visual inspection only; omit for images intended to be decoded as
    /// audio.
    #[arg(long)]
    pub octave_lines: bool,

    /// PDF page size for centred output (only used when output is a PDF).
    ///
    /// The `PhonoPaper` image is scaled uniformly to fit the chosen page size
    /// with 10 mm margins on all sides, and centred both horizontally and
    /// vertically.  Defaults to `a4-landscape`.
    ///
    /// Use `pixel-perfect` to instead produce a PDF whose page dimensions
    /// exactly match the image pixel dimensions (1 px = 1 pt), as in older
    /// versions of this tool.  This is useful for software pipelines that need
    /// the raw image without any page framing.
    #[arg(long, default_value = "a4-landscape", value_name = "SIZE")]
    pub pdf_page: PdfPageArg,
}

/// PDF page size choices for the `--pdf-page` flag.
#[derive(Clone, Debug, clap::ValueEnum)]
pub enum PdfPageArg {
    /// A4 landscape — 297 × 210 mm (default).
    A4Landscape,
    /// A4 portrait — 210 × 297 mm.
    A4Portrait,
    /// A3 landscape — 420 × 297 mm.
    A3Landscape,
    /// A3 portrait — 297 × 420 mm.
    A3Portrait,
    /// US Letter landscape — 11 × 8.5 in.
    LetterLandscape,
    /// US Letter portrait — 8.5 × 11 in.
    LetterPortrait,
    /// No page framing: PDF page = image pixels (1 px = 1 pt).
    PixelPerfect,
}

impl PdfPageArg {
    /// Convert to the library's [`PdfPageLayout`] type.
    ///
    /// All paper sizes use a 10 mm (≈ 28.35 pt) margin on each side.
    #[must_use]
    pub fn to_layout(&self) -> PdfPageLayout {
        /// 10 mm in PDF points (1 pt = 1/72 inch; 1 mm ≈ 2.835 pt).
        const MARGIN_PT: f32 = 28.35;

        let (w, h) = match self {
            Self::A4Landscape => page_size::A4_LANDSCAPE,
            Self::A4Portrait => page_size::A4_PORTRAIT,
            Self::A3Landscape => page_size::A3_LANDSCAPE,
            Self::A3Portrait => page_size::A3_PORTRAIT,
            Self::LetterLandscape => page_size::LETTER_LANDSCAPE,
            Self::LetterPortrait => page_size::LETTER_PORTRAIT,
            Self::PixelPerfect => return PdfPageLayout::PixelPerfect,
        };

        PdfPageLayout::FitToPage {
            page_width_pt: w,
            page_height_pt: h,
            margin_pt: MARGIN_PT,
        }
    }
}

/// Run the `encode` subcommand.
///
/// # Errors
///
/// Returns a [`phonopaper_rs::PhonoPaperError`] if the audio file cannot be
/// read, the STFT fails, or the output image cannot be written.
pub fn run(args: EncodeArgs) -> phonopaper_rs::Result<()> {
    let output = args
        .output
        .unwrap_or_else(|| default_output(&args.input, "png"));

    let analysis = AnalysisOptions {
        fft_window: args.fft_window,
        hop_size: args.hop_size,
        min_db: args.min_db,
        max_db: args.max_db,
    };
    let render = RenderOptions {
        px_per_octave: args.px_per_octave,
        gamma: args.gamma,
        draw_octave_lines: args.octave_lines,
        ..RenderOptions::default()
    };

    let fmt = output_format(&output);

    eprintln!("Encoding  {} → {} ({fmt})", args.input, output);
    eprintln!(
        "  fft window: {}   hop: {}   px/octave: {}   gamma: {:.2}   dB range: [{:.0}, {:.0}]",
        analysis.fft_window,
        analysis.hop_size,
        render.px_per_octave,
        render.gamma,
        analysis.min_db,
        analysis.max_db,
    );

    match fmt {
        "svg" | "pdf" => {
            let (mono, sample_rate) = phonopaper_rs::audio::read_audio_file(&args.input)?;
            let spec = audio_to_spectrogram(&mono, sample_rate, &analysis)?;

            if fmt == "svg" {
                let svg = spectrogram_to_svg(&spec, &render);
                std::fs::write(&output, svg.as_bytes())
                    .map_err(phonopaper_rs::PhonoPaperError::IoError)?;
            } else {
                let layout = args.pdf_page.to_layout();
                let pdf = spectrogram_to_pdf(&spec, &render, layout);
                std::fs::write(&output, &pdf).map_err(phonopaper_rs::PhonoPaperError::IoError)?;
            }
        }
        _ => encode_audio_to_image(&args.input, &output, analysis, render)?,
    }

    eprintln!("Done.");
    Ok(())
}
