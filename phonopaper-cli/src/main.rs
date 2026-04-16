//! `phonopaper` — command-line tool for the `PhonoPaper` audio format.
//!
//! Converts between `PhonoPaper` images/vectors and audio files (WAV, MP3),
//! and can generate blank printable templates for hand-drawn music.
//!
//! ## Usage
//!
//! **Decode** a `PhonoPaper` image (PNG, JPEG, SVG, or PDF) to a WAV file:
//! ```text
//! phonopaper decode code.png output.wav
//! phonopaper decode code.svg output.wav
//! phonopaper decode code.pdf output.wav
//! ```
//!
//! **Encode** a WAV or MP3 file to a `PhonoPaper` image (PNG, SVG, or PDF):
//! ```text
//! phonopaper encode input.wav code.png
//! phonopaper encode input.mp3 code.png
//! phonopaper encode input.wav code.svg
//! phonopaper encode input.wav code.pdf
//! ```
//!
//! **Decode a camera photo** with perspective correction:
//! ```text
//! phonopaper robust-decode --input photo.jpg --output decoded.wav
//! ```
//!
//! **Generate a blank template** for hand-drawn music:
//! ```text
//! phonopaper blank
//! phonopaper blank --paper a3 --output my_template.pdf
//! ```
//!
//! Run `phonopaper --help` (or `phonopaper <subcommand> --help`) for a full
//! list of options.

mod cmd;

use std::path::Path;

use clap::{Parser, Subcommand};
use color_eyre::eyre::WrapErr as _;

use cmd::{
    blank::BlankArgs, decode::DecodeArgs, encode::EncodeArgs, robust_decode::RobustDecodeArgs,
};

// ─── Top-level CLI ────────────────────────────────────────────────────────────

/// Convert between `PhonoPaper` images/vectors and audio files (WAV, MP3).
///
/// `PhonoPaper` encodes audio as a printable grayscale spectrogram image.
/// White pixels represent silence and black pixels represent maximum amplitude.
/// The image spans 8 octaves (~16.6 Hz – ~4186 Hz) on a logarithmic scale.
///
/// Supported input audio formats for encoding: WAV and MP3.
/// Supported output image formats: PNG, JPEG, SVG, PDF.  JPEG is accepted for
/// both encode and decode, but its lossy compression can introduce artefacts
/// that degrade decode quality; PNG is strongly preferred for any software
/// round-trip.  The output format is inferred from the output file extension.
#[derive(Parser)]
#[command(
    name = "phonopaper",
    version,
    about,
    long_about = None,
    propagate_version = true,
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Decode a `PhonoPaper` image (PNG, JPEG, SVG, or PDF) into a WAV audio file.
    Decode(DecodeArgs),
    /// Encode a WAV or MP3 audio file into a `PhonoPaper` image (PNG, JPEG, SVG, or PDF).
    Encode(EncodeArgs),
    /// Decode a (possibly skewed or perspective-distorted) camera photo of a `PhonoPaper` print.
    ///
    /// Uses per-column marker detection so that keystone distortion, camera
    /// tilt, and paper curl are handled without an explicit de-warp step.
    RobustDecode(RobustDecodeArgs),
    /// Generate a blank, ready-to-print `PhonoPaper` template PDF.
    ///
    /// The output contains the `PhonoPaper` marker bands, a blank white data
    /// area for hand-drawn music, octave separator lines, and octave labels
    /// (C2–C8) on the left margin.
    Blank(BlankArgs),
}

// ─── Format helpers ───────────────────────────────────────────────────────────

/// Infer the output format from a file extension (`"svg"`, `"pdf"`, `"jpeg"`, or `"png"`).
pub(crate) fn output_format(path: &str) -> &'static str {
    match Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
        .as_deref()
    {
        Some("svg") => "svg",
        Some("pdf") => "pdf",
        Some("jpg" | "jpeg") => "jpeg",
        _ => "png",
    }
}

/// Derive an output path from the input path by swapping the extension.
pub(crate) fn default_output(input: &str, ext: &str) -> String {
    let stem = std::path::Path::new(input)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(input);
    let dir = std::path::Path::new(input)
        .parent()
        .and_then(|p| p.to_str())
        .unwrap_or(".");
    if dir == "." {
        format!("{stem}.{ext}")
    } else {
        format!("{dir}/{stem}.{ext}")
    }
}

// ─── Entry point ──────────────────────────────────────────────────────────────

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    let cli = Cli::parse();

    match cli.command {
        Command::Decode(args) => cmd::decode::run(args).wrap_err("decode failed")?,
        Command::Encode(args) => cmd::encode::run(args).wrap_err("encode failed")?,
        Command::RobustDecode(args) => {
            cmd::robust_decode::run(&args).wrap_err("robust-decode failed")?;
        }
        Command::Blank(args) => cmd::blank::run(&args).wrap_err("blank failed")?,
    }

    Ok(())
}
