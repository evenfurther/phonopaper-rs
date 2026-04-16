//! `phonopaper decode` — convert a `PhonoPaper` image to a WAV audio file.

use std::path::Path;

use clap::Args;
use phonopaper_rs::{
    decode::{AmplitudeMode, SynthesisOptions, image_to_spectrogram, write_wav},
    vector::{image_from_pdf, image_from_svg},
};

use crate::cmd::{dispatch_sps, dispatch_sps_synth};
use crate::default_output;

/// Arguments for the `decode` sub-command.
#[derive(Args)]
pub struct DecodeArgs {
    /// Input `PhonoPaper` image file (PNG, JPEG, SVG, or PDF).
    pub input: String,

    /// Output WAV file path.
    ///
    /// Defaults to the input filename with its extension replaced by `.wav`.
    pub output: Option<String>,

    /// Number of audio samples to synthesise per image column.
    ///
    /// Controls playback speed and synthesis resolution.  At 44 100 Hz:
    ///
    /// - `353` (default) → matches Android's exact rate of 352.8 (= 44 100 ÷ 125)
    /// - `512` → each column is ~11.6 ms (higher synthesis resolution, slower playback)
    /// - `256` → each column is ~5.8 ms (faster playback)
    ///
    /// The value is used directly; no rounding occurs.
    #[arg(long, default_value_t = 353, value_name = "N")]
    pub samples_per_column: usize,

    /// Output audio sample rate in Hz.
    #[arg(long, default_value_t = 44_100, value_name = "HZ")]
    pub sample_rate: u32,

    /// Master output gain (linear).
    ///
    /// Scales the synthesized waveform before writing to WAV.  With dB
    /// inversion active (the default), `gain = 4.0` would give unity amplitude
    /// for a perfectly-aligned single FFT bin, but in practice `PhonoPaper` bins
    /// overlap (≈1.5 PP bins share each FFT bin), boosting output by ≈1.22×.
    /// The default `3.0` compensates to give approximately unity round-trip
    /// amplitude on typical music content.  Lower this if you hear distortion.
    #[arg(long, default_value_t = 3.0, value_name = "GAIN")]
    pub gain: f32,

    /// Binary amplitude threshold (e.g. 0.85 for Android-like decoding).
    ///
    /// When set, the decoder uses [`AmplitudeMode::Linear`] and treats each
    /// pixel as either fully on (amplitude ≥ threshold) or silent (amplitude
    /// < threshold).  Empirical testing found that `~0.85` gives the best
    /// spectral fidelity against the Android reference app (see
    /// `PHONOPAPER_SPEC.md §6.2`).
    ///
    /// If neither `--amplitude-threshold` nor `--linear` is given, the
    /// decoder defaults to [`AmplitudeMode::Db`] with the configured dB window.
    ///
    /// Default: disabled.
    #[arg(long, value_name = "T", conflicts_with = "linear")]
    pub amplitude_threshold: Option<f32>,

    /// Use linear (non-dB) amplitude mode with fractional pixel values.
    ///
    /// Selects [`AmplitudeMode::Linear`] with no threshold.  Use this for
    /// images that were encoded without a dB window (linear normalisation).
    /// Ignored if `--amplitude-threshold` is set.
    #[arg(long)]
    pub linear: bool,

    /// Inverse gamma correction for pixel→amplitude decoding.
    ///
    /// Must match the `--gamma` value used when the image was encoded.
    /// The decoding formula is `amplitude = (1 − pixel/255) ^ (1/gamma)`.
    ///
    /// Default: `1.0` (linear, no correction).  Use `0.15` to decode images
    /// that were encoded with `--gamma 0.15`.
    #[arg(long, default_value_t = 1.0, value_name = "G")]
    pub gamma: f32,

    /// Lower bound of the dB window used when the image was encoded.
    ///
    /// Must match the `--min-db` value used during encoding.
    /// Defaults to `-60.0` dB (matching the encoder default).
    /// Ignored when `--amplitude-threshold` or `--linear` is set.
    #[arg(long, default_value_t = -60.0, value_name = "DB")]
    pub min_db: f32,

    /// Upper bound of the dB window used when the image was encoded.
    ///
    /// Must match the `--max-db` value used during encoding.
    /// Defaults to `-10.0` dB (matching the encoder default).
    /// Ignored when `--amplitude-threshold` or `--linear` is set.
    #[arg(long, default_value_t = -10.0, value_name = "DB")]
    pub max_db: f32,
}

/// Extract a [`image::DynamicImage`] from the input file, handling PNG/JPEG,
/// SVG, and PDF.
///
/// For SVG and PDF inputs the embedded raster image is extracted.  PNG, JPEG,
/// and any other format recognised by the `image` crate are opened directly.
///
/// # Errors
///
/// Returns a [`phonopaper_rs::PhonoPaperError`] if the file cannot be read or
/// the image cannot be decoded.
pub fn load_phonopaper_image(path: &str) -> phonopaper_rs::Result<image::DynamicImage> {
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase);

    match ext.as_deref() {
        Some("svg") => {
            let svg =
                std::fs::read_to_string(path).map_err(phonopaper_rs::PhonoPaperError::IoError)?;
            image_from_svg(&svg)
        }
        Some("pdf") => {
            let pdf = std::fs::read(path).map_err(phonopaper_rs::PhonoPaperError::IoError)?;
            image_from_pdf(&pdf)
        }
        _ => image::open(path).map_err(phonopaper_rs::PhonoPaperError::ImageError),
    }
}

/// Run the `decode` subcommand.
///
/// # Errors
///
/// Returns a [`phonopaper_rs::PhonoPaperError`] if the image cannot be loaded,
/// the marker bands are not found, or the output WAV file cannot be written.
pub fn run(args: DecodeArgs) -> phonopaper_rs::Result<()> {
    let output = args
        .output
        .unwrap_or_else(|| default_output(&args.input, "wav"));

    let mode = if let Some(t) = args.amplitude_threshold {
        AmplitudeMode::Linear { threshold: Some(t) }
    } else if args.linear {
        AmplitudeMode::Linear { threshold: None }
    } else {
        AmplitudeMode::Db {
            min_db: args.min_db,
            max_db: args.max_db,
        }
    };

    let options = SynthesisOptions {
        sample_rate: args.sample_rate,
        gain: args.gain,
        decode_gamma: args.gamma,
        mode,
    };

    let ext = Path::new(&args.input)
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase);
    let fmt = match ext.as_deref() {
        Some("svg") => "svg",
        Some("pdf") => "pdf",
        _ => "png/jpeg",
    };

    eprintln!("Decoding  {} ({fmt}) → {}", args.input, output);
    eprintln!(
        "  samples/column: {}   sample rate: {} Hz   gain: {:.3}   gamma: {:.3}   mode: {:?}",
        args.samples_per_column,
        options.sample_rate,
        options.gain,
        options.decode_gamma,
        options.mode,
    );

    // For SVG and PDF inputs, extract the embedded raster image into memory and
    // run the full decode pipeline without touching the file system.
    // For PNG/JPEG use the existing file-based pipeline directly.
    match ext.as_deref() {
        Some("svg" | "pdf") => {
            let img = load_phonopaper_image(&args.input)?;
            let spec = image_to_spectrogram(&img, None)?;
            let num_samples = spec.num_columns() * args.samples_per_column;
            let mut samples = vec![0.0_f32; num_samples];
            dispatch_sps_synth!(args.samples_per_column, &spec, &options, &mut samples);
            write_wav(&output, &samples, options.sample_rate)?;
        }
        _ => {
            dispatch_sps!(args.samples_per_column, &args.input, &output, options)?;
        }
    }

    eprintln!("Done.");
    Ok(())
}
