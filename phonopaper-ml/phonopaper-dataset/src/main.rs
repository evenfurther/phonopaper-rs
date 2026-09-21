//! Command-line front end of the deterministic `PhonoPaper` dataset generator.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use phonopaper_dataset::{GeneratorConfig, generate_dataset};

/// Generate a deterministic synthetic dataset of `PhonoPaper` photographs.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// Output directory (created if missing).
    #[arg(short, long, default_value = "dataset")]
    output: PathBuf,
    /// Number of images to generate.
    #[arg(short = 'n', long, default_value_t = GeneratorConfig::default().count)]
    count: u64,
    /// Side of the square images, in pixels.
    #[arg(long, default_value_t = GeneratorConfig::default().size)]
    size: u32,
    /// Master seed; the dataset is a pure function of the seed and options.
    #[arg(long, default_value_t = GeneratorConfig::default().seed)]
    seed: u64,
    /// Probability that an image contains a pattern.
    #[arg(long, default_value_t = GeneratorConfig::default().positive_ratio)]
    positive_ratio: f64,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if cli.size < 16 {
        eprintln!("error: --size must be at least 16");
        return ExitCode::FAILURE;
    }
    if !(0.0..=1.0).contains(&cli.positive_ratio) {
        eprintln!("error: --positive-ratio must be in [0, 1]");
        return ExitCode::FAILURE;
    }
    let cfg = GeneratorConfig {
        count: cli.count,
        size: cli.size,
        seed: cli.seed,
        positive_ratio: cli.positive_ratio,
    };
    match generate_dataset(&cfg, &cli.output) {
        Ok(manifest) => {
            println!(
                "wrote {} images ({} positive, {} negative) to {}",
                cfg.count,
                manifest.positives,
                manifest.negatives,
                cli.output.display()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
