//! Command-line front end: `train`, `eval` and `infer`.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use phonopaper_train::model::DetectorConfig;
use phonopaper_train::training::{
    TrainingConfig, export_checkpoint, load_trained, train, validation_losses,
};
use phonopaper_train::{eval, infer};

// ─── Backend selection ────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
mod backend {
    pub type Inference = burn::backend::Cuda;
    pub fn device() -> burn::backend::cuda::CudaDevice {
        burn::backend::cuda::CudaDevice::default()
    }
    pub const NAME: &str = "cuda";
}

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
mod backend {
    pub type Inference = burn::backend::Wgpu;
    pub fn device() -> burn::backend::wgpu::WgpuDevice {
        burn::backend::wgpu::WgpuDevice::default()
    }
    pub const NAME: &str = "wgpu";
}

#[cfg(all(feature = "flex", not(any(feature = "wgpu", feature = "cuda"))))]
mod backend {
    pub type Inference = burn::backend::Flex;
    pub fn device() -> burn::backend::flex::FlexDevice {
        burn::backend::flex::FlexDevice
    }
    pub const NAME: &str = "flex (CPU)";
}

#[cfg(not(any(feature = "flex", feature = "wgpu", feature = "cuda")))]
mod backend {
    pub type Inference = burn::backend::NdArray;
    pub fn device() -> burn::backend::ndarray::NdArrayDevice {
        burn::backend::ndarray::NdArrayDevice::Cpu
    }
    pub const NAME: &str = "ndarray (CPU)";
}

type Training = burn::backend::Autodiff<backend::Inference>;

// ─── CLI ──────────────────────────────────────────────────────────────────────

/// Train and use the `PhonoPaper` pattern detector.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Train a new detector on a dataset produced by `phonopaper-dataset`.
    Train {
        /// Dataset directory (contains `labels.csv` and `manifest.json`).
        #[arg(short, long, default_value = "dataset")]
        dataset: PathBuf,
        /// Where to write logs, checkpoints and the exported model.
        #[arg(short, long, default_value = "artifacts")]
        artifacts: PathBuf,
        /// Number of epochs.
        #[arg(long, default_value_t = 30)]
        epochs: usize,
        /// Mini-batch size.
        #[arg(long, default_value_t = 32)]
        batch_size: usize,
        /// Adam learning rate.
        #[arg(long, default_value_t = 1e-3)]
        learning_rate: f64,
        /// Seed for weight initialisation and shuffling.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Data-loader worker threads.
        #[arg(long, default_value_t = 4)]
        workers: usize,
        /// Network input side in pixels; must match the dataset `--size`.
        #[arg(long, default_value_t = 128)]
        input_size: usize,
        /// Stop early when the validation loss has not improved for this
        /// many epochs.
        #[arg(long, default_value_t = 8)]
        patience: usize,
    },
    /// Convert a training checkpoint into `model.bin` (on the CPU).
    ///
    /// Use this to recover a model when training was interrupted, or to pick
    /// a specific epoch.  Without `--epoch`, the epoch with the lowest mean
    /// validation loss is used.
    Export {
        /// Artifact directory of a `train` run.
        #[arg(short, long, default_value = "artifacts")]
        artifacts: PathBuf,
        /// Epoch to export; must still exist in `<artifacts>/checkpoint/`.
        #[arg(long)]
        epoch: Option<usize>,
    },
    /// Evaluate a trained detector on the validation split of a dataset.
    Eval {
        /// Dataset directory.
        #[arg(short, long, default_value = "dataset")]
        dataset: PathBuf,
        /// Artifact directory containing `model.json` and `model.bin`.
        #[arg(short, long, default_value = "artifacts")]
        artifacts: PathBuf,
        /// Mini-batch size.
        #[arg(long, default_value_t = 64)]
        batch_size: usize,
    },
    /// Run a trained detector on image files and print JSON results.
    Infer {
        /// Artifact directory containing `model.json` and `model.bin`.
        #[arg(short, long, default_value = "artifacts")]
        artifacts: PathBuf,
        /// Presence probability threshold.
        #[arg(long, default_value_t = 0.5)]
        threshold: f32,
        /// Image files (any format supported by the `image` crate build).
        #[arg(required = true)]
        images: Vec<PathBuf>,
    },
}

fn run(cli: Cli) -> Result<(), String> {
    let device = backend::device();
    eprintln!("backend: {}", backend::NAME);
    match cli.command {
        Command::Train {
            dataset,
            artifacts,
            epochs,
            batch_size,
            learning_rate,
            seed,
            workers,
            input_size,
            patience,
        } => {
            let config = TrainingConfig {
                model: DetectorConfig::new().with_input_size(input_size),
                num_epochs: epochs,
                batch_size,
                learning_rate,
                seed,
                num_workers: workers,
                patience,
                ..TrainingConfig::default()
            };
            train::<Training>(&dataset, &artifacts, &config, &device)
        }
        Command::Export { artifacts, epoch } => {
            for (e, loss) in validation_losses(&artifacts).unwrap_or_default() {
                println!("epoch {e:>3}: mean validation loss {loss:.4}");
            }
            export_checkpoint(&artifacts, epoch)
        }
        Command::Eval {
            dataset,
            artifacts,
            batch_size,
        } => {
            let model = load_trained::<backend::Inference>(&artifacts, &device)?;
            let metrics = eval::evaluate(&model, &dataset, batch_size, &device)?;
            println!("{metrics}");
            Ok(())
        }
        Command::Infer {
            artifacts,
            threshold,
            images,
        } => {
            let model = load_trained::<backend::Inference>(&artifacts, &device)?;
            for path in &images {
                let det = infer::infer_file(&model, path, threshold, &device)?;
                let json = serde_json::to_string(&det).map_err(|e| e.to_string())?;
                println!("{json}");
            }
            Ok(())
        }
    }
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
