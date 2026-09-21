//! Loss function and training loop.

use std::path::Path;

use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::activation::log_sigmoid;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::LossMetric;
use burn::train::{
    InferenceStep, Learner, RegressionOutput, SupervisedTraining, TrainOutput, TrainStep,
};
use serde::{Deserialize, Serialize};

use crate::data::{DetectionBatch, DetectionBatcher, Split, count_positives, load_split};
use crate::model::{Detector, DetectorConfig};

/// File name of the training checkpoint (`MessagePack`, full precision).
pub const CHECKPOINT_FILE: &str = "model.mpk";
/// File name of the exported weights for embedding (`BinFileRecorder`).
pub const EXPORT_FILE: &str = "model.bin";
/// File name of the model hyper-parameters.
pub const MODEL_CONFIG_FILE: &str = "model.json";
/// File name of the training hyper-parameters.
pub const TRAIN_CONFIG_FILE: &str = "training.json";

/// Weight of the corner regression term relative to the presence term.
///
/// Corner errors are measured in normalised units (an error of `0.01` is
/// about one pixel on a 128 px input), so they need a large weight to matter
/// as much as the classification term.
const CORNER_LOSS_WEIGHT: f64 = 20.0;

/// Transition point of the smooth-L1 (Huber) corner loss, in normalised units.
const HUBER_DELTA: f64 = 0.05;

/// Training hyper-parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Network hyper-parameters.
    pub model: DetectorConfig,
    /// Optimiser settings.
    pub optimizer: AdamConfig,
    /// Number of passes over the training split.
    pub num_epochs: usize,
    /// Mini-batch size.
    pub batch_size: usize,
    /// Data-loading worker threads.
    pub num_workers: usize,
    /// Seed for weight initialisation and shuffling.
    pub seed: u64,
    /// Adam learning rate.
    pub learning_rate: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model: DetectorConfig::new(),
            optimizer: AdamConfig::new(),
            num_epochs: 30,
            batch_size: 32,
            num_workers: 4,
            seed: 42,
            learning_rate: 1e-3,
        }
    }
}

/// Compute the training loss.
///
/// * presence: binary cross-entropy with logits, averaged over the batch;
/// * corners: smooth-L1 between predicted and true normalised coordinates,
///   averaged over the positive samples only (there is no meaningful corner
///   target for negatives).
pub fn detection_loss<B: Backend>(output: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
    let logits = output.clone().narrow(1, 0, 1);
    let present = targets.clone().narrow(1, 0, 1);
    let absent = present.clone().neg().add_scalar(1.0);
    let bce = (present.clone() * log_sigmoid(logits.clone()) + absent * log_sigmoid(logits.neg()))
        .neg()
        .mean();

    let diff = output.narrow(1, 1, 8) - targets.narrow(1, 1, 8);
    let abs = diff.abs();
    let quadratic = abs.clone().clamp_max(HUBER_DELTA);
    let huber = quadratic.clone().powi_scalar(2).mul_scalar(0.5)
        + (abs - quadratic).mul_scalar(HUBER_DELTA);
    // Per-sample mean over the 8 coordinates, then mask negatives.
    let per_sample = huber.mean_dim(1) * present.clone();
    let positives = present.sum().clamp_min(1.0);
    let corner = per_sample.sum() / positives;

    bce + corner.mul_scalar(CORNER_LOSS_WEIGHT)
}

impl<B: Backend> Detector<B> {
    /// Forward pass plus loss, packaged for burn's metrics.
    pub fn forward_regression(&self, batch: DetectionBatch<B>) -> RegressionOutput<B> {
        let output = self.forward(batch.images);
        let loss = detection_loss(output.clone(), batch.targets.clone());
        RegressionOutput {
            loss,
            output,
            targets: batch.targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep for Detector<B> {
    type Input = DetectionBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: DetectionBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(batch);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for Detector<B> {
    type Input = DetectionBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: DetectionBatch<B>) -> RegressionOutput<B> {
        self.forward_regression(batch)
    }
}

/// Train a detector and write the artifacts.
///
/// `artifact_dir` receives burn's logs and checkpoints plus:
/// `model.json` (architecture), `training.json` (hyper-parameters),
/// `model.mpk` (checkpoint) and `model.bin` (weights to embed).
///
/// # Errors
///
/// Returns a message when the dataset cannot be loaded, when its image size
/// does not match `config.model.input_size`, or when artifacts cannot be
/// written.
pub fn train<B: AutodiffBackend>(
    dataset_dir: &Path,
    artifact_dir: &Path,
    config: &TrainingConfig,
    device: &B::Device,
) -> Result<(), String> {
    std::fs::create_dir_all(artifact_dir).map_err(|e| e.to_string())?;
    let artifact_str = artifact_dir
        .to_str()
        .ok_or("artifact directory path is not valid UTF-8")?;
    config
        .model
        .save(artifact_dir.join(MODEL_CONFIG_FILE))
        .map_err(|e| e.to_string())?;
    let training_json = serde_json::to_string_pretty(config).map_err(|e| e.to_string())?;
    std::fs::write(artifact_dir.join(TRAIN_CONFIG_FILE), training_json + "\n")
        .map_err(|e| e.to_string())?;

    B::seed(device, config.seed);

    let (train_set, size) = load_split(dataset_dir, Split::Train)?;
    let (valid_set, _) = load_split(dataset_dir, Split::Valid)?;
    if size != config.model.input_size {
        return Err(format!(
            "dataset images are {size} px but the model expects {} px; \
             pass --input-size {size} or regenerate the dataset with --size {}",
            config.model.input_size, config.model.input_size
        ));
    }
    println!(
        "train: {} images ({} positive); valid: {} images ({} positive)",
        train_set.len(),
        count_positives(&train_set),
        valid_set.len(),
        count_positives(&valid_set)
    );

    let dataloader_train = DataLoaderBuilder::new(DetectionBatcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_set);
    let dataloader_valid = DataLoaderBuilder::new(DetectionBatcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(valid_set);

    let training = SupervisedTraining::new(artifact_str, dataloader_train, dataloader_valid)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(NamedMpkFileRecorder::<FullPrecisionSettings>::new())
        .num_epochs(config.num_epochs)
        .summary();

    let model = config.model.init::<B>(device);
    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        config.learning_rate,
    ));

    let trained = result.model;
    trained
        .clone()
        .save_file(
            artifact_dir.join(CHECKPOINT_FILE),
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
        )
        .map_err(|e| e.to_string())?;
    trained
        .save_file(
            artifact_dir.join(EXPORT_FILE),
            &BinFileRecorder::<FullPrecisionSettings>::new(),
        )
        .map_err(|e| e.to_string())?;
    println!(
        "saved {} and {} in {}",
        CHECKPOINT_FILE,
        EXPORT_FILE,
        artifact_dir.display()
    );
    Ok(())
}

/// Load a trained detector from an artifact directory (`model.json` +
/// `model.bin`).
///
/// # Errors
///
/// Returns a message when either file is missing or invalid.
pub fn load_trained<B: Backend>(
    artifact_dir: &Path,
    device: &B::Device,
) -> Result<Detector<B>, String> {
    let model_config =
        DetectorConfig::load(artifact_dir.join(MODEL_CONFIG_FILE)).map_err(|e| e.to_string())?;
    model_config
        .init::<B>(device)
        .load_file(
            artifact_dir.join(EXPORT_FILE),
            &BinFileRecorder::<FullPrecisionSettings>::new(),
            device,
        )
        .map_err(|e| e.to_string())
}
