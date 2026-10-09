//! Loss function and training loop.

use std::path::Path;

use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::tensor::Device;
use burn::tensor::activation::{log_sigmoid, log_softmax};
use burn::train::checkpoint::KeepLastNCheckpoints;
use burn::train::metric::LossMetric;
use burn::train::metric::store::{Aggregate, Direction, Split as MetricSplit};
use burn::train::{
    InferenceStep, Learner, MetricEarlyStoppingStrategy, RegressionOutput, StoppingCondition,
    SupervisedTraining, TrainOutput, TrainStep,
};
use serde::{Deserialize, Serialize};

use crate::data::{
    DetectionBatch, DetectionBatcher, HostBatch, Split, count_positives, load_split,
};
use crate::model::{Detector, DetectorConfig, DetectorOutput, HEATMAP_GRID_MAX, HEATMAP_GRID_MIN};

/// File name of the exported Burnpack model artifact.
pub const CHECKPOINT_FILE: &str = "model.bpk";
/// File name of the exported weights for embedding.
pub const EXPORT_FILE: &str = CHECKPOINT_FILE;
/// File name of the model hyper-parameters.
pub const MODEL_CONFIG_FILE: &str = "model.json";
/// File name of the training hyper-parameters.
pub const TRAIN_CONFIG_FILE: &str = "training.json";

/// Weight of the corner regression term relative to the presence term.
///
/// Corner errors are measured in normalised units (an error of `0.01` is
/// about one pixel on a 128 px input), so they need a large weight to matter
/// as much as the classification term.
pub const CORNER_LOSS_WEIGHT: f64 = 20.0;

/// Weight of direct heat-map supervision relative to presence classification.
///
/// Unit-normalized Gaussian targets make this a cross-entropy in natural-log
/// units, so one keeps it comparable to the presence binary cross-entropy.
pub const HEATMAP_LOSS_WEIGHT: f64 = 1.0;

/// Standard deviation of target Gaussians in normalized image coordinates.
///
/// At the default 128-pixel input this is 2.56 pixels, broad enough to provide
/// gradients around a corner while still rewarding a sharply localized peak.
pub const HEATMAP_SIGMA: f32 = 0.02;

/// Transition point of the smooth-L1 (Huber) corner loss, in normalised units
/// (`0.01` ≈ 1.3 px on a 128 px input).
///
/// Below δ the loss is quadratic and its gradient fades, so δ is effectively
/// the precision the network stops caring about: with δ = 0.05 the mean
/// corner error plateaued at exactly 6.4 px = 0.05 × 128.
pub const HUBER_DELTA: f64 = 0.01;

/// Training hyper-parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
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
    /// Stop when the validation loss has not improved for this many epochs.
    pub patience: usize,
    /// Final learning rate of the cosine schedule, as a fraction of
    /// `learning_rate`.  The rate decays from `learning_rate` to
    /// `learning_rate × min_lr_fraction` over the `num_epochs` epochs.
    pub min_lr_fraction: f64,
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
            patience: 8,
            min_lr_fraction: 0.05,
        }
    }
}

/// Compute the training loss from one trunk pass.
///
/// The objective combines presence binary cross-entropy, direct normalized
/// Gaussian heat-map cross-entropy, and smooth-L1 decoded-coordinate loss.
/// Spatial terms are averaged over positive samples only. Both spatial terms
/// take the minimum over direct corner order and a two-channel rotation, since
/// a `PhonoPaper` sheet is indistinguishable after a 180° turn.
#[must_use]
pub fn detection_loss(prediction: DetectorOutput, targets: Tensor<2>) -> Tensor<1> {
    let logits = prediction.output.clone().narrow(1, 0, 1);
    let present = targets.clone().narrow(1, 0, 1);
    let absent = present.clone().neg().add_scalar(1.0);
    let bce = (present.clone() * log_sigmoid(logits.clone()) + absent * log_sigmoid(logits.neg()))
        .neg()
        .mean();

    let predicted = prediction.output.narrow(1, 1, 8);
    let truth = targets.clone().narrow(1, 1, 8);
    let direct = corner_huber(predicted.clone(), truth.clone());
    let rotated = corner_huber(predicted, rotate_180(truth));
    let positives = present.clone().sum().clamp_min(1.0);
    let corner = (direct.min_pair(rotated) * present.clone()).sum() / positives.clone();
    let heatmap = heatmap_loss(prediction.heatmaps, targets);

    bce + corner.mul_scalar(CORNER_LOSS_WEIGHT) + heatmap.mul_scalar(HEATMAP_LOSS_WEIGHT)
}

/// Gaussian cross-entropy for raw `[batch, 4, h, w]` corner heat-map logits.
///
/// Target Gaussians are normalized independently per corner. Negative samples
/// are masked, and an all-negative batch returns a finite zero heat-map loss.
/// A two-channel rotation represents the equivalent 180° corner ordering.
///
/// # Panics
///
/// Panics unless `heatmaps` has exactly four corner channels, or if `targets`
/// cannot be read as `f32` values.
#[must_use]
pub fn heatmap_loss(heatmaps: Tensor<4>, targets: Tensor<2>) -> Tensor<1> {
    let [batch, corners, h, w] = heatmaps.dims();
    assert_eq!(corners, 4, "expected four corner heat-map channels");
    let device = heatmaps.device();
    let target_values: Vec<f32> = targets
        .clone()
        .narrow(1, 1, 8)
        .into_data()
        .try_into_vec()
        .expect("f32 corner targets");
    let mut maps = vec![0.0_f32; batch * corners * h * w];
    for sample in 0..batch {
        for corner in 0..corners {
            let x = target_values[sample * 8 + corner * 2];
            let y = target_values[sample * 8 + corner * 2 + 1];
            let offset = (sample * corners + corner) * h * w;
            let mut sum = 0.0;
            for row in 0..h {
                for col in 0..w {
                    #[expect(clippy::cast_precision_loss, reason = "heat-map dimensions are small")]
                    let gx = HEATMAP_GRID_MIN
                        + (HEATMAP_GRID_MAX - HEATMAP_GRID_MIN) * (col as f32 + 0.5) / w as f32;
                    #[expect(clippy::cast_precision_loss, reason = "heat-map dimensions are small")]
                    let gy = HEATMAP_GRID_MIN
                        + (HEATMAP_GRID_MAX - HEATMAP_GRID_MIN) * (row as f32 + 0.5) / h as f32;
                    let value = (-((gx - x).powi(2) + (gy - y).powi(2))
                        / (2.0 * HEATMAP_SIGMA.powi(2)))
                    .exp();
                    maps[offset + row * w + col] = value;
                    sum += value;
                }
            }
            for value in &mut maps[offset..offset + h * w] {
                *value /= sum;
            }
        }
    }
    let target_maps =
        Tensor::<1>::from_floats(maps.as_slice(), &device).reshape([batch, corners, h * w]);
    let log_probabilities = log_softmax(heatmaps.reshape([batch, corners, h * w]), 2);
    let direct = (target_maps.clone() * log_probabilities.clone())
        .sum_dim(2)
        .mean_dim(1)
        .neg()
        .reshape([batch, 1]);
    let rotated_target = Tensor::cat(
        vec![
            target_maps.clone().narrow(1, 2, 2),
            target_maps.narrow(1, 0, 2),
        ],
        1,
    );
    let rotated = (rotated_target * log_probabilities)
        .sum_dim(2)
        .mean_dim(1)
        .neg()
        .reshape([batch, 1]);
    let present = targets.narrow(1, 0, 1);
    let positives = present.clone().sum().clamp_min(1.0);
    (direct.min_pair(rotated) * present).sum() / positives
}

/// Per-sample mean smooth-L1 between two `[batch, 8]` corner tensors →
/// `[batch, 1]`.
fn corner_huber(predicted: Tensor<2>, truth: Tensor<2>) -> Tensor<2> {
    let abs = (predicted - truth).abs();
    let quadratic = abs.clone().clamp_max(HUBER_DELTA);
    let huber = quadratic.clone().powi_scalar(2).mul_scalar(0.5)
        + (abs - quadratic).mul_scalar(HUBER_DELTA);
    huber.mean_dim(1)
}

/// Reorder `[batch, 8]` corners `TL, TR, BR, BL` into `BR, BL, TL, TR` — the
/// same quadrilateral seen from a sheet turned by 180°.
#[must_use]
pub fn rotate_180(corners: Tensor<2>) -> Tensor<2> {
    let first_half = corners.clone().narrow(1, 0, 4);
    let second_half = corners.narrow(1, 4, 4);
    Tensor::cat(vec![second_half, first_half], 1)
}

impl Detector {
    /// Forward pass plus loss, packaged for burn's metrics.
    #[must_use]
    pub fn forward_regression(&self, batch: DetectionBatch) -> RegressionOutput {
        let prediction = self.forward_with_heatmaps(batch.images);
        let output = prediction.output.clone();
        let loss = detection_loss(prediction, batch.targets.clone());
        RegressionOutput {
            loss,
            output,
            targets: batch.targets,
        }
    }

    /// Device the model's parameters live on.
    fn device(&self) -> Device {
        self.devices().into_iter().next().unwrap_or_default()
    }
}

impl TrainStep for Detector {
    type Input = HostBatch;
    type Output = RegressionOutput;

    fn step(&self, batch: HostBatch) -> TrainOutput<RegressionOutput> {
        let item = self.forward_regression(batch.to_device(&self.device()));
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl InferenceStep for Detector {
    type Input = HostBatch;
    type Output = RegressionOutput;

    fn step(&self, batch: HostBatch) -> RegressionOutput {
        self.forward_regression(batch.to_device(&self.device()))
    }
}

/// Train a detector and write the artifacts.
///
/// `artifact_dir` receives burn's logs and checkpoints plus:
/// `model.json` (architecture), `training.json` (hyper-parameters),
/// `model.bpk` (checkpoint and weights to embed).
///
/// # Errors
///
/// Returns a message when the dataset cannot be loaded, when its image size
/// does not match `config.model.input_size`, or when artifacts cannot be
/// written.
pub fn train(
    dataset_dir: &Path,
    artifact_dir: &Path,
    config: &TrainingConfig,
    device: &Device,
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

    device.seed(config.seed);
    let training_device = device.clone().autodiff();

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
    let train_len = train_set.len();

    let dataloader_train = DataLoaderBuilder::new(DetectionBatcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .set_device(device.clone())
        .build(train_set);
    let dataloader_valid = DataLoaderBuilder::new(DetectionBatcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .set_device(device.clone())
        .build(valid_set);

    let training = SupervisedTraining::new(artifact_str, dataloader_train, dataloader_valid)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_default_checkpointers()
        // Keep every checkpoint while training runs: burn's metric-based
        // strategy only saves an epoch that is already the best when the
        // checkpoint decision is made and can never rescue it later, which
        // lost the best epoch in practice.  The best epoch is exported from
        // the metric logs below and the rest is pruned afterwards.
        .with_checkpointing_strategy(KeepLastNCheckpoints::new(config.num_epochs.max(1)))
        .early_stopping(MetricEarlyStoppingStrategy::new(
            &LossMetric::new(),
            Aggregate::Mean,
            Direction::Lowest,
            MetricSplit::Valid,
            StoppingCondition::NoImprovementSince {
                n_epochs: config.patience,
            },
        ))
        .num_epochs(config.num_epochs)
        .summary();

    let model = config.model.init(&training_device);
    // Cosine decay over the whole run (one scheduler step per iteration).
    let iterations_per_epoch = train_len.div_ceil(config.batch_size.max(1));
    let scheduler = CosineAnnealingLrSchedulerConfig::new(
        config.learning_rate,
        (config.num_epochs * iterations_per_epoch).max(1),
    )
    .with_min_lr(config.learning_rate * config.min_lr_fraction)
    .init()
    .map_err(|e| format!("learning-rate schedule: {e}"))?;
    // The trained model returned here lives on the training backend; we do
    // not read it back (GPU read-back has proven fragile).  The checkpoints
    // on disk are the source of truth for the export below.
    let result = training.launch(Learner::new(model, config.optimizer.init(), scheduler));
    if let Some(error) = result.error {
        return Err(error.to_string());
    }

    let best = best_epoch(artifact_dir)?;
    println!("best validation loss at epoch {best}; exporting it");
    export_checkpoint(artifact_dir, Some(best))?;

    // Free disk space: keep the best and the most recent checkpoint only.
    let last = validation_losses(artifact_dir)?
        .last()
        .map_or(best, |&(epoch, _)| epoch);
    let removed = prune_checkpoints(artifact_dir, &[best, last])?;
    if removed > 0 {
        if best == last {
            println!("pruned {removed} checkpoint files (kept epoch {best})");
        } else {
            println!("pruned {removed} checkpoint files (kept epochs {best} and {last})");
        }
    }
    Ok(())
}

/// Delete checkpoint files of every epoch not listed in `keep`.
///
/// Returns the number of files removed.
///
/// # Errors
///
/// Returns a message when a file cannot be removed.
pub fn prune_checkpoints(artifact_dir: &Path, keep: &[usize]) -> Result<usize, String> {
    let dir = artifact_dir.join("checkpoint");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Ok(0);
    };
    let mut removed = 0;
    for entry in entries.filter_map(Result::ok) {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        // Files are named `<kind>-<epoch>.bpk` (model, optim, scheduler…).
        let Some(epoch) = name
            .strip_suffix(".bpk")
            .and_then(|stem| stem.rsplit_once('-'))
            .and_then(|(_, epoch)| epoch.parse::<usize>().ok())
        else {
            continue;
        };
        if !keep.contains(&epoch) {
            std::fs::remove_file(entry.path())
                .map_err(|e| format!("{}: {e}", entry.path().display()))?;
            removed += 1;
        }
    }
    Ok(removed)
}

/// Mean validation loss per epoch, read from burn's metric logs
/// (`<artifacts>/valid/epoch-N/Loss.log`).
///
/// # Errors
///
/// Returns a message when no validation log can be found.
pub fn validation_losses(artifact_dir: &Path) -> Result<Vec<(usize, f64)>, String> {
    let valid_dir = artifact_dir.join("valid");
    let entries =
        std::fs::read_dir(&valid_dir).map_err(|e| format!("{}: {e}", valid_dir.display()))?;
    let mut losses = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name();
        let Some(epoch) = name
            .to_str()
            .and_then(|n| n.strip_prefix("epoch-"))
            .and_then(|n| n.parse::<usize>().ok())
        else {
            continue;
        };
        let log = entry.path().join("Loss.log");
        let Ok(text) = std::fs::read_to_string(&log) else {
            continue;
        };
        let values: Vec<f64> = text
            .lines()
            .filter_map(|l| l.split(',').next()?.trim().parse().ok())
            .collect();
        if !values.is_empty() {
            #[expect(clippy::cast_precision_loss, reason = "batch counts are small")]
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            losses.push((epoch, mean));
        }
    }
    if losses.is_empty() {
        return Err(format!(
            "no validation logs found under {}",
            valid_dir.display()
        ));
    }
    losses.sort_by_key(|&(epoch, _)| epoch);
    Ok(losses)
}

/// Epoch with the lowest mean validation loss.
///
/// # Errors
///
/// See [`validation_losses`].
pub fn best_epoch(artifact_dir: &Path) -> Result<usize, String> {
    validation_losses(artifact_dir)?
        .into_iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(epoch, _)| epoch)
        .ok_or_else(|| "no validation losses".to_owned())
}

/// Path (without extension) of the checkpoint written for `epoch`.
#[must_use]
pub fn checkpoint_path(artifact_dir: &Path, epoch: usize) -> std::path::PathBuf {
    artifact_dir
        .join("checkpoint")
        .join(format!("model-{epoch}"))
}

/// Convert a training checkpoint into `model.bpk`, entirely on the CPU.
///
/// `epoch` defaults to the epoch with the best validation loss.  The
/// checkpoint must still exist in `<artifacts>/checkpoint/` (all epochs are
/// kept while training runs; afterwards only the best and the last remain).
///
/// # Errors
///
/// Returns a message when the checkpoint or `model.json` is missing or
/// cannot be read/written.
pub fn export_checkpoint(artifact_dir: &Path, epoch: Option<usize>) -> Result<(), String> {
    let device = Device::flex();

    let epoch = match epoch {
        Some(e) => e,
        None => best_epoch(artifact_dir)?,
    };
    let model = load_model(artifact_dir, Some(epoch), &device)?;
    model
        .save_file(artifact_dir.join(EXPORT_FILE))
        .map_err(|e| e.to_string())?;
    println!(
        "exported epoch {epoch} to {} in {}",
        EXPORT_FILE,
        artifact_dir.display()
    );
    Ok(())
}

/// Names of the model checkpoints present in `<artifacts>/checkpoint/`.
#[must_use]
pub fn available_checkpoints(artifact_dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(artifact_dir.join("checkpoint")) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .filter_map(Result::ok)
        .filter_map(|e| e.file_name().to_str().map(str::to_owned))
        .filter(|n| n.starts_with("model-"))
        .collect();
    names.sort();
    names
}

/// Load a detector from an artifact directory.
///
/// With `epoch = None`, the exported `model.bpk` is used; with
/// `Some(epoch)`, the training checkpoint `checkpoint/model-<epoch>.bpk` is
/// loaded directly, which allows comparing epochs without re-exporting.
///
/// # Errors
///
/// Returns a message when `model.json` or the requested weights are missing
/// or invalid.
pub fn load_model(
    artifact_dir: &Path,
    epoch: Option<usize>,
    device: &Device,
) -> Result<Detector, String> {
    let config_path = artifact_dir.join(MODEL_CONFIG_FILE);
    if !config_path.is_file() {
        return Err(format!("{} not found", config_path.display()));
    }
    let model_config = DetectorConfig::load(&config_path)
        .map_err(|e| format!("{}: {e}", config_path.display()))?;
    let model = model_config.init(device);

    match epoch {
        None => {
            let weights_path = artifact_dir.join(EXPORT_FILE);
            if !weights_path.is_file() {
                return Err(format!(
                    "{} not found; run `train` to completion, `export [--epoch N]` to convert a \
                     checkpoint from {}, or pass `--epoch N` to use a checkpoint directly",
                    weights_path.display(),
                    artifact_dir.join("checkpoint").display()
                ));
            }
            model
                .try_load_file(&weights_path)
                .map_err(|e| format!("{}: {e}", weights_path.display()))
        }
        Some(epoch) => {
            let ckpt = checkpoint_path(artifact_dir, epoch);
            if !ckpt.with_extension("bpk").is_file() {
                return Err(format!(
                    "checkpoint {} does not exist (available: {})",
                    ckpt.with_extension("bpk").display(),
                    available_checkpoints(artifact_dir).join(", ")
                ));
            }
            model
                .try_load_file(&ckpt)
                .map_err(|e| format!("{}: {e}", ckpt.display()))
        }
    }
}

/// Load the exported detector (`model.json` + `model.bpk`).
///
/// Shorthand for [`load_model`] with `epoch = None`.
///
/// # Errors
///
/// See [`load_model`].
pub fn load_trained(artifact_dir: &Path, device: &Device) -> Result<Detector, String> {
    load_model(artifact_dir, None, device)
}
