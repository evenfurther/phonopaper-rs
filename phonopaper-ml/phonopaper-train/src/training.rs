//! Loss function and training loop.

use std::path::Path;

use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::tensor::Device;
use burn::tensor::Int;
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

/// Weight of compact target-cell classification relative to presence BCE.
pub const HEATMAP_LOSS_WEIGHT: f64 = 1.0;
/// Weight of x/y offset supervision at the four target cells.
pub const OFFSET_LOSS_WEIGHT: f64 = 2.0;
/// Weight of decoded-coordinate Charbonnier supervision.
pub const COORDINATE_LOSS_WEIGHT: f64 = 5.0;
/// Weight of translation-invariant directed edge-vector supervision.
pub const EDGE_LOSS_WEIGHT: f64 = 2.0;
/// Weight of relative quadrilateral width/height supervision.
pub const SIZE_LOSS_WEIGHT: f64 = 1.0;
/// Charbonnier smoothing in normalized coordinate units.
pub const CHARBONNIER_EPSILON: f64 = 1.0e-3;

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

/// Individually inspectable terms of the detector objective.
///
/// `total` uses [`HEATMAP_LOSS_WEIGHT`], [`OFFSET_LOSS_WEIGHT`],
/// [`COORDINATE_LOSS_WEIGHT`], [`EDGE_LOSS_WEIGHT`], and
/// [`SIZE_LOSS_WEIGHT`]. All fields are scalar tensors.
pub struct DetectionLoss {
    /// Presence binary cross-entropy.
    pub presence: Tensor<1>,
    /// Hard nearest-cell cross-entropy.
    pub heatmap: Tensor<1>,
    /// Selected target-cell x/y offset error.
    pub offset: Tensor<1>,
    /// Decoded corner-coordinate Charbonnier error.
    pub coordinate: Tensor<1>,
    /// Translation-invariant directed edge-vector error.
    pub edge: Tensor<1>,
    /// Relative width/height error.
    pub size: Tensor<1>,
    /// Weighted sum used for back-propagation and learner metrics.
    pub total: Tensor<1>,
}

/// Target cells and center-relative displacements for four corners.
pub struct CornerCellTargets {
    /// Flattened cell indices with shape `[batch, 4, 1]`.
    pub indices: Tensor<3, Int>,
    /// Desired x displacement from each selected cell center.
    pub x_offsets: Tensor<3>,
    /// Desired y displacement from each selected cell center.
    pub y_offsets: Tensor<3>,
}

/// Compute the complete structured detector loss from one trunk pass.
///
/// One decoded-coordinate comparison chooses either the direct target ordering
/// or its equivalent 180° ordering per sample. That assigned target is shared
/// by every spatial term. Negatives are excluded from all spatial terms, and
/// an all-negative batch therefore has finite, exactly zero spatial losses.
#[must_use]
pub fn detection_loss_breakdown(prediction: DetectorOutput, targets: Tensor<2>) -> DetectionLoss {
    let logits = prediction.output.clone().narrow(1, 0, 1);
    let present = targets.clone().narrow(1, 0, 1);
    let absent = present.clone().neg().add_scalar(1.0);
    let presence = (present.clone() * log_sigmoid(logits.clone())
        + absent * log_sigmoid(logits.neg()))
    .neg()
    .mean();

    let predicted = prediction.output.narrow(1, 1, 8);
    let truth = targets.narrow(1, 1, 8);
    let assigned = assign_corner_targets(predicted.clone(), truth);
    let [_, _, h, w] = prediction.heatmaps.dims();
    let cells = corner_cell_targets(assigned.clone(), h, w);
    let heatmap = assigned_heatmap_loss(prediction.heatmaps, &cells, present.clone());
    let offset = offset_loss(
        prediction.x_offsets,
        prediction.y_offsets,
        &cells,
        present.clone(),
    );
    let coordinate = masked_mean(
        charbonnier(predicted.clone() - assigned.clone()).mean_dim(1),
        present.clone(),
    );
    let edge = edge_vector_loss(predicted.clone(), assigned.clone(), present.clone());
    let size = relative_size_loss(predicted, assigned, present);
    let total = presence.clone()
        + heatmap.clone().mul_scalar(HEATMAP_LOSS_WEIGHT)
        + offset.clone().mul_scalar(OFFSET_LOSS_WEIGHT)
        + coordinate.clone().mul_scalar(COORDINATE_LOSS_WEIGHT)
        + edge.clone().mul_scalar(EDGE_LOSS_WEIGHT)
        + size.clone().mul_scalar(SIZE_LOSS_WEIGHT);
    DetectionLoss {
        presence,
        heatmap,
        offset,
        coordinate,
        edge,
        size,
        total,
    }
}

/// Compute the scalar detector objective used by Burn's learner.
#[must_use]
pub fn detection_loss(prediction: DetectorOutput, targets: Tensor<2>) -> Tensor<1> {
    detection_loss_breakdown(prediction, targets).total
}

/// Choose one direct-vs-180° target ordering per sample.
///
/// The choice minimizes decoded-coordinate Charbonnier error and is then used
/// unchanged by all spatial objectives.
#[must_use]
pub fn assign_corner_targets(predicted: Tensor<2>, truth: Tensor<2>) -> Tensor<2> {
    let rotated = rotate_180(truth.clone());
    let direct_error = charbonnier(predicted.clone() - truth.clone()).mean_dim(1);
    let rotated_error = charbonnier(predicted - rotated.clone()).mean_dim(1);
    let use_rotated = rotated_error.lower(direct_error).repeat_dim(1, 8);
    truth.mask_where(use_rotated, rotated)
}

/// Build hard nearest-cell targets and exact center-relative offsets.
///
/// Coordinates outside the grid are assigned to its nearest boundary cell;
/// their displacement remains explicit instead of being silently clipped.
///
/// # Panics
///
/// Panics if `corners` cannot be read as `f32` values.
#[must_use]
pub fn corner_cell_targets(corners: Tensor<2>, h: usize, w: usize) -> CornerCellTargets {
    let [batch, values] = corners.dims();
    assert_eq!(values, 8, "expected four x/y corner pairs");
    let device = corners.device();
    let values: Vec<f32> = corners
        .into_data()
        .try_into_vec()
        .expect("f32 corner targets");
    let mut indices = Vec::with_capacity(batch * 4);
    let mut x_offsets = Vec::with_capacity(batch * 4);
    let mut y_offsets = Vec::with_capacity(batch * 4);
    let span = HEATMAP_GRID_MAX - HEATMAP_GRID_MIN;
    for sample in 0..batch {
        for corner in 0..4 {
            let x = values[sample * 8 + corner * 2];
            let y = values[sample * 8 + corner * 2 + 1];
            #[expect(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_possible_wrap,
                clippy::cast_sign_loss,
                reason = "tiny grid dimensions and explicit clamping make the nearest-cell index bounded and nonnegative"
            )]
            let col = (((x - HEATMAP_GRID_MIN) / span * w as f32 - 0.5).round() as isize)
                .clamp(0, w as isize - 1) as usize;
            #[expect(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_possible_wrap,
                clippy::cast_sign_loss,
                reason = "tiny grid dimensions and explicit clamping make the nearest-cell index bounded and nonnegative"
            )]
            let row = (((y - HEATMAP_GRID_MIN) / span * h as f32 - 0.5).round() as isize)
                .clamp(0, h as isize - 1) as usize;
            #[expect(clippy::cast_precision_loss, reason = "heat-map dimensions are small")]
            let center_x = HEATMAP_GRID_MIN + span * (col as f32 + 0.5) / w as f32;
            #[expect(clippy::cast_precision_loss, reason = "heat-map dimensions are small")]
            let center_y = HEATMAP_GRID_MIN + span * (row as f32 + 0.5) / h as f32;
            indices.push(i64::try_from(row * w + col).expect("heat-map index fits i64"));
            x_offsets.push(x - center_x);
            y_offsets.push(y - center_y);
        }
    }
    CornerCellTargets {
        indices: Tensor::<1, Int>::from_ints(indices.as_slice(), &device).reshape([batch, 4, 1]),
        x_offsets: Tensor::<1>::from_floats(x_offsets.as_slice(), &device).reshape([batch, 4, 1]),
        y_offsets: Tensor::<1>::from_floats(y_offsets.as_slice(), &device).reshape([batch, 4, 1]),
    }
}

/// Hard target-cell cross-entropy with a per-sample 180° assignment.
///
/// This standalone component chooses the assignment from its own cell losses;
/// [`detection_loss_breakdown`] instead shares the coordinate-based assignment
/// with every spatial component.
#[must_use]
pub fn heatmap_loss(heatmaps: Tensor<4>, targets: Tensor<2>) -> Tensor<1> {
    let present = targets.clone().narrow(1, 0, 1);
    let truth = targets.narrow(1, 1, 8);
    let [_, _, h, w] = heatmaps.dims();
    let direct =
        heatmap_loss_per_sample(heatmaps.clone(), &corner_cell_targets(truth.clone(), h, w));
    let rotated = heatmap_loss_per_sample(heatmaps, &corner_cell_targets(rotate_180(truth), h, w));
    masked_mean(direct.min_pair(rotated), present)
}

fn assigned_heatmap_loss(
    heatmaps: Tensor<4>,
    cells: &CornerCellTargets,
    present: Tensor<2>,
) -> Tensor<1> {
    masked_mean(heatmap_loss_per_sample(heatmaps, cells), present)
}

fn heatmap_loss_per_sample(heatmaps: Tensor<4>, cells: &CornerCellTargets) -> Tensor<2> {
    let [batch, corners, h, w] = heatmaps.dims();
    assert_eq!(corners, 4, "expected four corner heat-map channels");
    log_softmax(heatmaps.reshape([batch, corners, h * w]), 2)
        .gather(2, cells.indices.clone())
        .neg()
        .mean_dim(1)
        .reshape([batch, 1])
}

fn offset_loss(
    x_offsets: Tensor<4>,
    y_offsets: Tensor<4>,
    cells: &CornerCellTargets,
    present: Tensor<2>,
) -> Tensor<1> {
    let [batch, corners, h, w] = x_offsets.dims();
    #[expect(clippy::cast_precision_loss, reason = "heat-map dimensions are small")]
    let half_x = (HEATMAP_GRID_MAX - HEATMAP_GRID_MIN) / (2 * w) as f32;
    #[expect(clippy::cast_precision_loss, reason = "heat-map dimensions are small")]
    let half_y = (HEATMAP_GRID_MAX - HEATMAP_GRID_MIN) / (2 * h) as f32;
    let x = x_offsets
        .reshape([batch, corners, h * w])
        .gather(2, cells.indices.clone())
        .tanh()
        .mul_scalar(half_x);
    let y = y_offsets
        .reshape([batch, corners, h * w])
        .gather(2, cells.indices.clone())
        .tanh()
        .mul_scalar(half_y);
    let per_sample = (charbonnier(x - cells.x_offsets.clone())
        + charbonnier(y - cells.y_offsets.clone()))
    .mean_dim(1)
    .reshape([batch, 1]);
    masked_mean(per_sample, present)
}

/// Translation-invariant loss on the four directed quadrilateral edges.
#[must_use]
pub fn edge_vector_loss(predicted: Tensor<2>, truth: Tensor<2>, present: Tensor<2>) -> Tensor<1> {
    let edges = |corners: Tensor<2>| {
        let points = corners.reshape([-1, 4, 2]);
        let next = Tensor::cat(
            vec![
                points.clone().narrow(1, 1, 3),
                points.clone().narrow(1, 0, 1),
            ],
            1,
        );
        (next - points).flatten::<2>(1, 2)
    };
    masked_mean(
        charbonnier(edges(predicted) - edges(truth)).mean_dim(1),
        present,
    )
}

/// Relative width/height loss that penalizes uniformly contracted predictions.
#[must_use]
pub fn relative_size_loss(predicted: Tensor<2>, truth: Tensor<2>, present: Tensor<2>) -> Tensor<1> {
    let sizes = |corners: Tensor<2>| {
        let points = corners.reshape([-1, 4, 2]);
        let edge_length = |a: usize, b: usize| {
            (points.clone().narrow(1, b, 1) - points.clone().narrow(1, a, 1))
                .powi_scalar(2)
                .sum_dim(2)
                .add_scalar(CHARBONNIER_EPSILON * CHARBONNIER_EPSILON)
                .sqrt()
        };
        let width = (edge_length(0, 1) + edge_length(3, 2)).mul_scalar(0.5);
        let height = (edge_length(1, 2) + edge_length(0, 3)).mul_scalar(0.5);
        Tensor::cat(vec![width, height], 2).reshape([-1, 2])
    };
    let target_size = sizes(truth);
    let relative = (sizes(predicted) - target_size.clone()) / target_size.clamp_min(1.0e-3);
    masked_mean(charbonnier(relative).mean_dim(1), present)
}

fn charbonnier<const D: usize>(error: Tensor<D>) -> Tensor<D> {
    error
        .powi_scalar(2)
        .add_scalar(CHARBONNIER_EPSILON * CHARBONNIER_EPSILON)
        .sqrt()
        .sub_scalar(CHARBONNIER_EPSILON)
}

fn masked_mean(values: Tensor<2>, present: Tensor<2>) -> Tensor<1> {
    let positives = present.clone().sum().clamp_min(1.0);
    (values * present).sum() / positives
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
