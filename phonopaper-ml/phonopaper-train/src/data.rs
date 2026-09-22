//! Loading the generated dataset and batching it into tensors.

use std::path::Path;

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::{Dataset, InMemDataset};
use burn::prelude::*;
use phonopaper_dataset::labels::{read_labels, read_manifest};

use crate::model::OUTPUT_SIZE;

/// One image with its ground truth, fully loaded in memory.
#[derive(Debug, Clone)]
pub struct Item {
    /// Row-major 8-bit grayscale pixels, `size × size`.
    pub pixels: Vec<u8>,
    /// Image side in pixels.
    pub size: usize,
    /// Presence flag followed by eight normalised corner coordinates (zeros
    /// when absent) — exactly the layout the network predicts.
    pub target: [f32; OUTPUT_SIZE],
}

impl Item {
    /// Whether the image contains a pattern.
    #[must_use]
    pub fn present(&self) -> bool {
        self.target[0] > 0.5
    }
}

/// Which part of the dataset to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    /// Images whose index is **not** a multiple of ten.
    Train,
    /// Every tenth image (index `0, 10, 20, …`).
    Valid,
}

impl Split {
    fn contains(self, index: usize) -> bool {
        let is_valid = index.is_multiple_of(10);
        match self {
            Self::Train => !is_valid,
            Self::Valid => is_valid,
        }
    }
}

/// Load one split of a dataset directory produced by `phonopaper-dataset`.
///
/// Returns the dataset together with the image side read from the manifest.
///
/// # Errors
///
/// Returns a message if the manifest, labels or any image cannot be read, or
/// if an image does not have the size announced by the manifest.
pub fn load_split(dir: &Path, split: Split) -> Result<(InMemDataset<Item>, usize), String> {
    let manifest = read_manifest(dir)?;
    let size = usize::try_from(manifest.config.size).map_err(|e| e.to_string())?;
    let labels = read_labels(dir)?;
    let norm = 1.0 / f64::from(manifest.config.size);

    let mut items = Vec::with_capacity(labels.len() / 2);
    for (index, label) in labels.iter().enumerate() {
        if !split.contains(index) {
            continue;
        }
        let img = image::open(dir.join(&label.file))
            .map_err(|e| format!("{}: {e}", label.file))?
            .into_luma8();
        if img.width() != manifest.config.size || img.height() != manifest.config.size {
            return Err(format!(
                "{}: expected {0}×{0} pixels, got {1}×{2}",
                label.file,
                img.width(),
                img.height()
            ));
        }
        let mut target = [0.0_f32; OUTPUT_SIZE];
        if let Some(q) = &label.corners {
            target[0] = 1.0;
            for (i, p) in q.iter().enumerate() {
                // Coordinates are pixels of a ≤ 4096 px image: f32 is plenty.
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "normalised coordinates lie in roughly [-0.1, 1.1]"
                )]
                let (x, y) = ((p.x * norm) as f32, (p.y * norm) as f32);
                target[1 + 2 * i] = x;
                target[2 + 2 * i] = y;
            }
        }
        items.push(Item {
            pixels: img.into_raw(),
            size,
            target,
        });
    }
    Ok((InMemDataset::new(items), size))
}

/// A mini-batch assembled on the host, ready to be uploaded.
///
/// The data-loader workers only produce this plain struct; the upload to the
/// training device happens in the training/inference step, on the caller's
/// thread.  Creating device tensors inside worker threads made the CUDA
/// backend allocate one pinned host-memory pool per worker thread *per epoch*
/// (streams are per thread and workers are respawned every epoch), which
/// grew without bound until the job was OOM-killed.
#[derive(Debug, Clone)]
pub struct HostBatch {
    /// Number of images.
    pub batch: usize,
    /// Image side in pixels.
    pub size: usize,
    /// `batch × size × size` pixels in `[0, 1]`, row-major, image-major.
    pub pixels: Vec<f32>,
    /// `batch × OUTPUT_SIZE` targets.
    pub targets: Vec<f32>,
}

impl HostBatch {
    /// Upload to `device` as a [`DetectionBatch`].
    #[must_use]
    pub fn to_device<B: Backend>(&self, device: &B::Device) -> DetectionBatch<B> {
        DetectionBatch {
            images: Tensor::<B, 1>::from_floats(self.pixels.as_slice(), device)
                .reshape([self.batch, 1, self.size, self.size]),
            targets: Tensor::<B, 1>::from_floats(self.targets.as_slice(), device)
                .reshape([self.batch, OUTPUT_SIZE]),
        }
    }
}

/// A mini-batch on device `B`.
#[derive(Debug, Clone)]
pub struct DetectionBatch<B: Backend> {
    /// `[batch, 1, size, size]` in `[0, 1]`.
    pub images: Tensor<B, 4>,
    /// `[batch, 9]`: presence flag + normalised corners.
    pub targets: Tensor<B, 2>,
}

/// Turns [`Item`]s into a [`HostBatch`].
#[derive(Debug, Clone, Default)]
pub struct DetectionBatcher;

impl DetectionBatcher {
    /// Assemble a batch on the host.
    #[must_use]
    pub fn assemble(items: &[Item]) -> HostBatch {
        let batch = items.len();
        let size = items.first().map_or(0, |item| item.size);
        let mut pixels = Vec::with_capacity(batch * size * size);
        let mut targets = Vec::with_capacity(batch * OUTPUT_SIZE);
        for item in items {
            debug_assert_eq!(item.size, size, "all images in a batch share one size");
            pixels.extend(item.pixels.iter().map(|&p| f32::from(p) / 255.0));
            targets.extend_from_slice(&item.target);
        }
        HostBatch {
            batch,
            size,
            pixels,
            targets,
        }
    }
}

impl<B: Backend> Batcher<B, Item, HostBatch> for DetectionBatcher {
    fn batch(&self, items: Vec<Item>, _device: &B::Device) -> HostBatch {
        Self::assemble(&items)
    }
}

/// Count the positive samples of a dataset.
#[must_use]
pub fn count_positives(dataset: &InMemDataset<Item>) -> usize {
    dataset.iter().filter(Item::present).count()
}
