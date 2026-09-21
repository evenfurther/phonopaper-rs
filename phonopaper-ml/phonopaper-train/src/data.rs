//! Loading the generated dataset and batching it into tensors.

use std::path::Path;

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::{Dataset, InMemDataset};
use burn::prelude::*;
use phonopaper_dataset::labels::{read_labels, read_manifest};

use crate::model::{OUTPUT_SIZE, image_tensor};

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

/// A mini-batch on device `B`.
#[derive(Debug, Clone)]
pub struct DetectionBatch<B: Backend> {
    /// `[batch, 1, size, size]` in `[0, 1]`.
    pub images: Tensor<B, 4>,
    /// `[batch, 9]`: presence flag + normalised corners.
    pub targets: Tensor<B, 2>,
}

/// Turns [`Item`]s into a [`DetectionBatch`].
#[derive(Debug, Clone, Default)]
pub struct DetectionBatcher;

impl<B: Backend> Batcher<B, Item, DetectionBatch<B>> for DetectionBatcher {
    fn batch(&self, items: Vec<Item>, device: &B::Device) -> DetectionBatch<B> {
        let images: Vec<Tensor<B, 3>> = items
            .iter()
            .map(|item| image_tensor::<B>(&item.pixels, item.size, device))
            .collect();
        let targets: Vec<Tensor<B, 2>> = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_floats(item.target, device).unsqueeze::<2>())
            .collect();
        DetectionBatch {
            images: Tensor::stack(images, 0),
            targets: Tensor::cat(targets, 0),
        }
    }
}

/// Count the positive samples of a dataset.
#[must_use]
pub fn count_positives(dataset: &InMemDataset<Item>) -> usize {
    dataset.iter().filter(Item::present).count()
}
