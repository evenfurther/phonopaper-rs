//! # phonopaper-dataset
//!
//! Deterministic synthetic dataset generator for training a neural network
//! that locates `PhonoPaper` patterns in camera frames.
//!
//! Every image is a pure function of `(seed, index)`: the generator uses an
//! in-crate PRNG and only IEEE-754 basic arithmetic (`+ - * /` and `sqrt`), so
//! the same command produces byte-identical PNGs and labels on every
//! platform and toolchain.
//!
//! See [`generate_dataset`] for the on-disk layout and the [`labels`] module
//! for the label file format.

pub mod background;
pub mod canvas;
pub mod geometry;
pub mod labels;
pub mod num;
pub mod pattern;
pub mod rng;
pub mod sample;

use std::path::Path;

use rayon::prelude::*;

use labels::{Label, Manifest, labels_to_csv};
pub use sample::{GeneratorConfig, Sample, generate_sample};

/// Current `manifest.json` / `labels.csv` format version.
pub const FORMAT_VERSION: u32 = 1;

/// File name of the `index`-th image.
#[must_use]
pub fn image_file_name(index: u64) -> String {
    format!("{index:06}.png")
}

/// Generate a whole dataset into `out_dir`.
///
/// Writes `<index>.png` for every image, plus `labels.csv` and
/// `manifest.json`.  Images are generated in parallel; because each one has
/// an independent RNG stream the output does not depend on scheduling.
///
/// # Errors
///
/// Returns an I/O error if the directory cannot be created or a file cannot
/// be written, or an encoding error from the PNG encoder.
pub fn generate_dataset(cfg: &GeneratorConfig, out_dir: &Path) -> Result<Manifest, String> {
    std::fs::create_dir_all(out_dir)
        .map_err(|e| format!("cannot create {}: {e}", out_dir.display()))?;

    let labels: Vec<Label> = (0..cfg.count)
        .into_par_iter()
        .map(|index| {
            let sample = generate_sample(cfg, index);
            let file = image_file_name(index);
            sample
                .image
                .save(out_dir.join(&file))
                .map_err(|e| format!("cannot write {file}: {e}"))?;
            Ok(Label {
                file,
                corners: sample.corners,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    let positives = labels.iter().filter(|l| l.corners.is_some()).count() as u64;
    let manifest = Manifest {
        format_version: FORMAT_VERSION,
        config: cfg.clone(),
        corner_order: "top-left, top-right, bottom-right, bottom-left of the pattern \
                       (pattern orientation: the top marker band lies between x0y0 and x1y1); \
                       pixel coordinates with pixel centres at +0.5"
            .to_owned(),
        positives,
        negatives: cfg.count - positives,
    };

    std::fs::write(out_dir.join("labels.csv"), labels_to_csv(&labels))
        .map_err(|e| format!("cannot write labels.csv: {e}"))?;
    let manifest_json =
        serde_json::to_string_pretty(&manifest).map_err(|e| format!("manifest: {e}"))?;
    std::fs::write(out_dir.join("manifest.json"), manifest_json + "\n")
        .map_err(|e| format!("cannot write manifest.json: {e}"))?;
    Ok(manifest)
}
