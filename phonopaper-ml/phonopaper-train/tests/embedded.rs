//! Consistency of the model embedded in `phonopaper-rs` with this crate.
//!
//! `phonopaper-rs/src/decode/nn/model.rs` must be a verbatim copy of
//! `src/model.rs` (burn matches weights by field name), and the embedded
//! `model.bpk` / `model.json` must have been exported from that definition.
//! These tests catch a retraining that changed the architecture without
//! re-copying the model file or re-exporting the weights, and vice versa.

use std::path::{Path, PathBuf};

use burn::config::Config as _;
use burn::module::{Module, ModuleVisitor, Param};
use burn::prelude::*;
use burn::store::ModuleRecord;
use burn::tensor::Device;
use phonopaper_train::model::{DetectorConfig, OUTPUT_SIZE};

/// `phonopaper-rs/src/decode/nn/` in the sibling library crate.
fn embedded_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../phonopaper-rs/src/decode/nn")
        .canonicalize()
        .expect("phonopaper-rs/src/decode/nn exists next to phonopaper-ml")
}

/// Collects the shape of every parameter of a module, in traversal order.
#[derive(Default)]
struct Shapes(Vec<Vec<usize>>);

impl ModuleVisitor for Shapes {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        self.0.push(param.val().dims().to_vec());
    }
    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<D, Int>>) {
        self.0.push(param.val().dims().to_vec());
    }
    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<D, Bool>>) {
        self.0.push(param.val().dims().to_vec());
    }
}

fn parameter_shapes<M: Module>(module: &M) -> Vec<Vec<usize>> {
    let mut shapes = Shapes::default();
    module.visit(&mut shapes);
    shapes.0
}

#[test]
fn embedded_model_definition_is_identical_to_the_training_one() {
    let ours = include_str!("../src/model.rs");
    let theirs = std::fs::read_to_string(embedded_dir().join("model.rs"))
        .expect("read phonopaper-rs/src/decode/nn/model.rs");
    assert!(
        ours == theirs,
        "phonopaper-rs/src/decode/nn/model.rs differs from phonopaper-train/src/model.rs; \
         copy the training file over and re-export model.bpk from it"
    );
}

#[test]
fn embedded_weights_load_into_the_current_model_definition() {
    let dir = embedded_dir();
    let config = DetectorConfig::load(dir.join("model.json")).expect("parse embedded model.json");
    let device = Device::flex();
    let fresh = config.init(&device);
    let expected_shapes = parameter_shapes(&fresh);

    // A structural mismatch (renamed / added / removed fields) fails here …
    let record = ModuleRecord::load(dir.join("model.bpk")).expect(
        "embedded model.bpk must deserialise into the current Detector; \
             re-export it after changing the architecture",
    );
    let model = fresh
        .try_load_record(record)
        .expect("matching parameter names");

    // … but burn takes the tensors of the record as they are, so a shape
    // mismatch (e.g. a different `hidden` or channel width) must be checked
    // explicitly against a model freshly built from `model.json`.
    assert_eq!(
        parameter_shapes(&model),
        expected_shapes,
        "embedded model.bpk was exported from a Detector whose parameter shapes differ \
         from those of model.json + model.rs"
    );

    // The loaded network must actually run at the configured input size.
    let n = model.input_size();
    let out = model.forward(Tensor::<4>::zeros([1, 1, n, n], &device));
    assert_eq!(out.dims(), [1, OUTPUT_SIZE]);
    let values: Vec<f32> = out.into_data().try_into_vec::<f32>().expect("f32 output");
    assert!(
        values.iter().all(|v| v.is_finite()),
        "network output must be finite: {values:?}"
    );
}
