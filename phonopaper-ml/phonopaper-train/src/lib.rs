//! # phonopaper-train
//!
//! Train, evaluate and run a small convolutional network that locates a
//! `PhonoPaper` pattern in a camera frame, using the
//! [burn](https://burn.dev) deep-learning framework.
//!
//! The network takes a `128 × 128` grayscale image and predicts whether a
//! pattern is present plus the four corners of its ink box.  See
//! [`model`] for the architecture and output layout, and the crate README for
//! the end-to-end workflow (dataset generation → training → embedding the
//! weights in `phonopaper-rs`).

pub mod data;
pub mod eval;
pub mod infer;
pub mod model;
pub mod training;
