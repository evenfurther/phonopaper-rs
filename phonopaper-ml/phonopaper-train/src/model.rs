//! The detector network.
//!
//! This module depends only on `burn`'s core (`Module`, `nn`, `Tensor`) so it
//! can be copied verbatim into `phonopaper-rs` for inference.  Nothing here
//! requires the `train` feature.
//!
//! # Architecture
//!
//! A small VGG-style convolutional network for a single-channel square input
//! of side `input_size` (default 128):
//!
//! ```text
//! [conv3×3 → BatchNorm → ReLU → maxpool2] × 5   (channels 16, 32, 64, 128, 128)
//! flatten → Linear(2048 → 128) → ReLU → Dropout → Linear(128 → 9)
//! ```
//!
//! # Output
//!
//! A `[batch, 9]` tensor:
//!
//! | index | meaning |
//! |---|---|
//! | `0`      | presence **logit** (apply a sigmoid to get a probability) |
//! | `1..=8`  | `x0 y0 x1 y1 x2 y2 x3 y3` — corners **normalised by the input side** (`0.0` = left/top edge, `1.0` = right/bottom edge), clockwise, with `(x0,y0)→(x1,y1)` and `(x2,y2)→(x3,y3)` being the two marker-band edges |
//!
//! Because a sheet is symmetric under a 180° turn, the network may return
//! either `TL, TR, BR, BL` or `BR, BL, TL, TR`; use [`Detection::canonical`]
//! for a deterministic choice.

use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
};
use burn::prelude::*;
use burn::tensor::activation::sigmoid;

/// Number of output values: one presence logit plus eight coordinates.
pub const OUTPUT_SIZE: usize = 9;

/// Channel width of each convolutional stage.
const STAGE_CHANNELS: [usize; 5] = [16, 32, 64, 128, 128];

/// Hyper-parameters of the [`Detector`].
#[derive(Config, Debug)]
pub struct DetectorConfig {
    /// Side of the square grayscale input, in pixels.  Must be a multiple of
    /// 32 (five 2× pooling stages).
    #[config(default = 128)]
    pub input_size: usize,
    /// Width of the hidden fully connected layer.
    #[config(default = 128)]
    pub hidden: usize,
    /// Dropout probability applied before the output layer.
    #[config(default = 0.2)]
    pub dropout: f64,
}

/// One `conv → BatchNorm → ReLU → maxpool` stage.
#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: Conv2d<B>,
    norm: BatchNorm<B>,
    activation: Relu,
    pool: MaxPool2d,
}

impl<B: Backend> ConvBlock<B> {
    fn new(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        Self {
            conv: Conv2dConfig::new([in_channels, out_channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .with_bias(false)
                .init(device),
            norm: BatchNormConfig::new(out_channels).init(device),
            activation: Relu::new(),
            pool: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        let x = self.norm.forward(x);
        let x = self.activation.forward(x);
        self.pool.forward(x)
    }
}

/// The `PhonoPaper` pattern detector.
#[derive(Module, Debug)]
pub struct Detector<B: Backend> {
    blocks: Vec<ConvBlock<B>>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    activation: Relu,
    dropout: Dropout,
    input_size: usize,
}

impl DetectorConfig {
    /// Instantiate a detector with freshly initialised weights.
    ///
    /// # Panics
    ///
    /// Panics if `input_size` is not a positive multiple of 32.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Detector<B> {
        assert!(
            self.input_size > 0 && self.input_size.is_multiple_of(32),
            "input_size must be a positive multiple of 32, got {}",
            self.input_size
        );
        let mut blocks = Vec::with_capacity(STAGE_CHANNELS.len());
        let mut in_channels = 1;
        for &out_channels in &STAGE_CHANNELS {
            blocks.push(ConvBlock::new(in_channels, out_channels, device));
            in_channels = out_channels;
        }
        let spatial = self.input_size >> STAGE_CHANNELS.len();
        let flat = in_channels * spatial * spatial;
        Detector {
            blocks,
            fc1: LinearConfig::new(flat, self.hidden).init(device),
            fc2: LinearConfig::new(self.hidden, OUTPUT_SIZE).init(device),
            activation: Relu::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
            input_size: self.input_size,
        }
    }
}

impl<B: Backend> Detector<B> {
    /// Side of the expected square input.
    #[must_use]
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Run the network.
    ///
    /// `images` has shape `[batch, 1, input_size, input_size]` with values in
    /// `[0, 1]`.  Returns `[batch, 9]` (see the module documentation).
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut x = images;
        for block in &self.blocks {
            x = block.forward(x);
        }
        let x = x.flatten(1, 3);
        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        self.fc2.forward(x)
    }

    /// Run the network on a single grayscale image and decode the result.
    ///
    /// `pixels` is the row-major `input_size × input_size` 8-bit image.
    ///
    /// # Panics
    ///
    /// Panics if `pixels.len() != input_size²`.
    pub fn detect(&self, pixels: &[u8], device: &B::Device) -> Detection {
        let n = self.input_size;
        assert_eq!(pixels.len(), n * n, "expected {n}×{n} pixels");
        let input = image_tensor::<B>(pixels, n, device).unsqueeze::<4>();
        let output = self.forward(input);
        decode_output(&output)[0]
    }
}

/// Convert an 8-bit grayscale image to a `[1, size, size]` tensor in `[0, 1]`.
#[must_use]
pub fn image_tensor<B: Backend>(pixels: &[u8], size: usize, device: &B::Device) -> Tensor<B, 3> {
    let floats: Vec<f32> = pixels.iter().map(|&p| f32::from(p) / 255.0).collect();
    Tensor::<B, 1>::from_floats(floats.as_slice(), device).reshape([1, size, size])
}

/// A decoded detector output for one image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    /// Probability that a pattern is present, in `[0, 1]`.
    pub probability: f32,
    /// Corners `[TL, TR, BR, BL]` as `[x, y]`, normalised by the input side.
    pub corners: [[f32; 2]; 4],
}

impl Detection {
    /// Corners scaled to an image of `width × height` pixels (the image that
    /// was resized to the network input).
    #[must_use]
    pub fn corners_in_pixels(&self, width: f32, height: f32) -> [[f32; 2]; 4] {
        self.corners.map(|[x, y]| [x * width, y * height])
    }

    /// The same quadrilateral with corners rotated by two positions
    /// (`BR, BL, TL, TR`) — the sheet seen upside down.
    #[must_use]
    pub fn rotated_180(&self) -> Self {
        let c = self.corners;
        Self {
            probability: self.probability,
            corners: [c[2], c[3], c[0], c[1]],
        }
    }

    /// Canonical ordering: of the two equivalent orderings, the one whose
    /// first corner is higher in the image (smaller `y`; ties broken by
    /// smaller `x`).
    ///
    /// A `PhonoPaper` sheet looks identical when turned by 180°, so the
    /// network's choice between `TL, TR, BR, BL` and `BR, BL, TL, TR` is
    /// arbitrary; this picks a deterministic one.  Which end is really the
    /// top (high frequencies) cannot be recovered from the image alone.
    #[must_use]
    pub fn canonical(&self) -> Self {
        let [a, b] = [self.corners[0], self.corners[2]];
        let first_is_higher = match a[1].total_cmp(&b[1]) {
            std::cmp::Ordering::Less => true,
            std::cmp::Ordering::Greater => false,
            std::cmp::Ordering::Equal => a[0] <= b[0],
        };
        if first_is_higher {
            *self
        } else {
            self.rotated_180()
        }
    }
}

/// Decode a `[batch, 9]` raw output into one [`Detection`] per row.
///
/// # Panics
///
/// Panics if the tensor data cannot be read back as `f32`, which cannot
/// happen for a float tensor produced by [`Detector::forward`].
#[must_use]
pub fn decode_output<B: Backend>(output: &Tensor<B, 2>) -> Vec<Detection> {
    let [batch, _] = output.dims();
    let logits = output.clone().narrow(1, 0, 1);
    let probabilities: Vec<f32> = sigmoid(logits).into_data().to_vec().expect("f32 data");
    let coords: Vec<f32> = output
        .clone()
        .narrow(1, 1, 8)
        .into_data()
        .to_vec()
        .expect("f32 data");
    (0..batch)
        .map(|i| {
            let c = &coords[i * 8..(i + 1) * 8];
            Detection {
                probability: probabilities[i],
                corners: [[c[0], c[1]], [c[2], c[3]], [c[4], c[5]], [c[6], c[7]]],
            }
        })
        .collect()
}
