//! The detector network.
//!
//! This module depends only on `burn`'s core (`Module`, `nn`, `Tensor`) so it
//! can be copied verbatim into `phonopaper-rs` for inference.  Nothing here
//! requires the `train` feature.
//!
//! # Architecture
//!
//! A small convolutional network for a single-channel square input of side
//! `input_size` (default 128), with two heads:
//!
//! ```text
//! trunk:    [conv3×3 → BatchNorm → ReLU → maxpool2] × 5   (channels 16, 32, 64, 128, 128)
//! presence: global average pool of the last stage → Linear(128 → 1)
//! corners:  stage 0 (stride 2, 64×64×16) ⊕ stage 1 upsampled ×2
//!           ⊕ stage 2 upsampled ×4 → conv3×3 → BatchNorm → ReLU → conv1×1
//!           → 4 heat-maps (64×64) → soft-argmax
//! ```
//!
//! Corners are **not** regressed by a fully connected layer: that discards
//! spatial precision and plateaus around 10 % of the frame.  Instead each
//! corner gets a heat-map over a genuine stride-2 grid and its coordinate is
//! the soft-argmax (softmax-weighted mean of the cell centres), which is
//! continuous and localises to a fraction of a cell.  The heat-map input
//! merges stage 0's fine edges with stages 1 and 2 upsampled from strides 4
//! and 8 to provide progressively wider context, U-Net style.  The grid spans
//! `[-0.1, 1.1]` so
//! corners slightly outside the frame stay representable.
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
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu};
use burn::prelude::*;
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};

/// Number of output values: one presence logit plus eight coordinates.
pub const OUTPUT_SIZE: usize = 9;

/// Channel width of each convolutional stage.
const STAGE_CHANNELS: [usize; 5] = [16, 32, 64, 128, 128];

/// Trunk stages (0-based) feeding the corner head. Stage 0 sets the stride-2
/// heat-map resolution; stages 1 and 2 contribute deeper spatial context.
const FINE_STAGE: usize = 0;
const CONTEXT_STAGES: [usize; 2] = [1, 2];

/// Stride of the heat-map grid relative to the input.
const HEATMAP_STRIDE: usize = 2;

/// Lower extent of the soft-argmax coordinate grid, in normalised units.
///
/// The grid extends beyond the frame so corners up to 10% outside remain
/// representable.
pub const HEATMAP_GRID_MIN: f32 = -0.1;
/// Upper extent of the soft-argmax and heat-map supervision grid.
pub const HEATMAP_GRID_MAX: f32 = 1.1;

/// Hyper-parameters of the [`Detector`].
#[derive(Config, Debug)]
pub struct DetectorConfig {
    /// Side of the square grayscale input, in pixels.  Must be a multiple of
    /// 32 (five 2× pooling stages).
    #[config(default = 128)]
    pub input_size: usize,
    /// Channel width of the corner head's hidden convolution.
    #[config(default = 64)]
    pub hidden: usize,
}

/// One `conv → BatchNorm → ReLU → maxpool` stage.
#[derive(Module, Debug)]
pub struct ConvBlock {
    conv: Conv2d,
    norm: BatchNorm,
    activation: Relu,
    pool: MaxPool2d,
}

impl ConvBlock {
    fn new(in_channels: usize, out_channels: usize, device: &Device) -> Self {
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

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.conv.forward(x);
        let x = self.norm.forward(x);
        let x = self.activation.forward(x);
        self.pool.forward(x)
    }
}

/// The `PhonoPaper` pattern detector.
#[derive(Module, Debug)]
pub struct Detector {
    blocks: Vec<ConvBlock>,
    presence: Linear,
    corner_conv: Conv2d,
    corner_norm: BatchNorm,
    corner_out: Conv2d,
    activation: Relu,
    input_size: usize,
}

impl DetectorConfig {
    /// Instantiate a detector with freshly initialised weights.
    ///
    /// # Panics
    ///
    /// Panics if `input_size` is not a positive multiple of 32.
    #[must_use]
    pub fn init(&self, device: &Device) -> Detector {
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
        Detector {
            blocks,
            presence: LinearConfig::new(in_channels, 1).init(device),
            corner_conv: Conv2dConfig::new(
                [
                    STAGE_CHANNELS[FINE_STAGE]
                        + STAGE_CHANNELS[CONTEXT_STAGES[0]]
                        + STAGE_CHANNELS[CONTEXT_STAGES[1]],
                    self.hidden,
                ],
                [3, 3],
            )
            .with_padding(PaddingConfig2d::Same)
            .with_bias(false)
            .init(device),
            corner_norm: BatchNormConfig::new(self.hidden).init(device),
            corner_out: Conv2dConfig::new([self.hidden, 4], [1, 1]).init(device),
            activation: Relu::new(),
            input_size: self.input_size,
        }
    }
}

/// Outputs produced by one detector trunk pass.
///
/// `output` preserves the public `[batch, 9]` inference representation, while
/// `heatmaps` exposes `[batch, 4, input_size / 2, input_size / 2]` logits for
/// direct spatial supervision during training.
pub struct DetectorOutput {
    /// Presence logit followed by eight soft-argmax corner coordinates.
    pub output: Tensor<2>,
    /// Raw stride-2 corner heat-map logits.
    pub heatmaps: Tensor<4>,
}

impl Detector {
    /// Side of the expected square input.
    #[must_use]
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Side of the corner heat-maps (`input_size / 2`).
    #[must_use]
    pub fn heatmap_size(&self) -> usize {
        self.input_size / HEATMAP_STRIDE
    }

    /// Run the network.
    ///
    /// `images` has shape `[batch, 1, input_size, input_size]` with values in
    /// `[0, 1]`.  Returns `[batch, 9]` (see the module documentation).
    #[must_use]
    pub fn forward(&self, images: Tensor<4>) -> Tensor<2> {
        self.forward_with_heatmaps(images).output
    }

    /// Run the network once and return both inference output and heat-map logits.
    ///
    /// This is the training entry point: unlike separately requesting decoded
    /// coordinates and heat-maps, it computes the convolutional trunk only once.
    #[must_use]
    pub fn forward_with_heatmaps(&self, images: Tensor<4>) -> DetectorOutput {
        let mut x = images;
        let mut features = Vec::with_capacity(3);
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(x);
            if i <= CONTEXT_STAGES[1] {
                features.push(x.clone());
            }
        }
        let pooled = x.mean_dim(3).mean_dim(2).flatten::<2>(1, 3);
        let logit = self.presence.forward(pooled);
        let heatmaps = self.heatmaps(features);
        let coords = soft_argmax(heatmaps.clone());
        DetectorOutput {
            output: Tensor::cat(vec![logit, coords], 1),
            heatmaps,
        }
    }

    /// Build raw stride-2 corner heat-maps from the first three trunk stages.
    fn heatmaps(&self, features: Vec<Tensor<4>>) -> Tensor<4> {
        let [_, _, h, w] = features[0].dims();
        let mut fused = Vec::with_capacity(features.len());
        for (i, feature) in features.into_iter().enumerate() {
            if i == FINE_STAGE {
                fused.push(feature);
            } else {
                fused.push(interpolate(
                    feature,
                    InterpolateOptions::new(InterpolateMode::Nearest).with_output_size([h, w]),
                ));
            }
        }
        let x = Tensor::cat(fused, 1);
        let x = self.corner_conv.forward(x);
        let x = self.corner_norm.forward(x);
        let x = self.activation.forward(x);
        self.corner_out.forward(x)
    }

    /// Run the network on a single grayscale image and decode the result.
    ///
    /// `pixels` is the row-major `input_size × input_size` 8-bit image.
    ///
    /// # Panics
    ///
    /// Panics if `pixels.len() != input_size²`.
    #[must_use]
    pub fn detect(&self, pixels: &[u8], device: &Device) -> Detection {
        let n = self.input_size;
        assert_eq!(pixels.len(), n * n, "expected {n}×{n} pixels");
        let input = image_tensor(pixels, n, device).unsqueeze::<4>();
        let output = self.forward(input);
        decode_output(&output)[0]
    }
}

/// Convert an 8-bit grayscale image to a `[1, size, size]` tensor in `[0, 1]`.
#[must_use]
pub fn image_tensor(pixels: &[u8], size: usize, device: &Device) -> Tensor<3> {
    let floats: Vec<f32> = pixels.iter().map(|&p| f32::from(p) / 255.0).collect();
    Tensor::<1>::from_floats(floats.as_slice(), device).reshape([1, size, size])
}

/// Soft-argmax of `[batch, 4, h, w]` heat-maps → `[batch, 8]` coordinates
/// `x0 y0 … x3 y3` in normalised units.
///
/// Each heat-map is soft-maxed over its `h·w` cells and the coordinate is the
/// probability-weighted mean of the cell centres, laid out on a grid spanning
/// `[GRID_MIN, GRID_MAX]` in both directions.
#[must_use]
pub fn soft_argmax(heatmaps: Tensor<4>) -> Tensor<2> {
    let [batch, corners, h, w] = heatmaps.dims();
    let device = heatmaps.device();
    let flat = heatmaps.reshape([batch, corners, h * w]);
    let weights = softmax(flat, 2);

    let centre = |i: usize, n: usize| {
        #[expect(clippy::cast_precision_loss, reason = "grid sizes are tiny integers")]
        let t = (i as f32 + 0.5) / n as f32;
        HEATMAP_GRID_MIN + (HEATMAP_GRID_MAX - HEATMAP_GRID_MIN) * t
    };
    let mut xs = Vec::with_capacity(h * w);
    let mut ys = Vec::with_capacity(h * w);
    for row in 0..h {
        for col in 0..w {
            xs.push(centre(col, w));
            ys.push(centre(row, h));
        }
    }
    let grid_x = Tensor::<1>::from_floats(xs.as_slice(), &device).reshape([1, 1, h * w]);
    let grid_y = Tensor::<1>::from_floats(ys.as_slice(), &device).reshape([1, 1, h * w]);

    let x = (weights.clone() * grid_x).sum_dim(2); // [batch, 4, 1]
    let y = (weights * grid_y).sum_dim(2); // [batch, 4, 1]
    // Interleave as x0 y0 x1 y1 …
    Tensor::cat(vec![x, y], 2).reshape([batch, corners * 2])
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
pub fn decode_output(output: &Tensor<2>) -> Vec<Detection> {
    let [batch, _] = output.dims();
    let logits = output.clone().narrow(1, 0, 1);
    let probabilities: Vec<f32> = sigmoid(logits)
        .into_data()
        .try_into_vec::<f32>()
        .expect("f32 data");
    let coords: Vec<f32> = output
        .clone()
        .narrow(1, 1, 8)
        .into_data()
        .try_into_vec::<f32>()
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
