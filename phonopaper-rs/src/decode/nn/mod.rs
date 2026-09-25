//! Neural-network `PhonoPaper` pattern detector.
//!
//! A small convolutional network (≈ 280 k parameters, trained on synthetic
//! data by the `phonopaper-ml` workspace) that tells whether a camera frame
//! contains a `PhonoPaper` sheet and, if so, where its four corners are.  It
//! complements the hand-written stripe detector
//! ([`detect_markers`](crate::decode::detect_markers)), which assumes an
//! upright, axis-aligned pattern and is prone to false positives on stripy
//! backgrounds.
//!
//! The trained weights are **embedded in the library** (`model.bin`, ≈ 1.2 MB)
//! so no file access is needed at run time.  Inference runs on the CPU with
//! burn's `ndarray` backend and takes a few milliseconds per frame.
//!
//! This module is only available with the `nn-detector` Cargo feature.
//!
//! # Example
//!
//! ```no_run
//! use phonopaper_rs::decode::nn::PatternDetector;
//!
//! let detector = PatternDetector::new();
//! let frame = image::open("photo.jpg").unwrap();
//! let detection = detector.detect(&frame);
//! if detection.probability >= 0.5 {
//!     #[expect(clippy::cast_precision_loss, reason = "image dimensions are small")]
//!     let corners = detection.corners_in_pixels(frame.width() as f32, frame.height() as f32);
//!     println!("pattern corners: {corners:?}");
//! }
//! ```
//!
//! # Corner convention
//!
//! [`Detection::corners`] are `[TL, TR, BR, BL]` **in pattern orientation**,
//! clockwise in image coordinates, normalised by the network input side
//! (`0.0` = left/top edge, `1.0` = right/bottom edge; values slightly outside
//! `[0, 1]` are legitimate for corners just outside the frame).  The edges
//! `c0→c1` and `c2→c3` carry the marker bands.  Because a sheet is symmetric
//! under a 180° turn, the image alone cannot tell which of those two edges is
//! the *top* (high-frequency) band; [`PatternDetector::detect`] returns the
//! [canonical](Detection::canonical) ordering, whose first corner is the
//! higher one in the image.

mod model;

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::config::Config as _;
use burn::module::Module;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use image::DynamicImage;
use image::imageops::FilterType;

pub use model::{
    Detection, Detector, DetectorConfig, OUTPUT_SIZE, decode_output, image_tensor, soft_argmax,
};

/// The burn backend used for inference.
pub type Backend = NdArray;

/// Trained weights, exported by `phonopaper-train` with a
/// `BinFileRecorder<FullPrecisionSettings>`.
static WEIGHTS: &[u8] = include_bytes!("model.bin");

/// The [`DetectorConfig`] the weights were trained with.
static CONFIG: &[u8] = include_bytes!("model.json");

/// Configuration of the embedded model.
///
/// # Panics
///
/// Panics if the embedded `model.json` is not a valid [`DetectorConfig`],
/// which would be a packaging error of the library itself.
#[must_use]
pub fn embedded_config() -> DetectorConfig {
    DetectorConfig::load_binary(CONFIG).expect("embedded model.json is a valid DetectorConfig")
}

/// Load the embedded detector on the CPU.
///
/// Prefer [`PatternDetector`], which wraps the model with image preparation;
/// use this when you need the raw [`Detector`] (e.g. to run batches or to
/// inspect heat-maps).
///
/// # Panics
///
/// Panics if the embedded weights do not match the model definition, which
/// would be a packaging error of the library itself.
#[must_use]
pub fn load() -> Detector<Backend> {
    let device = NdArrayDevice::Cpu;
    let record = BinBytesRecorder::<FullPrecisionSettings, &'static [u8]>::default()
        .load(WEIGHTS, &device)
        .expect("embedded weights match the model definition");
    embedded_config()
        .init::<Backend>(&device)
        .load_record(record)
}

/// Prepare an image for the network: grayscale, stretched to `size × size`.
///
/// The aspect ratio is **not** preserved (the training data covers arbitrary
/// shear and anisotropic scaling); the predicted normalised corners are
/// mapped back by multiplying with the original width and height
/// ([`Detection::corners_in_pixels`]).  This is the same preparation the
/// training crate applies, so the results are identical to `phonopaper-train
/// infer`.
#[must_use]
pub fn prepare_image(img: &DynamicImage, size: u32) -> Vec<u8> {
    img.resize_exact(size, size, FilterType::Triangle)
        .into_luma8()
        .into_raw()
}

/// The embedded detector, ready to run on arbitrary images.
///
/// Construction ([`PatternDetector::new`]) deserialises the ≈ 1.2 MB of
/// weights; keep one instance around and call [`detect`](Self::detect) for
/// every frame.
#[derive(Debug)]
pub struct PatternDetector {
    model: Detector<Backend>,
    device: NdArrayDevice,
    input_size: u32,
}

impl Default for PatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternDetector {
    /// Load the embedded model.
    ///
    /// # Panics
    ///
    /// Panics if the embedded weights do not match the model definition,
    /// which would be a packaging error of the library itself.
    #[must_use]
    pub fn new() -> Self {
        let model = load();
        let input_size = u32::try_from(model.input_size()).expect("input size fits in u32");
        Self {
            model,
            device: NdArrayDevice::Cpu,
            input_size,
        }
    }

    /// Side of the square network input, in pixels (128 for the embedded
    /// model).
    #[must_use]
    pub fn input_size(&self) -> u32 {
        self.input_size
    }

    /// The underlying network.
    #[must_use]
    pub fn model(&self) -> &Detector<Backend> {
        &self.model
    }

    /// Run the detector on an image of any size and colour type.
    ///
    /// The image is converted to grayscale and stretched to the network
    /// input size (see [`prepare_image`]); the returned corners are
    /// normalised and in [canonical](Detection::canonical) order.  Multiply
    /// them with the original width and height
    /// ([`Detection::corners_in_pixels`]) to get frame coordinates.
    #[must_use]
    pub fn detect(&self, image: &DynamicImage) -> Detection {
        let pixels = prepare_image(image, self.input_size());
        self.detect_prepared(&pixels)
    }

    /// Run the detector on an already prepared `input_size × input_size`
    /// 8-bit grayscale image (row-major), as produced by [`prepare_image`].
    ///
    /// Use this on a camera pipeline that can deliver a downscaled
    /// luminance plane directly, to skip the conversion done by
    /// [`detect`](Self::detect).  The result is in canonical order.
    ///
    /// # Panics
    ///
    /// Panics if `pixels.len()` is not `input_size²`.
    #[must_use]
    pub fn detect_prepared(&self, pixels: &[u8]) -> Detection {
        self.model.detect(pixels, &self.device).canonical()
    }

    /// Detect a pattern and return its corners in pixels of `image`, or
    /// `None` when the presence probability is below `threshold` (0.5 is a
    /// sensible default).
    ///
    /// The corners are `[TL, TR, BR, BL]` in pattern orientation and
    /// canonical order (see the [module documentation](self)).
    #[must_use]
    pub fn find_corners(&self, image: &DynamicImage, threshold: f32) -> Option<[[f32; 2]; 4]> {
        let detection = self.detect(image);
        if detection.probability < threshold {
            return None;
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "image dimensions are far below 2^24"
        )]
        let corners = detection.corners_in_pixels(image.width() as f32, image.height() as f32);
        Some(corners)
    }
}
