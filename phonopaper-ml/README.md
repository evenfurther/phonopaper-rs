# phonopaper-ml — neural-network pattern detector

This directory is a **separate Cargo workspace** with the tooling used to
train a small convolutional neural network that finds a `PhonoPaper` pattern
in a camera frame and returns its four corners.  It is kept out of the main
workspace so that the (large) [burn](https://burn.dev) dependency tree never
affects the library, its lock file or its quality gates.

| Crate | Purpose |
|---|---|
| `phonopaper-dataset` | Deterministic synthetic dataset generator (images + `labels.csv`) |
| `phonopaper-train`   | Model definition, training, evaluation and inference with burn 0.21 |

The end-to-end workflow is:

1. [generate the dataset](#1-generate-the-dataset) — seconds;
2. [train the network](#2-train-the-network) — minutes to an hour depending on backend;
3. [evaluate / try it](#3-evaluate-and-try-the-model);
4. [transfer the model to `phonopaper-rs`](#4-transfer-the-model-to-phonopaper-rs).

All commands below are run from this directory (`phonopaper-ml/`).

---

## 1. Generate the dataset

```bash
cargo run --release -p phonopaper-dataset -- --output dataset --count 4000
```

| Option | Default | Description |
|---|---|---|
| `-o, --output <DIR>` | `dataset` | Output directory (created if missing) |
| `-n, --count <N>` | `4000` | Number of images |
| `--size <PX>` | `128` | Side of the square images; must be a multiple of 32 for the default network |
| `--seed <U64>` | `1592590337` | Master seed |
| `--positive-ratio <P>` | `0.6` | Probability that an image contains a pattern |

The output directory contains:

* `000000.png`, `000001.png`, … — 8-bit grayscale PNGs;
* `labels.csv` — one row per image (see below);
* `manifest.json` — the generator parameters, corner convention and counts.

### Determinism

The dataset is a **pure function of the command-line options**.  Every image
derives its own random stream from `(seed, index)` using an in-crate
`xoshiro256**` generator, and the whole pipeline uses only IEEE-754
`+ - * /` and `sqrt` (no `sin`, `exp`, `pow` — rotations are built from random
tangents, Gaussian noise from sums of uniforms, blur from binomial kernels).
Running the same command twice, on any machine, in any order or degree of
parallelism, produces byte-identical PNGs and label files.  This is verified
by `cargo test -p phonopaper-dataset`.

### What the images contain

**Positive samples** (`present = 1`) contain one genuine `PhonoPaper` sheet
rendered by `phonopaper_rs::render::spectrogram_to_image_buf` — the same code
the encoder uses — with:

* random marker geometry within the ranges accepted by
  `phonopaper_rs::decode::detect_markers` (thin/thick/gap/margin,
  pixels per octave, optional octave lines);
* random musical content (blank, notes with harmonics and glides, dense
  texture, printer dust) and random width (60 – 1600 columns);
* random paper tone and ink darkness, optional print blur;
* a random **projective transform**: scale (25 % – 95 % of the frame),
  rotation (±31°, plus a quarter/half turn 15 % of the time), shear
  (±0.25) and independent per-corner perspective jitter;
* proper anti-aliasing (box pre-filter + 2×2 super-sampling), so heavily
  minified stripes are averaged like a real camera would;
* optionally a small occluder drawn over the pattern (12 %).

Corners may lie up to 5 % of the frame outside the image.

**Negative samples** (`present = 0`) are backgrounds only, or backgrounds with
a *decoy* sheet: barcode-like stripes, text-like blocks, grids, or a broken
`PhonoPaper` with one marker band erased.  The decoys teach the network the
actual marker topology rather than "dark horizontal bars on white".

Every image starts from a random **background** (flat, gradient, blurred
noise, tiles, stripes) with random rectangles and lines, and ends with
**camera-like degradations**: contrast/brightness change, vignette, blur,
Gaussian noise, salt-and-pepper noise.

### `labels.csv`

```text
file,present,x0,y0,x1,y1,x2,y2,x3,y3
000000.png,1,39.90,1.02,119.65,0.38,119.13,46.16,38.55,48.42
000001.png,0,0,0,0,0,0,0,0,0
```

* `present` — `1` if a pattern is in the image.
* `(x0,y0) (x1,y1) (x2,y2) (x3,y3)` — the pattern's **top-left, top-right,
  bottom-right, bottom-left corners in pattern orientation**, in pixels of
  the image (origin at the top-left, pixel centres at `+0.5`), with two
  decimals.  All zeros for negatives.

"Pattern orientation" means the order is defined on the printed sheet, not in
the image: the edge `(x0,y0)→(x1,y1)` is always the outer edge of the **top
marker band**, whatever the rotation.  A consumer therefore knows which way is
up.  The corners delimit the *ink box* — from the first stripe of the top
band to the last stripe of the bottom band, across all data columns — the
white margins are not included.

---

## 2. Train the network

### Choose a backend

| Cargo features | Device | Notes |
|---|---|---|
| *(default)* `flex` | CPU | Pure Rust, works everywhere; ~3 s per batch of 32 |
| `--no-default-features --features wgpu` | GPU via Vulkan/Metal/DX12 | Needs only the graphics driver; ~0.35 s per batch on an RTX A2000 at full clocks |
| `--no-default-features --features cuda` | NVIDIA GPU | Needs the CUDA toolkit (`libnvrtc`) installed |
| `--no-default-features --features ndarray` | CPU | Legacy backend, ~10× slower than `flex` |
| add `--features tui` | — | Interactive terminal dashboard instead of plain logs |

> **NixOS / Vulkan:** the Vulkan loader must be reachable, e.g.
> `export LD_LIBRARY_PATH=/run/opengl-driver/lib`.

> **Laptop GPUs** may be power/thermally throttled (`nvidia-smi -q -d
> PERFORMANCE`); training can be several times slower than the numbers above.

### Run

```bash
# CPU
cargo run --release -p phonopaper-train -- train --dataset dataset --artifacts artifacts --epochs 30

# GPU (wgpu)
cargo run --release -p phonopaper-train --no-default-features --features wgpu -- \
    train --dataset dataset --artifacts artifacts --epochs 30
```

| Option | Default | Description |
|---|---|---|
| `-d, --dataset <DIR>` | `dataset` | Dataset directory |
| `-a, --artifacts <DIR>` | `artifacts` | Output directory |
| `--epochs <N>` | `30` | Training epochs |
| `--batch-size <N>` | `32` | Mini-batch size |
| `--learning-rate <F>` | `0.001` | Adam learning rate |
| `--seed <U64>` | `42` | Weight initialisation / shuffling seed |
| `--workers <N>` | `4` | Data-loader threads |
| `--input-size <PX>` | `128` | Network input side; must equal the dataset `--size` |

Images whose index is a multiple of 10 form the **validation split**; the
rest is the training split.

### How much data / how long?

A quick smoke test (3 000 images, 8 epochs, ≈ 20 min on a throttled laptop
GPU) reaches ≈ 80 % presence accuracy with a ≈ 30 px mean corner error —
enough to verify the pipeline, not to ship.  For a usable detector plan on:

* **20 000 – 50 000 images** (`--count`; generation takes a few seconds per
  thousand images and the dataset is ≈ 10 kB per image);
* **30 – 60 epochs**, watching the validation loss in `artifacts/valid/`;
* lowering `--learning-rate` to `3e-4` for the last third of training if the
  validation loss plateaus.

Because the dataset is deterministic, "more data" simply means a larger
`--count` with the same seed: the first *N* images are unchanged.

The artifact directory receives burn's logs and per-epoch checkpoints plus:

| File | Content |
|---|---|
| `model.json` | `DetectorConfig` (input size, hidden width, dropout) |
| `training.json` | All training hyper-parameters |
| `model.mpk` | Full-precision checkpoint (`NamedMpkFileRecorder`) |
| `model.bin` | **Weights to embed** (`BinFileRecorder`, full precision, ≈ 2 MB) |

### The network

`phonopaper-train/src/model.rs` — a VGG-style CNN of ≈ 500 k parameters:

```text
input  [1 × 128 × 128]  (grayscale, values in [0, 1])
5 × ( conv 3×3 → BatchNorm → ReLU → maxpool 2 )   channels 16, 32, 64, 128, 128
flatten (2048) → Linear 128 → ReLU → Dropout 0.2 → Linear 9
```

Output row layout: `[presence logit, x0, y0, x1, y1, x2, y2, x3, y3]` with
corners normalised by the input side (`0` = left/top edge, `1` = right/bottom
edge; values slightly outside `[0, 1]` are legitimate).

Loss = binary cross-entropy on the presence logit + 20 × smooth-L1 (δ = 0.05)
on the corners, the latter averaged over positive samples only.

---

## 3. Evaluate and try the model

```bash
cargo run --release -p phonopaper-train -- eval --dataset dataset --artifacts artifacts
```

prints presence accuracy / precision / recall on the validation split, the
mean corner error in pixels over true positives and the fraction of corners
within 3 px and 6 px.

```bash
cargo run --release -p phonopaper-train -- infer --artifacts artifacts photo1.jpg photo2.png
```

prints one JSON object per image with `probability`, `present` (threshold
`--threshold`, default 0.5) and `corners` in **original image pixels**.  The
image is converted to grayscale and stretched (aspect ratio not preserved) to
the network input size; the normalised corners are mapped back by multiplying
with the original width and height.

> `infer` only understands PNG out of the box; enable more `image` codecs in
> `phonopaper-train/Cargo.toml` if needed.

---

## 4. Transfer the model to `phonopaper-rs`

The detector is meant to replace / complement the hand-written stripe
detector (`phonopaper_rs::decode::detect_markers`) inside the library and,
through `phonopaper-android`, in the Android app.  The steps are:

### 4.1 Add burn (inference only) as an optional dependency

In `phonopaper-rs/Cargo.toml`:

```toml
[features]
# Neural-network corner detector (pulls in burn for inference).
nn-detector = ["dep:burn"]

[dependencies]
burn = { version = "0.21", default-features = false, features = ["std", "ndarray"], optional = true }
```

Only the `ndarray` (or `flex`) CPU backend is needed for inference; neither
`train` nor `autodiff` are required, which keeps the dependency tree small
enough for the Android `cdylib`.

### 4.2 Copy the model definition

Copy `phonopaper-ml/phonopaper-train/src/model.rs` to
`phonopaper-rs/src/decode/nn/model.rs` unchanged.  It depends only on
`burn::nn`, `burn::prelude` and `burn::tensor::activation`.  Make sure the
struct field names and order stay identical to the training crate — burn's
recorders match weights by field name.

### 4.3 Embed the weights

Copy `artifacts/model.bin` to `phonopaper-rs/src/decode/nn/model.bin` (≈ 2 MB)
and `artifacts/model.json` to `phonopaper-rs/src/decode/nn/model.json`, then:

```rust
//! phonopaper-rs/src/decode/nn/mod.rs
mod model;

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::module::Module;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};

pub use model::{Detection, Detector, DetectorConfig};

static WEIGHTS: &[u8] = include_bytes!("model.bin");
const CONFIG: &str = include_str!("model.json");

/// Load the embedded detector.
pub fn load() -> Detector<NdArray> {
    let device = NdArrayDevice::Cpu;
    // `DetectorConfig` derives serde, so it can be read from the embedded
    // JSON; if you trained with the defaults, `DetectorConfig::new()` works.
    let config: DetectorConfig = serde_json::from_str(CONFIG).expect("valid model.json");
    let record = BinBytesRecorder::<FullPrecisionSettings, &'static [u8]>::default()
        .load(WEIGHTS, &device)
        .expect("embedded weights match the model definition");
    config.init::<NdArray>(&device).load_record(record)
}
```

`Detector::detect(&pixels, &device)` then returns a `Detection` with
`probability` and normalised `corners`.  For a `W × H` camera frame:

1. convert to grayscale and resize (stretch) to `input_size × input_size`
   (same as `phonopaper_train::infer::prepare_image`);
2. call `detect`;
3. if `probability ≥ 0.5`, `detection.corners_in_pixels(W, H)` gives the four
   corners `TL, TR, BR, BL` in frame coordinates, in pattern orientation.

### 4.4 Use the corners

With the four corners you can either:

* **rectify** the pattern: compute the homography mapping the detected quad to
  an upright rectangle (`phonopaper_dataset::geometry::Homography::from_quads`
  is a dependency-free reference implementation you may copy) and sample the
  image column by column — then feed the columns to
  `phonopaper_rs::decode::column_amplitudes_from_image` with `DataBounds`
  derived from the marker geometry; or
* **seed the existing detector**: run `detect_markers_at_column` only on
  columns inside the detected quad, which removes the false positives that
  motivated this work.

Corner accuracy is roughly ±2 px at the 128 px network resolution, i.e.
±1.5 % of the frame.  For sub-pixel precision, refine each corner with a local
edge search in the full-resolution frame.

### 4.5 Keep the gates green

Adding burn to the library changes the dependency set, so run the six checks
from `AGENTS.md`, add tests for the new public functions (e.g. a round-trip
that renders a pattern with `spectrogram_to_image`, warps it and checks that
the detector finds its corners), and update the coverage baselines.

---

## Development

The same quality bar as the main workspace applies here:

```bash
cargo fmt --check --all
cargo clippy --workspace --all-targets
cargo test --workspace
```

(`cargo test` for `phonopaper-train` always uses the small `ndarray` backend
through a dev-dependency, whatever features are selected.)
