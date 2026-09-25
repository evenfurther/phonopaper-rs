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
marker band**, whatever the rotation.  The corners delimit the *ink box* —
from the first stripe of the top band to the last stripe of the bottom band,
across all data columns — the white margins are not included.

> Note that a `PhonoPaper` sheet looks identical when turned by 180°, so the
> labels carry information the image itself does not (which end is the top).
> The trainer accounts for this with a symmetric loss; see *Orientation
> ambiguity* below.

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

> Neither GPU feature enables burn's `fusion` layer: with burn 0.21 it has
> produced an invalid CUDA kernel at export time and crashed mid-training on
> wgpu (`Should have handle for tensor …`).  Plain kernels are somewhat
> slower but reliable.

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
| `--patience <N>` | `8` | Stop early when the validation loss has not improved for N epochs |
| `--min-lr-fraction <F>` | `0.05` | Final learning rate of the cosine decay, as a fraction of `--learning-rate` |

Images whose index is a multiple of 10 form the **validation split**; the
rest is the training split.

Training keeps **every** epoch's checkpoint in `artifacts/checkpoint/`
(≈ 4 MB each with optimiser state) while it runs.  When it ends (after
`--epochs` or by early stopping), the epoch with the lowest mean validation
loss is exported to `model.bin` — on the CPU, independently of the training
backend — and the checkpoints are pruned to that epoch and the last one.

> burn's metric-based checkpointing strategy is deliberately not used: it
> only saves an epoch that is already the best when the checkpoint decision
> is taken and cannot rescue it later, which lost the best epoch in practice.

### Recovering a model from checkpoints

If a run was interrupted, or you want a specific epoch:

```bash
cargo run --release -p phonopaper-train -- export --artifacts artifacts            # best validation epoch
cargo run --release -p phonopaper-train -- export --artifacts artifacts --epoch 12 # a specific epoch
```

`export` prints the mean validation loss of every epoch it can find in
`artifacts/valid/`, then writes `model.mpk` and `model.bin`.  Only epochs
still present in `artifacts/checkpoint/` can be exported.

### Overfitting

Synthetic data is cheap, so the remedy for a validation loss that rises
while the training loss keeps falling is simply **more images**: a 30 k
dataset overfits within ~12 epochs, a 300 k one (≈ 3 GB, minutes to
generate) does not.  Early stopping (`--patience`) and best-epoch export
make sure an over-long run still yields the best model.

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
| `model.json` | `DetectorConfig` (input size, corner-head width) |
| `training.json` | All training hyper-parameters |
| `model.mpk` | Full-precision copy of the exported epoch (`NamedMpkFileRecorder`) |
| `model.bin` | **Weights to embed** (`BinFileRecorder`, full precision, ≈ 1.1 MB) |
| `checkpoint/` | Per-epoch checkpoints (all during training; best + last afterwards) |

### The network

`phonopaper-train/src/model.rs` — a small CNN of ≈ 280 k parameters with two
heads:

```text
input     [1 × 128 × 128]  (grayscale, values in [0, 1])
trunk     5 × ( conv 3×3 → BatchNorm → ReLU → maxpool 2 )   channels 16, 32, 64, 128, 128
presence  global average pool of the last stage (128) → Linear → 1 logit
corners   stage 2 (32 × 32 × 32, stride 4) ⊕ stage 3 (64 × 16 × 16) upsampled ×2
          → conv 3×3 → BatchNorm → ReLU → conv 1×1 → 4 heat-maps (32 × 32) → soft-argmax
```

Corners are localised with **heat-maps + soft-argmax** rather than a fully
connected regression: the coordinate of each corner is the softmax-weighted
mean of the heat-map cell centres, which keeps the spatial information of
the feature map and is continuous (sub-cell precision).  The heat-map input
merges a fine stage (stride 4, sharp edges) with a coarser one (stride 8,
context), U-Net style.  A fully connected head was tried first and plateaued
at ≈ 13 px mean error on 128 px inputs, identically on the training and
validation splits — a capacity limit, not over-fitting; stride-8 heat-maps
brought that to 6.4 px.  The soft-argmax grid spans `[-0.1, 1.1]` so corners
slightly outside the frame remain representable.

Output row layout: `[presence logit, x0, y0, x1, y1, x2, y2, x3, y3]` with
corners normalised by the input side (`0` = left/top edge, `1` = right/bottom
edge; values slightly outside `[0, 1]` are legitimate).

Loss = binary cross-entropy on the presence logit + 20 × smooth-L1 on the
corners, the latter averaged over positive samples only.  The smooth-L1
transition δ is **0.01** (≈ 1.3 px): below δ the loss turns quadratic and
its gradient fades, so δ is effectively the precision the network stops
caring about — with δ = 0.05 the mean error plateaued at exactly 6.4 px.

The learning rate follows a **cosine decay** from `--learning-rate` to
`--learning-rate × --min-lr-fraction` (default 0.05) over `--epochs`; the
plateau of a constant rate showed up as the best epoch sitting in the
middle of the run with no further progress.

#### Orientation ambiguity

A `PhonoPaper` sheet is symmetric under a 180° turn: the bottom marker band
is the mirror image of the top one and every stripe spans the full width.
The image therefore cannot tell `TL, TR, BR, BL` from `BR, BL, TL, TR`.  The
corner loss (and `eval`) take the **minimum over both orderings**, so the
network is free to commit to either; without this the network hedges towards
the average of the two — the pattern centre — and corner errors of ~20 px
result.  At inference, `Detection::canonical()` picks the ordering whose first
corner is higher in the image.  What the network *does* tell you is which two
edges carry the marker bands (`c0→c1` and `c2→c3`); which of them is the
high-frequency end must come from elsewhere (the phone's orientation, or
decoding both ways and keeping the one that sounds right).

### Batch job on a Slurm cluster

`slurm/train-a40.sbatch`, `slurm/train-h100.sbatch` and
`slurm/train-rtx6000pro.sbatch` do steps 1–3 unattended on one GPU: build,
generate the dataset if missing, train with early stopping, export the best
epoch and evaluate it on both splits.  They only differ in resource
directives and default backend / batch size / learning rate / worker count;
the shared job body is `slurm/lib/train-and-eval.sh`.

```bash
cd phonopaper-ml
sbatch slurm/train-a40.sbatch                 # A40, CUDA:          batch 256, lr 2e-3,  8 workers
sbatch slurm/train-h100.sbatch                # H100, CUDA:         batch 512, lr 3e-3, 16 workers
sbatch slurm/train-rtx6000pro.sbatch          # RTX 6000 Pro, wgpu: batch 512, lr 3e-3, 16 workers
# tunables are environment variables:
DATASET=dataset-200k EPOCHS=60 BATCH_SIZE=512 LEARNING_RATE=3e-3 sbatch slurm/train-a40.sbatch
# other partition / GPU with the A40 defaults:
sbatch --partition=V100-32GB slurm/train-a40.sbatch
```

Variables: `REPO`, `DATASET`, `DATASET_COUNT`, `ARTIFACTS` (default
`artifacts-<jobid>`), `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `PATIENCE`,
`WORKERS`, `SEED`, `BACKEND` (`cuda` or `wgpu`), `CUDA_MODULE`,
`CARGO_TARGET_DIR`.  Output lands in `phonopaper-train-<jobid>.out` in the
submission directory; the trained model is `<ARTIFACTS>/model.bin`.

> The RTX 6000 Pro script uses the Vulkan `wgpu` backend, which needs only
> the graphics driver.  With `BACKEND=cuda` on that (Blackwell) GPU the CUDA
> module must provide `libnvrtc` ≥ 12.8 (`CUDA_MODULE=cuda/12.8 …`).

> The workspace compiles with `-C target-cpu=native` and Cargo does not
> notice when a cached binary was built on a different CPU.  The script
> therefore builds into `target/cpu-<cpu model>/` on the node itself; do not
> point `CARGO_TARGET_DIR` at a directory shared with builds from other
> machines.

---

## 3. Evaluate and try the model

```bash
cargo run --release -p phonopaper-train -- eval --dataset dataset --artifacts artifacts
cargo run --release -p phonopaper-train -- eval --dataset dataset --artifacts artifacts --split train
cargo run --release -p phonopaper-train -- eval --dataset dataset --artifacts artifacts --epoch 12
```

prints, for the validation split (default) or the training split, presence
accuracy / precision / recall, the mean corner error in pixels over true
positives, the fraction of corners within 3 px and 6 px, and the same error
**relative to the pattern's longest edge** (mean, and fractions within 2 %
and 5 %) — a 6 px error means something different on a 30 px pattern than
on a 120 px one.  Comparing the
two splits tells **under-fitting** (both poor → train longer / stronger
signal) from **over-fitting** (train good, valid poor → more data).
`--epoch N` evaluates the checkpoint of epoch `N` (it must still exist in
`artifacts/checkpoint/`) instead of the exported `model.bin`, so epochs can
be compared without re-exporting.

```bash
cargo run --release -p phonopaper-train -- infer --artifacts artifacts photo1.jpg photo2.png
```

prints one JSON object per image with `probability`, `present` (threshold
`--threshold`, default 0.5) and `corners` in **original image pixels**.
`--epoch N` uses a checkpoint instead of `model.bin`, as for `eval`.  The
image is converted to grayscale and stretched (aspect ratio not preserved) to
the network input size; the normalised corners are mapped back by multiplying
with the original width and height.

> `infer` only understands PNG out of the box; enable more `image` codecs in
> `phonopaper-train/Cargo.toml` if needed.

---

## 4. Transfer the model to `phonopaper-rs`

The detector lives in the library as `phonopaper_rs::decode::nn`, behind the
**`nn-detector`** Cargo feature (off by default so that burn never affects the
default build).  It complements the hand-written stripe detector
(`phonopaper_rs::decode::detect_markers`) and is what the Android app uses
through `phonopaper-android`: the network locates the sheet in every preview
frame (outline overlay, auto-play trigger) and, for decoding, the detected
quadrilateral is rectified with a homography before the stripe detector reads
it (`phonopaper_android::rectify_pattern`).  The pieces in the library are:

| File in `phonopaper-rs/` | Content |
|---|---|
| `Cargo.toml` | `nn-detector = ["dep:burn"]`; `burn` with only `std` + `ndarray` (inference on the CPU, no `train` / `autodiff`) |
| `src/decode/nn/model.rs` | **Verbatim copy** of `phonopaper-train/src/model.rs` |
| `src/decode/nn/model.bin`, `model.json` | The exported weights and their `DetectorConfig`, embedded with `include_bytes!` |
| `src/decode/nn/mod.rs` | `load()`, `prepare_image()` and the `PatternDetector` wrapper |
| `tests/nn.rs` | Renders a pattern with `spectrogram_to_image`, warps it into a scene and checks that the detector finds its corners; negatives; ordering helpers |

### 4.1 Refresh the embedded model

After a new training run:

```bash
cp artifacts/model.bin  ../phonopaper-rs/src/decode/nn/model.bin
cp artifacts/model.json ../phonopaper-rs/src/decode/nn/model.json
# only if the architecture changed:
cp phonopaper-train/src/model.rs ../phonopaper-rs/src/decode/nn/model.rs
```

burn's recorders match weights **by field name**, so `model.rs` must be
identical in both crates and the weights must have been exported from that
very definition, or `load()` panics at start-up.  Note that burn does *not*
check tensor shapes when loading: weights exported with a different `hidden`
or channel width load silently and only misbehave later.
`phonopaper-train/tests/embedded.rs` guards against both mistakes — it
compares the two `model.rs` byte for byte and checks that the embedded
`model.bin` loads with exactly the parameter shapes of `model.json`:

```bash
cargo test -p phonopaper-train --test embedded
```

Then, from the repository root:

```bash
cargo clippy -p phonopaper-rs --all-targets --features nn-detector
cargo test -p phonopaper-rs --features nn-detector
```

`tests/nn.rs` places synthetic patterns at known positions and asserts on the
presence probability and mean corner error; a weaker model shows up there.

### 4.2 Use the detector

```rust
use phonopaper_rs::decode::nn::PatternDetector;

let detector = PatternDetector::new();          // deserialises the embedded weights once
let frame = image::open("photo.jpg")?;
if let Some(corners) = detector.find_corners(&frame, 0.5) {
    // [TL, TR, BR, BL] in frame pixels, canonical order (see *Orientation ambiguity*)
}
```

`PatternDetector::detect` returns the raw normalised `Detection`;
`detect_prepared` accepts an already downscaled `input_size × input_size`
luminance plane (what a camera pipeline can deliver directly), and `load()`
gives the bare `Detector<NdArray>` for batches or heat-map inspection.  For a
`W × H` frame the pipeline is: grayscale + stretch to `input_size ×
input_size` (`prepare_image`, same as `phonopaper_train::infer`), `detect`,
then `detection.corners_in_pixels(W, H)`.

### 4.3 Use the corners

With the four corners you can either:

* **rectify** the pattern: compute the homography mapping the detected quad to
  an upright rectangle and resample the image, then run the usual upright
  decoding on the result — this is what `phonopaper-android` does
  (`rectify_pattern` in `phonopaper-android/src/lib.rs`, with a small white
  border so the stripe detector sees a light run before the first stripe;
  `phonopaper_dataset::geometry::Homography` is the same maths); or
* **seed the existing detector**: run `detect_markers_at_column` only on
  columns inside the detected quad, which removes the false positives that
  motivated this work.

Corner accuracy is roughly ±2 px at the 128 px network resolution, i.e.
±1.5 % of the frame.  For sub-pixel precision, refine each corner with a local
edge search in the full-resolution frame.

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
