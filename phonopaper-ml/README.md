# phonopaper-ml — neural-network pattern detector

This directory is a **separate Cargo workspace** with the tooling used to
train a small convolutional neural network that finds a `PhonoPaper` pattern
in a camera frame and returns its four corners.  It is kept out of the main
workspace so that the (large) [burn](https://burn.dev) dependency tree never
affects the library, its lock file or its quality gates.

| Crate | Purpose |
|---|---|
| `phonopaper-dataset` | Deterministic synthetic dataset generator (images + `labels.csv`) |
| `phonopaper-train`   | Model definition, training, evaluation and inference with Burn 0.22 |

The end-to-end workflow is:

1. [generate the dataset](#1-generate-the-dataset) — seconds;
2. [train the network](#2-train-the-network) — minutes to an hour depending on backend;
3. [evaluate / try it](#3-evaluate-and-try-the-model);
4. [transfer the model to `phonopaper-rs`](#4-transfer-the-model-to-phonopaper-rs).

All commands below are run from this directory (`phonopaper-ml/`). For the
complete local workflow, use the checked automation rather than transcribing
individual commands:

```bash
# Generates dataset-v2-scale3-50k only when needed, trains, then evaluates
# both validation and training splits. Defaults to the portable CPU backend.
./scripts/train-and-eval.sh

# Typical GPU run and environment overrides:
BACKEND=wgpu DATASET_COUNT=200000 BATCH_SIZE=16 EPOCHS=50 \
    ARTIFACTS=artifacts-v2 ./scripts/train-and-eval.sh

# After reviewing metrics, safely stage and test all three embedded artifacts:
ARTIFACTS=artifacts-v2 ./scripts/embed-model.sh
```

Every script uses Bash strict mode, resolves repository paths relative to its
own location, and accepts only non-secret environment configuration. The
shared dataset check reuses a dataset only when `manifest.json` says format
version 2 with the requested `SOURCE_SCALE` (default 3) and `labels.csv` is
present. A missing, malformed, v1, or differently scaled dataset is removed
and deterministically rebuilt instead of being silently reused.

> **Current embedded-model status:** the checked-in application model is
> **not loadable with the current stride-2 heat-map architecture** until a new
> model is trained and embedded with `scripts/embed-model.sh`. Do not ship or
> run the Android detector from this worktree before completing that step.

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
| `--source-scale <N>` | `3` | Integer scale of the synthetic camera frame before production `Triangle` resizing; must be at least 2 |

The output directory contains:

* `000000.png`, `000001.png`, … — 8-bit grayscale PNGs;
* `labels.csv` — one row per image (see below);
* `manifest.json` — format version, generator parameters, exact preprocessing,
  curriculum policy, corner convention and counts.

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
the encoder uses. The deterministic positive curriculum assigns 20% to clean,
high-contrast localization scenes, 20% to deliberate narrow-margin/frame-filling
scenes, and 60% to the existing varied photo-like path. Each complete scene is
rendered at `--source-scale` times the stored dimensions and then resized with
`image::FilterType::Triangle`, exactly matching production preprocessing;
labels are divided by the same scale into final network-sized pixel coordinates
without clipping, so out-of-frame corners remain valid. The varied path retains:

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
| add `--features tui` | — | Interactive terminal dashboard instead of plain logs |

Burn 0.22 selects the backend at runtime through `Device`; model and tensor
types no longer carry backend parameters. The CPU `flex` backend is also used
for loading and exporting models regardless of the training device. Burn 0.22
requires Rust 1.95 or newer.

> Neither GPU feature enables Burn's `fusion` layer: it previously produced an
> invalid CUDA kernel at export time and crashed mid-training on wgpu
> (`Should have handle for tensor …`). Plain kernels are somewhat slower but
> reliable.

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
loss is exported to `model.bpk` — on the CPU, independently of the training
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
`artifacts/valid/`, then writes `model.bpk`. Only epochs
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
| `model.bpk` | **Weights to embed** (Burnpack format) |
| `checkpoint/` | Per-epoch model, optimizer and scheduler checkpoints (all during training; best + last afterwards) |

Burn 0.22 stores checkpoints in Burnpack format. Older Burn recorder files
(`.mpk` / `.bin`) cannot be loaded directly; migrate their model weights via
SafeTensors as described in the [Burn 0.22 migration guide](https://burn.dev/books/burn/migrating-to-0.22.html#migrating-checkpoints).

### The network

`phonopaper-train/src/model.rs` defines a compact CNN with two heads:

```text
input     [1 × 128 × 128] (grayscale, values in [0, 1])
trunk     5 × (conv 3×3 → BatchNorm → ReLU → maxpool 2)
          channels 16, 32, 64, 128, 128
presence  global average pool of final stride-32 stage → Linear(128 → 1 logit)
corners   stride-2 stage 0 (16 × 64 × 64)
          ⊕ stage 1 (32 × 32 × 32) nearest-upsampled ×2
          ⊕ stage 2 (64 × 16 × 16) nearest-upsampled ×4
          → conv 3×3 (64 channels) → BatchNorm → ReLU → conv 1×1
          → 4 heat-maps (64 × 64) → soft-argmax
```

Corners are localised with **heat-maps + soft-argmax**, not a fully connected
regressor. Each coordinate is the probability-weighted mean of one heat-map,
so it remains continuous while preserving fine spatial evidence. The U-Net-like
fusion combines stride-2 edges with stride-4 and stride-8 context. Its
64×64 maps consume substantially more training activation memory than the old
stride-4 head, so automation deliberately defaults to batches of 16 locally,
32 on A40, and 64 on the larger cluster GPUs. Increase these only after
measuring peak memory. The coordinate grid spans `[-0.1, 1.1]`, preserving
labels for corners just outside the frame.

Output rows are `[presence logit, x0, y0, x1, y1, x2, y2, x3, y3]`; corner
coordinates are divided by the input side, and may legitimately fall outside
`[0, 1]`.

The objective is:

```text
presence BCE
+ 1 × normalized-Gaussian heat-map cross-entropy
+ 20 × smooth-L1 decoded-coordinate loss
```

Spatial terms are averaged over positive samples only and take the minimum of
the labelled ordering and its 180°-rotated equivalent. Heat-map targets use
σ = 0.02 in normalized coordinates (2.56 px at 128 px). Smooth-L1 uses
δ = 0.01 (≈1.3 px), retaining useful localization gradients below the older
δ = 0.05 plateau. An all-negative batch contributes finite zero spatial loss.

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

### Batch jobs on a Slurm cluster

Generate the dataset independently on the CPU partition with:

```bash
cd phonopaper-ml
sbatch slurm/generate-dataset.sbatch
# Override the output and size when needed:
DATASET=dataset-v2-scale3-300k DATASET_COUNT=300000 \
    sbatch slurm/generate-dataset.sbatch
```

The generator already parallelizes images with Rayon. Each image has an
independent RNG stream derived from `(seed, index)`, so thread scheduling does
not affect the bytes written. The job sets `RAYON_NUM_THREADS` to the allocated
`SLURM_CPUS_PER_TASK` (32 by default). Override `--partition` at submission time
if the cluster's CPU partition has a different name.

`slurm/train-a40.sbatch`, `slurm/train-h100.sbatch` and
`slurm/train-rtx6000pro.sbatch` do steps 1–3 unattended on one GPU: build,
generate the dataset if missing, train with early stopping, export the best
epoch and evaluate it on both splits. They only differ in resource directives
and default backend / batch size / learning rate / worker count; the shared job
body is `slurm/lib/train-and-eval.sh`. Submit the CPU generation job first when
you do not want dataset generation to consume GPU allocation time.

```bash
cd phonopaper-ml
sbatch slurm/train-a40.sbatch                 # A40, CUDA:          batch 32, lr 1e-3,  8 workers
sbatch slurm/train-h100.sbatch                # H100, CUDA:         batch 64, lr 1e-3, 16 workers
sbatch slurm/train-rtx6000pro.sbatch          # RTX 6000 Pro, wgpu: batch 64, lr 1e-3, 16 workers
# tunables are environment variables:
DATASET=dataset-v2-scale3-200k EPOCHS=60 BATCH_SIZE=48 sbatch slurm/train-a40.sbatch
# other partition / GPU with the A40 defaults:
sbatch --partition=V100-32GB slurm/train-a40.sbatch
```

Variables: `REPO`, `DATASET` (default `dataset-v2-scale3-200k`),
`DATASET_COUNT`, `SOURCE_SCALE`, `IMAGE_SIZE`, `POSITIVE_RATIO`,
`DATASET_SEED`, `ARTIFACTS` (default `artifacts-<jobid>`), `EPOCHS`,
`BATCH_SIZE`, `LEARNING_RATE`, `PATIENCE`, `WORKERS`, `SEED`, `BACKEND`
(`cuda` or `wgpu`), `CUDA_MODULE`, `CARGO_TARGET_DIR`, and `PYTHON`. Before
reuse, the shared script parses the manifest and requires format v2 plus the
requested source scale; incompatible directories are rebuilt. Output lands in
`phonopaper-train-<jobid>.out` in the submission directory; the trained model
is `<ARTIFACTS>/model.bpk`.

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

prints, for the validation split (default) or the training split:

* presence accuracy / precision / recall;
* mean corner error, fractions within 3 px and 6 px, and corner-error p50,
  p90, p95 and worst case, all over correctly detected positive images;
* corner error **relative to the truth quadrilateral's longest edge** (mean,
  and fractions within 2 % and 5 %) — a 6 px error means something different
  on a 30 px pattern than on a 120 px one;
* mean, p10 and worst-case **quadrilateral IoU**. This is exact polygon IoU for
  convex predicted and truth quadrilaterals, computed by convex clipping;
  degenerate, self-intersecting or non-convex predictions score zero;
* signed bias and mean absolute error of width and height scale, where width is
  the mean length of the two marker-band edges, height is the mean length of
  the other two edges, and scale error is `predicted / truth - 1`. Positive
  bias therefore exposes systematic expanded-box predictions even when corner
  errors alone look tolerable.

Corner correspondence uses whichever of the direct and 180-degree-equivalent
sheet orderings has lower total corner error. IoU and dimensions describe the
quadrilateral geometry and are unchanged by that equivalent reordering.
Comparing the two splits tells **under-fitting** (both poor → train longer /
stronger signal) from **over-fitting** (train good, valid poor → more data).
`--epoch N` evaluates the checkpoint of epoch `N` (it must still exist in
`artifacts/checkpoint/`) instead of the exported `model.bpk`, so epochs can
be compared without re-exporting.

```bash
cargo run --release -p phonopaper-train -- infer --artifacts artifacts photo1.jpg photo2.png
```

prints one JSON object per image with `probability`, `present` (threshold
`--threshold`, default 0.5) and `corners` in **original image pixels**.
`--epoch N` uses a checkpoint instead of `model.bpk`, as for `eval`. The
image is converted to grayscale and stretched (aspect ratio not preserved) to
the network input size; the normalised corners are mapped back by multiplying
with the original width and height.

For a repeatable fixture set, the wrapper writes JSON Lines and can compare it
byte-for-byte with a reviewed baseline:

```bash
ARTIFACTS=artifacts-v2 OUTPUT=results.jsonl \
    ./scripts/regression-infer.sh fixtures/positive.png fixtures/negative.png
ARTIFACTS=artifacts-v2 EXPECTED=baselines/results.jsonl \
    ./scripts/regression-infer.sh fixtures/positive.png fixtures/negative.png
```

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
| `Cargo.toml` | `nn-detector = ["dep:burn"]`; `burn` with only `std` + `flex` (inference on the CPU, no `train` / `autodiff`) |
| `src/decode/nn/model.rs` | **Verbatim copy** of `phonopaper-train/src/model.rs` |
| `src/decode/nn/model.bpk`, `model.json` | The exported Burnpack weights and their `DetectorConfig`, embedded with `include_bytes!` |
| `src/decode/nn/mod.rs` | `load()`, `prepare_image()` and the `PatternDetector` wrapper |
| `tests/nn.rs` | Renders a pattern with `spectrogram_to_image`, warps it into a scene and checks that the detector finds its corners; negatives; ordering helpers |

### 4.1 Refresh the embedded model

After reviewing a completed training run, use the safe transfer script:

```bash
ARTIFACTS=artifacts-v2 ./scripts/embed-model.sh
```

It requires non-empty `model.bpk` and `model.json`, stages those files together
with the current `phonopaper-train/src/model.rs`, installs all three, and runs
`cargo test -p phonopaper-train --test embedded`. If validation fails or the
script is interrupted, it restores the previous embedded files. This avoids a
partially copied or architecture-mismatched model.

Burn's recorders match weights **by field name**, so `model.rs` must be
identical in both crates and the weights must come from that exact definition,
or `load()` fails at start-up. The embedded test compares `model.rs` byte for
byte and loads `model.bpk` against the parameter shapes in `model.json`.

> The current checked-in weights predate the stride-2 head and are not loadable.
> A newly trained model must pass this script before the app detector is usable.

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
gives the bare backend-free `Detector` for batches or heat-map inspection. For a
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

(`cargo test` for `phonopaper-train` always uses the portable `flex` backend
through a dev-dependency, whatever features are selected.)
