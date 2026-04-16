# PhonoPaper Format Specification

> **Author of the format:** Alexander Zolotov (NightRadio) —
> <https://warmplace.ru/soft/phonopaper/>
>
> **This document:** a working specification for this library, assembled by
> reverse-engineering published sample images, measuring the official Android
> application's output, and reading a third-party JavaScript reimplementation
> (<https://github.com/zz85/phonopaper.js>).  **The original author has not
> published a formal specification or source code.**  Claims in this document
> are best-effort observations, not authoritative definitions.  See §0 for a
> per-section confidence assessment.

---

## Table of Contents

0. [Status of This Document](#0-status-of-this-document)
1. [Overview](#1-overview)
2. [Image Anatomy](#2-image-anatomy)
3. [Marker Bands](#3-marker-bands)
4. [Data Area](#4-data-area)
5. [Frequency Scale](#5-frequency-scale)
6. [Pixel-to-Amplitude Mapping](#6-pixel-to-amplitude-mapping)
7. [Decoding: Image → Audio](#7-decoding-image--audio)
8. [Encoding: Audio → Image](#8-encoding-audio--image)
9. [Reference Constants](#9-reference-constants)
10. [Worked Examples](#10-worked-examples)
11. [Implementation Notes and Edge Cases](#11-implementation-notes-and-edge-cases)

---

## 0. Status of This Document

This section summarises **how each part of the spec was derived** and how much
confidence we have in it.  Three confidence levels are used throughout:

| Symbol | Meaning |
|--------|---------|
| ✅ **Confirmed** | Directly measured from multiple reference images or the official Android application output; consistent with the JS reimplementation. |
| ⚠️ **Inferred** | Derived from indirect evidence (pixel counting, frequency arithmetic, cross-checking with the JS reimplementation).  Likely correct but unverified against the original source. |
| ❓ **Uncertain** | Based on a single data point, an assumption, or conflicting evidence.  The real behaviour may differ. |

### Per-section summary

| Section | Claim | Status |
|---------|-------|--------|
| §2 Image anatomy — pixel dimensions | Measured from official Android-app images and warmplace.ru samples | ✅ |
| §3 Marker detection — stripe pattern | Directly observed in every reference image | ✅ |
| §3 Marker detection — data boundary rule | Deduced from pixel measurement; validated against Android output | ✅ |
| §4 Frequency grid (384 bins, 8 oct, 4 sub) | Consistent across all reference material | ✅ |
| §5 Frequency formula | Corrected to pivot=252 by measuring Android output (sines.jpg round-trip) | ✅ |
| §6 Pixel → amplitude: **fractional** `(255−L)/255` | **Our default implementation** — see §6.2 for discussion | ❓ |
| §6 Pixel → amplitude: **binary threshold** | Empirically fits Android output better than fractional | ⚠️ |
| §7 `samples_per_column = 353` | Matches Android's confirmed rate of 352.8 (= 44 100 ÷ 125) | ✅ |
| §7 `gain = 3.0` | This library's chosen default; not observed in reference output | ❓ |
| §7 Phase-continuous synthesis | Inferred from quality considerations; not confirmed in reference app | ⚠️ |
| §7 Frequency range synthesised | Android app synthesises **≈ 131–4186 Hz** (bins 96–336, C3–C8) | ⚠️ |
| §8 STFT parameters (FFT=4096, hop=353) | hop=353 chosen to match Android column rate; FFT size is a reasonable default | ⚠️ |
| §11 Reading direction left-to-right | Observed in all reference material; no contrary evidence | ✅ |

---

## 1. Overview

PhonoPaper encodes a mono audio signal as a **grayscale spectrogram image**. The
image can be printed on paper and decoded in real time by sweeping a camera from
left to right over it, or processed offline by software.

The format is essentially an **analog, logarithmically-scaled, printable
spectrogram** with the following conventions:

| Axis | Meaning |
|------|---------|
| X (left → right) | Time |
| Y (top → bottom) | Frequency — highest at top, lowest at bottom |
| Pixel brightness | **White = silence, Black = maximum amplitude** |

The image is divided into three vertical zones stacked top to bottom:

```
┌─────────────────────────────┐
│       Top marker band       │
├─────────────────────────────┤
│                             │
│         Data area           │  ← audio content
│                             │
├─────────────────────────────┤
│      Bottom marker band     │
└─────────────────────────────┘
```

The marker bands are barcode-like stripes that allow a decoder to locate and
calibrate the data area — even when the image is printed at an unknown scale,
tilted slightly, or viewed through a camera.

---

## 2. Image Anatomy ✅

A well-formed PhonoPaper image has the following vertical layout (all measurements
are in pixels, using the reference default values, measured from official
Android-app output):

```
Row offset   Content                        Height (px)
──────────   ───────────────────────────    ──────────
0            White margin                   88
88           Thin black stripe              9
97           White gap                      10
107          Thin black stripe              9
116          White gap                      10
126          THICK black stripe             39    ← key identifier
165          White gap                      10
175          Thin black stripe              9
184          ── data area begins ──
184          Audio data rows                720   (= 8 octaves × 90 px/octave)
904          ── data area ends ──
904          Thin black stripe              9    ← bottom band is top band mirrored
913          White gap                      10
923          THICK black stripe             39    ← thick stripe closest to data area
962          White gap                      10
972          Thin black stripe              9
981          White gap                      10
991          Thin black stripe              9
1000         White margin                   88
1088         ── image bottom ──
```

Total image height with default settings: **1088 px**.

Image width is variable and corresponds directly to the number of time columns
(audio frames).  The Android application produces images with **1400 columns**
for its default recording length; the number of columns is not fixed by the
format.

> **Note:** The absolute pixel dimensions are not fixed by the format. A decoder
> must locate the data area dynamically using the marker bands (see §3).

---

## 3. Marker Bands

### 3.1 Structure ✅

Each marker band consists of four **horizontal black stripes** separated by white
gaps, arranged (top to bottom) as:

```
[white margin]
[thin stripe]
[white gap]
[thin stripe]
[white gap]
[THICK stripe]    ← the identifying feature
[white gap]
[thin stripe]
```

The **thick stripe** is the key: it must be **at least 3× the height of either
adjacent thin stripe**. This invariant holds regardless of printing scale.

The top and bottom marker bands are **vertically mirrored**. In both bands the
innermost stripe (closest to the data area) is a **thin stripe**, preceded by a
white gap and the thick stripe. The thick stripe is therefore two stripes away
from the data area edge, not directly adjacent to it.

### 3.2 Default Dimensions ✅

| Element | Height (px) |
|---------|-------------|
| White margin | 88 |
| Thin stripe | 9 |
| White gap | 10 |
| Thick stripe | 39 |

The thick stripe (39 px) is `39 / 9 ≈ 4.3×` the thin stripes — well above the
3× detection threshold.

### 3.3 Detection Algorithm ✅

Scan a **vertical strip at the horizontal centre** of the image:

1. Convert each pixel to grayscale luminance (BT.601: `L = 0.299R + 0.587G + 0.114B`).
2. Threshold: a pixel is *dark* if `L < 128`, otherwise *light*.
3. Run-length encode the column into alternating dark/light runs.
4. Extract the list of dark runs only.
5. In the **top 30%** of the image, find the dark run whose length is
   ≥ 3× the average of its immediate dark neighbours. This is the top thick
   stripe.
6. In the **bottom 30%**, repeat to find the bottom thick stripe.
7. The marker band layout around the data area is:
   - **Top:** `… thick stripe → white gap → thin stripe → [DATA]`
   - **Bottom:** `[DATA] → thin stripe → white gap → thick stripe …`

   Therefore:
   - **`data_top`** = start row of the thin stripe that follows the top thick stripe,
     **plus** its height (i.e. the row immediately after that trailing thin stripe).
   - **`data_bottom`** = start row of the thin stripe that precedes the bottom thick
     stripe (i.e. the row of the first dark pixel of that leading thin stripe).

   If no adjacent thin stripe is found (malformed image), fall back to the inner
   edge of the thick stripe itself.

The data area spans rows `[data_top, data_bottom)` (exclusive end).

> **Implementation note:** Searching by pixel row (top/bottom 30%) rather than
> by list-index (first/second half) is important for robustness: loud audio
> content produces many dark pixels in the middle of the image and can shift
> the dark-run midpoint significantly.

#### Robustness note

When no run satisfies the 3× criterion (e.g. due to print degradation), a
reasonable fallback is to pick the longest dark run in the top/bottom region.
Most well-printed images satisfy the strict criterion.

### 3.4 Reading Direction ✅

All reference material is read **left to right**.  No right-to-left or vertical
orientations have been observed.  Whether the format formally prohibits other
orientations is unknown.

---

## 4. Data Area ✅

### 4.1 Dimensions

- **Width:** variable — equals the number of time columns (one per audio frame)
- **Height:** `OCTAVES × pixels_per_octave = 8 × 90 = 720 px` (default, measured)

The data area height in pixels need not equal the number of frequency bins
(384). A decoder must resample vertically: for each image row `r` in
`[0, data_height)`, the corresponding frequency bin index is:

```
bin = floor(r × TOTAL_BINS / data_height)
bin = clamp(bin, 0, TOTAL_BINS - 1)
```

### 4.2 Coordinate System

```
         col=0      col=1    …   col=W-1
        ┌──────┬──────┬──────────────┐   row = 0    (bin 0,   highest freq ≈ 16 744 Hz, above hearing)
        │      │      │              │
        │      │      │              │
        │      │      │              │
        └──────┴──────┴──────────────┘   row = H-1  (bin 383, lowest freq ≈ 66.36 Hz, C2)
```

- Row 0 → highest frequency (≈ 16 744 Hz, bin index 0; above the human hearing range)
- Row `data_height - 1` → lowest frequency (≈ 66.36 Hz, bin index 383; C2)

### 4.3 Octave Separator Lines (optional)

Implementations **may** draw light-gray horizontal lines at each octave boundary
within the data area:

```
row = octave × pixels_per_octave,  for octave = 1, 2, …, 7
```

> ⚠️ **Caution:** Any pixel with luminance < 255 is decoded as non-zero
> amplitude under the fractional model (§6.2).  A line drawn at luminance
> 200/255 corresponds to amplitude ≈ 0.22, which produces a constant audible hum
> at each of the 7 octave-boundary frequencies during playback.  The **reference
> PhonoPaper application does not draw octave separator lines**.  Encoders should
> omit them (or use luminance 255) in any image intended for audio decoding.
> They are useful only for human visual inspection of the spectrogram.

---

## 5. Frequency Scale ✅

### 5.1 Grid Parameters

| Parameter | Value |
|-----------|-------|
| Total frequency bins (`TOTAL_BINS`) | **384** |
| Octaves | 8 |
| Semitones per octave | 12 |
| Subdivisions per semitone (`MULTITONES`) | 4 |
| Bins per octave | 96 |
| Frequency ratio between adjacent bins | 2^(1/48) ≈ 1.014 54 |

These values are consistent across all reference images and the JS
reimplementation.

### 5.2 Frequency Formula ✅

The centre frequency of bin index `i` (0 = top of image = highest frequency):

```
freq(i) = 2^((63 × MULTITONES − i) / (12 × MULTITONES)) × 440 Hz
         = 2^((252 − i) / 48) × 440 Hz
```

| i | freq (Hz) | Note |
|---|-----------|------|
| 0 | 16744.04 | ≈ C10 (top of image; above human hearing) |
| 96 | 4186.01 | C8 (top of the audible range) |
| 144 | 2093.00 | C7 |
| 192 | 1046.50 | C6 |
| 240 | 523.25 | C5 |
| 252 | 440.00 | A4 (concert pitch) |
| 288 | 261.63 | C4 (middle C) |
| 336 | 130.81 | C3 |
| 383 | 66.36 | ≈ C2 (bottom of image) |

### 5.3 Inverse Formula

To convert a frequency `f` in Hz to the nearest bin index:

```
i = round(252 − log₂(f / 440) × 48)
i = clamp(i, 0, 383)
```

### 5.4 Notes on the Frequency Range

- Bin 96 corresponds exactly to C8 ≈ 4186.009 Hz (the top of the audible range).
  Bins 0–95 represent frequencies above C8 (up to ≈ 16 744 Hz), which are above
  the range of human hearing; the reference Android application does not synthesise
  these bins.
- Bin 383 represents ≈ 66.36 Hz (C2), the bottom of the format's range.
- The reference Android application synthesises only bins 96–336 (approximately
  C3–C8, 5 audible octaves).

---

## 6. Pixel-to-Amplitude Mapping ❓

### 6.1 Colour Model ⚠️

PhonoPaper images contain **grayscale content** regardless of the stored colour
space. Colour images (RGB, RGBA) should be converted to luminance before
processing.  The standard formula is BT.601:

```
L = 0.299 × R + 0.587 × G + 0.114 × B
```

where R, G, B ∈ [0, 255].

> **Note:** For images where R = G = B (pure grayscale storage), BT.601 and
> BT.709 give identical results.  All observed reference images are pure
> grayscale, so the specific luminance formula is practically irrelevant.

### 6.2 Amplitude Convention — What We Know and What We Don't

**The pixel-to-amplitude mapping is the most uncertain part of this spec.**
The format's original author has not documented it, and the two available
secondary sources (the Android app output and the JS reimplementation) give
conflicting or incomplete evidence.

#### Fractional (this library's default) ❓

```
amplitude = (255 − L) / 255
```

| Pixel value L | Amplitude | Meaning |
|---------------|-----------|---------|
| 255 (white) | 0.0 | Silence |
| 128 (mid-gray) | ~0.502 | 50% amplitude |
| 0 (black) | 1.0 | Maximum amplitude |

This is a natural interpretation and is internally consistent: it treats every
non-white pixel as carrying signal, with brightness proportional to amplitude.

**However**, comparison of our output against the official Android application's
decoded WAV showed that fractional mapping produces significantly *lower*
spectral correlation than a binary threshold approach (see below).

#### Binary threshold ⚠️

A threshold value `t ∈ (0, 1)` is chosen:

```
amplitude = 1.0   if (255 − L) / 255 ≥ t    (dark enough → fully on)
amplitude = 0.0   otherwise                  (too light → silent)
```

This treats the image as an **on/off bitmap** rather than a continuous
amplitude map.  The JS reimplementation uses binary thresholding (threshold
≈ 0.55, i.e. `L < 140`), though it is itself a third-party interpretation.

**Empirical evidence from Android comparison (measured):**

A systematic threshold sweep was performed by comparing synthetic audio
(from `mozart2.jpg`) against the audio produced by the official Android app
decoding the same image.  Per-semitone spectral correlation was used as the
metric:

| Amplitude model | Mean correlation | Per-semitone correlation |
|----------------|-----------------|------------------------|
| Fractional (current default) | 0.80 | 0.76 |
| Binary, threshold 0.50 | 0.85 | 0.82 |
| Binary, threshold 0.60 | 0.86 | 0.84 |
| Binary, threshold 0.70 | 0.89 | 0.89 |
| Binary, threshold 0.75 | 0.90 | 0.91 |
| Binary, threshold 0.80 | 0.91 | 0.92 |
| **Binary, threshold 0.85** | **0.92** | **0.93** ← best |
| Binary, threshold 0.90 | 0.91 | 0.93 |

Binary thresholding at ≈ 0.85 (luminance < 38, i.e. only very dark pixels
produce sound) substantially outperforms fractional mapping in terms of
fidelity to the Android reference output.

> ⚠️ **Caveat:** This sweep used a simplified (non-phase-continuous) synthesiser.
> The improvement in the full phase-continuous synthesiser has not been
> independently verified.  The true threshold used by the Android app is unknown.

#### Recommendation

- **For maximum fidelity to the Android reference:** use binary threshold ≈ 0.85.
- **For encoding/decoding round-trips** where the encoder produces fractional
  amplitudes: use fractional mapping to preserve the gradient information.
- **Default in this library:** fractional, to be conservative and reversible.
  The `SynthesisOptions` struct provides a field to configure this.

### 6.3 Inverse (encoding)

Regardless of which decoding model is used, the canonical way to encode an
amplitude value back to a pixel is:

```
L = round((1.0 − amplitude) × 255)
L = clamp(L, 0, 255)
```

This is a straightforward reversal of the fractional model and is consistent
with how the reference application appears to write images.

---

## 7. Decoding: Image → Audio

### 7.1 Pipeline

```
Image file (PNG, JPEG, SVG, or PDF)
  │
  ▼
Detect marker bands  →  data_top, data_bottom
  │
  ▼
Crop data area  →  pixel grid [data_height × image_width]
  │
  ▼
Map each pixel to amplitude  →  Spectrogram [num_columns × 384]
  │
  ▼
Additive sine-wave synthesis  →  PCM samples []
  │
  ▼
WAV file (mono, 16-bit PCM — the only supported decode output format)
```

### 7.2 Step-by-Step

#### Step 1 — Load image

Accept PNG, JPEG, SVG, or PDF.  For SVG and PDF inputs, extract the embedded
raster image first.  Convert to a per-pixel luminance representation.

#### Step 2 — Detect markers

Apply the algorithm in §3.3 to obtain `data_top` and `data_bottom`.

#### Step 3 — Extract spectrogram

For each column `c` ∈ [0, image_width) and each bin `b` ∈ [0, 384):

1. Map bin `b` to image row using the centre-of-range formula (same formula
   used by the encoder to avoid a systematic per-bin offset):  
   `r = floor((2b + 1) × data_height / (2 × 384))`
2. Read luminance `L` at `(c, data_top + r)`.
3. Compute stored amplitude: `stored_amp = (255 − L) / 255`.
4. If the image was encoded with dB normalisation (§8 Step 3), invert the dB
   mapping to recover the true linear synthesis amplitude:
   ```
   dB = min_dB + stored_amp × (max_dB − min_dB)
   amplitude = 10 ^ (dB / 20)
   ```
   with the same `min_dB` / `max_dB` used during encoding.
   If the image was encoded with linear normalisation, use `stored_amp` directly.

#### Step 4 — Synthesize audio ⚠️

Use **additive sine-wave synthesis**.  The exact synthesis method used by the
reference application is unknown; the description below reflects this library's
implementation:

```
ω[b] = 2π × freq(b) / sample_rate          # angular frequency per sample
φ[b] = 0.0  for all b                       # initial phase

for each column c:
    for each sample s in [0, samples_per_column):
        output[c × spc + s] = gain × Σ_{b=0}^{383} A[c][b] × sin(φ[b])
        for each b: φ[b] += ω[b]
```

**Design choices (ours, not confirmed as the reference's):**

- **Phase continuity:** Phases are advanced every sample, even for silent bins,
  so oscillators remain coherent across column boundaries.  Resetting phases per
  column would cause audible clicks.  We infer the reference app also does this
  on quality grounds, but it is not confirmed.
- **`samples_per_column`:** This library defaults to **353** samples/column
  — the closest integer to 352.8 (= 44 100 ÷ 125), the Android app's
  confirmed rate.  The timing error versus Android is < 0.06 %.
- **`gain`:** This library defaults to **3.0**.  After dB inversion, per-bin
  amplitudes equal `web_audio_mag = A/4` (for a Hann-windowed sine of
  amplitude `A`), so `gain = 4.0` would give unity amplitude for a
  perfectly-aligned bin.  However, the PhonoPaper log-frequency grid has
  ≈1.5 PP bins per FFT bin, boosting the output by ≈√1.5 ≈ 1.22×; `gain = 3.0`
  (≈ 4/1.33) compensates and avoids clipping on typical music.
- **Frequency range:** The Android app synthesises approximately **C3–C8
  (~131–4186 Hz, bins 96–336)**, not the full 384-bin range.  This library
  synthesises all 384 bins by default.
- **Output format:** mono, 16-bit PCM. ✅

#### Step 5 — Write WAV

| Field | Value | Status |
|-------|-------|--------|
| Channels | 1 (mono) | ✅ |
| Sample rate | 44 100 Hz | ✅ |
| Bit depth | 16-bit signed integer | ✅ |
| Format | Microsoft PCM | ✅ |

### 7.3 Timing ⚠️

The timing relationship between image columns and audio samples is **not fixed
by the format**.  The column rate is an implementation choice:

| Source | Samples/column | Duration of 1400-col image |
|--------|---------------|--------------------------|
| This library default | 353 | 11.2 s |
| Android application (measured) | 352.8 (= 44 100 ÷ 125) | 11.2 s |

```
duration (seconds) = image_width × samples_per_column / sample_rate
```

The Android app plays back at a fixed **125 columns per second**, confirmed
by measuring the sample count of native Android recordings against their image
widths.  For a recording where the user presses Record for ~10 s, the image
will be ~1250 columns wide and decode to exactly 10.00 s at this rate.

---

## 8. Encoding: Audio → Image ⚠️

### 8.1 Pipeline

```
WAV or MP3 file
  │
  ▼
Read & downmix to mono
  │
  ▼
Short-Time Fourier Transform (STFT)  →  complex spectrum per frame
  │
  ▼
Map FFT bins → PhonoPaper bins  →  Spectrogram [num_frames × 384]
  │
  ▼
Render data area pixels  →  grayscale pixel grid
  │
  ▼
Composite: marker bands + data area + marker bands  →  RGB image
  │
  ▼
Save as PNG (or JPEG)
```

This pipeline reflects this library's encoder.  The original application's
encoder parameters are unknown.

### 8.2 Step-by-Step

#### Step 1 — Load and downmix ✅

Read the WAV or MP3 file (any sample rate; WAV any bit depth; mono or stereo).
Downmix stereo to mono by averaging channels:

```
mono[n] = (left[n] + right[n]) / 2
```

#### Step 2 — STFT analysis ❓

This library uses:

- Window function: **Hann window**
- FFT size: **4096 samples** (power of two; larger = better frequency resolution)
- Hop size: **353 samples** (= `round(44100 / 125)`, matching Android's column
  rate of 352.8; overlap = 1 − 353/4096 ≈ 91.4%)

These are reasonable choices but are **not confirmed** as the reference encoder's
actual values.  Different implementations may use different parameters.

The Hann window for a frame of length N:

```
w[n] = 0.5 × (1 − cos(2π × n / (N − 1)))
```

For each frame `f`:
1. Extract samples `[f × hop, f × hop + fft_size)`.
2. Zero-pad if the frame extends past the end of the signal.
3. Multiply by the Hann window.
4. Apply FFT → complex spectrum `X[k]` for k ∈ [0, fft_size).

#### Step 3 — Map to PhonoPaper bins ⚠️

For each PhonoPaper bin `b` ∈ [0, 384):

1. Compute centre frequency: `freq = 2^((252 − b) / 48) × 440`
2. Find the corresponding FFT bin using linear interpolation between adjacent
   bins (rather than rounding to the nearest bin):
   ```
   k_frac = freq × fft_size / sample_rate
   k0     = floor(k_frac)
   t      = k_frac − k0
   mag    = |X[k0]| × (1 − t) + |X[k0 + 1]| × t
   ```
3. Normalise to dB scale (matching the Web Audio API `AnalyserNode`):
   ```
   web_audio_mag = mag / fft_size
   dB = 20 × log₁₀(max(web_audio_mag, 1×10⁻¹⁰))
   A[f][b] = clamp((dB − min_dB) / (max_dB − min_dB), 0.0, 1.0)
   ```
   where `min_dB = −60.0` and `max_dB = −10.0`.

   The window `[−60, −10]` covers the practical dynamic range of Hann-windowed
   audio: bins quieter than −60 dBFS are treated as silence and produce a white
   pixel.  A full-scale sine (amplitude 1.0) produces
   `web_audio_mag ≈ 0.25` (= −12 dBFS), which maps to amplitude ≈ 0.98
   (nearly black) without clamping.

#### Step 4 — Render data area ✅

For each column `c` and each image row `r` in the data area:

1. Map to bin: `b = floor(r × 384 / data_height)`
2. Look up amplitude: `A = spectrogram[c][b]`
3. Optionally apply gamma: `A = A^γ` (γ = 1.0 means no correction)
4. Write pixel: `L = round((1.0 − A) × 255)`

#### Step 5 — Composite full image ✅

Stack vertically:

```
top_marker_band  (height = margin + thin + gap + thin + gap + thick + gap + thin)
data_area        (height = px_per_octave × 8)
bottom_marker_band  (top band mirrored: thin stripe innermost, thick stripe outer)
```

Convert the grayscale buffer to RGB by replicating the luminance across R, G, B
channels (required by common PNG/JPEG writers).

#### Step 6 — Save

Save as **PNG** (preferred, lossless), JPEG, **SVG**, or **PDF**.

> ⚠️ **JPEG compression artefacts** reduce fidelity but are acceptable for
> the original physical use case (camera scanning). For software round-trips,
> PNG is strongly recommended.
>
> **SVG and PDF** outputs embed the raster `PhonoPaper` image (with marker
> bands) inside a vector container, scaling it to fill the page.  The
> embedded raster is always lossless (PNG-equivalent), so decode quality is
> equivalent to PNG.

---

## 9. Reference Constants

Constants marked ✅ were measured directly.  Constants marked ❓ or ⚠️ are this
library's defaults, not confirmed values from the reference implementation.

```
# Frequency grid — measured/confirmed ✅
MULTITONES           = 4          # subdivisions per semitone
SEMITONES_PER_OCTAVE = 12
OCTAVES              = 8
TOTAL_BINS           = 384        # = 8 × 12 × 4
BINS_PER_OCTAVE      = 96         # = 12 × 4

HIGH_FREQ            = 16744.036 Hz  # bin 0   (≈ C10, above hearing)  ✅
LOW_FREQ             =    66.358 Hz  # bin 383 (≈ C2)                  ✅

SAMPLE_RATE          = 44 100 Hz  # ✅ confirmed in Android output

# Default image layout (pixels) — measured from reference images ✅
MARGIN               = 88
THIN_STRIPE          = 9
MARKER_GAP           = 10
THICK_STRIPE         = 39
PX_PER_OCTAVE        = 90
DATA_HEIGHT          = 720        # = 8 × 90

# Synthesis — this library's defaults
SAMPLES_PER_COLUMN   = 353        # = round(44100 / 125); matches Android's 352.8
GAIN                 = 3.0        # 4/1.33; compensates for ~1.5 PP bins sharing each FFT bin

# Analysis — this library's defaults ❓
FFT_WINDOW           = 4096
HOP_SIZE             = 353
MIN_DB               = -60.0     # maps to amplitude 0.0 (silence / white pixel)
MAX_DB               = -10.0     # maps to amplitude 1.0 (maximum / black pixel)
                                  # Chosen to avoid saturating full-scale audio bins:
                                  # a Hann-windowed full-scale sine gives web_audio_mag=0.25
                                  # (= -12 dBFS), safely below this ceiling.
```

---

## 10. Worked Examples

### 10.1 Frequency of a given bin

> What frequency corresponds to bin 200?

```
freq(200) = 2^((252 − 200) / 48) × 440
          = 2^(52 / 48) × 440
          = 2^(1.0833) × 440
          ≈ 2.117 × 440
          ≈ 931.5 Hz
```

That is approximately A5 / B♭5.

### 10.2 Bin for a given frequency

> Which bin is closest to 261.63 Hz (middle C, C4)?

```
i = round(252 − log₂(261.63 / 440) × 48)
  = round(252 − log₂(0.5946) × 48)
  = round(252 − (−0.7500) × 48)
  = round(252 + 36)
  = 288
```

Bin 288 corresponds to C4 (middle C).

### 10.3 Duration of a 1400-column image

At this library's default settings (`samples_per_column = 353`,
`sample_rate = 44 100 Hz`):

```
duration = 1400 × 353 / 44 100 ≈ 11.21 seconds
```

This matches the Android application's rate of 352.8 (= 44 100 ÷ 125) to
within 0.06 %.  Using `SPS = 512` instead gives ≈ 16.24 seconds.

### 10.4 Image height for default settings

```
marker_height = 88 + 9 + 10 + 9 + 10 + 39 + 10 + 9 = 184 px
data_height   = 8 × 90 = 720 px
total_height  = 184 + 720 + 184 = 1088 px
```

### 10.5 Pixel value for amplitude 0.6

```
L = round((1.0 − 0.6) × 255) = round(0.4 × 255) = round(102) = 102
```

A pixel of luminance 102 out of 255 — a fairly dark gray.

### 10.6 Simulating instrument timbre with harmonics

A single horizontal line in the image produces a pure sine wave — the simplest
possible timbre.  Real instruments produce a **harmonic series**: the fundamental
frequency plus overtones at integer multiples (2×, 3×, 4×, …) with decreasing
amplitude.  An encoder can approximate a specific instrument timbre by drawing
additional faint lines at the corresponding bins:

```
# Example: approximate a flute playing A4 (440 Hz)
fundamental   : bin 252   (440 Hz),  amplitude 1.0
2nd harmonic  : bin 204   (880 Hz),  amplitude 0.5
3rd harmonic  : bin 176   (1320 Hz), amplitude 0.25
4th harmonic  : bin 156   (1760 Hz), amplitude 0.12
```

The bin for harmonic `n` of fundamental frequency `f` is:

```
bin(n·f) = round(252 − log₂(n·f / 440) × 48)
```

Amplitude of each harmonic is left to the encoder; a `1/n` roll-off mimics a
sawtooth wave, while `1/n²` approximates a clarinet-like timbre.

---

## 11. Implementation Notes and Edge Cases

- **Blurry or low-resolution images:** The dark-threshold of 128 may need
  adjustment. A safer approach is to use Otsu's method for automatic thresholding.
- **Printed images scanned by camera:** The image may be trapezoidal (keystone
  distortion) or rotated. A robust mobile decoder should detect the thick
  stripes independently for *every* horizontal column (or a set of evenly
  spaced sample columns) and interpolate `data_top` / `data_bottom` per column.
  This compensates for keystone perspective, paper curl, and tilt without
  requiring an explicit de-warp step. A software decoder operating on clean
  digital files does not need this.
- **Search range:** To avoid misidentifying dark audio content as marker
  stripes, search for the top thick stripe only within the top ~30% of the
  image height and the bottom thick stripe only within the bottom ~30%.
- **Fallback:** If no run satisfies the 3× thick-stripe criterion, picking the
  longest dark run in the respective region is a reasonable fallback.

### 11.2 Vertical resampling ✅

The data area height in pixels (`720` by default) is not equal to `TOTAL_BINS`
(384). A decoder must therefore resample vertically.  This library uses a
**centre-of-range** formula for both decoder and encoder, so that
encode→decode round-trips have no systematic per-bin row offset:

**Decoder** — map image row `r` ∈ [0, data_height) to bin:
```
bin = floor(r × 384 / data_height)
bin = clamp(bin, 0, 383)
```

**Encoder** — map bin `b` ∈ [0, 384) to image row:
```
row = floor((2b + 1) × data_height / (2 × 384))
```

The encoder formula places each bin at the centre of its nominal row range
(`[b × H/384, (b+1) × H/384)`) rather than at its bottom edge, which avoids a
systematic ½-row downward bias.

Multiple rows may map to the same bin (since 720 > 384), so each bin is
represented by approximately `720 / 384 ≈ 1.875` rows.

### 11.3 Phase continuity ⚠️

When synthesising audio in this library, oscillator phases are maintained
across column boundaries to avoid audible clicks at the column rate.  This is
an inference about quality; whether the reference Android application does the
same is unconfirmed.

### 11.4 Gain and clipping ❓

With 384 simultaneous oscillators at full amplitude the raw sum would reach 384.
This library's default gain of `3.0` still allows clipping (`384 × 3.0 = 1152`
maximum, which is well above 1.0).  In practice, with dB-mode decoding and
typical music content, the per-bin amplitudes are much smaller and clipping is
uncommon; however, automatic peak normalisation (see below) is safer.

The reference application's output clips on loud content, consistent with either
a fixed gain or binary thresholding (§6.2) — binary thresholding means that a
column with 384 simultaneously active bins can sum to 384 without any gain
reduction.

For highest fidelity in offline decoding, **automatic peak normalisation** is
preferable: synthesise the full output, measure the peak absolute value, then
divide the entire buffer by that peak.  A fixed gain is simpler for real-time
streaming.

### 11.5 JPEG vs PNG

The reference Android application outputs JPEG (smaller file size, acceptable
for camera scanning).  PNG is preferred for software-to-software round-trips
because JPEG compression introduces DCT artefacts that spread energy across
frequency bins and corrupt amplitude data.

### 11.6 Sample rate independence ✅

The frequency grid is fixed — the 384 bins always cover the same musical range
regardless of the audio sample rate.  However:

- **Decoding:** the number of audio samples per image column
  (`samples_per_column`) scales the playback speed but not the pitch.  See §7.3
  for measured values from the Android app.
- **Encoding:** the STFT bin-to-PhonoPaper-bin mapping depends on both the FFT
  size and the input sample rate. The mapping must be recomputed for each source
  file if the sample rate differs from 44 100 Hz.

### 11.7 Stereo ✅

PhonoPaper is inherently **mono**. Stereo input should be downmixed to mono
before encoding. There is no stereo or multi-channel variant of the format.

### 11.8 Colour images ⚠️

All observed reference images are stored as pure grayscale (R = G = B).
Decoders should convert to luminance using the BT.601 formula.  The format
does not formally prohibit colour images, but none have been observed in the
wild.

### 11.9 Known divergences from the Android reference

This section documents cases where this library's behaviour is known to differ
from the official Android application:

| Behaviour | This library | Android app |
|-----------|-------------|-------------|
| `samples_per_column` | 353 (default, matches Android) | 352.8 (= 44 100 ÷ 125) |
| Amplitude model | Fractional (default) | Likely binary ~0.85 threshold |
| Frequency range synthesised | All 384 bins (≈ C2–inaudible) | Approximately bins 96–336 (C3–C8) |
| Output clipping | Clamped to ±1.0 | Clips at ±1.0 |
| Leading silence | None | ≈ 1152 samples (~26 ms) |

To approximate Android playback with this library:

```bash
phonopaper decode --samples-per-column 353 image.jpg output.wav
```
