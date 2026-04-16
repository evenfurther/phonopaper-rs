# phonopaper-rs

A Rust library and command-line tool for encoding and decoding audio in the
**[PhonoPaper](https://warmplace.ru/soft/phonopaper/)** format.

PhonoPaper (invented by Alexander Zolotov / NightRadio) represents audio as a
*camera-readable* grayscale spectrogram image that can be printed on paper and
played back by sweeping a phone camera across it.

> **Independence notice:** `phonopaper-rs` is an independent, clean-room
> implementation written from scratch by studying the publicly available
> PhonoPaper format specification.  It is not affiliated with, endorsed by,
> or derived from Alexander Zolotov's original work in any way.  The
> PhonoPaper name and concept belong to their respective author.

> **AI disclosure:** Large language models were used heavily during the
> development of this library — for code drafting, test generation, and
> documentation — under strict human supervision and validation.

---

## Repository layout

This is a Cargo workspace with two crates:

| Crate | Path | Description |
|---|---|---|
| `phonopaper-rs` | `phonopaper-rs/` | Core library — encode, decode, render, vector output |
| `phonopaper-cli` | `phonopaper-cli/` | `phonopaper` binary — four subcommands |

---

## Format summary

| Property | Value |
|---|---|
| X axis | Time, left → right |
| Y axis | Frequency, logarithmic musical scale (top = high, bottom = low) |
| Pixel brightness | White = silence, black = maximum amplitude |
| Frequency bins | 384 (8 octaves × 12 semitones × 4 subdivisions) |
| Frequency range | ≈ 66 Hz (C2) … ≈ 4186 Hz (C8); bins 0–95 above C8 are above human hearing |
| Marker bands | Alternating black/white stripes at top and bottom for auto-calibration |

For a detailed technical description see [`PHONOPAPER_SPEC.md`](PHONOPAPER_SPEC.md).

---

## Command-line tool

### Installation

```bash
cargo install --path phonopaper-cli
```

Or run directly from the workspace:

```bash
cargo run -p phonopaper-cli -- <subcommand> [options]
```

### `decode` — image → WAV

Convert a PhonoPaper image to a WAV audio file.

```bash
phonopaper decode <INPUT> [OUTPUT] [OPTIONS]
```

Supported input formats: PNG, JPEG, SVG, PDF.
Default output: `<input-stem>.wav` (mono, 16-bit PCM).

| Option | Default | Description |
|---|---|---|
| `--samples-per-column <N>` | `353` | PCM samples synthesised per image column (353 ≈ Android's 44 100 ÷ 125) |
| `--sample-rate <HZ>` | `44100` | Output sample rate |
| `--gain <GAIN>` | `3.0` | Master output gain (linear) |
| `--gamma <G>` | `1.0` | Inverse gamma for pixel → amplitude; must match the `--gamma` used when encoding |
| `--min-db <DB>` | `-60.0` | Lower bound of the dB window used when the image was encoded |
| `--max-db <DB>` | `-10.0` | Upper bound of the dB window used when the image was encoded |
| `--amplitude-threshold <T>` | off | Binary threshold (e.g. `0.85` for Android-like fidelity); conflicts with `--linear` |
| `--linear` | off | Linear amplitude mode with no threshold; conflicts with `--amplitude-threshold` |

**Examples:**

```bash
phonopaper decode code.png
phonopaper decode code.png output.wav --gain 2.0
phonopaper decode scan.pdf output.wav
phonopaper decode code.svg output.wav --amplitude-threshold 0.85
```

### `encode` — WAV / MP3 → image

Convert an audio file to a PhonoPaper image.

```bash
phonopaper encode <INPUT> [OUTPUT] [OPTIONS]
```

Supported input formats: WAV (any bit depth, mono or stereo), MP3.
Supported output formats: PNG, JPEG, SVG, PDF (inferred from extension).
Default output: `<input-stem>.png`.

| Option | Default | Description |
|---|---|---|
| `--fft-window <N>` | `4096` | FFT window size in samples (power of two) |
| `--hop-size <N>` | `353` | Hop size between FFT frames; controls columns/second |
| `--px-per-octave <PX>` | `90` | Pixels per octave (total data height = value × 8) |
| `--gamma <G>` | `1.0` | Gamma correction applied to amplitudes before writing pixels |
| `--min-db <DB>` | `-60.0` | Bins quieter than this map to white (silence) |
| `--max-db <DB>` | `-10.0` | Bins at this level or louder map to black (maximum) |
| `--octave-lines` | off | Draw light-gray octave separator lines (⚠ produces audible hum on decode) |
| `--pdf-page <SIZE>` | `a4-landscape` | PDF page size for centred output; use `pixel-perfect` for a 1 px = 1 pt page (PDF only) |

Supported `--pdf-page` values: `a4-landscape` (default), `a4-portrait`, `a3-landscape`, `a3-portrait`, `letter-landscape`, `letter-portrait`, `pixel-perfect`.

**Examples:**

```bash
phonopaper encode input.wav
phonopaper encode input.mp3 code.png
phonopaper encode input.wav code.svg
phonopaper encode input.wav code.pdf --px-per-octave 120
```

> ⚠ **JPEG output** applies lossy compression that can corrupt amplitude data
> and degrade decode quality.  Prefer PNG for any software round-trip; JPEG
> is acceptable only when the image will be scanned by a phone camera.

### `robust-decode` — camera photo → WAV

Decode a PhonoPaper print photographed by a camera, tolerating keystone
distortion, tilt, and paper curl.  Uses per-column marker detection followed by
linear interpolation of the data boundaries — no explicit de-warp step is
required.

```bash
phonopaper robust-decode --input <FILE> --output <FILE> [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `-i, --input <FILE>` | required | Input image (PNG or JPEG) |
| `-o, --output <FILE>` | required | Output WAV file (mono, 16-bit PCM) |
| `--sample-columns <N>` | `50` | Number of evenly-spaced columns to sample for marker detection |
| `--gain <GAIN>` | `3.0` | Master output gain (linear) |
| `--sample-rate <HZ>` | `44100` | Output sample rate |
| `--samples-per-column <N>` | `353` | PCM samples synthesised per image column |
| `--min-db <DB>` | `-60.0` | Lower bound of the dB window |
| `--max-db <DB>` | `-10.0` | Upper bound of the dB window |
| `--debug-image <FILE>` | off | Save a PNG with detected `data_top`/`data_bottom` lines overlaid |
| `--rectified <FILE>` | off | Save a perspective-rectified PNG of the data area |

**Example:**

```bash
phonopaper robust-decode --input photo.jpg --output decoded.wav
phonopaper robust-decode --input photo.jpg --output decoded.wav \
    --debug-image debug.png --rectified rectified.png
```

### `blank` — generate a blank template PDF

Generate a ready-to-print PhonoPaper template: marker bands, blank white data
area for hand-drawn music, octave separator lines, octave labels (C2–C8), and
a footer.

```bash
phonopaper blank [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `-o, --output <FILE>` | `blank_phonopaper.pdf` | Output PDF path |
| `--paper <FORMAT>` | `a4` | Paper size: `a4`, `a3`, `letter`, `legal` |
| `--portrait` | off | Portrait orientation (default is landscape) |

**Example:**

```bash
phonopaper blank
phonopaper blank --paper a3 --output template_a3.pdf
phonopaper blank --paper letter --portrait
```

---

## Library usage

Add to your `Cargo.toml`:

```toml
[dependencies]
phonopaper-rs = { path = "/path/to/phonopaper-rs/phonopaper-rs" }
```

> The crate name on disk is `phonopaper-rs`; the Rust module name is
> `phonopaper_rs`.

### Decode an image to a WAV file

```rust
use phonopaper_rs::decode::{decode_image_to_wav, SynthesisOptions};

decode_image_to_wav("code.png", "output.wav", SynthesisOptions::default())?;
# Ok::<(), phonopaper_rs::PhonoPaperError>(())
```

### Encode an audio file to an image

```rust
use phonopaper_rs::encode::{encode_wav_to_image, AnalysisOptions};
use phonopaper_rs::render::RenderOptions;

encode_wav_to_image(
    "input.wav",
    "code.png",
    AnalysisOptions::default(),
    RenderOptions::default(),
)?;
# Ok::<(), phonopaper_rs::PhonoPaperError>(())
```

### Real-time / column-by-column playback

`Synthesizer` maintains oscillator phase across calls, avoiding clicks at
column boundaries — matching the PhonoPaper Android app's playback model.

```rust
use phonopaper_rs::decode::{
    Synthesizer, SynthesisOptions,
    column_amplitudes_from_image, detect_markers,
};

let image = image::open("code.png")?;
let bounds = detect_markers(&image)?;

let mut synth = Synthesizer::<353>::new(SynthesisOptions::default());
let mut pcm = [0.0_f32; 353];

for col_x in 0..image.width() {
    let amps = column_amplitudes_from_image(&image, Some(bounds), col_x)?;
    synth.synthesize_column(&amps, &mut pcm);
    // Send `pcm` to an audio device...
}
# Ok::<(), phonopaper_rs::PhonoPaperError>(())
```

### Vector output (SVG / PDF)

```rust
use phonopaper_rs::encode::{AnalysisOptions, audio_to_spectrogram};
use phonopaper_rs::render::RenderOptions;
use phonopaper_rs::vector::{PdfPageLayout, page_size, spectrogram_to_pdf, spectrogram_to_svg};

let (mono, sample_rate) = phonopaper_rs::audio::read_audio_file("input.wav")?;
let spec = audio_to_spectrogram(&mono, sample_rate, &AnalysisOptions::default())?;
let render = RenderOptions::default();

std::fs::write("code.svg", spectrogram_to_svg(&spec, &render))?;
// A4 landscape, 10 mm margins, image centred on the page:
std::fs::write("code.pdf", spectrogram_to_pdf(&spec, &render, PdfPageLayout::FitToPage {
    page_width_pt:  page_size::A4_LANDSCAPE.0,
    page_height_pt: page_size::A4_LANDSCAPE.1,
    margin_pt: 28.35,
}))?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

---

## Public API overview

| Module | Key items |
|---|---|
| `phonopaper_rs::error` | `PhonoPaperError`, `Result` |
| `phonopaper_rs::format` | `TOTAL_BINS`, `OCTAVES`, `SAMPLE_RATE`, `bin_to_freq`, `freq_to_bin` |
| `phonopaper_rs::spectrogram` | `Spectrogram<S>`, `SpectrogramVec`, `SpectrogramBuf`, `SpectrogramBufMut` |
| `phonopaper_rs::render` | `RenderOptions`, `spectrogram_to_image`, `spectrogram_to_image_buf` |
| `phonopaper_rs::vector` | `spectrogram_to_svg`, `spectrogram_to_pdf`, `image_from_svg`, `image_from_pdf`, `PdfPageLayout`, `page_size` |
| `phonopaper_rs::audio` | `read_audio_file` (WAV + MP3 → mono `f32` + sample rate) |
| `phonopaper_rs::encode` | `AnalysisOptions`, `audio_to_spectrogram`, `encode_wav_to_image` |
| `phonopaper_rs::decode` | `SynthesisOptions`, `AmplitudeMode`, `Synthesizer<SPS>`, `DataBounds`, `detect_markers`, `detect_markers_at_column`, `column_amplitudes_from_image`, `spectrogram_to_audio`, `decode_image_to_wav`, `decode_image_to_wav_sps` |

---

## Supported formats

| Direction | Format | Notes |
|---|---|---|
| Input audio | WAV | Any bit depth (8/16/24/32-bit int, 32-bit float); mono or stereo |
| Input audio | MP3 | Via [symphonia](https://github.com/pdeljanov/Symphonia); mono or stereo |
| Input image | PNG | Lossless; preferred for software round-trips |
| Input image | JPEG | Accepted; lossy compression may degrade decode quality |
| Input image | SVG | Embedded raster image extracted automatically |
| Input image | PDF | Embedded raster image extracted automatically |
| Output image | PNG | Lossless; strongly preferred |
| Output image | JPEG | Lossy; acceptable for camera-scanned use cases only |
| Output image | SVG | Vector; marker bands as `<rect>` elements + embedded PNG |
| Output image | PDF | Vector; marker bands as rectangles + deflate-compressed raster XObject |
| Output audio | WAV | Mono, 16-bit signed PCM |

---

## Development

All four commands must pass with zero warnings and zero errors:

```bash
cargo fmt --check --all
cargo clippy --workspace --all-targets
cargo test --workspace
cargo bench -p phonopaper-rs
```

Coverage (informational, not a hard gate):

```bash
cargo llvm-cov -p phonopaper-rs --tests \
    --ignore-filename-regex='(benches|examples)' --summary-only
```

---

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT License](LICENSE-MIT) at your option.
