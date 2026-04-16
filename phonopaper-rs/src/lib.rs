//! # phonopaper-rs
//!
//! A Rust library for encoding and decoding audio in the
//! [PhonoPaper](https://warmplace.ru/soft/phonopaper/) format.
//!
//! `PhonoPaper` is a camera-readable musical notation invented by Alexander
//! Zolotov (`NightRadio`).  It represents audio as a grayscale spectrogram
//! image that can be printed on paper and played back in real time by sweeping
//! a phone camera across it.
//!
//! ## Format overview
//!
//! | Property | Value |
//! |----------|-------|
//! | X axis   | Time (left → right) |
//! | Y axis   | Frequency, logarithmic (top = high, bottom = low) |
//! | Brightness | White = silence, black = maximum amplitude |
//! | Frequency range | ~16.35 Hz (C0) – ~4186 Hz (C8), 8 octaves |
//! | Frequency bins | 384 (12 semitones × 4 subdivisions × 8 octaves) |
//! | Markers | Alternating black/white stripes with one thick stripe at top and bottom |
//!
//! ## Quick start
//!
//! ### Decode an image to a WAV file
//!
//! ```no_run
//! use phonopaper_rs::decode::{decode_image_to_wav, SynthesisOptions};
//!
//! decode_image_to_wav("code.png", "output.wav", SynthesisOptions::default()).unwrap();
//! ```
//!
//! ### Encode a WAV or MP3 file to an image
//!
//! ```no_run
//! use phonopaper_rs::encode::{encode_audio_to_image, AnalysisOptions};
//! use phonopaper_rs::render::RenderOptions;
//!
//! encode_audio_to_image(
//!     "input.wav",
//!     "code.png",
//!     AnalysisOptions::default(),
//!     RenderOptions::default(),
//! )
//! .unwrap();
//! ```
//!
//! ### Real-time / sliding-camera playback
//!
//! The [`Synthesizer`] struct and [`decode::column_amplitudes_from_image`]
//! together support the **sliding-camera** use case — continuously decoding
//! one image column at a time and playing it back without phase discontinuities,
//! exactly like the `PhonoPaper` Android application:
//!
//! ```no_run
//! use phonopaper_rs::decode::{
//!     Synthesizer, SynthesisOptions, column_amplitudes_from_image, detect_markers,
//! };
//!
//! // Load the image and detect markers once.
//! let image = image::open("code.png").unwrap();
//! let bounds = detect_markers(&image).unwrap();
//!
//! // Create a stateful synthesizer; it keeps phasor state across columns.
//! let mut synth = Synthesizer::<353>::new(SynthesisOptions::default());
//! let mut pcm = [0.0_f32; 353];
//!
//! // Simulate the camera/playhead sweeping across columns 0..image_width.
//! let image_width = image.width();
//! for col_x in 0..image_width {
//!     let amps = column_amplitudes_from_image(&image, Some(bounds), col_x).unwrap();
//!     synth.synthesize_column(&amps, &mut pcm);
//!     // Send `pcm` to the audio output device…
//! }
//! ```
//!
//! ## Module structure
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`error`]        | Error type and `Result` alias |
//! | [`format`]       | Format constants and frequency ↔ index mapping |
//! | [`spectrogram`]  | The [`Spectrogram`] type (time × frequency amplitude matrix) |
//! | [`render`]       | Image rendering: [`render::RenderOptions`], [`render::spectrogram_to_image`], [`render::spectrogram_to_image_buf`] |
//! | [`vector`]       | Vector output: [`vector::spectrogram_to_svg`], [`vector::spectrogram_to_pdf`] |
//! | [`audio`]        | Audio file I/O: [`audio::read_audio_file`] (WAV, MP3) |
//! | [`decode`]       | Image → audio pipeline |
//! | [`encode`]       | Audio → image pipeline with file I/O |

pub mod audio;
pub mod decode;
pub mod encode;
pub mod error;
pub mod format;
pub mod render;
pub mod spectrogram;
pub mod vector;

// ─── Top-level re-exports ─────────────────────────────────────────────────────

pub use error::{PhonoPaperError, Result};
pub use spectrogram::{Spectrogram, SpectrogramBuf, SpectrogramBufMut, SpectrogramVec};

/// Stateful synthesizer for real-time, column-by-column audio playback.
///
/// This is a convenience re-export of [`decode::Synthesizer`].
/// See that type for full documentation.
pub use decode::Synthesizer;

/// Decode a `PhonoPaper` image file to a WAV audio file (353 samples/column).
///
/// This is a convenience wrapper around [`decode::decode_image_to_wav`].
/// See that function for full documentation.
pub use decode::decode_image_to_wav;

/// Decode a `PhonoPaper` image file to a WAV audio file with a configurable
/// number of samples per image column.
///
/// This is a convenience re-export of [`decode::decode_image_to_wav_sps`].
/// See that function for full documentation.
pub use decode::decode_image_to_wav_sps;

/// Encode an audio file (WAV or MP3) into a `PhonoPaper` PNG image.
///
/// This is a convenience wrapper around [`encode::encode_audio_to_image`].
/// See that function for full documentation.
pub use encode::encode_audio_to_image;
