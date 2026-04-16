//! Decode a `PhonoPaper` image into audio.
//!
//! # Pipeline
//!
//! ```text
//! image file  ──► detect_markers ──► image_to_spectrogram ──► spectrogram_to_audio ──► WAV file
//! ```
//!
//! The top-level entry point is [`decode_image_to_wav`].
//!
//! # Submodules
//!
//! | Submodule | Contents |
//! |-----------|---------|
//! | [`markers`] | [`DataBounds`] and [`detect_markers`] |
//! | [`image`]   | [`image_to_spectrogram`], [`column_amplitudes_from_image`], [`column_amplitudes_from_image_into`] |
//! | [`synth`]   | [`AmplitudeMode`], [`SynthesisOptions`], [`Synthesizer`], [`spectrogram_to_audio`] |
//! | [`wav`]     | [`decode_image_to_wav`], [`decode_image_to_wav_sps`], [`write_wav`] |

pub mod image;
pub mod markers;
pub mod synth;
pub mod wav;

// ─── Re-exports ───────────────────────────────────────────────────────────────

pub use image::{
    column_amplitudes_from_image, column_amplitudes_from_image_into, fill_spectrogram_from_pixels,
    image_to_spectrogram, spectrogram_from_pixels,
};
pub use markers::{DataBounds, detect_markers, detect_markers_at_column};
pub use synth::{AmplitudeMode, SynthesisOptions, Synthesizer, spectrogram_to_audio};
pub use wav::{decode_image_to_wav, decode_image_to_wav_sps, write_wav};
