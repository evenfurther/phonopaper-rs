//! Audio file reading: WAV and MP3 input for the encode pipeline.
//!
//! This module provides [`read_audio_file`], a single entry point that reads
//! any supported audio file and returns a mono `f32` sample vector along with
//! the source sample rate.
//!
//! ## Supported formats
//!
//! | Extension | Format | Decoder |
//! |-----------|--------|---------|
//! | `.wav`    | RIFF/WAVE | [`hound`] |
//! | `.mp3`    | MPEG Audio Layer III | [`symphonia`] |
//!
//! [`hound`]: https://docs.rs/hound
//! [`symphonia`]: https://docs.rs/symphonia

use std::path::Path;

use crate::error::{PhonoPaperError, Result};

// ─── WAV reading ──────────────────────────────────────────────────────────────

/// Read a WAV file and return `(mono_samples, sample_rate)`.
///
/// All standard WAV sample formats (8/16/24/32-bit integer and 32-bit float)
/// are supported.  Stereo and multi-channel files are downmixed to mono by
/// averaging all channels.
///
/// # Errors
///
/// Returns [`PhonoPaperError::AudioError`] if the file cannot be opened or
/// decoded, or [`PhonoPaperError::IoError`] for underlying I/O failures.
fn read_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = usize::from(spec.channels);

    let samples_raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(PhonoPaperError::AudioError)?,
        hound::SampleFormat::Int => {
            #[expect(
                clippy::cast_precision_loss,
                reason = "full-scale value for ≤24-bit audio is exact in f32; 32-bit int audio accepts minor rounding"
            )]
            let max_val = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(PhonoPaperError::AudioError)?
                .into_iter()
                .map(|s| {
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "i32 → f32 for audio samples; precision loss at the LSB of 32-bit int audio is inaudible"
                    )]
                    let sf = s as f32;
                    sf / max_val
                })
                .collect()
        }
    };

    let mut mono = samples_raw;
    downmix(&mut mono, channels);
    Ok((mono, sample_rate))
}

// ─── MP3 reading via symphonia ────────────────────────────────────────────────

/// Read an MP3 file and return `(mono_samples, sample_rate)`.
///
/// Uses the [`symphonia`] library with the `mp3` feature.  The first audio
/// track is decoded; all channels are downmixed to mono by averaging.
///
/// # Errors
///
/// Returns [`PhonoPaperError::InvalidFormat`] if the file cannot be probed,
/// no audio track is found, or decoding fails.
fn read_mp3(path: &Path) -> Result<(Vec<f32>, u32)> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::{MediaSource, MediaSourceStream, MediaSourceStreamOptions};
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    // Open the file as a MediaSource.
    let file = std::fs::File::open(path).map_err(PhonoPaperError::IoError)?;
    let mss = MediaSourceStream::new(
        Box::new(file) as Box<dyn MediaSource>,
        MediaSourceStreamOptions::default(),
    );

    // Give the prober a filename hint so it can pick the right demuxer faster.
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    // Probe the stream to identify the container format.
    let probe = symphonia::default::get_probe();
    let probe_result = probe
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| PhonoPaperError::InvalidFormat(format!("Failed to probe audio file: {e}")))?;
    let mut format = probe_result.format;

    // Find the default audio track.
    let track = format.default_track().ok_or_else(|| {
        PhonoPaperError::InvalidFormat("No audio track found in file".to_string())
    })?;

    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    let sample_rate = codec_params.sample_rate.ok_or_else(|| {
        PhonoPaperError::InvalidFormat("Audio track has no sample rate".to_string())
    })?;

    let n_channels = codec_params
        .channels
        .map_or(1, symphonia::core::audio::Channels::count);

    // Build the decoder.
    let codecs = symphonia::default::get_codecs();
    let mut decoder = codecs
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| PhonoPaperError::InvalidFormat(format!("Failed to create decoder: {e}")))?;

    // Decode all packets belonging to this track.
    let mut interleaved: Vec<f32> = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(pkt) => pkt,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break; // end of stream
            }
            Err(e) => {
                return Err(PhonoPaperError::InvalidFormat(format!(
                    "Error reading audio packet: {e}"
                )));
            }
        };

        // Skip packets that belong to other tracks.
        if packet.track_id() != track_id {
            continue;
        }

        let buf = match decoder.decode(&packet) {
            Ok(b) => b,
            // ResetRequired means the decoder was reset (e.g. gapless seek point);
            // the next decode call will succeed, so skip this packet.
            Err(symphonia::core::errors::Error::ResetRequired) => continue,
            Err(e) => {
                return Err(PhonoPaperError::InvalidFormat(format!(
                    "Failed to decode audio packet: {e}"
                )));
            }
        };

        // Initialise the SampleBuffer on the first decoded frame.
        let sb = sample_buf
            .get_or_insert_with(|| SampleBuffer::<f32>::new(buf.capacity() as u64, *buf.spec()));

        sb.copy_interleaved_ref(buf);
        interleaved.extend_from_slice(sb.samples());
    }

    if interleaved.is_empty() {
        return Err(PhonoPaperError::InvalidFormat(
            "Audio file contained no decodable samples".to_string(),
        ));
    }

    let mut mono = interleaved;
    downmix(&mut mono, n_channels);
    Ok((mono, sample_rate))
}

// ─── Shared helper ────────────────────────────────────────────────────────────

/// Downmix interleaved multi-channel samples to mono in-place by averaging all
/// channels.
///
/// The first `N / channels` positions of `samples` are overwritten with the
/// per-frame averages, then `samples` is truncated to that length.  This
/// halves peak memory usage compared to building a separate output `Vec` for
/// stereo input.
///
/// If `channels <= 1` the buffer is left unchanged.
fn downmix(samples: &mut Vec<f32>, channels: usize) {
    if channels <= 1 {
        return;
    }
    debug_assert!(
        samples.len().is_multiple_of(channels),
        "interleaved buffer length {} is not divisible by channel count {}",
        samples.len(),
        channels,
    );
    #[expect(
        clippy::cast_precision_loss,
        reason = "channel count comes from a u16 (WAV) or symphonia channel count; values ≤ 65535 are exact in f32"
    )]
    let ch_f = channels as f32;
    let n_frames = samples.len() / channels;
    for frame in 0..n_frames {
        let sum: f32 = (0..channels).map(|ch| samples[frame * channels + ch]).sum();
        samples[frame] = sum / ch_f;
    }
    samples.truncate(n_frames);
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Read an audio file and return the samples as a mono `f32` vector together
/// with the sample rate.
///
/// The format is inferred from the file extension:
///
/// | Extension | Format |
/// |-----------|--------|
/// | `.wav`    | WAV (any bit depth/sample format) |
/// | `.mp3`    | MPEG Audio Layer III |
///
/// Stereo and multi-channel files are automatically downmixed to mono by
/// averaging all channels.
///
/// # Errors
///
/// Returns [`PhonoPaperError::AudioError`] for WAV decoding errors,
/// [`PhonoPaperError::InvalidFormat`] for MP3 decoding errors,
/// [`PhonoPaperError::IoError`] for I/O failures, or
/// [`PhonoPaperError::InvalidFormat`] if the file extension is not recognised.
pub fn read_audio_file(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32)> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase);

    match ext.as_deref() {
        Some("mp3") => read_mp3(path),
        Some("wav") => read_wav(path),
        other => Err(PhonoPaperError::InvalidFormat(format!(
            "Unsupported audio format: {}",
            other.unwrap_or("<no extension>")
        ))),
    }
}
