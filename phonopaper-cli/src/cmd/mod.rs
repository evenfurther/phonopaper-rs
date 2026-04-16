//! Sub-command modules for the `phonopaper` CLI.

pub mod blank;
pub mod decode;
pub mod encode;
pub mod robust_decode;

/// Dispatch [`phonopaper_rs::decode::decode_image_to_wav_sps`] (or
/// [`phonopaper_rs::decode::spectrogram_to_audio`]) with a compile-time `SPS`
/// constant that is the smallest power-of-two-ish value ≥ the requested
/// runtime `samples_per_column`.
///
/// Usage: `dispatch_sps!(runtime_sps, img_path, wav_path, options)?`
///
/// The match arms cover 256, 353, 384, 512, 1024, and 2048.  Any value above
/// 1024 falls into the 2048 arm.
macro_rules! dispatch_sps {
    ($sps:expr, $img:expr, $wav:expr, $opts:expr) => {
        match $sps {
            n if n <= 256 => {
                phonopaper_rs::decode::decode_image_to_wav_sps::<256>($img, $wav, $opts)
            }
            n if n <= 353 => {
                phonopaper_rs::decode::decode_image_to_wav_sps::<353>($img, $wav, $opts)
            }
            n if n <= 384 => {
                phonopaper_rs::decode::decode_image_to_wav_sps::<384>($img, $wav, $opts)
            }
            n if n <= 512 => {
                phonopaper_rs::decode::decode_image_to_wav_sps::<512>($img, $wav, $opts)
            }
            n if n <= 1024 => {
                phonopaper_rs::decode::decode_image_to_wav_sps::<1024>($img, $wav, $opts)
            }
            _ => phonopaper_rs::decode::decode_image_to_wav_sps::<2048>($img, $wav, $opts),
        }
    };
}

pub(crate) use dispatch_sps;

/// Dispatch [`phonopaper_rs::decode::spectrogram_to_audio`] with a
/// compile-time `SPS` constant that is the smallest value ≥ the requested
/// runtime `samples_per_column`.
///
/// Usage: `dispatch_sps_synth!(runtime_sps, spec_ref, options_ref, output_slice)`
macro_rules! dispatch_sps_synth {
    ($sps:expr, $spec:expr, $opts:expr, $out:expr) => {
        match $sps {
            n if n <= 256 => {
                phonopaper_rs::decode::spectrogram_to_audio::<_, 256>($spec, $opts, $out)
            }
            n if n <= 353 => {
                phonopaper_rs::decode::spectrogram_to_audio::<_, 353>($spec, $opts, $out)
            }
            n if n <= 384 => {
                phonopaper_rs::decode::spectrogram_to_audio::<_, 384>($spec, $opts, $out)
            }
            n if n <= 512 => {
                phonopaper_rs::decode::spectrogram_to_audio::<_, 512>($spec, $opts, $out)
            }
            n if n <= 1024 => {
                phonopaper_rs::decode::spectrogram_to_audio::<_, 1024>($spec, $opts, $out)
            }
            _ => phonopaper_rs::decode::spectrogram_to_audio::<_, 2048>($spec, $opts, $out),
        }
    };
}

pub(crate) use dispatch_sps_synth;
