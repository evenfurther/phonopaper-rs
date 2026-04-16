//! Generate a `PhonoPaper` image containing a 440 Hz sine for 5 s followed by
//! an 880 Hz sine for 5 s, then save it to a temp file.
//!
//! Run with:
//! ```bash
//! cargo run --release --example gen_sines
//! ```

use std::f32::consts::PI;

use phonopaper_rs::encode::{AnalysisOptions, audio_to_spectrogram};
use phonopaper_rs::format::SAMPLE_RATE;
use phonopaper_rs::render::{RenderOptions, spectrogram_to_image};

fn main() -> phonopaper_rs::Result<()> {
    let sample_rate = SAMPLE_RATE;
    let sr = sample_rate as usize;

    // Build 10 seconds of mono PCM: 5 s at 440 Hz then 5 s at 880 Hz.
    let mut samples: Vec<f32> = Vec::with_capacity(10 * sr);

    for (freq, seconds) in [(440.0_f32, 5), (880.0_f32, 5)] {
        #[expect(
            clippy::cast_precision_loss,
            reason = "44_100 is exactly representable as f32 (< 2^24)"
        )]
        let omega = 2.0 * PI * freq / sample_rate as f32;
        for i in 0..seconds * sr {
            #[expect(
                clippy::cast_precision_loss,
                reason = "i < 5*44_100 = 220_500 < 2^24; every value is exact in f32"
            )]
            samples.push((omega * i as f32).sin());
        }
    }

    // Encode to a spectrogram, then render to an image.
    let spec = audio_to_spectrogram(&samples, sample_rate, &AnalysisOptions::default())?;
    let img = spectrogram_to_image(&spec, &RenderOptions::default());

    let out_path = std::env::temp_dir().join("sines.jpg");
    img.save(&out_path)
        .map_err(|e| phonopaper_rs::PhonoPaperError::IoError(std::io::Error::other(e)))?;

    println!(
        "Saved {}  ({}×{} px)",
        out_path.display(),
        img.width(),
        img.height()
    );
    Ok(())
}
