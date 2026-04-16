use std::time::Instant;

use phonopaper_rs::decode::{SynthesisOptions, image_to_spectrogram, spectrogram_to_audio};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: bench_decode <image>");

    let t0 = Instant::now();
    let img = image::open(&path).expect("failed to open image");
    eprintln!("image load:   {:?}", t0.elapsed());

    let t1 = Instant::now();
    let spec = image_to_spectrogram(&img, None).expect("marker detection failed");
    eprintln!(
        "img→spec:     {:?}  ({} columns)",
        t1.elapsed(),
        spec.num_columns()
    );

    let opts = SynthesisOptions::default();
    let t2 = Instant::now();
    let mut samples = vec![0.0_f32; spec.num_columns() * 512];
    spectrogram_to_audio::<_, 512>(&spec, &opts, &mut samples);
    eprintln!(
        "spec→audio:   {:?}  ({} samples)",
        t2.elapsed(),
        samples.len()
    );

    eprintln!("total:        {:?}", t0.elapsed());
    // prevent the compiler from optimising the result away
    eprintln!(
        "(checksum: {})",
        samples.iter().map(|s| f64::from(*s)).sum::<f64>()
    );
}
