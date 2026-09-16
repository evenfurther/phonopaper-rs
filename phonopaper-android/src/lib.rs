//! Phonopaper Android library - Rust JNI bindings for camera-based decoding

use jni::objects::{JClass, JObject, JString};
use jni::sys::jfloatArray;
use jni::JNIEnv;
use phonopaper_rs::decode::{column_amplitudes_from_image_into, detect_markers, fill_spectrogram_from_pixels, spectrogram_to_audio, SynthesisOptions, DataBounds};
use phonopaper_rs::format::TOTAL_BINS;
use phonopaper_rs::spectrogram::SpectrogramVec;

// Global state for holding decoded audio between calls
static mut DECODER_STATE: Option<DecoderState> = None;

struct DecoderState {
    spectrogram: SpectrogramVec,
    sample_rate: u32,
    samples_per_column: usize,
}

/// Initialize the decoder with an image
#[no_mangle]
pub extern "system" fn Java_com_example_phonopaper_PhonopaperDecoder_init(
    env: JNIEnv,
    _class: JClass,
    image_path: JString,
    sample_rate: jni::sys::jint,
    samples_per_column: jni::sys::jint,
) -> jni::sys::jobject {
    let path: String = env.get_string(&image_path).unwrap().into();
    
    match std::fs::read(&path) {
        Ok(image_data) => {
            match image::load_from_memory(&image_data) {
                Ok(dynamic_image) => {
                    let rgb = dynamic_image.to_rgb8();
                    let width = rgb.width() as usize;
                    let height = rgb.height() as usize;
                    
                    let mut spectrogram = SpectrogramVec::new(width);
                    
                    // Detect markers and extract data bounds
                    match detect_markers(&dynamic_image) {
                        Ok(bounds) => {
                            // Convert image to grayscale pixels
                            let pixels: Vec<u8> = rgb.pixels().map(|p| {
                                // Convert RGB to grayscale (luma)
                                let r = p[0] as f32;
                                let g = p[1] as f32;
                                let b = p[2] as f32;
                                // Use standard luma formula
                                let luma = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
                                // Invert: black (0) = amplitude 1.0, white (255) = amplitude 0.0
                                255 - luma
                            }).collect();
                            
                            fill_spectrogram_from_pixels(&mut spectrogram, &pixels, width, height);
                        }
                        Err(_) => {
                            // If marker detection fails, use the whole image
                            let pixels: Vec<u8> = rgb.pixels().map(|p| {
                                let r = p[0] as f32;
                                let g = p[1] as f32;
                                let b = p[2] as f32;
                                let luma = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
                                255 - luma
                            }).collect();
                            
                            fill_spectrogram_from_pixels(&mut spectrogram, &pixels, width, height);
                        }
                    }
                    
                    unsafe {
                        DECODER_STATE = Some(DecoderState {
                            spectrogram,
                            sample_rate: sample_rate as u32,
                            samples_per_column: samples_per_column as usize,
                        });
                    }
                    
                    // Return null (success)
                    JObject::null().into_raw()
                }
                Err(e) => {
                    env.throw_new("java/lang/Exception", format!("Failed to load image: {}", e)).unwrap();
                    JObject::null().into_raw()
                }
            }
        }
        Err(e) => {
            env.throw_new("java/lang/Exception", format!("Failed to read file: {}", e)).unwrap();
            JObject::null().into_raw()
        }
    }
}

/// Decode a column range to audio
#[no_mangle]
pub extern "system" fn Java_com_example_phonopaper_PhonopaperDecoder_decodeRange(
    env: JNIEnv,
    _class: JClass,
    start_col: jni::sys::jint,
    end_col: jni::sys::jint,
) -> jfloatArray {
    unsafe {
        if let Some(state) = &DECODER_STATE {
            let start = start_col as usize;
            let end = end_col as usize;
            let cols = end.saturating_sub(start);
            
            if cols == 0 {
                return env.new_float_array(0).unwrap().into_raw();
            }
            
            // Extract the column range from spectrogram
            let mut partial_spectrogram = SpectrogramVec::new(cols);
            for (dst_col, src_col) in (0..cols).zip(start..end) {
                if src_col < state.spectrogram.num_columns() {
                    for bin in 0..TOTAL_BINS {
                        partial_spectrogram.set(dst_col, bin, state.spectrogram.get(src_col, bin));
                    }
                }
            }
            
            // Synthesize audio - use the const generic version with Vec storage
            let options = SynthesisOptions::default();
            let mut audio = vec![0.0f32; cols * state.samples_per_column];
            spectrogram_to_audio::<Vec<f32>, { 353 }>(&partial_spectrogram, &options, &mut audio);
            
            // Convert to Java float array
            let result = env.new_float_array(audio.len() as i32).unwrap();
            env.set_float_array_region(&result, 0, &audio).unwrap();
            result.into_raw()
        } else {
            env.throw_new("java/lang/IllegalStateException", "Decoder not initialized").unwrap();
            JObject::null().into_raw()
        }
    }
}

/// Get the total number of columns in the image
#[no_mangle]
pub extern "system" fn Java_com_example_phonopaper_PhonopaperDecoder_getTotalColumns(
    env: JNIEnv,
    _class: JClass,
) -> jni::sys::jint {
    unsafe {
        if let Some(state) = &DECODER_STATE {
            state.spectrogram.num_columns() as jni::sys::jint
        } else {
            0
        }
    }
}

/// Clean up decoder state
#[no_mangle]
pub extern "system" fn Java_com_example_phonopaper_PhonopaperDecoder_cleanup(
    env: JNIEnv,
    _class: JClass,
) {
    unsafe {
        DECODER_STATE = None;
    }
}
