//! Phonopaper Android library - Rust JNI bindings for camera-based decoding

use jni::objects::{JClass, JObject, JString, JValue};
use jni::sys::{jbyteArray, jfloatArray, jint, jobject};
use jni::{JNIEnv, JNIEnvExt};
use phonopaper_rs::decode::{column_amplitudes_from_image_into, detect_markers_at_column, spectrogram_to_audio, SynthesisOptions};
use phonopaper_rs::format::TOTAL_BINS;
use phonopaper_rs::spectrogram::SpectrogramVec;
use std::path::Path;

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
    sample_rate: jint,
    samples_per_column: jint,
) -> jobject {
    let path: String = env.get_string(&image_path).unwrap().into();
    
    match std::fs::read(&path) {
        Ok(image_data) => {
            match image::load_from_memory(&image_data) {
                Ok(dynamic_image) => {
                    let rgb = dynamic_image.to_rgb8();
                    let width = rgb.width() as usize;
                    let height = rgb.height() as usize;
                    
                    let mut spectrogram = SpectrogramVec::new(width, TOTAL_BINS);
                    
                    // Detect markers and extract data bounds
                    let data_top = 0;
                    let data_bottom = height;
                    
                    // Fill spectrogram from image
                    for col in 0..width {
                        if let Some(amplitudes) = column_amplitudes_from_image_into(
                            &rgb,
                            col,
                            data_top,
                            data_bottom,
                            &mut spectrogram,
                        ).ok() {
                            // Successfully decoded column
                        }
                    }
                    
                    unsafe {
                        DECODER_STATE = Some(DecoderState {
                            spectrogram,
                            sample_rate: sample_rate as u32,
                            samples_per_column: samples_per_column as usize,
                        });
                    }
                    
                    // Return true (success)
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
    start_col: jint,
    end_col: jint,
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
            let mut partial_spectrogram = SpectrogramVec::new(cols, TOTAL_BINS);
            for (dst_col, src_col) in (0..cols).zip(start..end) {
                if src_col < state.spectrogram.num_columns() {
                    for bin in 0..TOTAL_BINS {
                        partial_spectrogram.set(dst_col, bin, state.spectrogram.get(src_col, bin));
                    }
                }
            }
            
            // Synthesize audio
            let options = SynthesisOptions::default();
            let audio = spectrogram_to_audio(&partial_spectrogram, state.sample_rate, state.samples_per_column, options);
            
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
) -> jint {
    unsafe {
        if let Some(state) = &DECODER_STATE {
            state.spectrogram.num_columns() as jint
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
