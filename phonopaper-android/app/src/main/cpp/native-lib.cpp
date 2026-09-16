// Native library - C++ JNI stubs that delegate to Rust

#include <jni.h>

// Forward declarations of Rust functions
extern "C" {
    void Java_com_example_phonopaper_PhonopaperDecoder_init(JNIEnv* env, jobject thiz, jstring image_path, jint sample_rate, jint samples_per_column);
    jfloatArray Java_com_example_phonopaper_PhonopaperDecoder_decodeRange(JNIEnv* env, jobject thiz, jint start_col, jint end_col);
    jint Java_com_example_phonopaper_PhonopaperDecoder_getTotalColumns(JNIEnv* env, jobject thiz);
    void Java_com_example_phonopaper_PhonopaperDecoder_cleanup(JNIEnv* env, jobject thiz);
}

// JNI entry points - these delegate directly to Rust implementations

JNIEXPORT void JNICALL
Java_com_example_phonopaper_PhonopaperDecoder_init(
    JNIEnv* env,
    jobject thiz,
    jstring image_path,
    jint sample_rate,
    jint samples_per_column
) {
    Java_com_example_phonopaper_PhonopaperDecoder_init(
        env, thiz, image_path, sample_rate, samples_per_column
    );
}

JNIEXPORT jfloatArray JNICALL
Java_com_example_phonopaper_PhonopaperDecoder_decodeRange(
    JNIEnv* env,
    jobject thiz,
    jint start_col,
    jint end_col
) {
    return Java_com_example_phonopaper_PhonopaperDecoder_decodeRange(
        env, thiz, start_col, end_col
    );
}

JNIEXPORT jint JNICALL
Java_com_example_phonopaper_PhonopaperDecoder_getTotalColumns(
    JNIEnv* env,
    jobject thiz
) {
    return Java_com_example_phonopaper_PhonopaperDecoder_getTotalColumns(
        env, thiz
    );
}

JNIEXPORT void JNICALL
Java_com_example_phonopaper_PhonopaperDecoder_cleanup(
    JNIEnv* env,
    jobject thiz
) {
    Java_com_example_phonopaper_PhonopaperDecoder_cleanup(
        env, thiz
    );
}
