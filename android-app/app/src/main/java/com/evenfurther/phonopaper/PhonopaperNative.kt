package com.evenfurther.phonopaper

object PhonopaperNative {
    init {
        System.loadLibrary("phonopaper_android")
    }

    @JvmStatic
    external fun decodeImageToPcm(imageBytes: ByteArray): ShortArray
}
