package com.evenfurther.phonopaper

object PhonopaperNative {
    init {
        System.loadLibrary("phonopaper_android")
    }

    @JvmStatic
    external fun decodeImageToPcm(imageBytes: ByteArray): ShortArray

    /**
     * Locate a PhonoPaper sheet with the neural-network detector.
     *
     * Returns the corners `x0 y0 x1 y1 x2 y2 x3 y3` (top-left, top-right,
     * bottom-right, bottom-left of the sheet, clockwise in the image) as
     * fractions of the image width and height, or `null` when no sheet is
     * visible.
     */
    @JvmStatic
    external fun detectPatternCorners(imageBytes: ByteArray): FloatArray?
}
