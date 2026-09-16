package com.example.phonopaper;

public class PhonopaperDecoder {
    static {
        System.loadLibrary("phonopaper_android");
    }

    // Native methods
    public native void init(String imagePath, int sampleRate, int samplesPerColumn);
    public native float[] decodeRange(int startCol, int endCol);
    public native int getTotalColumns();
    public native void cleanup();
}
