package com.example.phonopaper;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.provider.MediaStore;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class CameraActivity extends AppCompatActivity {
    private static final String TAG = "CameraActivity";
    private static final int REQUEST_CODE_CAMERA_PERMISSION = 100;
    private static final int REQUEST_CODE_IMAGE_CAPTURE = 101;
    private static final String[] REQUIRED_PERMISSIONS = {Manifest.permission.CAMERA};

    private Preview preview;
    private ImageCapture imageCapture;
    private SurfaceView previewView;
    private ImageView capturedImageView;
    private Button captureButton;
    private Button playButton;
    private Button manualModeButton;
    private SeekBar positionSeekBar;
    private TextView positionText;
    
    private PhonopaperDecoder decoder;
    private Handler decodeHandler;
    private HandlerThread decodeThread;
    private AudioTrack audioTrack;
    private String currentImagePath;
    private int totalColumns = 0;
    private boolean manualMode = false;
    private int currentPosition = 0;
    private int windowSize = 50; // Columns to decode at a time

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);
        
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        
        previewView = findViewById(R.id.preview_view);
        capturedImageView = findViewById(R.id.captured_image);
        captureButton = findViewById(R.id.capture_button);
        playButton = findViewById(R.id.play_button);
        manualModeButton = findViewById(R.id.manual_mode_button);
        positionSeekBar = findViewById(R.id.position_seekbar);
        positionText = findViewById(R.id.position_text);
        
        decoder = new PhonopaperDecoder();
        
        // Create handler thread for decoding
        decodeThread = new HandlerThread("DecodeThread");
        decodeThread.start();
        decodeHandler = new Handler(decodeThread.getLooper());
        
        captureButton.setOnClickListener(v -> captureImage());
        playButton.setOnClickListener(v -> playCurrentSelection());
        manualModeButton.setOnClickListener(v -> toggleManualMode());
        
        positionSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                currentPosition = progress;
                positionText.setText(String.format("Position: %d / %d", progress, totalColumns));
            }
            
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}
            
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                if (manualMode) {
                    playCurrentSelection();
                }
            }
        });
        
        // Check and request camera permission
        if (checkCameraPermission()) {
            startCamera();
        } else {
            requestCameraPermission();
        }
    }
    
    private boolean checkCameraPermission() {
        return ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == 
                PackageManager.PERMISSION_GRANTED;
    }
    
    private void requestCameraPermission() {
        ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_CAMERA_PERMISSION);
    }
    
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, 
            @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_CAMERA_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera();
            } else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }
    
    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = 
                ProcessCameraProvider.getInstance(this);
        
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                
                preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView::getHolder);
                
                imageCapture = new ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                        .build();
                
                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(
                        this, 
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        imageCapture
                );
            } catch (Exception e) {
                Log.e(TAG, "Error starting camera", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }
    
    private void captureImage() {
        if (imageCapture == null) return;
        
        File photoFile = new File(getExternalFilesDir(null), "phonopaper_" + System.currentTimeMillis() + ".jpg");
        
        ImageCapture.OutputFileOptions outputOptions = new ImageCapture.OutputFileOptions.Builder(photoFile).build();
        
        imageCapture.takePicture(outputOptions, Executors.newSingleThreadExecutor(), 
                new ImageCapture.OnImageSavedCallback() {
                    @Override
                    public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {
                        runOnUiThread(() -> {
                            currentImagePath = photoFile.getAbsolutePath();
                            displayCapturedImage();
                            initializeDecoder();
                        });
                    }
                    
                    @Override
                    public void onError(@NonNull ImageCaptureException exception) {
                        runOnUiThread(() -> 
                            Toast.makeText(CameraActivity.this, "Capture failed: " + exception.getMessage(), 
                                    Toast.LENGTH_SHORT).show());
                    }
                });
    }
    
    private void displayCapturedImage() {
        Bitmap bitmap = BitmapFactory.decodeFile(currentImagePath);
        if (bitmap != null) {
            // Rotate if needed
            Matrix matrix = new Matrix();
            matrix.postRotate(90);
            Bitmap rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
            capturedImageView.setImageBitmap(rotated);
            capturedImageView.setVisibility(View.VISIBLE);
        }
    }
    
    private void initializeDecoder() {
        decodeHandler.post(() -> {
            try {
                decoder.cleanup();
                decoder.init(currentImagePath, 44100, 353);
                totalColumns = decoder.getTotalColumns();
                runOnUiThread(() -> {
                    positionSeekBar.setMax(totalColumns - 1);
                    positionSeekBar.setProgress(0);
                    positionText.setText(String.format("Position: 0 / %d", totalColumns));
                    playButton.setEnabled(true);
                    manualModeButton.setEnabled(true);
                });
            } catch (Exception e) {
                runOnUiThread(() -> 
                    Toast.makeText(CameraActivity.this, "Decode error: " + e.getMessage(), 
                            Toast.LENGTH_SHORT).show());
            }
        });
    }
    
    private void toggleManualMode() {
        manualMode = !manualMode;
        manualModeButton.setText(manualMode ? "Auto Mode" : "Manual Mode");
        if (manualMode) {
            positionSeekBar.setVisibility(View.VISIBLE);
            positionText.setVisibility(View.VISIBLE);
        } else {
            positionSeekBar.setVisibility(View.GONE);
            positionText.setVisibility(View.GONE);
            currentPosition = 0;
            playFullAudio();
        }
    }
    
    private void playCurrentSelection() {
        decodeHandler.post(() -> {
            try {
                int endCol;
                if (manualMode) {
                    endCol = Math.min(currentPosition + windowSize, totalColumns);
                } else {
                    endCol = totalColumns;
                    currentPosition = 0;
                }
                
                float[] audioSamples = decoder.decodeRange(currentPosition, endCol);
                playAudio(audioSamples);
            } catch (Exception e) {
                runOnUiThread(() -> 
                    Toast.makeText(CameraActivity.this, "Play error: " + e.getMessage(), 
                            Toast.LENGTH_SHORT).show());
            }
        });
    }
    
    private void playFullAudio() {
        decodeHandler.post(() -> {
            try {
                float[] audioSamples = decoder.decodeRange(0, totalColumns);
                playAudio(audioSamples);
            } catch (Exception e) {
                runOnUiThread(() -> 
                    Toast.makeText(CameraActivity.this, "Play error: " + e.getMessage(), 
                            Toast.LENGTH_SHORT).show());
            }
        });
    }
    
    private void playAudio(float[] samples) {
        if (samples == null || samples.length == 0) return;
        
        runOnUiThread(() -> {
            // Release previous audio track
            if (audioTrack != null) {
                audioTrack.release();
            }
            
            int sampleRate = 44100;
            int channelConfig = AudioFormat.CHANNEL_OUT_MONO;
            int audioFormat = AudioFormat.ENCODING_PCM_FLOAT;
            int bufferSize = AudioTrack.getMinBufferSize(
                    sampleRate, 
                    channelConfig, 
                    audioFormat);
            
            audioTrack = new AudioTrack(
                    AudioManager.STREAM_MUSIC,
                    sampleRate,
                    channelConfig,
                    audioFormat,
                    Math.max(bufferSize, samples.length),
                    AudioTrack.MODE_STATIC
            );
            
            audioTrack.write(samples, 0, samples.length);
            audioTrack.setVolume(1.0f);
            audioTrack.play();
            
            audioTrack.setOnPlaybackPositionUpdateListener(position -> {
                // Could update UI with current playback position
            });
        });
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (audioTrack != null) {
            audioTrack.release();
            audioTrack = null;
        }
        if (decoder != null) {
            decoder.cleanup();
        }
        if (decodeThread != null) {
            decodeThread.quitSafely();
        }
    }
}
