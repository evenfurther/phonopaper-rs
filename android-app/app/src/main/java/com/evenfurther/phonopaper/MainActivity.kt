package com.evenfurther.phonopaper

import android.Manifest
import android.graphics.Bitmap
import android.graphics.RectF
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.MotionEvent
import android.widget.Button
import android.widget.ImageView
import android.widget.SeekBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.abs
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {
    private var decodedPcm: ShortArray? = null
    private var audioTrack: AudioTrack? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var isUserSeeking = false
    private var selectedPlaybackFraction = 0.0F
    private var playbackStartSample = 0
    private var lastTouchFraction = Float.NaN
    private var lastTouchPlayAtMs = 0L
    private var lastCameraAutoplayAtMs = 0L

    private val decodeGeneration = AtomicInteger(0)
    private val cameraFrameInFlight = AtomicBoolean(false)
    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())

    private lateinit var previewView: PreviewView
    private lateinit var detectionOverlay: DetectionOverlayView
    private lateinit var cameraStatusText: TextView
    private lateinit var autoplaySwitch: SwitchCompat
    private lateinit var captureFrameButton: Button
    private lateinit var previewImage: ImageView
    private lateinit var scrubOverlay: ScrubOverlayView
    private lateinit var pickImageButton: Button
    private lateinit var playButton: Button
    private lateinit var stopButton: Button
    private lateinit var scrubModeSwitch: SwitchCompat
    private lateinit var positionSeekBar: SeekBar
    private lateinit var statusText: TextView

    private val imagePicker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri == null) {
            return@registerForActivityResult
        }

        val generation = decodeGeneration.incrementAndGet()
        previewImage.setImageURI(uri)
        beginSelectedImageDecode(generation, uri.lastPathSegment ?: getString(R.string.library_image)) {
            contentResolver.openInputStream(uri)?.use { input ->
                input.readBytes()
            } ?: throw IOException(getString(R.string.error_open_image))
        }
    }

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startCamera()
            } else {
                cameraStatusText.text = getString(R.string.camera_permission_denied)
            }
        }

    private val cameraProbeRunnable = object : Runnable {
        override fun run() {
            if (hasCameraPermission()) {
                probeCameraPreview()
                mainHandler.postDelayed(this, CAMERA_POLL_INTERVAL_MS)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.cameraPreview)
        detectionOverlay = findViewById(R.id.detectionOverlay)
        cameraStatusText = findViewById(R.id.cameraStatusText)
        autoplaySwitch = findViewById(R.id.autoplaySwitch)
        captureFrameButton = findViewById(R.id.captureFrameButton)
        previewImage = findViewById(R.id.previewImage)
        scrubOverlay = findViewById(R.id.scrubOverlay)
        pickImageButton = findViewById(R.id.pickImageButton)
        playButton = findViewById(R.id.playButton)
        stopButton = findViewById(R.id.stopButton)
        scrubModeSwitch = findViewById(R.id.scrubModeSwitch)
        positionSeekBar = findViewById(R.id.positionSeekBar)
        statusText = findViewById(R.id.statusText)

        pickImageButton.setOnClickListener { imagePicker.launch("image/*") }
        playButton.setOnClickListener { playDecodedAudio(selectedPlaybackFraction) }
        stopButton.setOnClickListener {
            stopPlayback()
            updateSelectedStatus(getString(R.string.status_stopped))
        }
        captureFrameButton.setOnClickListener { captureCurrentFrame() }
        scrubModeSwitch.setOnCheckedChangeListener { _, isChecked ->
            scrubOverlay.setInteractive(isChecked)
            if (!isChecked) {
                updateSelectedStatusForCurrentPosition()
            }
        }
        previewImage.setOnTouchListener { view, event -> handlePreviewTouch(view as ImageView, event) }
        positionSeekBar.max = SEEK_BAR_MAX
        positionSeekBar.setOnSeekBarChangeListener(
            object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                    if (!fromUser) {
                        return
                    }
                    updateSelectedPlaybackFraction(progress / SEEK_BAR_MAX.toFloat())
                    updateSelectedStatusForCurrentPosition()
                }

                override fun onStartTrackingTouch(seekBar: SeekBar) {
                    isUserSeeking = true
                }

                override fun onStopTrackingTouch(seekBar: SeekBar) {
                    isUserSeeking = false
                    updateSelectedPlaybackFraction(seekBar.progress / SEEK_BAR_MAX.toFloat())
                    updateSelectedStatusForCurrentPosition()
                }
            },
        )
    }

    override fun onStart() {
        super.onStart()
        ensureCameraStarted()
    }

    override fun onStop() {
        stopCameraLoop()
        cameraProvider?.unbindAll()
        super.onStop()
    }

    override fun onDestroy() {
        stopPlayback()
        stopCameraLoop()
        cameraProvider?.unbindAll()
        cameraExecutor.shutdownNow()
        super.onDestroy()
    }

    private fun ensureCameraStarted() {
        if (hasCameraPermission()) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun hasCameraPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            android.content.pm.PackageManager.PERMISSION_GRANTED

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener(
            {
                runCatching {
                    providerFuture.get().also { provider ->
                        cameraProvider = provider
                        provider.unbindAll()
                        provider.bindToLifecycle(
                            this,
                            CameraSelector.DEFAULT_BACK_CAMERA,
                            Preview.Builder().build().also {
                                it.surfaceProvider = previewView.surfaceProvider
                            },
                        )
                    }
                }.onSuccess {
                    cameraStatusText.text = getString(R.string.camera_ready)
                    captureFrameButton.isEnabled = true
                    stopCameraLoop()
                    mainHandler.post(cameraProbeRunnable)
                }.onFailure { error ->
                    Log.e(TAG, "Could not start camera preview", error)
                    cameraStatusText.text = getString(R.string.camera_unavailable)
                }
            },
            ContextCompat.getMainExecutor(this),
        )
    }

    private fun stopCameraLoop() {
        mainHandler.removeCallbacks(cameraProbeRunnable)
    }

    private fun probeCameraPreview() {
        if (cameraFrameInFlight.getAndSet(true)) {
            return
        }

        val bitmap = previewView.bitmap
        if (bitmap == null) {
            cameraFrameInFlight.set(false)
            cameraStatusText.text = getString(R.string.camera_waiting)
            return
        }

        val bytes = bitmapToPng(bitmap)
        if (bytes == null) {
            cameraFrameInFlight.set(false)
            cameraStatusText.text = getString(R.string.camera_frame_error)
            return
        }
        val autoplayEnabled = autoplaySwitch.isChecked
        val playbackIdle = audioTrack == null
        val autoplayCooldownElapsed =
            SystemClock.elapsedRealtime() - lastCameraAutoplayAtMs >= CAMERA_AUTOPLAY_COOLDOWN_MS

        cameraExecutor.execute {
            try {
                val bounds = PhonopaperNative.detectPreviewBounds(bytes)
                val shouldAutoplay =
                    bounds != null && autoplayEnabled && playbackIdle && autoplayCooldownElapsed
                val livePcm = if (shouldAutoplay) PhonopaperNative.decodeImageToPcm(bytes) else null
                runOnUiThread {
                    updateCameraOverlay(bounds, bitmap.height)
                    if (bounds == null) {
                        cameraStatusText.text = getString(R.string.camera_searching)
                    } else {
                        cameraStatusText.text = getString(R.string.camera_detected)
                    }
                    if (livePcm != null && livePcm.isNotEmpty()) {
                        lastCameraAutoplayAtMs = SystemClock.elapsedRealtime()
                        playTransientAudio(livePcm, getString(R.string.camera_playing))
                    }
                }
            } catch (error: Exception) {
                Log.w(TAG, "Live preview frame could not be analyzed", error)
                runOnUiThread {
                    detectionOverlay.clearDetection()
                    cameraStatusText.text = getString(R.string.camera_frame_error)
                }
            } finally {
                cameraFrameInFlight.set(false)
            }
        }
    }

    private fun updateCameraOverlay(bounds: IntArray?, bitmapHeight: Int) {
        if (bounds == null || bitmapHeight <= 0) {
            detectionOverlay.clearDetection()
            return
        }
        val top = bounds[0] / bitmapHeight.toFloat()
        val bottom = bounds[1] / bitmapHeight.toFloat()
        detectionOverlay.setDetection(top, bottom)
    }

    private fun captureCurrentFrame() {
        val bitmap = previewView.bitmap
        if (bitmap == null) {
            cameraStatusText.text = getString(R.string.camera_waiting)
            return
        }
        previewImage.setImageBitmap(bitmap)
        val generation = decodeGeneration.incrementAndGet()
        beginSelectedImageDecode(generation, getString(R.string.captured_frame)) {
            bitmapToPng(bitmap) ?: throw IOException(getString(R.string.camera_frame_error))
        }
    }

    private fun beginSelectedImageDecode(
        generation: Int,
        label: String,
        bytesProvider: () -> ByteArray,
    ) {
        stopPlayback()
        decodedPcm = null
        playButton.isEnabled = false
        stopButton.isEnabled = false
        positionSeekBar.progress = 0
        updateSelectedPlaybackFraction(0.0F)
        updateSelectedStatus(getString(R.string.status_decoding, label))

        cameraExecutor.execute {
            try {
                val pcm = PhonopaperNative.decodeImageToPcm(bytesProvider())
                runOnUiThread {
                    if (generation != decodeGeneration.get()) {
                        return@runOnUiThread
                    }
                    decodedPcm = pcm
                    playButton.isEnabled = pcm.isNotEmpty()
                    stopButton.isEnabled = false
                    updateSelectedPlaybackFraction(0.0F)
                    updateSelectedStatus(getString(R.string.status_ready, formatDuration(pcm.size)))
                }
            } catch (error: Exception) {
                Log.e(TAG, "Decoding failed", error)
                runOnUiThread {
                    if (generation != decodeGeneration.get()) {
                        return@runOnUiThread
                    }
                    decodedPcm = null
                    playButton.isEnabled = false
                    stopButton.isEnabled = false
                    updateSelectedStatus(userVisibleDecodeError(error))
                }
            }
        }
    }

    private fun handlePreviewTouch(imageView: ImageView, event: MotionEvent): Boolean {
        if (!scrubModeSwitch.isChecked || decodedPcm == null) {
            return false
        }

        return when (event.actionMasked) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE -> {
                val fraction = fractionFromTouch(imageView, event.x, event.y) ?: return true
                updateSelectedPlaybackFraction(fraction)
                updateSelectedStatus(getString(R.string.status_touch_scrub, formatDuration(currentSampleIndex())))
                val now = SystemClock.elapsedRealtime()
                if (
                    audioTrack == null ||
                        abs(fraction - lastTouchFraction) >= TOUCH_SCRUB_MIN_DELTA ||
                        now - lastTouchPlayAtMs >= TOUCH_SCRUB_MIN_INTERVAL_MS
                ) {
                    lastTouchPlayAtMs = now
                    lastTouchFraction = fraction
                    playDecodedAudio(fraction)
                }
                true
            }
            MotionEvent.ACTION_UP -> {
                imageView.performClick()
                true
            }
            MotionEvent.ACTION_CANCEL -> true
            else -> false
        }
    }

    private fun playDecodedAudio(startFraction: Float) {
        val pcm = decodedPcm ?: return
        val startSample = sampleIndexForFraction(pcm, startFraction)
        stopPlayback()
        playBuffer(
            pcm = pcm,
            startSample = startSample,
            onStart = {
                playbackStartSample = startSample
                playButton.isEnabled = false
                stopButton.isEnabled = true
                updateSelectedPlaybackFraction(startSample / pcm.lastIndex.coerceAtLeast(1).toFloat())
                updateSelectedStatus(
                    if (scrubModeSwitch.isChecked) {
                        getString(R.string.status_scrubbing)
                    } else {
                        getString(R.string.status_playing)
                    },
                )
            },
            onComplete = {
                updateSelectedStatus(getString(R.string.status_finished))
            },
        )
    }

    private fun playTransientAudio(pcm: ShortArray, status: String) {
        stopPlayback()
        playBuffer(
            pcm = pcm,
            startSample = 0,
            onStart = {
                playButton.isEnabled = decodedPcm != null
                stopButton.isEnabled = true
                cameraStatusText.text = status
            },
            onComplete = {
                cameraStatusText.text = getString(R.string.camera_detected)
            },
        )
    }

    private fun playBuffer(
        pcm: ShortArray,
        startSample: Int,
        onStart: () -> Unit,
        onComplete: () -> Unit,
    ) {
        val clippedStart = startSample.coerceIn(0, pcm.lastIndex.coerceAtLeast(0))
        val slice = pcm.copyOfRange(clippedStart, pcm.size)
        if (slice.isEmpty()) {
            return
        }

        val track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                    .build(),
            ).setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(SAMPLE_RATE)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build(),
            ).setBufferSizeInBytes(slice.size * 2)
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()

        val writtenSamples = writeAll(track, slice)
        track.notificationMarkerPosition = writtenSamples
        track.positionNotificationPeriod = SAMPLE_RATE / 10
        track.setPlaybackPositionUpdateListener(
            object : AudioTrack.OnPlaybackPositionUpdateListener {
                override fun onMarkerReached(track: AudioTrack) {
                    runOnUiThread {
                        stopPlayback()
                        onComplete()
                    }
                }

                override fun onPeriodicNotification(track: AudioTrack) {
                    runOnUiThread {
                        if (!isUserSeeking && decodedPcm != null && scrubModeSwitch.isChecked.not()) {
                            val absoluteSample = playbackStartSample + track.playbackHeadPosition
                            updateSelectedPlaybackFraction(
                                absoluteSample.coerceAtMost(decodedPcm!!.lastIndex.coerceAtLeast(0)) /
                                    decodedPcm!!.lastIndex.coerceAtLeast(1).toFloat(),
                            )
                        }
                    }
                }
            },
        )
        track.play()

        audioTrack = track
        onStart()
    }

    private fun stopPlayback() {
        audioTrack?.runCatching {
            stop()
        }
        audioTrack?.release()
        audioTrack = null
        stopButton.isEnabled = false
        playButton.isEnabled = decodedPcm != null
    }

    private fun updateSelectedPlaybackFraction(fraction: Float) {
        selectedPlaybackFraction = fraction.coerceIn(0.0F, 1.0F)
        if (!isUserSeeking) {
            positionSeekBar.progress = (selectedPlaybackFraction * SEEK_BAR_MAX).roundToInt()
        }
        scrubOverlay.setScrubFraction(selectedPlaybackFraction)
    }

    private fun currentSampleIndex(): Int {
        val pcm = decodedPcm ?: return 0
        return sampleIndexForFraction(pcm, selectedPlaybackFraction)
    }

    private fun sampleIndexForFraction(pcm: ShortArray, fraction: Float): Int {
        val numColumns = (pcm.size / SAMPLES_PER_COLUMN).coerceAtLeast(1)
        val column = (fraction.coerceIn(0.0F, 1.0F) * (numColumns - 1)).roundToInt()
        return (column * SAMPLES_PER_COLUMN).coerceAtMost(pcm.lastIndex.coerceAtLeast(0))
    }

    private fun updateSelectedStatusForCurrentPosition() {
        val pcm = decodedPcm
        if (pcm == null) {
            return
        }
        updateSelectedStatus(
            getString(
                R.string.status_position,
                formatDuration(currentSampleIndex()),
                formatDuration(pcm.size),
            ),
        )
    }

    private fun updateSelectedStatus(message: String) {
        statusText.text = message
    }

    private fun userVisibleDecodeError(error: Exception): String {
        val details = error.message.orEmpty()
        return when {
            error is IOException -> getString(R.string.error_read_image)
            "marker pattern" in details -> getString(R.string.error_marker_pattern)
            "image" in details.lowercase(Locale.US) -> getString(R.string.error_unsupported_image)
            else -> getString(R.string.error_decode_failed)
        }
    }

    private fun fractionFromTouch(imageView: ImageView, x: Float, y: Float): Float? {
        val rect = displayedImageRect(imageView) ?: return null
        if (!rect.contains(x, y)) {
            return null
        }
        return ((x - rect.left) / rect.width()).coerceIn(0.0F, 1.0F)
    }

    private fun displayedImageRect(imageView: ImageView): RectF? {
        val drawable = imageView.drawable ?: return null
        val drawableWidth = drawable.intrinsicWidth.toFloat()
        val drawableHeight = drawable.intrinsicHeight.toFloat()
        if (drawableWidth <= 0.0F || drawableHeight <= 0.0F) {
            return null
        }

        val viewWidth = imageView.width.toFloat()
        val viewHeight = imageView.height.toFloat()
        if (viewWidth <= 0.0F || viewHeight <= 0.0F) {
            return null
        }

        val scale = minOf(viewWidth / drawableWidth, viewHeight / drawableHeight)
        val width = drawableWidth * scale
        val height = drawableHeight * scale
        val left = (viewWidth - width) / 2.0F
        val top = (viewHeight - height) / 2.0F
        return RectF(left, top, left + width, top + height)
    }

    private fun bitmapToPng(bitmap: Bitmap): ByteArray? {
        val output = ByteArrayOutputStream()
        return if (bitmap.compress(Bitmap.CompressFormat.PNG, 100, output)) {
            output.toByteArray()
        } else {
            null
        }
    }

    private fun formatDuration(sampleCount: Int): String {
        val seconds = sampleCount / SAMPLE_RATE.toFloat()
        return String.format(Locale.US, "%.1fs", seconds)
    }

    private fun writeAll(track: AudioTrack, pcm: ShortArray): Int {
        var totalWritten = 0
        while (totalWritten < pcm.size) {
            val written = track.write(pcm, totalWritten, pcm.size - totalWritten)
            if (written <= 0) {
                throw IOException(getString(R.string.error_audio_playback))
            }
            totalWritten += written
        }
        return totalWritten
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val SAMPLE_RATE = 44_100
        private const val SAMPLES_PER_COLUMN = 353
        private const val SEEK_BAR_MAX = 1_000
        private const val CAMERA_POLL_INTERVAL_MS = 400L
        private const val CAMERA_AUTOPLAY_COOLDOWN_MS = 3_000L
        private const val TOUCH_SCRUB_MIN_INTERVAL_MS = 120L
        private const val TOUCH_SCRUB_MIN_DELTA = 0.015F
    }
}
