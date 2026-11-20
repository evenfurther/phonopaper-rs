package com.evenfurther.phonopaper

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.provider.Settings
import android.util.Log
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.SystemBarStyle
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.annotation.ColorRes
import androidx.annotation.StringRes
import androidx.annotation.WorkerThread
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.isVisible
import androidx.core.view.updateLayoutParams
import androidx.core.view.updatePadding
import com.google.android.material.button.MaterialButton
import com.google.android.material.chip.Chip
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.android.material.progressindicator.LinearProgressIndicator
import com.google.android.material.slider.Slider
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs

class MainActivity : AppCompatActivity() {
    /** What the live viewfinder is currently showing, reflected by the status pill. */
    private enum class CameraState(@field:ColorRes val dotColor: Int) {
        WAITING(R.color.status_idle),
        SEARCHING(R.color.status_idle),
        DETECTED(R.color.status_detected),
        PLAYING(R.color.status_playing),
        ERROR(R.color.status_error),
    }

    private val model: PlayerViewModel by viewModels()
    private val decodedPcm: ShortArray?
        get() = model.decodedPcm
    private var selectedPlaybackFraction: Float
        get() = model.playbackFraction
        set(value) {
            model.playbackFraction = value
        }

    private var audioTrack: AudioTrack? = null
    private var playingLive = false
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraState = CameraState.WAITING
    private var isUserSeeking = false
    private var playbackStartSample = 0
    private var lastTouchFraction = Float.NaN
    private var lastTouchPlayAtMs = 0L
    private var lastCameraAutoplayAtMs = 0L

    private val cameraFrameInFlight = AtomicBoolean(false)
    private val cameraExecutor: ExecutorService
        get() = model.executor
    private val mainHandler = Handler(Looper.getMainLooper())

    private lateinit var previewView: PreviewView
    private lateinit var detectionOverlay: DetectionOverlayView
    private lateinit var cameraStatusPill: View
    private lateinit var cameraStatusDot: ImageView
    private lateinit var cameraStatusText: TextView
    private lateinit var cameraUnavailablePanel: View
    private lateinit var cameraUnavailableText: TextView
    private lateinit var grantCameraButton: MaterialButton
    private lateinit var autoplayChip: Chip
    private lateinit var captureFrameButton: FloatingActionButton
    private lateinit var pickImageButton: MaterialButton
    private lateinit var previewImage: ImageView
    private lateinit var previewPlaceholder: View
    private lateinit var decodeProgress: LinearProgressIndicator
    private lateinit var scrubOverlay: ScrubOverlayView
    private lateinit var scrubModeChip: Chip
    private lateinit var positionSlider: Slider
    private lateinit var elapsedText: TextView
    private lateinit var durationText: TextView
    private lateinit var playButton: MaterialButton
    private lateinit var statusText: TextView

    private val imagePicker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri == null) {
            return@registerForActivityResult
        }

        showSelectedImage(PlayerViewModel.Source.Library(uri))
        beginSelectedImageDecode(getString(R.string.library_image)) {
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
                showCameraUnavailable(R.string.camera_permission_denied, offerGrant = true)
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
        // The viewfinder is dark, so keep light status-bar icons over it.
        enableEdgeToEdge(statusBarStyle = SystemBarStyle.dark(Color.TRANSPARENT))
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.cameraPreview)
        detectionOverlay = findViewById(R.id.detectionOverlay)
        cameraStatusPill = findViewById(R.id.cameraStatusPill)
        cameraStatusDot = findViewById(R.id.cameraStatusDot)
        cameraStatusText = findViewById(R.id.cameraStatusText)
        cameraUnavailablePanel = findViewById(R.id.cameraUnavailablePanel)
        cameraUnavailableText = findViewById(R.id.cameraUnavailableText)
        grantCameraButton = findViewById(R.id.grantCameraButton)
        autoplayChip = findViewById(R.id.autoplayChip)
        captureFrameButton = findViewById(R.id.captureFrameButton)
        pickImageButton = findViewById(R.id.pickImageButton)
        previewImage = findViewById(R.id.previewImage)
        previewPlaceholder = findViewById(R.id.previewPlaceholder)
        decodeProgress = findViewById(R.id.decodeProgress)
        scrubOverlay = findViewById(R.id.scrubOverlay)
        scrubModeChip = findViewById(R.id.scrubModeChip)
        positionSlider = findViewById(R.id.positionSlider)
        elapsedText = findViewById(R.id.elapsedText)
        durationText = findViewById(R.id.durationText)
        playButton = findViewById(R.id.playButton)
        statusText = findViewById(R.id.statusText)

        applyWindowInsets()

        pickImageButton.setOnClickListener { imagePicker.launch("image/*") }
        captureFrameButton.setOnClickListener {
            it.performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY)
            captureCurrentFrame()
        }
        grantCameraButton.setOnClickListener { requestCameraAgain() }
        autoplayChip.setOnCheckedChangeListener { _, _ -> renderCameraState() }
        playButton.setOnClickListener {
            if (audioTrack != null) {
                stopPlayback()
                updateSelectedStatus(getString(R.string.status_stopped))
            } else {
                playDecodedAudio(selectedPlaybackFraction)
            }
        }
        scrubModeChip.setOnCheckedChangeListener { _, isChecked ->
            scrubOverlay.setInteractive(isChecked)
            if (decodedPcm != null) {
                updateSelectedStatus(
                    getString(if (isChecked) R.string.status_ready_scrub else R.string.status_ready),
                )
            }
        }
        previewImage.setOnTouchListener { view, event -> handlePreviewTouch(view as ImageView, event) }
        previewImage.addOnLayoutChangeListener { _, _, _, _, _, _, _, _, _ -> syncScrubOverlayBounds() }

        positionSlider.setLabelFormatter { value -> formatDuration(sampleIndexForFraction(value)) }
        positionSlider.addOnChangeListener { _, value, fromUser ->
            if (fromUser) {
                updateSelectedPlaybackFraction(value)
            }
        }
        positionSlider.addOnSliderTouchListener(
            object : Slider.OnSliderTouchListener {
                override fun onStartTrackingTouch(slider: Slider) {
                    isUserSeeking = true
                }

                override fun onStopTrackingTouch(slider: Slider) {
                    isUserSeeking = false
                    updateSelectedPlaybackFraction(slider.value)
                    // Moving the cursor while playing restarts from there.
                    if (audioTrack != null && !playingLive) {
                        playDecodedAudio(selectedPlaybackFraction)
                    }
                }
            },
        )

        model.onDecodeFinished = ::onDecodeFinished
        restorePlayer()
        renderCameraState()
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
        if (model.onDecodeFinished == this::onDecodeFinished) {
            model.onDecodeFinished = null
        }
        super.onDestroy()
    }

    /**
     * The app draws edge to edge: the viewfinder runs under the status bar,
     * so only its floating controls are pushed inside the safe area, while
     * the player panel is padded away from the navigation bar.
     */
    private fun applyWindowInsets() {
        val root = findViewById<LinearLayout>(R.id.root)
        val playerScroll = findViewById<View>(R.id.playerScroll)
        val sideBySide = root.orientation == LinearLayout.HORIZONTAL
        val pillMargin = (cameraStatusPill.layoutParams as ViewGroup.MarginLayoutParams).topMargin
        ViewCompat.setOnApplyWindowInsetsListener(root) { view, windowInsets ->
            val insets = windowInsets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout(),
            )
            view.updatePadding(left = insets.left, right = insets.right, bottom = insets.bottom)
            if (sideBySide) {
                // The side panel is next to the viewfinder, not below it.
                playerScroll.updatePadding(top = insets.top)
            }
            cameraStatusPill.updateLayoutParams<ViewGroup.MarginLayoutParams> {
                topMargin = pillMargin + insets.top
            }
            WindowInsetsCompat.CONSUMED
        }
    }

    private fun ensureCameraStarted() {
        if (hasCameraPermission()) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun requestCameraAgain() {
        if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        } else {
            // The system no longer shows the prompt: the user has to allow it in the settings.
            startActivity(
                Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", packageName, null)),
            )
        }
    }

    private fun hasCameraPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun startCamera() {
        cameraUnavailablePanel.isVisible = false
        cameraStatusPill.isVisible = true
        setCameraState(CameraState.WAITING)
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
                    captureFrameButton.isEnabled = true
                    setCameraState(CameraState.SEARCHING)
                    stopCameraLoop()
                    mainHandler.post(cameraProbeRunnable)
                }.onFailure { error ->
                    Log.e(TAG, "Could not start camera preview", error)
                    showCameraUnavailable(R.string.camera_unavailable, offerGrant = false)
                }
            },
            ContextCompat.getMainExecutor(this),
        )
    }

    private fun showCameraUnavailable(@StringRes message: Int, offerGrant: Boolean) {
        captureFrameButton.isEnabled = false
        detectionOverlay.clearDetection()
        cameraStatusPill.isVisible = false
        cameraUnavailableText.setText(message)
        grantCameraButton.isVisible = offerGrant
        cameraUnavailablePanel.isVisible = true
    }

    private fun stopCameraLoop() {
        mainHandler.removeCallbacks(cameraProbeRunnable)
    }

    private fun setCameraState(state: CameraState) {
        cameraState = state
        renderCameraState()
    }

    private fun renderCameraState() {
        val text = when (cameraState) {
            CameraState.WAITING -> R.string.camera_waiting
            CameraState.SEARCHING -> R.string.camera_searching
            CameraState.DETECTED ->
                if (autoplayChip.isChecked) R.string.camera_detected else R.string.camera_detected_capture
            CameraState.PLAYING -> R.string.camera_playing
            CameraState.ERROR -> R.string.camera_frame_error
        }
        val newText = getString(text)
        // Avoid re-announcing an unchanged status through the live region.
        if (cameraStatusText.text != newText) {
            cameraStatusText.text = newText
        }
        cameraStatusDot.imageTintList = ColorStateList.valueOf(ContextCompat.getColor(this, cameraState.dotColor))
    }

    private fun probeCameraPreview() {
        if (cameraFrameInFlight.getAndSet(true)) {
            return
        }

        val bitmap = previewView.bitmap
        if (bitmap == null) {
            cameraFrameInFlight.set(false)
            setCameraState(CameraState.WAITING)
            return
        }

        val autoplayEnabled = autoplayChip.isChecked
        val playbackIdle = audioTrack == null
        val autoplayCooldownElapsed =
            SystemClock.elapsedRealtime() - lastCameraAutoplayAtMs >= CAMERA_AUTOPLAY_COOLDOWN_MS

        cameraExecutor.execute {
            try {
                val bytes = bitmapToPng(bitmap) ?: throw IOException("PNG compression failed")
                val corners = PhonopaperNative.detectPatternCorners(bytes)
                val shouldAutoplay =
                    corners != null && autoplayEnabled && playbackIdle && autoplayCooldownElapsed
                val livePcm = if (shouldAutoplay) PhonopaperNative.decodeImageToPcm(bytes) else null
                runOnUiThread {
                    updateCameraOverlay(corners)
                    if (!playingLive) {
                        setCameraState(if (corners == null) CameraState.SEARCHING else CameraState.DETECTED)
                    }
                    if (livePcm != null && livePcm.isNotEmpty()) {
                        lastCameraAutoplayAtMs = SystemClock.elapsedRealtime()
                        playTransientAudio(livePcm)
                    }
                }
            } catch (error: Exception) {
                Log.w(TAG, "Live preview frame could not be analyzed", error)
                runOnUiThread {
                    detectionOverlay.clearDetection()
                    if (!playingLive) {
                        setCameraState(CameraState.ERROR)
                    }
                }
            } finally {
                cameraFrameInFlight.set(false)
            }
        }
    }

    private fun updateCameraOverlay(corners: FloatArray?) {
        if (corners == null) {
            detectionOverlay.clearDetection()
            return
        }
        // The analysed bitmap is a snapshot of the preview view, so fractions
        // of its size are fractions of the overlay's size.
        detectionOverlay.setDetection(corners)
    }

    private fun captureCurrentFrame() {
        val bitmap = previewView.bitmap
        if (bitmap == null) {
            setCameraState(CameraState.WAITING)
            return
        }
        previewView.animate().alpha(0.3F).setDuration(60L).withEndAction {
            previewView.animate().alpha(1.0F).setDuration(160L)
        }
        showSelectedImage(PlayerViewModel.Source.Captured(bitmap))
        beginSelectedImageDecode(getString(R.string.captured_frame)) {
            bitmapToPng(bitmap) ?: throw IOException(getString(R.string.camera_frame_error))
        }
    }

    private fun showSelectedImage(source: PlayerViewModel.Source) {
        model.source = source
        when (source) {
            is PlayerViewModel.Source.Captured -> previewImage.setImageBitmap(source.bitmap)
            is PlayerViewModel.Source.Library -> previewImage.setImageURI(source.uri)
        }
        previewPlaceholder.isVisible = false
        syncScrubOverlayBounds()
    }

    /** Rebuild the player from the view model after the activity was recreated. */
    private fun restorePlayer() {
        model.source?.let(::showSelectedImage)
        decodeProgress.isVisible = model.decoding
        statusText.text = model.statusMessage
            ?: getString(if (decodedPcm != null) R.string.status_ready else R.string.status_idle_hint)
        updateSelectedPlaybackFraction(selectedPlaybackFraction)
        scrubOverlay.setInteractive(scrubModeChip.isChecked)
        renderPlayerControls()
    }

    private fun beginSelectedImageDecode(label: String, bytesProvider: () -> ByteArray) {
        stopPlayback()
        val generation = model.nextGeneration()
        model.decodedPcm = null
        model.decoding = true
        updateSelectedPlaybackFraction(0.0F)
        decodeProgress.isVisible = true
        renderPlayerControls()
        updateSelectedStatus(getString(R.string.status_decoding, label))

        // Results go to the view model, which outlives this activity on
        // rotation; whichever activity is current then refreshes itself.
        val app = applicationContext
        cameraExecutor.execute {
            val result = runCatching { PhonopaperNative.decodeImageToPcm(bytesProvider()) }
            mainHandler.post {
                if (!model.isCurrent(generation)) {
                    return@post
                }
                model.decoding = false
                result.onSuccess { pcm ->
                    model.decodedPcm = pcm.takeIf { it.isNotEmpty() }
                    model.playbackFraction = 0.0F
                    model.statusMessage = null
                }.onFailure { error ->
                    Log.e(TAG, "Decoding failed", error)
                    model.decodedPcm = null
                    model.statusMessage = userVisibleDecodeError(app, error)
                }
                model.onDecodeFinished?.invoke()
            }
        }
    }

    private fun onDecodeFinished() {
        decodeProgress.isVisible = false
        updateSelectedPlaybackFraction(selectedPlaybackFraction)
        renderPlayerControls()
        updateSelectedStatus(
            model.statusMessage
                ?: getString(if (scrubModeChip.isChecked) R.string.status_ready_scrub else R.string.status_ready),
        )
    }

    /** Enable, label and fill the player controls from the current decode and playback state. */
    private fun renderPlayerControls() {
        val pcm = decodedPcm
        val hasAudio = pcm != null
        val playing = audioTrack != null
        playButton.isEnabled = hasAudio || playing
        playButton.setText(if (playing) R.string.stop_audio else R.string.play_audio)
        playButton.setIconResource(if (playing) R.drawable.ic_stop else R.drawable.ic_play)
        positionSlider.isEnabled = hasAudio
        scrubModeChip.isEnabled = hasAudio
        scrubOverlay.setActive(hasAudio)
        durationText.text = formatDuration(pcm?.size ?: 0)
        elapsedText.text = formatDuration(currentSampleIndex())
    }

    private fun handlePreviewTouch(imageView: ImageView, event: MotionEvent): Boolean {
        if (!scrubModeChip.isChecked || decodedPcm == null) {
            return false
        }

        return when (event.actionMasked) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE -> {
                // Keep the enclosing scroll view from stealing horizontal drags.
                imageView.parent.requestDisallowInterceptTouchEvent(true)
                val fraction = fractionFromTouch(imageView, event.x, event.y) ?: return true
                updateSelectedPlaybackFraction(fraction)
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
        val startSample = sampleIndexForFraction(startFraction)
        stopPlayback()
        playBuffer(
            pcm = pcm,
            startSample = startSample,
            onStart = {
                playbackStartSample = startSample
                updateSelectedPlaybackFraction(startSample / pcm.lastIndex.coerceAtLeast(1).toFloat())
                updateSelectedStatus(
                    getString(if (scrubModeChip.isChecked) R.string.status_scrubbing else R.string.status_playing),
                )
            },
            onComplete = {
                updateSelectedStatus(getString(R.string.status_finished))
            },
        )
    }

    private fun playTransientAudio(pcm: ShortArray) {
        stopPlayback()
        playBuffer(
            pcm = pcm,
            startSample = 0,
            onStart = {
                playingLive = true
                setCameraState(CameraState.PLAYING)
            },
            onComplete = {
                setCameraState(CameraState.DETECTED)
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

        val writtenSamples = try {
            writeAll(track, slice)
        } catch (error: IOException) {
            Log.e(TAG, "Could not fill the audio track", error)
            track.release()
            updateSelectedStatus(getString(R.string.error_audio_playback))
            return
        }
        track.notificationMarkerPosition = writtenSamples
        track.positionNotificationPeriod = SAMPLE_RATE / PROGRESS_UPDATES_PER_SECOND
        track.setPlaybackPositionUpdateListener(
            object : AudioTrack.OnPlaybackPositionUpdateListener {
                override fun onMarkerReached(track: AudioTrack) {
                    runOnUiThread {
                        if (track !== audioTrack) {
                            return@runOnUiThread
                        }
                        stopPlayback()
                        onComplete()
                    }
                }

                override fun onPeriodicNotification(track: AudioTrack) {
                    runOnUiThread {
                        val decoded = decodedPcm
                        if (track !== audioTrack || playingLive || decoded == null) {
                            return@runOnUiThread
                        }
                        if (!isUserSeeking && !scrubModeChip.isChecked) {
                            val absoluteSample = playbackStartSample + track.playbackHeadPosition
                            val last = decoded.lastIndex.coerceAtLeast(1)
                            updateSelectedPlaybackFraction(absoluteSample.coerceAtMost(last) / last.toFloat())
                        }
                    }
                }
            },
        )
        track.play()

        audioTrack = track
        onStart()
        renderPlayerControls()
    }

    private fun stopPlayback() {
        audioTrack?.runCatching {
            stop()
        }
        audioTrack?.release()
        audioTrack = null
        if (playingLive) {
            playingLive = false
            setCameraState(CameraState.DETECTED)
        }
        renderPlayerControls()
    }

    private fun updateSelectedPlaybackFraction(fraction: Float) {
        selectedPlaybackFraction = fraction.coerceIn(0.0F, 1.0F)
        if (!isUserSeeking) {
            positionSlider.value = selectedPlaybackFraction
        }
        scrubOverlay.setScrubFraction(selectedPlaybackFraction)
        elapsedText.text = formatDuration(currentSampleIndex())
    }

    private fun syncScrubOverlayBounds() {
        scrubOverlay.setContentRect(displayedImageRect(previewImage))
    }

    private fun currentSampleIndex(): Int = sampleIndexForFraction(selectedPlaybackFraction)

    private fun sampleIndexForFraction(fraction: Float): Int {
        val pcm = decodedPcm ?: return 0
        val numColumns = (pcm.size / SAMPLES_PER_COLUMN).coerceAtLeast(1)
        val column = Math.round(fraction.coerceIn(0.0F, 1.0F) * (numColumns - 1))
        return (column * SAMPLES_PER_COLUMN).coerceAtMost(pcm.lastIndex.coerceAtLeast(0))
    }

    private fun updateSelectedStatus(message: String) {
        statusText.text = message
        model.statusMessage = message
    }

    private fun fractionFromTouch(imageView: ImageView, x: Float, y: Float): Float? {
        val rect = displayedImageRect(imageView) ?: return null
        if (y < rect.top || y > rect.bottom) {
            return null
        }
        // Dragging past the sides clamps to the start or end of the sound.
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

    @WorkerThread
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
        return String.format(Locale.getDefault(), "%.1f s", seconds)
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
        private fun userVisibleDecodeError(context: Context, error: Throwable): String {
            val details = error.message.orEmpty()
            return context.getString(
                when {
                    error is IOException -> R.string.error_read_image
                    "marker" in details -> R.string.error_marker_pattern
                    "image" in details.lowercase(Locale.US) -> R.string.error_unsupported_image
                    else -> R.string.error_decode_failed
                },
            )
        }

        private const val TAG = "MainActivity"
        private const val SAMPLE_RATE = 44_100
        private const val SAMPLES_PER_COLUMN = 353
        private const val PROGRESS_UPDATES_PER_SECOND = 20
        private const val CAMERA_POLL_INTERVAL_MS = 400L
        private const val CAMERA_AUTOPLAY_COOLDOWN_MS = 3_000L
        private const val TOUCH_SCRUB_MIN_INTERVAL_MS = 120L
        private const val TOUCH_SCRUB_MIN_DELTA = 0.015F
    }
}
