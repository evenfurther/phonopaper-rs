package com.evenfurther.phonopaper

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.io.IOException

class MainActivity : AppCompatActivity() {
    private var decodedPcm: ShortArray? = null
    private var audioTrack: AudioTrack? = null

    private lateinit var previewImage: ImageView
    private lateinit var pickImageButton: Button
    private lateinit var playButton: Button
    private lateinit var stopButton: Button
    private lateinit var statusText: TextView

    private val imagePicker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri == null) {
            return@registerForActivityResult
        }

        previewImage.setImageURI(uri)
        stopPlayback()
        pickImageButton.isEnabled = false
        playButton.isEnabled = false
        stopButton.isEnabled = false
        statusText.text = "Decoding ${uri.lastPathSegment ?: "image"}…"

        Thread {
            try {
                val bytes = contentResolver.openInputStream(uri)?.use { input ->
                    input.readBytes()
                } ?: throw IOException("Could not open the selected image.")
                val pcm = PhonopaperNative.decodeImageToPcm(bytes)
                runOnUiThread {
                    decodedPcm = pcm
                    pickImageButton.isEnabled = true
                    playButton.isEnabled = pcm.isNotEmpty()
                    stopButton.isEnabled = false
                    statusText.text = "Decoded ${pcm.size} mono PCM samples at 44.1 kHz."
                }
            } catch (error: Exception) {
                runOnUiThread {
                    decodedPcm = null
                    pickImageButton.isEnabled = true
                    playButton.isEnabled = false
                    stopButton.isEnabled = false
                    statusText.text = error.message ?: "Decoding failed."
                }
            }
        }.start()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewImage = findViewById(R.id.previewImage)
        pickImageButton = findViewById(R.id.pickImageButton)
        playButton = findViewById(R.id.playButton)
        stopButton = findViewById(R.id.stopButton)
        statusText = findViewById(R.id.statusText)

        pickImageButton.setOnClickListener {
            imagePicker.launch("image/*")
        }
        playButton.setOnClickListener {
            playDecodedAudio()
        }
        stopButton.setOnClickListener {
            stopPlayback()
        }
    }

    override fun onDestroy() {
        stopPlayback()
        super.onDestroy()
    }

    private fun playDecodedAudio() {
        val pcm = decodedPcm ?: return
        stopPlayback()

        val track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                    .build(),
            ).setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(44_100)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build(),
            ).setBufferSizeInBytes(pcm.size * 2)
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()

        track.write(pcm, 0, pcm.size)
        track.notificationMarkerPosition = pcm.size
        track.setPlaybackPositionUpdateListener(
            object : AudioTrack.OnPlaybackPositionUpdateListener {
                override fun onMarkerReached(track: AudioTrack) {
                    runOnUiThread {
                        stopPlayback()
                        statusText.text = "Playback finished."
                    }
                }

                override fun onPeriodicNotification(track: AudioTrack) = Unit
            },
        )
        track.play()

        audioTrack = track
        playButton.isEnabled = false
        stopButton.isEnabled = true
        statusText.text = "Playing decoded audio…"
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
}
