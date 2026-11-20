package com.evenfurther.phonopaper

import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.ViewModel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

/**
 * Decoded image and playback position, kept across configuration changes
 * (rotation, dark mode, …) so that the activity can be recreated without
 * decoding the image again.
 */
class PlayerViewModel : ViewModel() {
    /** Image shown in the player: either a captured frame or a library image. */
    sealed interface Source {
        data class Captured(val bitmap: Bitmap) : Source

        data class Library(val uri: Uri) : Source
    }

    var source: Source? = null
    var decodedPcm: ShortArray? = null
    var decoding = false
    var playbackFraction = 0.0F
    var statusMessage: String? = null

    /** Called on the main thread when a decode started by [decode] finishes. */
    var onDecodeFinished: (() -> Unit)? = null

    /** Background thread for native detection and decoding. */
    val executor: ExecutorService = Executors.newSingleThreadExecutor()

    private val generation = AtomicInteger(0)

    /** Start a new decode; the result of any decode still running is discarded. */
    fun nextGeneration(): Int = generation.incrementAndGet()

    fun isCurrent(candidate: Int): Boolean = candidate == generation.get()

    override fun onCleared() {
        executor.shutdownNow()
        onDecodeFinished = null
    }
}
