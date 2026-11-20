package com.evenfurther.phonopaper

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.google.android.material.color.MaterialColors

/**
 * Draws the playback position over the decoded image: the part already
 * played is dimmed and a playhead marks the current column.
 *
 * The overlay is limited to [setContentRect], the area where the image is
 * actually drawn inside its letterboxed view.
 */
class ScrubOverlayView(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {
    private val density = resources.displayMetrics.density
    private val accent = MaterialColors.getColor(
        context,
        androidx.appcompat.R.attr.colorPrimary,
        Color.rgb(245, 158, 11),
    )
    private val shadePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(80, Color.red(accent), Color.green(accent), Color.blue(accent))
        style = Paint.Style.FILL
    }
    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = accent
        style = Paint.Style.STROKE
        strokeWidth = density * 3.0F
        strokeCap = Paint.Cap.ROUND
        setShadowLayer(density * 2.0F, 0.0F, 0.0F, Color.argb(160, 0, 0, 0))
    }
    private val knobPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = accent
        style = Paint.Style.FILL
    }

    private var scrubFraction = 0.0F
    private var active = false
    private var interactive = false
    private var contentRect: RectF? = null
    private val fullRect = RectF()

    /** Show the playhead (a sound has been decoded from the image). */
    fun setActive(enabled: Boolean) {
        active = enabled
        invalidate()
    }

    /** Emphasise the playhead while touch scrubbing is enabled. */
    fun setInteractive(enabled: Boolean) {
        interactive = enabled
        invalidate()
    }

    fun setScrubFraction(fraction: Float) {
        scrubFraction = fraction.coerceIn(0.0F, 1.0F)
        invalidate()
    }

    /** Area of this view covered by the image, or `null` for the whole view. */
    fun setContentRect(rect: RectF?) {
        contentRect = rect
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (!active) {
            return
        }

        val rect = contentRect ?: fullRect.apply { set(0.0F, 0.0F, width.toFloat(), height.toFloat()) }
        val x = rect.left + scrubFraction * rect.width()
        canvas.drawRect(rect.left, rect.top, x, rect.bottom, shadePaint)
        canvas.drawLine(x, rect.top, x, rect.bottom, linePaint)
        if (interactive) {
            val radius = density * 6.0F
            canvas.drawCircle(x, rect.top + radius, radius, knobPaint)
            canvas.drawCircle(x, rect.bottom - radius, radius, knobPaint)
        }
    }
}
