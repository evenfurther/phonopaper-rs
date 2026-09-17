package com.evenfurther.phonopaper

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class ScrubOverlayView(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {
    private val shadePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(70, 33, 150, 243)
        style = Paint.Style.FILL
    }
    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = resources.displayMetrics.density * 2.0F
    }

    private var scrubFraction = 0.0F
    private var interactive = false

    fun setInteractive(enabled: Boolean) {
        interactive = enabled
        invalidate()
    }

    fun setScrubFraction(fraction: Float) {
        scrubFraction = fraction.coerceIn(0.0F, 1.0F)
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (!interactive) {
            return
        }

        val x = scrubFraction * width
        canvas.drawRect(0.0F, 0.0F, x, height.toFloat(), shadePaint)
        canvas.drawLine(x, 0.0F, x, height.toFloat(), linePaint)
    }
}
