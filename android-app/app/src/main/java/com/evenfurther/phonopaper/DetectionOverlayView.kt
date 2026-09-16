package com.evenfurther.phonopaper

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.max
import kotlin.math.min

class DetectionOverlayView(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(90, 255, 193, 7)
        style = Paint.Style.FILL
    }
    private val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = resources.displayMetrics.density * 2.0F
    }
    private val bandRect = RectF()

    private var topFraction = 0.0F
    private var bottomFraction = 0.0F
    private var hasDetection = false

    fun clearDetection() {
        hasDetection = false
        invalidate()
    }

    fun setDetection(top: Float, bottom: Float) {
        topFraction = top.coerceIn(0.0F, 1.0F)
        bottomFraction = bottom.coerceIn(0.0F, 1.0F)
        hasDetection = bottomFraction > topFraction
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (!hasDetection) {
            return
        }

        val top = min(topFraction, bottomFraction) * height
        val bottom = max(topFraction, bottomFraction) * height
        bandRect.set(0.0F, top, width.toFloat(), bottom)
        val radius = resources.displayMetrics.density * 12.0F
        canvas.drawRoundRect(bandRect, radius, radius, fillPaint)
        canvas.drawRoundRect(bandRect, radius, radius, strokePaint)
    }
}
