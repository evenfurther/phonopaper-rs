package com.evenfurther.phonopaper

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View

/**
 * Draws the quadrilateral of the PhonoPaper sheet found by the neural-network
 * detector over the camera preview.
 *
 * The two marker-band edges (top-left → top-right and bottom-right →
 * bottom-left) are stroked thicker so the user can see how the sheet is
 * oriented.
 */
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
    private val bandPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = resources.displayMetrics.density * 5.0F
        strokeCap = Paint.Cap.ROUND
    }
    private val outline = Path()

    /** `x0 y0 … x3 y3` as fractions of the view size, or `null`. */
    private var corners: FloatArray? = null

    fun clearDetection() {
        corners = null
        invalidate()
    }

    /**
     * Show a detection.  [fractions] holds eight values, `x0 y0 … x3 y3`, as
     * fractions of the analysed image size (which is also the preview size).
     */
    fun setDetection(fractions: FloatArray) {
        if (fractions.size != 8) {
            clearDetection()
            return
        }
        corners = fractions.copyOf()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val c = corners ?: return
        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0.0F || h <= 0.0F) {
            return
        }

        outline.rewind()
        outline.moveTo(c[0] * w, c[1] * h)
        for (i in 1 until 4) {
            outline.lineTo(c[2 * i] * w, c[2 * i + 1] * h)
        }
        outline.close()
        canvas.drawPath(outline, fillPaint)
        canvas.drawPath(outline, strokePaint)
        // Marker-band edges: TL→TR and BR→BL.
        canvas.drawLine(c[0] * w, c[1] * h, c[2] * w, c[3] * h, bandPaint)
        canvas.drawLine(c[4] * w, c[5] * h, c[6] * w, c[7] * h, bandPaint)
    }
}
