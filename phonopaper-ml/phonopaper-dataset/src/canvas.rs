//! A floating-point grayscale canvas with the image operations needed by the
//! generator: projective warping, box down-sampling, blur, noise and simple
//! drawing primitives.
//!
//! Pixel values are `f32` in `[0, 255]`; quantisation to `u8` happens once at
//! the very end (see [`Canvas::to_image`]).  Only `+ - * /` and `sqrt` are
//! used, so results are bit-identical across platforms.

use image::{GrayImage, Luma};

use crate::geometry::{Homography, Point, Quad};
use crate::num::{f32_of, px};
use crate::rng::Rng;

/// A grayscale raster with `f32` samples.
#[derive(Debug, Clone)]
pub struct Canvas {
    width: usize,
    height: usize,
    data: Vec<f32>,
}

impl Canvas {
    /// Create a canvas filled with a constant value.
    #[must_use]
    pub fn filled(width: usize, height: usize, value: f32) -> Self {
        Self {
            width,
            height,
            data: vec![value; width * height],
        }
    }

    /// Wrap a row-major `u8` buffer.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != width * height`.
    #[must_use]
    pub fn from_u8(width: usize, height: usize, data: &[u8]) -> Self {
        assert_eq!(data.len(), width * height, "buffer size mismatch");
        Self {
            width,
            height,
            data: data.iter().map(|&v| f32::from(v)).collect(),
        }
    }

    /// Width in pixels.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Height in pixels.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Read a pixel; out-of-range coordinates return `None`.
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> Option<f32> {
        (x < self.width && y < self.height).then(|| self.data[y * self.width + x])
    }

    /// Write a pixel (no-op when out of range).
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        if x < self.width && y < self.height {
            self.data[y * self.width + x] = value;
        }
    }

    /// Apply `f` to every pixel, passing its coordinates.
    pub fn map_in_place(&mut self, mut f: impl FnMut(usize, usize, f32) -> f32) {
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y * self.width + x;
                self.data[idx] = f(x, y, self.data[idx]);
            }
        }
    }

    /// Bilinearly interpolated sample at continuous coordinates, where pixel
    /// centres sit at integer + 0.5.  Returns `None` outside the canvas.
    #[must_use]
    pub fn sample_bilinear(&self, x: f64, y: f64) -> Option<f32> {
        if !(x >= 0.0 && y >= 0.0) {
            return None;
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "canvas dimensions are far below 2^52"
        )]
        let (w, h) = (self.width as f64, self.height as f64);
        if x >= w || y >= h {
            return None;
        }
        let fx = (x - 0.5).clamp(0.0, w - 1.0);
        let fy = (y - 0.5).clamp(0.0, h - 1.0);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "fx/fy are clamped to [0, dim-1] so floor fits in usize"
        )]
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let (tx, ty) = (f32_of(fx - px(x0)), f32_of(fy - px(y0)));
        let p00 = self.data[y0 * self.width + x0];
        let p10 = self.data[y0 * self.width + x1];
        let p01 = self.data[y1 * self.width + x0];
        let p11 = self.data[y1 * self.width + x1];
        let top = p00 + (p10 - p00) * tx;
        let bottom = p01 + (p11 - p01) * tx;
        Some(top + (bottom - top) * ty)
    }

    /// Down-sample by an integer factor with a box filter (area averaging).
    ///
    /// The last partial row/column is dropped.  A factor of `1` returns a
    /// clone.
    #[must_use]
    pub fn box_downsample(&self, factor: usize) -> Self {
        if factor <= 1 {
            return self.clone();
        }
        let nw = (self.width / factor).max(1);
        let nh = (self.height / factor).max(1);
        let mut out = Self::filled(nw, nh, 0.0);
        #[expect(clippy::cast_precision_loss, reason = "factor² is a small integer")]
        let inv = 1.0 / (factor * factor) as f32;
        for oy in 0..nh {
            for ox in 0..nw {
                let mut acc = 0.0_f32;
                for dy in 0..factor {
                    let sy = (oy * factor + dy).min(self.height - 1);
                    let row = &self.data[sy * self.width..(sy + 1) * self.width];
                    for dx in 0..factor {
                        acc += row[(ox * factor + dx).min(self.width - 1)];
                    }
                }
                out.data[oy * nw + ox] = acc * inv;
            }
        }
        out
    }

    /// Composite `source` onto `self` through the projective transform mapping
    /// source pixel coordinates to destination pixel coordinates.
    ///
    /// Every destination pixel is inverse-mapped and sampled `ss × ss` times
    /// (super-sampling) for anti-aliasing.  Destination pixels whose samples
    /// all fall outside the source are left untouched; partially covered
    /// pixels are blended proportionally.
    pub fn warp_onto(&mut self, source: &Self, src_to_dst: &Homography, ss: usize) {
        let Some(inv) = src_to_dst.inverse() else {
            return;
        };
        let ss = ss.max(1);
        #[expect(clippy::cast_precision_loss, reason = "ss is a tiny integer")]
        let step = 1.0 / ss as f64;
        #[expect(clippy::cast_precision_loss, reason = "ss² is a small integer")]
        let inv_samples = 1.0 / (ss * ss) as f32;
        for y in 0..self.height {
            for x in 0..self.width {
                let mut acc = 0.0_f32;
                let mut covered = 0_usize;
                for sy in 0..ss {
                    for sx in 0..ss {
                        #[expect(
                            clippy::cast_precision_loss,
                            reason = "pixel coordinates are small integers"
                        )]
                        let dst = Point::new(
                            x as f64 + (sx as f64 + 0.5) * step,
                            y as f64 + (sy as f64 + 0.5) * step,
                        );
                        let src = inv.apply(dst);
                        if let Some(v) = source.sample_bilinear(src.x, src.y) {
                            acc += v;
                            covered += 1;
                        }
                    }
                }
                if covered > 0 {
                    let idx = y * self.width + x;
                    let covered = f32_of(px(covered));
                    let coverage = covered * inv_samples;
                    let mean = acc / covered;
                    self.data[idx] = self.data[idx] * (1.0 - coverage) + mean * coverage;
                }
            }
        }
    }

    /// Separable blur with a binomial kernel.
    ///
    /// `radius = 0` is a no-op, `1` uses `[1 2 1] / 4`, `2` uses
    /// `[1 4 6 4 1] / 16`, larger radii repeat the radius-1 pass.
    pub fn blur(&mut self, radius: usize) {
        match radius {
            0 => {}
            1 => self.convolve_separable(&[0.25, 0.5, 0.25]),
            2 => self.convolve_separable(&[0.0625, 0.25, 0.375, 0.25, 0.0625]),
            n => {
                for _ in 0..n {
                    self.convolve_separable(&[0.25, 0.5, 0.25]);
                }
            }
        }
    }

    fn convolve_separable(&mut self, kernel: &[f32]) {
        let r = kernel.len() / 2;
        let mut tmp = vec![0.0_f32; self.data.len()];
        // Horizontal pass.
        for y in 0..self.height {
            for x in 0..self.width {
                let mut acc = 0.0;
                for (k, &w) in kernel.iter().enumerate() {
                    let sx = (x + k).saturating_sub(r).min(self.width - 1);
                    acc += w * self.data[y * self.width + sx];
                }
                tmp[y * self.width + x] = acc;
            }
        }
        // Vertical pass.
        for y in 0..self.height {
            for x in 0..self.width {
                let mut acc = 0.0;
                for (k, &w) in kernel.iter().enumerate() {
                    let sy = (y + k).saturating_sub(r).min(self.height - 1);
                    acc += w * tmp[sy * self.width + x];
                }
                self.data[y * self.width + x] = acc;
            }
        }
    }

    /// Add zero-mean Gaussian noise with the given standard deviation.
    pub fn add_gaussian_noise(&mut self, rng: &mut Rng, sigma: f32) {
        if sigma <= 0.0 {
            return;
        }
        for v in &mut self.data {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "noise sample is bounded to ±6"
            )]
            let n = rng.gaussian() as f32;
            *v += n * sigma;
        }
    }

    /// Replace a fraction `p` of pixels with pure black or white.
    pub fn add_salt_and_pepper(&mut self, rng: &mut Rng, p: f64) {
        for v in &mut self.data {
            if rng.chance(p) {
                *v = if rng.chance(0.5) { 0.0 } else { 255.0 };
            }
        }
    }

    /// Fill an axis-aligned rectangle (clipped to the canvas).
    pub fn fill_rect(&mut self, x0: i64, y0: i64, w: i64, h: i64, value: f32) {
        let xa = x0.max(0);
        let ya = y0.max(0);
        let xb = (x0 + w).min(i64::try_from(self.width).unwrap_or(i64::MAX));
        let yb = (y0 + h).min(i64::try_from(self.height).unwrap_or(i64::MAX));
        for y in ya..yb {
            // `ya ≥ 0` and `yb ≤ height`, so both conversions always succeed.
            let row = usize::try_from(y).unwrap_or(0) * self.width;
            for x in xa..xb {
                let col = usize::try_from(x).unwrap_or(0);
                self.data[row + col] = value;
            }
        }
    }

    /// Fill a convex quadrilateral (clipped to the canvas) using a half-plane
    /// test per pixel centre.
    pub fn fill_convex_quad(&mut self, q: &Quad, value: f32) {
        let min_x = q
            .iter()
            .map(|p| p.x)
            .fold(f64::INFINITY, f64::min)
            .floor()
            .max(0.0);
        let min_y = q
            .iter()
            .map(|p| p.y)
            .fold(f64::INFINITY, f64::min)
            .floor()
            .max(0.0);
        #[expect(clippy::cast_precision_loss, reason = "canvas dimensions are small")]
        let max_x = q
            .iter()
            .map(|p| p.x)
            .fold(f64::NEG_INFINITY, f64::max)
            .ceil()
            .min(self.width as f64);
        #[expect(clippy::cast_precision_loss, reason = "canvas dimensions are small")]
        let max_y = q
            .iter()
            .map(|p| p.y)
            .fold(f64::NEG_INFINITY, f64::max)
            .ceil()
            .min(self.height as f64);
        if !(min_x < max_x && min_y < max_y) {
            return;
        }
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "bounds are clamped to [0, dim]"
        )]
        let (x0, y0, x1, y1) = (
            min_x as usize,
            min_y as usize,
            max_x as usize,
            max_y as usize,
        );
        let clockwise = crate::geometry::signed_area(q) > 0.0;
        for y in y0..y1 {
            for x in x0..x1 {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "pixel coordinates are small integers"
                )]
                let p = Point::new(x as f64 + 0.5, y as f64 + 0.5);
                if point_in_convex_quad(q, p, clockwise) {
                    self.data[y * self.width + x] = value;
                }
            }
        }
    }

    /// Draw a thick line segment.
    pub fn draw_line(&mut self, a: Point, b: Point, thickness: f64, value: f32) {
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-9 {
            return;
        }
        // Perpendicular offset of half the thickness.
        let nx = -dy / len * thickness * 0.5;
        let ny = dx / len * thickness * 0.5;
        let q: Quad = [
            Point::new(a.x + nx, a.y + ny),
            Point::new(b.x + nx, b.y + ny),
            Point::new(b.x - nx, b.y - ny),
            Point::new(a.x - nx, a.y - ny),
        ];
        self.fill_convex_quad(&q, value);
    }

    /// Multiply every pixel by a radial falloff centred at `(cx, cy)`:
    /// `1 - strength · (d / d_max)²`.
    pub fn vignette(&mut self, cx: f64, cy: f64, strength: f32) {
        #[expect(clippy::cast_precision_loss, reason = "canvas dimensions are small")]
        let (w, h) = (self.width as f64, self.height as f64);
        let d_max_sq = w * w + h * h;
        self.map_in_place(|x, y, v| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "pixel coordinates are small integers"
            )]
            let (px, py) = (x as f64 + 0.5 - cx, y as f64 + 0.5 - cy);
            #[expect(clippy::cast_possible_truncation, reason = "ratio is in [0, 1]")]
            let t = ((px * px + py * py) / d_max_sq) as f32;
            v * (1.0 - strength * t)
        });
    }

    /// Quantise to an 8-bit grayscale image, clamping to `[0, 255]`.
    ///
    /// # Panics
    ///
    /// Panics if the canvas dimensions do not fit in `u32`.
    #[must_use]
    pub fn to_image(&self) -> GrayImage {
        let w = u32::try_from(self.width).expect("width fits in u32");
        let h = u32::try_from(self.height).expect("height fits in u32");
        GrayImage::from_fn(w, h, |x, y| {
            let v = self.data[y as usize * self.width + x as usize];
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "value is clamped to [0, 255] before the cast"
            )]
            let q = v.clamp(0.0, 255.0).round() as u8;
            Luma([q])
        })
    }
}

fn point_in_convex_quad(q: &Quad, p: Point, clockwise: bool) -> bool {
    for i in 0..4 {
        let a = q[i];
        let b = q[(i + 1) % 4];
        let cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
        if (cross < 0.0) == clockwise {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::rect_quad;

    #[test]
    fn bilinear_sample_is_exact_at_pixel_centres() {
        let c = Canvas::from_u8(2, 2, &[0, 100, 200, 50]);
        assert_eq!(c.sample_bilinear(0.5, 0.5), Some(0.0));
        assert_eq!(c.sample_bilinear(1.5, 0.5), Some(100.0));
        assert_eq!(c.sample_bilinear(0.5, 1.5), Some(200.0));
        assert_eq!(c.sample_bilinear(1.0, 0.5), Some(50.0));
        assert_eq!(c.sample_bilinear(-0.1, 0.5), None);
        assert_eq!(c.sample_bilinear(2.0, 0.5), None);
    }

    #[test]
    fn box_downsample_averages() {
        let c = Canvas::from_u8(4, 2, &[0, 100, 200, 200, 100, 200, 200, 200]);
        let d = c.box_downsample(2);
        assert_eq!((d.width(), d.height()), (2, 1));
        assert_eq!(d.get(0, 0), Some(100.0));
        assert_eq!(d.get(1, 0), Some(200.0));
    }

    #[test]
    fn identity_warp_copies_source() {
        let src = Canvas::from_u8(3, 3, &[0, 50, 100, 150, 200, 250, 10, 20, 30]);
        let mut dst = Canvas::filled(3, 3, 255.0);
        dst.warp_onto(&src, &Homography::identity(), 1);
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(dst.get(x, y), src.get(x, y));
            }
        }
    }

    #[test]
    fn warp_leaves_uncovered_pixels_untouched() {
        let src = Canvas::filled(2, 2, 0.0);
        let mut dst = Canvas::filled(6, 6, 255.0);
        let h = Homography::from_quads(
            &rect_quad(0.0, 0.0, 2.0, 2.0),
            &rect_quad(2.0, 2.0, 4.0, 4.0),
        )
        .unwrap();
        dst.warp_onto(&src, &h, 2);
        assert_eq!(dst.get(0, 0), Some(255.0));
        assert_eq!(dst.get(2, 2), Some(0.0));
        assert_eq!(dst.get(3, 3), Some(0.0));
        assert_eq!(dst.get(5, 5), Some(255.0));
    }

    #[test]
    fn fill_rect_clips() {
        let mut c = Canvas::filled(4, 4, 0.0);
        c.fill_rect(-2, -2, 4, 4, 255.0);
        assert_eq!(c.get(0, 0), Some(255.0));
        assert_eq!(c.get(1, 1), Some(255.0));
        assert_eq!(c.get(2, 2), Some(0.0));
    }

    #[test]
    fn fill_convex_quad_fills_interior() {
        let mut c = Canvas::filled(4, 4, 0.0);
        c.fill_convex_quad(&rect_quad(1.0, 1.0, 3.0, 3.0), 200.0);
        assert_eq!(c.get(1, 1), Some(200.0));
        assert_eq!(c.get(2, 2), Some(200.0));
        assert_eq!(c.get(0, 0), Some(0.0));
        assert_eq!(c.get(3, 3), Some(0.0));
    }

    #[test]
    fn blur_preserves_constant_image() {
        let mut c = Canvas::filled(5, 5, 42.0);
        c.blur(2);
        for y in 0..5 {
            for x in 0..5 {
                assert!((c.get(x, y).unwrap() - 42.0).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn to_image_clamps() {
        let mut c = Canvas::filled(2, 1, 0.0);
        c.set(0, 0, -20.0);
        c.set(1, 0, 300.0);
        let img = c.to_image();
        assert_eq!(img.get_pixel(0, 0).0, [0]);
        assert_eq!(img.get_pixel(1, 0).0, [255]);
    }
}
