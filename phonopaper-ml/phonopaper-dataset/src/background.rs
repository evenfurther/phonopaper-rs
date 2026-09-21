//! Synthetic backgrounds: flat, gradients, textures and clutter.

use crate::canvas::Canvas;
use crate::geometry::Point;
use crate::num::{f32_of, px};
use crate::rng::Rng;

/// Produce a random square background of side `size`.
#[must_use]
pub fn random_background(rng: &mut Rng, size: usize) -> Canvas {
    let base = f32_of(rng.range_f64(20.0, 240.0));
    let mut canvas = Canvas::filled(size, size, base);

    match rng.range_u32(0, 4) {
        // Flat.
        0 => {}
        // Linear gradient in a random direction.
        1 => {
            let gx = rng.range_f64(-1.0, 1.0);
            let gy = rng.range_f64(-1.0, 1.0);
            let amp = f32_of(rng.range_f64(20.0, 120.0));
            let inv = 1.0 / px(size);
            canvas.map_in_place(|x, y, v| {
                let t = (px(x) * gx + px(y) * gy) * inv;
                v + amp * f32_of(t)
            });
        }
        // Blurred noise texture (fabric, wood, wall).
        2 => {
            let sigma = f32_of(rng.range_f64(10.0, 60.0));
            canvas.add_gaussian_noise(rng, sigma);
            canvas.blur(rng.range_usize(0, 3));
        }
        // Tiles / checkerboard.
        3 => {
            let tile = rng.range_usize(8, size / 2);
            let alt = f32_of(rng.range_f64(20.0, 240.0));
            canvas.map_in_place(|x, y, v| {
                if ((x / tile) + (y / tile)).is_multiple_of(2) {
                    v
                } else {
                    alt
                }
            });
        }
        // Vertical / horizontal stripes (wallpaper, blinds).
        _ => {
            let period = rng.range_usize(4, size / 3);
            let duty = rng.range_usize(1, period);
            let horizontal = rng.chance(0.5);
            let alt = f32_of(rng.range_f64(0.0, 255.0));
            canvas.map_in_place(|x, y, v| {
                let c = if horizontal { y } else { x };
                if c % period < duty { alt } else { v }
            });
        }
    }

    add_clutter(rng, &mut canvas);
    canvas
}

/// Scatter random rectangles and lines.
fn add_clutter(rng: &mut Rng, canvas: &mut Canvas) {
    let size = canvas.width();
    let quarter = i64::try_from(size / 4).expect("size fits in i64");
    let limit = u32::try_from(size).expect("size fits in u32");
    let rects = rng.range_usize(0, 6);
    for _ in 0..rects {
        let x = i64::from(rng.range_u32(0, limit)) - quarter;
        let y = i64::from(rng.range_u32(0, limit)) - quarter;
        let rect_w = i64::from(rng.range_u32(2, limit / 2));
        let rect_h = i64::from(rng.range_u32(2, limit / 2));
        let v = f32_of(rng.range_f64(0.0, 255.0));
        canvas.fill_rect(x, y, rect_w, rect_h, v);
    }
    let lines = rng.range_usize(0, 5);
    let sf = px(size);
    for _ in 0..lines {
        let a = Point::new(rng.range_f64(0.0, sf), rng.range_f64(0.0, sf));
        let b = Point::new(rng.range_f64(0.0, sf), rng.range_f64(0.0, sf));
        let v = f32_of(rng.range_f64(0.0, 255.0));
        canvas.draw_line(a, b, rng.range_f64(1.0, 6.0), v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn background_has_requested_size_and_is_deterministic() {
        let a = random_background(&mut Rng::from_seed(11), 32);
        let b = random_background(&mut Rng::from_seed(11), 32);
        assert_eq!((a.width(), a.height()), (32, 32));
        assert_eq!(a.to_image().into_raw(), b.to_image().into_raw());
    }

    #[test]
    fn all_background_kinds_render() {
        for seed in 0..40 {
            let c = random_background(&mut Rng::from_seed(seed), 24);
            assert_eq!(c.width(), 24);
        }
    }
}
