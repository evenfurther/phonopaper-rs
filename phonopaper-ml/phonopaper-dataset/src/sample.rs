//! The per-image generation pipeline.
//!
//! Every sample is a pure function of `(dataset seed, index)`: the image's RNG
//! stream is derived from those two values only, so the dataset can be
//! generated in any order or in parallel with identical results.

use image::GrayImage;
use serde::{Deserialize, Serialize};

use crate::background::random_background;
use crate::canvas::Canvas;
use crate::geometry::{Homography, Point, Quad, is_convex_clockwise};
use crate::pattern::{Sheet, decoy_sheet, phonopaper_sheet};
use crate::rng::Rng;

/// Parameters of a dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeneratorConfig {
    /// Number of images to generate.
    pub count: u64,
    /// Side of the square output images, in pixels.
    pub size: u32,
    /// Master seed; images are a pure function of `(seed, index)`.
    pub seed: u64,
    /// Probability that an image contains a `PhonoPaper` pattern.
    pub positive_ratio: f64,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            count: 4000,
            size: 128,
            seed: 0x5EED_0001,
            positive_ratio: 0.6,
        }
    }
}

/// A generated image and its ground truth.
#[derive(Debug, Clone)]
pub struct Sample {
    /// The grayscale image.
    pub image: GrayImage,
    /// Corners of the `PhonoPaper` ink box (TL, TR, BR, BL in pattern
    /// orientation), in output pixel coordinates.  `None` when the image
    /// contains no pattern.
    pub corners: Option<Quad>,
}

/// Fraction of the image side by which a corner may lie outside the frame
/// while the pattern is still considered present.
const OUTSIDE_TOLERANCE: f64 = 0.05;

/// Super-sampling factor for the warp.
const SUPER_SAMPLES: usize = 2;

/// Generate the sample with the given index.
#[must_use]
pub fn generate_sample(cfg: &GeneratorConfig, index: u64) -> Sample {
    let mut rng = Rng::for_item(cfg.seed, index);
    let size = cfg.size as usize;
    let mut canvas = random_background(&mut rng, size);

    let mut corners = None;
    if rng.chance(cfg.positive_ratio) {
        let sheet = phonopaper_sheet(&mut rng);
        corners = place_sheet(&mut rng, &mut canvas, &sheet);
        // Placement fails only for pathological random draws; in that case the
        // image is (deterministically) a negative sample.
    } else if rng.chance(0.5) {
        let sheet = decoy_sheet(&mut rng);
        let _ = place_sheet(&mut rng, &mut canvas, &sheet);
    }

    if corners.is_some() && rng.chance(0.12) {
        add_occluder(&mut rng, &mut canvas);
    }
    photometric_degradation(&mut rng, &mut canvas);

    Sample {
        image: canvas.to_image(),
        corners,
    }
}

/// Recolour a sheet: paper becomes an off-white tone, ink a dark grey.
fn colour_grade(rng: &mut Rng, sheet: &Canvas) -> Canvas {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "luma values are in [0, 255]"
    )]
    let paper = rng.range_f64(160.0, 255.0) as f32;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "luma values are in [0, 255]"
    )]
    let ink = rng.range_f64(0.0, 90.0) as f32;
    let mut out = sheet.clone();
    out.map_in_place(|_, _, v| ink + (paper - ink) * (v / 255.0));
    // Slight print / focus blur before down-scaling.
    if rng.chance(0.3) {
        out.blur(1);
    }
    out
}

/// Draw a random unit direction with tilt limited to about ±31° (|tan| ≤ 0.6),
/// then optionally add a quarter or half turn.  Uses only `sqrt`.
fn random_rotation(rng: &mut Rng) -> (f64, f64) {
    let t = rng.range_f64(-0.6, 0.6);
    let n = (1.0 + t * t).sqrt();
    let (mut c, mut s) = (1.0 / n, t / n);
    let quarter_turns = if rng.chance(0.15) {
        rng.range_u32(1, 3)
    } else {
        0
    };
    for _ in 0..quarter_turns {
        // Rotate by +90°: (c, s) → (-s, c).
        (c, s) = (-s, c);
    }
    (c, s)
}

/// Compute the destination quadrilateral of a sheet's ink box.
///
/// Returns `None` if no placement fitting the tolerance could be found.
fn random_target_quad(rng: &mut Rng, ink_box: &Quad, size: usize) -> Option<Quad> {
    #[expect(clippy::cast_precision_loss, reason = "image size is a small integer")]
    let sf = size as f64;
    let ink_w = ink_box[1].x - ink_box[0].x;
    let ink_h = ink_box[3].y - ink_box[0].y;
    let cx = f64::midpoint(ink_box[0].x, ink_box[2].x);
    let cy = f64::midpoint(ink_box[0].y, ink_box[2].y);

    let (c, s) = random_rotation(rng);
    let shear = rng.range_f64(-0.25, 0.25);
    let mut longest = sf * rng.range_f64(0.25, 0.95);
    let lo = -OUTSIDE_TOLERANCE * sf;
    let hi = (1.0 + OUTSIDE_TOLERANCE) * sf;

    for _attempt in 0..6 {
        let scale = longest / ink_w.max(ink_h);
        // Affine part: scale, shear, rotate — about the ink-box centre.
        let mapped: Vec<Point> = ink_box
            .iter()
            .map(|p| {
                let x = (p.x - cx) * scale;
                let y = (p.y - cy) * scale;
                let xs = x + shear * y;
                Point::new(c * xs - s * y, s * xs + c * y)
            })
            .collect();
        let min_x = mapped.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
        let max_x = mapped.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
        let min_y = mapped.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let max_y = mapped.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
        let bw = max_x - min_x;
        let bh = max_y - min_y;
        if bw <= hi - lo && bh <= hi - lo {
            let tx = rng.range_f64(lo - min_x, hi - max_x);
            let ty = rng.range_f64(lo - min_y, hi - max_y);
            let base: Quad = [
                Point::new(mapped[0].x + tx, mapped[0].y + ty),
                Point::new(mapped[1].x + tx, mapped[1].y + ty),
                Point::new(mapped[2].x + tx, mapped[2].y + ty),
                Point::new(mapped[3].x + tx, mapped[3].y + ty),
            ];
            // Perspective: jitter each corner independently.
            let jitter = longest * rng.range_f64(0.0, 0.08);
            let mut warped = base;
            for p in &mut warped {
                p.x += rng.range_f64(-jitter, jitter);
                p.y += rng.range_f64(-jitter, jitter);
            }
            let in_bounds = warped
                .iter()
                .all(|p| p.x >= lo && p.x <= hi && p.y >= lo && p.y <= hi);
            return Some(if in_bounds && is_convex_clockwise(&warped) {
                warped
            } else {
                base
            });
        }
        longest *= 0.8;
    }
    None
}

/// Warp a sheet onto the canvas; returns the ink-box corners in canvas
/// coordinates on success.
fn place_sheet(rng: &mut Rng, canvas: &mut Canvas, sheet: &Sheet) -> Option<Quad> {
    let target = random_target_quad(rng, &sheet.ink_box, canvas.width())?;
    let graded = colour_grade(rng, &sheet.canvas);

    // Pre-shrink the sheet with a box filter when it is heavily minified so
    // thin stripes are averaged rather than aliased (as a camera would do).
    let ink_w = sheet.ink_box[1].x - sheet.ink_box[0].x;
    let target_w = {
        let dx = target[1].x - target[0].x;
        let dy = target[1].y - target[0].y;
        (dx * dx + dy * dy).sqrt()
    };
    let minification = ink_w / target_w.max(1e-6);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "minification is positive and small"
    )]
    let factor = minification.floor().clamp(1.0, 64.0) as usize;
    let source = graded.box_downsample(factor);
    #[expect(clippy::cast_precision_loss, reason = "factor is a small integer")]
    let inv_factor = 1.0 / factor as f64;
    let src_box: Quad = sheet
        .ink_box
        .map(|p| Point::new(p.x * inv_factor, p.y * inv_factor));

    let h = Homography::from_quads(&src_box, &target)?;
    canvas.warp_onto(&source, &h, SUPER_SAMPLES);
    Some(target)
}

/// Paint a small random quadrilateral over the image (finger, cable…).
fn add_occluder(rng: &mut Rng, canvas: &mut Canvas) {
    #[expect(clippy::cast_precision_loss, reason = "image size is a small integer")]
    let sf = canvas.width() as f64;
    let cx = rng.range_f64(0.0, sf);
    let cy = rng.range_f64(0.0, sf);
    let w = sf * rng.range_f64(0.05, 0.25);
    let h = sf * rng.range_f64(0.05, 0.25);
    let quad: Quad = [
        Point::new(cx - w * 0.5, cy - h * 0.5),
        Point::new(cx + w * 0.5, cy - h * 0.5 + rng.range_f64(-h, h) * 0.3),
        Point::new(cx + w * 0.5, cy + h * 0.5),
        Point::new(cx - w * 0.5, cy + h * 0.5 + rng.range_f64(-h, h) * 0.3),
    ];
    #[expect(
        clippy::cast_possible_truncation,
        reason = "luma values are in [0, 255]"
    )]
    let v = rng.range_f64(0.0, 255.0) as f32;
    if is_convex_clockwise(&quad) {
        canvas.fill_convex_quad(&quad, v);
    }
}

/// Camera-like degradations applied to the whole frame.
fn photometric_degradation(rng: &mut Rng, canvas: &mut Canvas) {
    #[expect(clippy::cast_possible_truncation, reason = "values are bounded")]
    let (contrast, brightness) = (
        rng.range_f64(0.6, 1.3) as f32,
        rng.range_f64(-40.0, 40.0) as f32,
    );
    canvas.map_in_place(|_, _, v| (v - 128.0) * contrast + 128.0 + brightness);

    if rng.chance(0.5) {
        #[expect(clippy::cast_precision_loss, reason = "image size is a small integer")]
        let sf = canvas.width() as f64;
        #[expect(clippy::cast_possible_truncation, reason = "strength is in [0, 0.6]")]
        let strength = rng.range_f64(0.1, 0.6) as f32;
        canvas.vignette(rng.range_f64(0.0, sf), rng.range_f64(0.0, sf), strength);
    }
    if rng.chance(0.4) {
        canvas.blur(rng.range_usize(1, 2));
    }
    if rng.chance(0.7) {
        #[expect(clippy::cast_possible_truncation, reason = "sigma is bounded")]
        let sigma = rng.range_f64(1.0, 20.0) as f32;
        canvas.add_gaussian_noise(rng, sigma);
    }
    if rng.chance(0.1) {
        canvas.add_salt_and_pepper(rng, 0.005);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> GeneratorConfig {
        GeneratorConfig {
            count: 10,
            size: 64,
            seed: 1234,
            positive_ratio: 0.6,
        }
    }

    #[test]
    fn samples_are_deterministic() {
        let cfg = cfg();
        for idx in 0..6 {
            let a = generate_sample(&cfg, idx);
            let b = generate_sample(&cfg, idx);
            assert_eq!(a.image.as_raw(), b.image.as_raw(), "index {idx}");
            assert_eq!(a.corners, b.corners, "index {idx}");
        }
    }

    #[test]
    fn samples_have_requested_size() {
        let s = generate_sample(&cfg(), 0);
        assert_eq!(s.image.dimensions(), (64, 64));
    }

    #[test]
    fn positive_corners_are_within_tolerance_and_convex() {
        let cfg = cfg();
        let mut positives = 0;
        for idx in 0..60 {
            if let Some(q) = generate_sample(&cfg, idx).corners {
                positives += 1;
                let lo = -OUTSIDE_TOLERANCE * 64.0 - 1e-9;
                let hi = (1.0 + OUTSIDE_TOLERANCE) * 64.0 + 1e-9;
                for p in &q {
                    assert!(p.x >= lo && p.x <= hi && p.y >= lo && p.y <= hi, "{q:?}");
                }
                assert!(is_convex_clockwise(&q), "{q:?}");
            }
        }
        assert!(positives > 20, "only {positives} positives out of 60");
    }

    #[test]
    fn different_seeds_give_different_images() {
        let a = generate_sample(&cfg(), 0);
        let b = generate_sample(
            &GeneratorConfig {
                seed: 4321,
                ..cfg()
            },
            0,
        );
        assert_ne!(a.image.as_raw(), b.image.as_raw());
    }
}
