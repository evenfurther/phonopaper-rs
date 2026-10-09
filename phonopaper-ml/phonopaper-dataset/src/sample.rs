//! The per-image generation pipeline.
//!
//! Every sample is a pure function of `(dataset seed, index)`: the image's RNG
//! stream is derived from those two values only, so the dataset can be
//! generated in any order or in parallel with identical results.

use image::imageops::FilterType;
use image::{DynamicImage, GrayImage};
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
    /// Integer scale of the synthetic camera frame relative to the stored image.
    ///
    /// The complete scene is rendered at `size * source_scale`, then resized
    /// to `size` with the production `Triangle` filter.
    pub source_scale: u32,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            count: 4000,
            size: 128,
            seed: 0x5EED_0001,
            positive_ratio: 0.6,
            source_scale: 3,
        }
    }
}

/// Positive-sample curriculum category selected before scene generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PositiveKind {
    /// Simple, high-contrast, nearly axis-aligned localization example.
    CleanLocalization,
    /// Pattern deliberately fills the frame with a narrow or clipped margin.
    FrameFilling,
    /// Existing varied, cluttered and photo-like generation path.
    VariedPhoto,
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
    /// Curriculum category for a positive sample.
    pub positive_kind: Option<PositiveKind>,
}

/// Share of positive samples reserved for clean localization.
pub const CLEAN_LOCALIZATION_SHARE: f64 = 0.20;
/// Share of positive samples reserved for narrow-margin/frame-filling scenes.
pub const FRAME_FILLING_SHARE: f64 = 0.20;

/// Fraction of the image side by which a corner may lie outside the frame
/// while the pattern is still considered present.
const OUTSIDE_TOLERANCE: f64 = 0.05;

/// Super-sampling factor for the warp.
const SUPER_SAMPLES: usize = 2;

/// Generate the sample with the given index.
///
/// # Panics
///
/// Panics if `source_scale` is less than two or the scaled dimensions overflow.
#[must_use]
pub fn generate_sample(cfg: &GeneratorConfig, index: u64) -> Sample {
    assert!(
        cfg.source_scale > 1,
        "source_scale must be greater than one"
    );
    let mut rng = Rng::for_item(cfg.seed, index);
    let source_side = cfg
        .size
        .checked_mul(cfg.source_scale)
        .expect("source image dimensions fit u32");
    let canvas_size = usize::try_from(source_side).expect("image size fits usize");
    let is_positive = rng.chance(cfg.positive_ratio);
    let positive_kind = is_positive.then(|| select_positive_kind(&mut rng));
    let mut canvas = if positive_kind == Some(PositiveKind::CleanLocalization) {
        Canvas::filled(canvas_size, canvas_size, 225.0)
    } else {
        random_background(&mut rng, canvas_size)
    };

    let mut corners = None;
    if let Some(kind) = positive_kind {
        let sheet = phonopaper_sheet(&mut rng);
        corners = place_sheet(&mut rng, &mut canvas, &sheet, kind);
        // Placement fails only for pathological random draws; in that case the
        // image is (deterministically) a negative sample.
    } else if rng.chance(0.5) {
        let sheet = decoy_sheet(&mut rng);
        let _ = place_sheet(&mut rng, &mut canvas, &sheet, PositiveKind::VariedPhoto);
    }

    if positive_kind == Some(PositiveKind::VariedPhoto) && corners.is_some() && rng.chance(0.12) {
        add_occluder(&mut rng, &mut canvas);
    }
    if positive_kind != Some(PositiveKind::CleanLocalization) {
        photometric_degradation(&mut rng, &mut canvas);
    }

    let label_scale = 1.0 / f64::from(cfg.source_scale);
    let corners =
        corners.map(|quad| quad.map(|p| Point::new(p.x * label_scale, p.y * label_scale)));
    let source = DynamicImage::ImageLuma8(canvas.to_image());
    let image = source
        .resize_exact(cfg.size, cfg.size, FilterType::Triangle)
        .into_luma8();
    Sample {
        image,
        corners,
        positive_kind: corners.and(positive_kind),
    }
}

fn select_positive_kind(rng: &mut Rng) -> PositiveKind {
    let draw = rng.next_f64();
    if draw < CLEAN_LOCALIZATION_SHARE {
        PositiveKind::CleanLocalization
    } else if draw < CLEAN_LOCALIZATION_SHARE + FRAME_FILLING_SHARE {
        PositiveKind::FrameFilling
    } else {
        PositiveKind::VariedPhoto
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
fn random_target_quad(
    rng: &mut Rng,
    ink_box: &Quad,
    size: usize,
    kind: PositiveKind,
) -> Option<Quad> {
    #[expect(clippy::cast_precision_loss, reason = "image size is a small integer")]
    let sf = size as f64;
    let ink_w = ink_box[1].x - ink_box[0].x;
    let ink_h = ink_box[3].y - ink_box[0].y;
    let cx = f64::midpoint(ink_box[0].x, ink_box[2].x);
    let cy = f64::midpoint(ink_box[0].y, ink_box[2].y);

    let (c, s, shear, mut longest) = match kind {
        PositiveKind::CleanLocalization => {
            let t = rng.range_f64(-0.08, 0.08);
            let n = (1.0 + t * t).sqrt();
            (
                1.0 / n,
                t / n,
                rng.range_f64(-0.03, 0.03),
                sf * rng.range_f64(0.55, 0.75),
            )
        }
        PositiveKind::FrameFilling => {
            let (c, s) = random_rotation(rng);
            (
                c,
                s,
                rng.range_f64(-0.15, 0.15),
                sf * rng.range_f64(0.90, 1.05),
            )
        }
        PositiveKind::VariedPhoto => {
            let (c, s) = random_rotation(rng);
            (
                c,
                s,
                rng.range_f64(-0.25, 0.25),
                sf * rng.range_f64(0.25, 0.95),
            )
        }
    };
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
            let max_jitter = match kind {
                PositiveKind::CleanLocalization => 0.01,
                PositiveKind::FrameFilling => 0.04,
                PositiveKind::VariedPhoto => 0.08,
            };
            let jitter = longest * rng.range_f64(0.0, max_jitter);
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
fn place_sheet(
    rng: &mut Rng,
    canvas: &mut Canvas,
    sheet: &Sheet,
    kind: PositiveKind,
) -> Option<Quad> {
    let target = random_target_quad(rng, &sheet.ink_box, canvas.width(), kind)?;
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
            source_scale: 3,
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

    #[test]
    fn positive_curriculum_contains_all_explicit_subsets() {
        let cfg = GeneratorConfig {
            positive_ratio: 1.0,
            ..cfg()
        };
        let mut seen = [false; 3];
        for index in 0..100 {
            match generate_sample(&cfg, index).positive_kind.unwrap() {
                PositiveKind::CleanLocalization => seen[0] = true,
                PositiveKind::FrameFilling => seen[1] = true,
                PositiveKind::VariedPhoto => seen[2] = true,
            }
        }
        assert_eq!(seen, [true; 3]);
    }

    #[test]
    fn frame_filling_labels_have_a_narrow_margin() {
        let cfg = GeneratorConfig {
            positive_ratio: 1.0,
            ..cfg()
        };
        let mut checked = 0;
        for index in 0..200 {
            let sample = generate_sample(&cfg, index);
            if sample.positive_kind != Some(PositiveKind::FrameFilling) {
                continue;
            }
            let quad = sample.corners.unwrap();
            let min_x = quad.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
            let max_x = quad.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
            let min_y = quad.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
            let max_y = quad.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
            assert!((max_x - min_x).max(max_y - min_y) >= f64::from(cfg.size) * 0.72);
            checked += 1;
        }
        assert!(checked > 20, "only checked {checked} frame-filling samples");
    }

    #[test]
    fn labels_scale_from_source_pixels_without_clipping() {
        let ink_box = [
            Point::new(0.0, 0.0),
            Point::new(200.0, 0.0),
            Point::new(200.0, 80.0),
            Point::new(0.0, 80.0),
        ];
        let mut found_outside = false;
        for seed in 0..200 {
            let source = random_target_quad(
                &mut Rng::from_seed(seed),
                &ink_box,
                384,
                PositiveKind::FrameFilling,
            )
            .unwrap();
            let final_quad = source.map(|p| Point::new(p.x / 3.0, p.y / 3.0));
            for (source_point, final_point) in source.iter().zip(final_quad) {
                assert!((final_point.x * 3.0 - source_point.x).abs() < 1e-12);
                assert!((final_point.y * 3.0 - source_point.y).abs() < 1e-12);
                found_outside |= !(0.0..=128.0).contains(&final_point.x)
                    || !(0.0..=128.0).contains(&final_point.y);
            }
        }
        assert!(
            found_outside,
            "expected at least one permitted out-of-frame label"
        );
    }
}
