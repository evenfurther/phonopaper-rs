//! Planar geometry: points, quadrilaterals and projective transforms.

use serde::{Deserialize, Serialize};

/// A 2-D point in pixel coordinates (`x` to the right, `y` downwards).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Point {
    /// Horizontal coordinate.
    pub x: f64,
    /// Vertical coordinate.
    pub y: f64,
}

impl Point {
    /// Construct a point.
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// Four corners of a quadrilateral, in the order
/// top-left, top-right, bottom-right, bottom-left **of the pattern**.
///
/// The order is defined in the pattern's own frame, not the image frame:
/// after a 90° rotation the "top-left" corner may well be the lowest point
/// in the image.  This is deliberate — it tells the consumer which side of
/// the detected pattern carries the top marker band.
pub type Quad = [Point; 4];

/// Axis-aligned rectangle as a [`Quad`].
#[must_use]
pub fn rect_quad(x0: f64, y0: f64, x1: f64, y1: f64) -> Quad {
    [
        Point::new(x0, y0),
        Point::new(x1, y0),
        Point::new(x1, y1),
        Point::new(x0, y1),
    ]
}

/// A 3×3 projective transform (homography) in row-major order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Homography {
    m: [f64; 9],
}

impl Homography {
    /// Identity transform.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Compute the homography mapping each `src[i]` onto `dst[i]`.
    ///
    /// Returns `None` if the points are degenerate (three collinear points
    /// or a self-intersecting quadrilateral make the system singular).
    #[must_use]
    pub fn from_quads(src: &Quad, dst: &Quad) -> Option<Self> {
        // Direct linear transform with h33 fixed to 1: 8 unknowns, 8 equations.
        let mut system = [[0.0_f64; 9]; 8];
        for i in 0..4 {
            let (sx, sy) = (src[i].x, src[i].y);
            let (dx, dy) = (dst[i].x, dst[i].y);
            system[2 * i] = [sx, sy, 1.0, 0.0, 0.0, 0.0, -dx * sx, -dx * sy, dx];
            system[2 * i + 1] = [0.0, 0.0, 0.0, sx, sy, 1.0, -dy * sx, -dy * sy, dy];
        }
        let h = solve_8x8(&mut system)?;
        Some(Self {
            m: [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0],
        })
    }

    /// Apply the transform to a point.
    ///
    /// Points mapped to infinity (`w ≈ 0`) are returned as `NaN` coordinates.
    #[must_use]
    pub fn apply(&self, p: Point) -> Point {
        let m = &self.m;
        let w = m[6] * p.x + m[7] * p.y + m[8];
        Point::new(
            (m[0] * p.x + m[1] * p.y + m[2]) / w,
            (m[3] * p.x + m[4] * p.y + m[5]) / w,
        )
    }

    /// Apply the transform to every corner of a quadrilateral.
    #[must_use]
    pub fn apply_quad(&self, q: &Quad) -> Quad {
        [
            self.apply(q[0]),
            self.apply(q[1]),
            self.apply(q[2]),
            self.apply(q[3]),
        ]
    }

    /// Matrix inverse (adjugate / determinant).  Returns `None` if singular.
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let m = &self.m;
        let c00 = m[4] * m[8] - m[5] * m[7];
        let c01 = m[5] * m[6] - m[3] * m[8];
        let c02 = m[3] * m[7] - m[4] * m[6];
        let det = m[0] * c00 + m[1] * c01 + m[2] * c02;
        if det.abs() < 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(Self {
            m: [
                c00 * inv_det,
                (m[2] * m[7] - m[1] * m[8]) * inv_det,
                (m[1] * m[5] - m[2] * m[4]) * inv_det,
                c01 * inv_det,
                (m[0] * m[8] - m[2] * m[6]) * inv_det,
                (m[2] * m[3] - m[0] * m[5]) * inv_det,
                c02 * inv_det,
                (m[1] * m[6] - m[0] * m[7]) * inv_det,
                (m[0] * m[4] - m[1] * m[3]) * inv_det,
            ],
        })
    }
}

/// Solve the 8×8 augmented system `a·h = b` (column 8 holds `b`) by Gaussian
/// elimination with partial pivoting.
fn solve_8x8(a: &mut [[f64; 9]; 8]) -> Option<[f64; 8]> {
    for col in 0..8 {
        // Partial pivoting.
        let pivot = (col..8).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        for row in (col + 1)..8 {
            let factor = a[row][col] / a[col][col];
            if factor != 0.0 {
                for k in col..9 {
                    a[row][k] -= factor * a[col][k];
                }
            }
        }
    }
    let mut h = [0.0_f64; 8];
    for col in (0..8).rev() {
        let mut acc = a[col][8];
        for k in (col + 1)..8 {
            acc -= a[col][k] * h[k];
        }
        h[col] = acc / a[col][col];
    }
    Some(h)
}

/// Signed area of a quadrilateral (shoelace formula).  Positive when the
/// corners are listed clockwise in image coordinates (`y` down).
#[must_use]
pub fn signed_area(q: &Quad) -> f64 {
    let mut acc = 0.0;
    for i in 0..4 {
        let a = q[i];
        let b = q[(i + 1) % 4];
        acc += a.x * b.y - b.x * a.y;
    }
    acc * 0.5
}

/// `true` when the quadrilateral is convex and its corners are listed in
/// clockwise order (image coordinates).
#[must_use]
pub fn is_convex_clockwise(q: &Quad) -> bool {
    for i in 0..4 {
        let a = q[i];
        let b = q[(i + 1) % 4];
        let c = q[(i + 2) % 4];
        let cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
        if cross <= 0.0 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Point, b: Point) -> bool {
        (a.x - b.x).abs() < 1e-9 && (a.y - b.y).abs() < 1e-9
    }

    #[test]
    fn identity_maps_points_to_themselves() {
        let h = Homography::identity();
        let p = Point::new(3.5, -2.0);
        assert!(close(h.apply(p), p));
    }

    #[test]
    fn from_quads_maps_corners_exactly() {
        let src = rect_quad(0.0, 0.0, 100.0, 50.0);
        let dst = [
            Point::new(10.0, 12.0),
            Point::new(95.0, 5.0),
            Point::new(110.0, 70.0),
            Point::new(2.0, 60.0),
        ];
        let h = Homography::from_quads(&src, &dst).unwrap();
        for i in 0..4 {
            assert!(close(h.apply(src[i]), dst[i]), "corner {i}");
        }
    }

    #[test]
    fn inverse_round_trips() {
        let src = rect_quad(0.0, 0.0, 30.0, 20.0);
        let dst = [
            Point::new(5.0, 7.0),
            Point::new(40.0, 3.0),
            Point::new(45.0, 33.0),
            Point::new(1.0, 28.0),
        ];
        let h = Homography::from_quads(&src, &dst).unwrap();
        let inv = h.inverse().unwrap();
        let p = Point::new(12.3, 4.5);
        assert!(close(inv.apply(h.apply(p)), p));
    }

    #[test]
    fn degenerate_quad_is_rejected() {
        // Three collinear source points make the linear system singular.
        let src = [
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 2.0),
            Point::new(0.0, 5.0),
        ];
        let dst = rect_quad(0.0, 0.0, 10.0, 10.0);
        assert!(Homography::from_quads(&src, &dst).is_none());
        // Four identical points as well.
        let same = [Point::new(1.0, 1.0); 4];
        assert!(Homography::from_quads(&same, &dst).is_none());
    }

    #[test]
    fn rect_quad_is_clockwise_and_convex() {
        let q = rect_quad(0.0, 0.0, 4.0, 2.0);
        assert!(is_convex_clockwise(&q));
        assert!((signed_area(&q) - 8.0).abs() < 1e-12);
        let mut reversed = q;
        reversed.reverse();
        assert!(!is_convex_clockwise(&reversed));
    }
}
