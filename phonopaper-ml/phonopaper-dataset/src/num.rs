//! Narrow numeric conversion helpers.
//!
//! Every lossy cast used by the generator goes through one of these functions
//! so the (single) safety argument lives in one place instead of being
//! repeated at each call site.

/// Convert a pixel count or index to `f64`.
///
/// Image dimensions are far below 2⁵², so the conversion is exact.
#[inline]
#[must_use]
pub fn px(n: usize) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "pixel counts are far below 2^52 and convert exactly"
    )]
    let v = n as f64;
    v
}

/// Convert a bounded `f64` (luma, amplitude, sigma…) to `f32`.
#[inline]
#[must_use]
pub fn f32_of(v: f64) -> f32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "callers pass bounded values for which f32 precision is sufficient"
    )]
    let r = v as f32;
    r
}

/// Round a non-negative `f64` that is known to be below `max` to an index.
///
/// The value is clamped to `[0, max]` before conversion so the result is
/// always in range.
#[inline]
#[must_use]
pub fn index_of(v: f64, max: usize) -> usize {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the value is clamped to [0, max] before the cast"
    )]
    let r = v.round().clamp(0.0, px(max)) as usize;
    r
}
