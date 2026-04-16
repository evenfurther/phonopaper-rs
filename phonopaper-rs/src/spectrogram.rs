//! The [`Spectrogram`] type: a time × frequency amplitude matrix.
//!
//! A `Spectrogram` stores the audio content of a `PhonoPaper` image as a 2D
//! grid of `f32` amplitudes with shape `[time_columns][TOTAL_BINS]`.
//!
//! - Amplitude `0.0` = silence (white pixel)
//! - Amplitude `1.0` = maximum loudness (black pixel)
//!
//! The backing storage `S` is generic and must implement `AsRef<[f32]>` for
//! read access and `AsMut<[f32]>` for write access.  The default `S = Vec<f32>`
//! is used via the [`SpectrogramVec`] type alias.  For zero-allocation use
//! cases, `Spectrogram<&mut [f32]>` accepts a caller-supplied buffer.

use crate::format::TOTAL_BINS;

/// A 2D amplitude matrix representing audio content in the `PhonoPaper` format.
///
/// The outer dimension is time (one entry per image column), and the inner
/// dimension is frequency (one entry per frequency bin, index 0 = highest
/// frequency, index `TOTAL_BINS-1` = lowest frequency).
///
/// `S` is the backing storage — any type implementing `AsRef<[f32]>`.  Mutable
/// operations additionally require `AsMut<[f32]>`.
///
/// ## Common storage types
///
/// | Constructor | Storage type |
/// |-------------|--------------|
/// | [`Spectrogram::new`] | `Vec<f32>` (heap-allocated, owned) |
/// | [`Spectrogram::from_storage`] with `&[f32]` | borrowed slice — [`SpectrogramBuf`] |
/// | [`Spectrogram::from_storage`] with `&mut [f32]` | mutable borrowed slice — [`SpectrogramBufMut`] |
#[derive(Debug, Clone)]
pub struct Spectrogram<S> {
    /// Number of time columns.
    num_columns: usize,
    /// Amplitude data stored in row-major order: `data[col * TOTAL_BINS + bin]`.
    data: S,
}

// ─── Vec-backed convenience type alias (requires allocator) ──────────────────

/// A heap-backed [`Spectrogram`] using `Vec<f32>` as storage.
pub type SpectrogramVec = Spectrogram<Vec<f32>>;

/// A read-only borrowed [`Spectrogram`] backed by a `&[f32]` slice.
///
/// Construct with [`Spectrogram::from_storage`]:
/// ```
/// # use phonopaper_rs::spectrogram::{SpectrogramBuf, Spectrogram};
/// # use phonopaper_rs::format::TOTAL_BINS;
/// let data = vec![0.0_f32; 4 * TOTAL_BINS];
/// let spec: SpectrogramBuf<'_> = Spectrogram::from_storage(4, data.as_slice()).unwrap();
/// ```
pub type SpectrogramBuf<'a> = Spectrogram<&'a [f32]>;

/// A mutable borrowed [`Spectrogram`] backed by a `&mut [f32]` slice.
///
/// Construct with [`Spectrogram::from_storage`]:
/// ```
/// # use phonopaper_rs::spectrogram::{SpectrogramBufMut, Spectrogram};
/// # use phonopaper_rs::format::TOTAL_BINS;
/// let mut data = vec![0.0_f32; 4 * TOTAL_BINS];
/// let mut spec: SpectrogramBufMut<'_> = Spectrogram::from_storage(4, data.as_mut_slice()).unwrap();
/// ```
pub type SpectrogramBufMut<'a> = Spectrogram<&'a mut [f32]>;

// ─── Constructors ─────────────────────────────────────────────────────────────

impl<S: AsRef<[f32]>> Spectrogram<S> {
    /// Wrap an existing storage buffer as a read-only spectrogram view.
    ///
    /// `data.as_ref()` must have length `num_columns * TOTAL_BINS`.
    ///
    /// Returns `None` if the length does not match.
    #[must_use]
    pub fn from_storage(num_columns: usize, data: S) -> Option<Self> {
        if data.as_ref().len() != num_columns * TOTAL_BINS {
            return None;
        }
        Some(Self { num_columns, data })
    }
}

impl Spectrogram<Vec<f32>> {
    /// Create a new, silent spectrogram with the given number of time columns.
    ///
    /// All amplitudes are initialised to `0.0` (silence).
    #[must_use]
    pub fn new(num_columns: usize) -> Self {
        Self {
            num_columns,
            data: vec![0.0f32; num_columns * TOTAL_BINS],
        }
    }
}

// ─── Read-only methods ────────────────────────────────────────────────────────

impl<S: AsRef<[f32]>> Spectrogram<S> {
    /// The number of time columns (image width of the data area).
    #[must_use]
    pub fn num_columns(&self) -> usize {
        self.num_columns
    }

    /// Get the amplitude at a specific time column and frequency bin.
    ///
    /// Returns `0.0` if either index is out of range (silent/white pixel is
    /// the safe default for missing data).  Use [`Self::column`] to read an
    /// entire column at once — it returns `None` for an out-of-range column
    /// rather than silently returning zeros.
    #[must_use]
    pub fn get(&self, col: usize, bin: usize) -> f32 {
        if col < self.num_columns && bin < TOTAL_BINS {
            self.data.as_ref()[col * TOTAL_BINS + bin]
        } else {
            0.0
        }
    }

    /// Return a slice of amplitudes for a given time column, or `None` if
    /// `col` is out of range.
    ///
    /// The slice has length [`TOTAL_BINS`], with index 0 at the highest
    /// frequency.
    #[must_use]
    pub fn column(&self, col: usize) -> Option<&[f32]> {
        if col < self.num_columns {
            let start = col * TOTAL_BINS;
            Some(&self.data.as_ref()[start..start + TOTAL_BINS])
        } else {
            None
        }
    }

    /// Return a slice of amplitudes for a given time column.
    ///
    /// The slice has length [`TOTAL_BINS`], with index 0 at the highest
    /// frequency.
    ///
    /// Unlike `_unchecked` functions in the standard library, this function is
    /// **entirely safe** — it performs a standard slice index and panics with a
    /// clear message on out-of-bounds access.  The name `_or_panic` reflects
    /// that it is an infallible alternative to [`Self::column`] that panics
    /// instead of returning `None`.
    ///
    /// # Panics
    ///
    /// Panics if `col >= self.num_columns()`.
    #[must_use]
    pub fn column_or_panic(&self, col: usize) -> &[f32] {
        let start = col * TOTAL_BINS;
        &self.data.as_ref()[start..start + TOTAL_BINS]
    }
}

// ─── Write methods ────────────────────────────────────────────────────────────

impl<S: AsRef<[f32]> + AsMut<[f32]>> Spectrogram<S> {
    /// Set the amplitude at a specific time column and frequency bin.
    ///
    /// The `amplitude` value is **silently clamped** to `[0.0, 1.0]` before
    /// storage.  Reading the value back via [`Self::get`] will therefore return
    /// the clamped result if the input was outside that range — e.g. `set(c, b,
    /// 1.5)` followed by `get(c, b)` returns `1.0`.
    ///
    /// Does nothing if the indices are out of range.
    pub fn set(&mut self, col: usize, bin: usize, amplitude: f32) {
        if col < self.num_columns && bin < TOTAL_BINS {
            self.data.as_mut()[col * TOTAL_BINS + bin] = amplitude.clamp(0.0, 1.0);
        }
    }

    /// Return a mutable slice of amplitudes for a given time column.
    ///
    /// # Panics
    /// Panics if `col >= self.num_columns()`.
    pub fn column_mut(&mut self, col: usize) -> &mut [f32] {
        let start = col * TOTAL_BINS;
        &mut self.data.as_mut()[start..start + TOTAL_BINS]
    }
}
