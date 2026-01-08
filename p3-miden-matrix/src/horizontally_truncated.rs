use core::marker::PhantomData;
use core::ops::Range;

use crate::Matrix;

/// A matrix wrapper that exposes a contiguous range of columns from an inner matrix.
///
/// This struct:
/// - wraps another matrix,
/// - restricts access to only the columns within the specified `column_range`.
pub struct HorizontallyTruncated<T, Inner> {
    /// The underlying full matrix being wrapped.
    inner: Inner,
    /// The range of columns to expose from the inner matrix.
    column_range: Range<usize>,
    /// Marker for the element type `T`, not used at runtime.
    _phantom: PhantomData<T>,
}

impl<T, Inner: Matrix<T>> HorizontallyTruncated<T, Inner>
where
    T: Send + Sync + Clone,
{
    /// Construct a new horizontally truncated view of a matrix.
    ///
    /// # Arguments
    /// - `inner`: The full inner matrix to be wrapped.
    /// - `truncated_width`: The number of columns to expose from the start (must be ≤ `inner.width()`).
    ///
    /// This is equivalent to `new_with_range(inner, 0..truncated_width)`.
    ///
    /// Returns `None` if `truncated_width` is greater than the width of the inner matrix.
    pub fn new(inner: Inner, truncated_width: usize) -> Option<Self> {
        Self::new_with_range(inner, 0..truncated_width)
    }

    /// Construct a new view exposing a specific column range of a matrix.
    ///
    /// # Arguments
    /// - `inner`: The full inner matrix to be wrapped.
    /// - `column_range`: The range of columns to expose (must satisfy `column_range.end <= inner.width()`).
    ///
    /// Returns `None` if the column range extends beyond the width of the inner matrix.
    pub fn new_with_range(inner: Inner, column_range: Range<usize>) -> Option<Self> {
        (column_range.end <= inner.width()).then(|| Self {
            inner,
            column_range,
            _phantom: PhantomData,
        })
    }
}

impl<T, Inner> Matrix<T> for HorizontallyTruncated<T, Inner>
where
    T: Send + Sync + Clone,
    Inner: Matrix<T>,
{
    /// Returns the number of columns exposed by the truncated matrix.
    #[inline(always)]
    fn width(&self) -> usize {
        self.column_range.len()
    }

    /// Returns the number of rows in the matrix (same as the inner matrix).
    #[inline(always)]
    fn height(&self) -> usize {
        self.inner.height()
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, r: usize, c: usize) -> T {
        unsafe {
            // Safety: The caller must ensure that `c < self.width()` and `r < self.height()`.
            //
            // We translate the column index by adding `column_range.start`.
            self.inner.get_unchecked(r, self.column_range.start + c)
        }
    }

    unsafe fn row_unchecked(
        &self,
        r: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        unsafe {
            // Safety: The caller must ensure that `r < self.height()`.
            self.inner
                .row_subseq_unchecked(r, self.column_range.start, self.column_range.end)
        }
    }

    unsafe fn row_subseq_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        unsafe {
            // Safety: The caller must ensure that r < self.height() and start <= end <= self.width().
            //
            // We translate the column indices by adding `column_range.start`.
            self.inner.row_subseq_unchecked(
                r,
                self.column_range.start + start,
                self.column_range.start + end,
            )
        }
    }

    unsafe fn row_subslice_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl core::ops::Deref<Target = [T]> {
        unsafe {
            // Safety: The caller must ensure that `r < self.height()` and `start <= end <= self.width()`.
            //
            // We translate the column indices by adding `column_range.start`.
            self.inner.row_subslice_unchecked(
                r,
                self.column_range.start + start,
                self.column_range.start + end,
            )
        }
    }
}
