use core::ops::Deref;

use p3_miden_field::PackedValue;

use crate::Matrix;
use crate::dense::RowMajorMatrix;

/// A trait for remapping row indices of a matrix.
///
/// Implementations can change the number of visible rows (`height`)
/// and define how a given logical row index maps to a physical one.
pub trait RowIndexMap: Send + Sync {
    /// Returns the number of rows exposed by the mapping.
    fn height(&self) -> usize;

    /// Maps a visible row index `r` to the corresponding row index in the underlying matrix.
    ///
    /// The input `r` is assumed to lie in the range `0..self.height()` and the output
    /// will lie in the range `0..self.inner.height()`.
    ///
    /// It is considered undefined behaviour to call `map_row_index` with `r >= self.height()`.
    fn map_row_index(&self, r: usize) -> usize;

    /// Converts the mapped matrix into a dense row-major matrix.
    ///
    /// This default implementation iterates over all mapped rows,
    /// collects them in order, and builds a dense representation.
    fn to_row_major_matrix<T: Clone + Send + Sync, Inner: Matrix<T>>(
        &self,
        inner: Inner,
    ) -> RowMajorMatrix<T> {
        RowMajorMatrix::new(
            unsafe {
                // Safety: The output of `map_row_index` is less than `inner.height()` for all inputs in the range `0..self.height()`.
                (0..self.height())
                    .flat_map(|r| inner.row_unchecked(self.map_row_index(r)))
                    .collect()
            },
            inner.width(),
        )
    }
}

/// A matrix view that applies a row index mapping to an inner matrix.
///
/// The mapping changes which rows are visible and in what order.
/// The width remains unchanged.
#[derive(Copy, Clone, Debug)]
pub struct RowIndexMappedView<IndexMap, Inner> {
    /// A row index mapping that defines the number and order of visible rows.
    pub index_map: IndexMap,
    /// The inner matrix that holds actual data.
    pub inner: Inner,
}

impl<T: Send + Sync + Clone, IndexMap: RowIndexMap, Inner: Matrix<T>> Matrix<T>
    for RowIndexMappedView<IndexMap, Inner>
{
    fn width(&self) -> usize {
        self.inner.width()
    }

    fn height(&self) -> usize {
        self.index_map.height()
    }

    unsafe fn get_unchecked(&self, r: usize, c: usize) -> T {
        unsafe {
            // Safety: The caller must ensure that r < self.height() and c < self.width().
            self.inner.get_unchecked(self.index_map.map_row_index(r), c)
        }
    }

    unsafe fn row_unchecked(
        &self,
        r: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        unsafe {
            // Safety: The caller must ensure that r < self.height().
            self.inner.row_unchecked(self.index_map.map_row_index(r))
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
            self.inner
                .row_subseq_unchecked(self.index_map.map_row_index(r), start, end)
        }
    }

    unsafe fn row_slice_unchecked(&self, r: usize) -> impl Deref<Target = [T]> {
        unsafe {
            // Safety: The caller must ensure that r < self.height().
            self.inner
                .row_slice_unchecked(self.index_map.map_row_index(r))
        }
    }

    unsafe fn row_subslice_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl Deref<Target = [T]> {
        unsafe {
            // Safety: The caller must ensure that r < self.height() and start <= end <= self.width().
            self.inner
                .row_subslice_unchecked(self.index_map.map_row_index(r), start, end)
        }
    }

    fn to_row_major_matrix(self) -> RowMajorMatrix<T>
    where
        Self: Sized,
        T: Clone,
    {
        // Use Perm's optimized permutation routine, if it has one.
        self.index_map.to_row_major_matrix(self.inner)
    }

    fn horizontally_packed_row<'a, P>(
        &'a self,
        r: usize,
    ) -> (
        impl Iterator<Item = P> + Send + Sync,
        impl Iterator<Item = T> + Send + Sync,
    )
    where
        P: PackedValue<Value = T>,
        T: Clone + 'a,
    {
        self.inner
            .horizontally_packed_row(self.index_map.map_row_index(r))
    }

    fn padded_horizontally_packed_row<'a, P>(
        &'a self,
        r: usize,
    ) -> impl Iterator<Item = P> + Send + Sync
    where
        P: PackedValue<Value = T>,
        T: Clone + Default + 'a,
    {
        self.inner
            .padded_horizontally_packed_row(self.index_map.map_row_index(r))
    }
}
