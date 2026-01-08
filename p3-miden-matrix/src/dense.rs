use alloc::borrow::Cow;
use alloc::vec;
use alloc::vec::Vec;
use core::borrow::{Borrow, BorrowMut};
use core::iter;
use core::marker::PhantomData;
use core::ops::Deref;

use p3_maybe_rayon::prelude::*;
use p3_miden_field::{
    ExtensionField, Field, PackedValue, par_scale_slice_in_place, scale_slice_in_place_single_core,
};
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::Matrix;

/// A dense matrix in row-major format, with customizable backing storage.
///
/// The data is stored as a flat buffer, where rows are laid out consecutively.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DenseMatrix<T, V = Vec<T>> {
    /// Flat buffer of matrix values in row-major order.
    pub values: V,
    /// Number of columns in the matrix.
    ///
    /// The number of rows is implicitly determined as `values.len() / width`.
    pub width: usize,
    /// Marker for the element type `T`, unused directly.
    ///
    /// Required to retain type information when `V` does not own or contain `T`.
    _phantom: PhantomData<T>,
}

pub type RowMajorMatrix<T> = DenseMatrix<T>;
pub type RowMajorMatrixView<'a, T> = DenseMatrix<T, &'a [T]>;
pub type RowMajorMatrixViewMut<'a, T> = DenseMatrix<T, &'a mut [T]>;
pub type RowMajorMatrixCow<'a, T> = DenseMatrix<T, Cow<'a, [T]>>;

pub trait DenseStorage<T>: Borrow<[T]> + Send + Sync {
    fn to_vec(self) -> Vec<T>;
}

// Cow doesn't impl IntoOwned so we can't blanket it
impl<T: Clone + Send + Sync> DenseStorage<T> for Vec<T> {
    fn to_vec(self) -> Self {
        self
    }
}

impl<T: Clone + Send + Sync> DenseStorage<T> for &[T] {
    fn to_vec(self) -> Vec<T> {
        <[T]>::to_vec(self)
    }
}

impl<T: Clone + Send + Sync> DenseStorage<T> for &mut [T] {
    fn to_vec(self) -> Vec<T> {
        <[T]>::to_vec(self)
    }
}

impl<T: Clone + Send + Sync> DenseStorage<T> for Cow<'_, [T]> {
    fn to_vec(self) -> Vec<T> {
        self.into_owned()
    }
}

impl<T: Clone + Send + Sync + Default> DenseMatrix<T> {
    /// Create a new dense matrix of the given dimensions, backed by a `Vec`, and filled with
    /// default values.
    #[must_use]
    pub fn default(width: usize, height: usize) -> Self {
        Self::new(vec![T::default(); width * height], width)
    }
}

impl<T: Clone + Send + Sync, S: DenseStorage<T>> DenseMatrix<T, S> {
    /// Create a new dense matrix of the given dimensions, backed by the given storage.
    ///
    /// Note that it is undefined behavior to create a matrix such that
    /// `values.len() % width != 0`.
    #[must_use]
    pub fn new(values: S, width: usize) -> Self {
        debug_assert!(values.borrow().len().is_multiple_of(width));
        Self {
            values,
            width,
            _phantom: PhantomData,
        }
    }

    /// Create a new RowMajorMatrix containing a single row.
    #[must_use]
    pub fn new_row(values: S) -> Self {
        let width = values.borrow().len();
        Self::new(values, width)
    }

    /// Create a new RowMajorMatrix containing a single column.
    #[must_use]
    pub fn new_col(values: S) -> Self {
        Self::new(values, 1)
    }

    /// Get a view of the matrix, i.e. a reference to the underlying data.
    pub fn as_view(&self) -> RowMajorMatrixView<'_, T> {
        RowMajorMatrixView::new(self.values.borrow(), self.width)
    }

    /// Get a mutable view of the matrix, i.e. a mutable reference to the underlying data.
    pub fn as_view_mut(&mut self) -> RowMajorMatrixViewMut<'_, T>
    where
        S: BorrowMut<[T]>,
    {
        RowMajorMatrixViewMut::new(self.values.borrow_mut(), self.width)
    }

    /// Copy the values from the given matrix into this matrix.
    pub fn copy_from<S2>(&mut self, source: &DenseMatrix<T, S2>)
    where
        T: Copy,
        S: BorrowMut<[T]>,
        S2: DenseStorage<T>,
    {
        assert_eq!(self.dimensions(), source.dimensions());
        // Equivalent to:
        // self.values.borrow_mut().copy_from_slice(source.values.borrow());
        self.par_rows_mut()
            .zip(source.par_row_slices())
            .for_each(|(dst, src)| {
                dst.copy_from_slice(src);
            });
    }

    /// Flatten an extension field matrix to a base field matrix.
    pub fn flatten_to_base<F: Field>(self) -> RowMajorMatrix<F>
    where
        T: ExtensionField<F>,
    {
        let width = self.width * T::DIMENSION;
        let values = T::flatten_to_base(self.values.to_vec());
        RowMajorMatrix::new(values, width)
    }

    /// Get an iterator over the rows of the matrix.
    pub fn row_slices(&self) -> impl DoubleEndedIterator<Item = &[T]> {
        self.values.borrow().chunks_exact(self.width)
    }

    /// Get a parallel iterator over the rows of the matrix.
    pub fn par_row_slices(&self) -> impl IndexedParallelIterator<Item = &[T]>
    where
        T: Sync,
    {
        self.values.borrow().par_chunks_exact(self.width)
    }

    /// Returns a slice of the given row.
    ///
    /// # Panics
    /// Panics if `r` larger than self.height().
    pub fn row_mut(&mut self, r: usize) -> &mut [T]
    where
        S: BorrowMut<[T]>,
    {
        &mut self.values.borrow_mut()[r * self.width..(r + 1) * self.width]
    }

    /// Get a mutable iterator over the rows of the matrix.
    pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut [T]>
    where
        S: BorrowMut<[T]>,
    {
        self.values.borrow_mut().chunks_exact_mut(self.width)
    }

    /// Get a mutable parallel iterator over the rows of the matrix.
    pub fn par_rows_mut<'a>(&'a mut self) -> impl IndexedParallelIterator<Item = &'a mut [T]>
    where
        T: 'a + Send,
        S: BorrowMut<[T]>,
    {
        self.values.borrow_mut().par_chunks_exact_mut(self.width)
    }

    /// Get a mutable iterator over the rows of the matrix which packs the rows into packed values.
    ///
    /// If `P::WIDTH` does not divide `self.width`, the remainder of the row will be returned as a
    /// base slice.
    pub fn horizontally_packed_row_mut<P>(&mut self, r: usize) -> (&mut [P], &mut [T])
    where
        P: PackedValue<Value = T>,
        S: BorrowMut<[T]>,
    {
        P::pack_slice_with_suffix_mut(self.row_mut(r))
    }

    /// Scale the given row by the given value.
    ///
    /// # Panics
    /// Panics if `r` larger than `self.height()`.
    pub fn scale_row(&mut self, r: usize, scale: T)
    where
        T: Field,
        S: BorrowMut<[T]>,
    {
        scale_slice_in_place_single_core(self.row_mut(r), scale);
    }

    /// Scale the given row by the given value.
    ///
    /// # Performance
    /// This function is parallelized, which may introduce some overhead compared to
    /// [`Self::scale_row`] when the width is small.
    ///
    /// # Panics
    /// Panics if `r` larger than `self.height()`.
    pub fn par_scale_row(&mut self, r: usize, scale: T)
    where
        T: Field,
        S: BorrowMut<[T]>,
    {
        par_scale_slice_in_place(self.row_mut(r), scale);
    }

    /// Scale the entire matrix by the given value.
    pub fn scale(&mut self, scale: T)
    where
        T: Field,
        S: BorrowMut<[T]>,
    {
        par_scale_slice_in_place(self.values.borrow_mut(), scale);
    }

    /// Split the matrix into two matrix views, one with the first `r` rows and one with the remaining rows.
    ///
    /// # Panics
    /// Panics if `r` larger than `self.height()`.
    pub fn split_rows(&self, r: usize) -> (RowMajorMatrixView<'_, T>, RowMajorMatrixView<'_, T>) {
        let (lo, hi) = self.values.borrow().split_at(r * self.width);
        (
            DenseMatrix::new(lo, self.width),
            DenseMatrix::new(hi, self.width),
        )
    }

    /// Split the matrix into two mutable matrix views, one with the first `r` rows and one with the remaining rows.
    ///
    /// # Panics
    /// Panics if `r` larger than `self.height()`.
    pub fn split_rows_mut(
        &mut self,
        r: usize,
    ) -> (RowMajorMatrixViewMut<'_, T>, RowMajorMatrixViewMut<'_, T>)
    where
        S: BorrowMut<[T]>,
    {
        let (lo, hi) = self.values.borrow_mut().split_at_mut(r * self.width);
        (
            DenseMatrix::new(lo, self.width),
            DenseMatrix::new(hi, self.width),
        )
    }

    /// Get an iterator over the rows of the matrix which takes `chunk_rows` rows at a time.
    ///
    /// If `chunk_rows` does not divide the height of the matrix, the last chunk will be smaller.
    pub fn par_row_chunks(
        &self,
        chunk_rows: usize,
    ) -> impl IndexedParallelIterator<Item = RowMajorMatrixView<'_, T>>
    where
        T: Send,
    {
        self.values
            .borrow()
            .par_chunks(self.width * chunk_rows)
            .map(|slice| RowMajorMatrixView::new(slice, self.width))
    }

    /// Get a parallel iterator over the rows of the matrix which takes `chunk_rows` rows at a time.
    ///
    /// If `chunk_rows` does not divide the height of the matrix, the last chunk will be smaller.
    pub fn par_row_chunks_exact(
        &self,
        chunk_rows: usize,
    ) -> impl IndexedParallelIterator<Item = RowMajorMatrixView<'_, T>>
    where
        T: Send,
    {
        self.values
            .borrow()
            .par_chunks_exact(self.width * chunk_rows)
            .map(|slice| RowMajorMatrixView::new(slice, self.width))
    }

    /// Get a mutable iterator over the rows of the matrix which takes `chunk_rows` rows at a time.
    ///
    /// If `chunk_rows` does not divide the height of the matrix, the last chunk will be smaller.
    pub fn par_row_chunks_mut(
        &mut self,
        chunk_rows: usize,
    ) -> impl IndexedParallelIterator<Item = RowMajorMatrixViewMut<'_, T>>
    where
        T: Send,
        S: BorrowMut<[T]>,
    {
        self.values
            .borrow_mut()
            .par_chunks_mut(self.width * chunk_rows)
            .map(|slice| RowMajorMatrixViewMut::new(slice, self.width))
    }

    /// Get a mutable iterator over the rows of the matrix which takes `chunk_rows` rows at a time.
    ///
    /// If `chunk_rows` does not divide the height of the matrix, the last up to `chunk_rows - 1` rows
    /// of the matrix will be omitted.
    pub fn row_chunks_exact_mut(
        &mut self,
        chunk_rows: usize,
    ) -> impl Iterator<Item = RowMajorMatrixViewMut<'_, T>>
    where
        T: Send,
        S: BorrowMut<[T]>,
    {
        self.values
            .borrow_mut()
            .chunks_exact_mut(self.width * chunk_rows)
            .map(|slice| RowMajorMatrixViewMut::new(slice, self.width))
    }

    /// Get a parallel mutable iterator over the rows of the matrix which takes `chunk_rows` rows at a time.
    ///
    /// If `chunk_rows` does not divide the height of the matrix, the last up to `chunk_rows - 1` rows
    /// of the matrix will be omitted.
    pub fn par_row_chunks_exact_mut(
        &mut self,
        chunk_rows: usize,
    ) -> impl IndexedParallelIterator<Item = RowMajorMatrixViewMut<'_, T>>
    where
        T: Send,
        S: BorrowMut<[T]>,
    {
        self.values
            .borrow_mut()
            .par_chunks_exact_mut(self.width * chunk_rows)
            .map(|slice| RowMajorMatrixViewMut::new(slice, self.width))
    }

    /// Get a pair of mutable slices of the given rows.
    ///
    /// # Panics
    /// Panics if `row_1` or `row_2` are out of bounds or if `row_1 >= row_2`.
    pub fn row_pair_mut(&mut self, row_1: usize, row_2: usize) -> (&mut [T], &mut [T])
    where
        S: BorrowMut<[T]>,
    {
        debug_assert_ne!(row_1, row_2);
        let start_1 = row_1 * self.width;
        let start_2 = row_2 * self.width;
        let (lo, hi) = self.values.borrow_mut().split_at_mut(start_2);
        (&mut lo[start_1..][..self.width], &mut hi[..self.width])
    }

    /// Get a pair of mutable slices of the given rows, both packed into packed field elements.
    ///
    /// If `P:WIDTH` does not divide `self.width`, the remainder of the row will be returned as a base slice.
    ///
    /// # Panics
    /// Panics if `row_1` or `row_2` are out of bounds or if `row_1 >= row_2`.
    #[allow(clippy::type_complexity)]
    pub fn packed_row_pair_mut<P>(
        &mut self,
        row_1: usize,
        row_2: usize,
    ) -> ((&mut [P], &mut [T]), (&mut [P], &mut [T]))
    where
        S: BorrowMut<[T]>,
        P: PackedValue<Value = T>,
    {
        let (slice_1, slice_2) = self.row_pair_mut(row_1, row_2);
        (
            P::pack_slice_with_suffix_mut(slice_1),
            P::pack_slice_with_suffix_mut(slice_2),
        )
    }

    /// Append zeros to the "end" of the given matrix, except that the matrix is in bit-reversed order,
    /// so in actuality we're interleaving zero rows.
    #[instrument(level = "debug", skip_all)]
    pub fn bit_reversed_zero_pad(self, added_bits: usize) -> RowMajorMatrix<T>
    where
        T: Field,
    {
        if added_bits == 0 {
            return self.to_row_major_matrix();
        }

        // This is equivalent to:
        //     reverse_matrix_index_bits(mat);
        //     mat
        //         .values
        //         .resize(mat.values.len() << added_bits, F::ZERO);
        //     reverse_matrix_index_bits(mat);
        // But rather than implement it with bit reversals, we directly construct the resulting matrix,
        // whose rows are zero except for rows whose low `added_bits` bits are zero.

        let w = self.width;
        let mut padded =
            RowMajorMatrix::new(T::zero_vec(self.values.borrow().len() << added_bits), w);
        padded
            .par_row_chunks_exact_mut(1 << added_bits)
            .zip(self.par_row_slices())
            .for_each(|(mut ch, r)| ch.row_mut(0).copy_from_slice(r));

        padded
    }
}

impl<T: Clone + Send + Sync, S: DenseStorage<T>> Matrix<T> for DenseMatrix<T, S> {
    #[inline]
    fn width(&self) -> usize {
        self.width
    }

    #[inline]
    fn height(&self) -> usize {
        if self.width == 0 {
            0
        } else {
            self.values.borrow().len() / self.width
        }
    }

    #[inline]
    unsafe fn get_unchecked(&self, r: usize, c: usize) -> T {
        unsafe {
            // Safety: The caller must ensure that r < self.height() and c < self.width().
            self.values
                .borrow()
                .get_unchecked(r * self.width + c)
                .clone()
        }
    }

    #[inline]
    unsafe fn row_subseq_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        unsafe {
            // Safety: The caller must ensure that r < self.height() and start <= end <= self.width().
            self.values
                .borrow()
                .get_unchecked(r * self.width + start..r * self.width + end)
                .iter()
                .cloned()
        }
    }

    #[inline]
    unsafe fn row_subslice_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl Deref<Target = [T]> {
        unsafe {
            // Safety: The caller must ensure that r < self.height()
            self.values
                .borrow()
                .get_unchecked(r * self.width + start..r * self.width + end)
        }
    }

    fn to_row_major_matrix(self) -> RowMajorMatrix<T>
    where
        Self: Sized,
        T: Clone,
    {
        RowMajorMatrix::new(self.values.to_vec(), self.width)
    }

    #[inline]
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
        let buf = &self.values.borrow()[r * self.width..(r + 1) * self.width];
        let (packed, sfx) = P::pack_slice_with_suffix(buf);
        (packed.iter().copied(), sfx.iter().cloned())
    }

    #[inline]
    fn padded_horizontally_packed_row<'a, P>(
        &'a self,
        r: usize,
    ) -> impl Iterator<Item = P> + Send + Sync
    where
        P: PackedValue<Value = T>,
        T: Clone + Default + 'a,
    {
        let buf = &self.values.borrow()[r * self.width..(r + 1) * self.width];
        let (packed, sfx) = P::pack_slice_with_suffix(buf);
        packed.iter().copied().chain(iter::once(P::from_fn(|i| {
            sfx.get(i).cloned().unwrap_or_default()
        })))
    }
}

impl<T: Clone + Default + Send + Sync> DenseMatrix<T> {
    pub fn as_cow<'a>(self) -> RowMajorMatrixCow<'a, T> {
        RowMajorMatrixCow::new(Cow::Owned(self.values), self.width)
    }

    pub fn rand<R: Rng>(rng: &mut R, rows: usize, cols: usize) -> Self
    where
        StandardUniform: Distribution<T>,
    {
        let values = rng.sample_iter(StandardUniform).take(rows * cols).collect();
        Self::new(values, cols)
    }

    pub fn rand_nonzero<R: Rng>(rng: &mut R, rows: usize, cols: usize) -> Self
    where
        T: Field,
        StandardUniform: Distribution<T>,
    {
        let values = rng
            .sample_iter(StandardUniform)
            .filter(|x| !x.is_zero())
            .take(rows * cols)
            .collect();
        Self::new(values, cols)
    }

    pub fn pad_to_height(&mut self, new_height: usize, fill: T) {
        assert!(new_height >= self.height());
        self.values.resize(self.width * new_height, fill);
    }
}

impl<T: Copy + Default + Send + Sync, V: DenseStorage<T>> DenseMatrix<T, V> {
    /// Return the transpose of this matrix.
    pub fn transpose(&self) -> RowMajorMatrix<T> {
        let nelts = self.height() * self.width();
        let mut values = vec![T::default(); nelts];
        transpose::transpose(
            self.values.borrow(),
            &mut values,
            self.width(),
            self.height(),
        );
        RowMajorMatrix::new(values, self.height())
    }

    /// Transpose the matrix returning the result in `other` without intermediate allocation.
    pub fn transpose_into<W: DenseStorage<T> + BorrowMut<[T]>>(
        &self,
        other: &mut DenseMatrix<T, W>,
    ) {
        assert_eq!(self.height(), other.width());
        assert_eq!(other.height(), self.width());
        transpose::transpose(
            self.values.borrow(),
            other.values.borrow_mut(),
            self.width(),
            self.height(),
        );
    }
}

impl<'a, T: Clone + Default + Send + Sync> RowMajorMatrixView<'a, T> {
    pub fn as_cow(self) -> RowMajorMatrixCow<'a, T> {
        RowMajorMatrixCow::new(Cow::Borrowed(self.values), self.width)
    }
}
