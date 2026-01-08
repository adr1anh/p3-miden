//! Matrix library.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::fmt::{Debug, Display, Formatter};
use core::ops::Deref;

use itertools::Itertools;
use p3_maybe_rayon::prelude::*;
use p3_miden_field::{
    BasedVectorSpace, ExtensionField, Field, FieldArray, PackedFieldExtension, PackedValue,
    PrimeCharacteristicRing, dot_product,
};
use strided::{VerticallyStridedMatrixView, VerticallyStridedRowIndexMap};
use tracing::instrument;

use crate::dense::RowMajorMatrix;

pub mod bitrev;
pub mod dense;
pub mod extension;
pub mod horizontally_truncated;
pub mod row_index_mapped;
pub mod stack;
pub mod strided;
pub mod util;

/// A simple struct representing the shape of a matrix.
///
/// The `Dimensions` type stores the number of columns (`width`) and rows (`height`)
/// of a matrix. It is commonly used for querying and displaying matrix shapes.
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Dimensions {
    /// Number of columns in the matrix.
    pub width: usize,
    /// Number of rows in the matrix.
    pub height: usize,
}

impl Debug for Dimensions {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

impl Display for Dimensions {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

/// A generic trait for two-dimensional matrix-like data structures.
///
/// The `Matrix` trait provides a uniform interface for accessing rows, elements,
/// and computing with matrices in both sequential and parallel contexts. It supports
/// packing strategies for SIMD optimizations and interaction with extension fields.
pub trait Matrix<T: Send + Sync + Clone>: Send + Sync {
    /// Returns the number of columns in the matrix.
    fn width(&self) -> usize;

    /// Returns the number of rows in the matrix.
    fn height(&self) -> usize;

    /// Returns the dimensions (width, height) of the matrix.
    fn dimensions(&self) -> Dimensions {
        Dimensions {
            width: self.width(),
            height: self.height(),
        }
    }

    // The methods:
    // get, get_unchecked, row, row_unchecked, row_subseq_unchecked, row_slice, row_slice_unchecked, row_subslice_unchecked
    // are all defined in a circular manner so you only need to implement a subset of them.
    // In particular is is enough to implement just one of: row_unchecked, row_subseq_unchecked
    //
    // That being said, most implementations will want to implement several methods for performance reasons.

    /// Returns the element at the given row and column.
    ///
    /// Returns `None` if either `r >= height()` or `c >= width()`.
    #[inline]
    fn get(&self, r: usize, c: usize) -> Option<T> {
        (r < self.height() && c < self.width()).then(|| unsafe {
            // Safety: Clearly `r < self.height()` and `c < self.width()`.
            self.get_unchecked(r, c)
        })
    }

    /// Returns the element at the given row and column.
    ///
    /// For a safe alternative, see [`get`].
    ///
    /// # Safety
    /// The caller must ensure that `r < self.height()` and `c < self.width()`.
    /// Breaking any of these assumptions is considered undefined behaviour.
    #[inline]
    unsafe fn get_unchecked(&self, r: usize, c: usize) -> T {
        unsafe { self.row_slice_unchecked(r)[c].clone() }
    }

    /// Returns an iterator over the elements of the `r`-th row.
    ///
    /// The iterator will have `self.width()` elements.
    ///
    /// Returns `None` if `r >= height()`.
    #[inline]
    fn row(
        &self,
        r: usize,
    ) -> Option<impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync>> {
        (r < self.height()).then(|| unsafe {
            // Safety: Clearly `r < self.height()`.
            self.row_unchecked(r)
        })
    }

    /// Returns an iterator over the elements of the `r`-th row.
    ///
    /// The iterator will have `self.width()` elements.
    ///
    /// For a safe alternative, see [`row`].
    ///
    /// # Safety
    /// The caller must ensure that `r < self.height()`.
    /// Breaking this assumption is considered undefined behaviour.
    #[inline]
    unsafe fn row_unchecked(
        &self,
        r: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        unsafe { self.row_subseq_unchecked(r, 0, self.width()) }
    }

    /// Returns an iterator over the elements of the `r`-th row from position `start` to `end`.
    ///
    /// When `start = 0` and `end = width()`, this is equivalent to [`row_unchecked`].
    ///
    /// For a safe alternative, use [`row`], along with the `skip` and `take` iterator methods.
    ///
    /// # Safety
    /// The caller must ensure that `r < self.height()` and `start <= end <= self.width()`.
    /// Breaking any of these assumptions is considered undefined behaviour.
    #[inline]
    unsafe fn row_subseq_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        unsafe {
            self.row_unchecked(r)
                .into_iter()
                .skip(start)
                .take(end - start)
        }
    }

    /// Returns the elements of the `r`-th row as something which can be coerced to a slice.
    ///
    /// Returns `None` if `r >= height()`.
    #[inline]
    fn row_slice(&self, r: usize) -> Option<impl Deref<Target = [T]>> {
        (r < self.height()).then(|| unsafe {
            // Safety: Clearly `r < self.height()`.
            self.row_slice_unchecked(r)
        })
    }

    /// Returns the elements of the `r`-th row as something which can be coerced to a slice.
    ///
    /// For a safe alternative, see [`row_slice`].
    ///
    /// # Safety
    /// The caller must ensure that `r < self.height()`.
    /// Breaking this assumption is considered undefined behaviour.
    #[inline]
    unsafe fn row_slice_unchecked(&self, r: usize) -> impl Deref<Target = [T]> {
        unsafe { self.row_subslice_unchecked(r, 0, self.width()) }
    }

    /// Returns a subset of elements of the `r`-th row as something which can be coerced to a slice.
    ///
    /// When `start = 0` and `end = width()`, this is equivalent to [`row_slice_unchecked`].
    ///
    /// For a safe alternative, see [`row_slice`].
    ///
    /// # Safety
    /// The caller must ensure that `r < self.height()` and `start <= end <= self.width()`.
    /// Breaking any of these assumptions is considered undefined behaviour.
    #[inline]
    unsafe fn row_subslice_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl Deref<Target = [T]> {
        unsafe {
            self.row_subseq_unchecked(r, start, end)
                .into_iter()
                .collect_vec()
        }
    }

    /// Returns an iterator over all rows in the matrix.
    #[inline]
    fn rows(&self) -> impl Iterator<Item = impl Iterator<Item = T>> + Send + Sync {
        unsafe {
            // Safety: `r` always satisfies `r < self.height()`.
            (0..self.height()).map(move |r| self.row_unchecked(r).into_iter())
        }
    }

    /// Returns a parallel iterator over all rows in the matrix.
    #[inline]
    fn par_rows(
        &self,
    ) -> impl IndexedParallelIterator<Item = impl Iterator<Item = T>> + Send + Sync {
        unsafe {
            // Safety: `r` always satisfies `r < self.height()`.
            (0..self.height())
                .into_par_iter()
                .map(move |r| self.row_unchecked(r).into_iter())
        }
    }

    /// Collect the elements of the rows `r` through `r + c`. If anything is larger than `self.height()`
    /// simply wrap around to the beginning of the matrix.
    fn wrapping_row_slices(&self, r: usize, c: usize) -> Vec<impl Deref<Target = [T]>> {
        unsafe {
            // Safety: Thank to the `%`, the rows index is always less than `self.height()`.
            (0..c)
                .map(|i| self.row_slice_unchecked((r + i) % self.height()))
                .collect_vec()
        }
    }

    /// Returns an iterator over the first row of the matrix.
    ///
    /// Returns None if `height() == 0`.
    #[inline]
    fn first_row(
        &self,
    ) -> Option<impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync>> {
        self.row(0)
    }

    /// Returns an iterator over the last row of the matrix.
    ///
    /// Returns None if `height() == 0`.
    #[inline]
    fn last_row(
        &self,
    ) -> Option<impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync>> {
        if self.height() == 0 {
            None
        } else {
            // Safety: Clearly `self.height() - 1 < self.height()`.
            unsafe { Some(self.row_unchecked(self.height() - 1)) }
        }
    }

    /// Converts the matrix into a `RowMajorMatrix` by collecting all rows into a single vector.
    fn to_row_major_matrix(self) -> RowMajorMatrix<T>
    where
        Self: Sized,
        T: Clone,
    {
        RowMajorMatrix::new(self.rows().flatten().collect(), self.width())
    }

    /// Get a packed iterator over the `r`-th row.
    ///
    /// If the row length is not divisible by the packing width, the final elements
    /// are returned as a base iterator with length `<= P::WIDTH - 1`.
    ///
    /// # Panics
    /// Panics if `r >= height()`.
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
        assert!(r < self.height(), "Row index out of bounds.");
        let num_packed = self.width() / P::WIDTH;
        unsafe {
            // Safety: We have already checked that `r < height()`.
            let mut iter = self
                .row_subseq_unchecked(r, 0, num_packed * P::WIDTH)
                .into_iter();

            // array::from_fn is guaranteed to always call in order.
            let packed =
                (0..num_packed).map(move |_| P::from_fn(|_| iter.next().unwrap_unchecked()));

            let sfx = self
                .row_subseq_unchecked(r, num_packed * P::WIDTH, self.width())
                .into_iter();
            (packed, sfx)
        }
    }

    /// Get a packed iterator over the `r`-th row.
    ///
    /// If the row length is not divisible by the packing width, the final entry will be zero-padded.
    ///
    /// # Panics
    /// Panics if `r >= height()`.
    fn padded_horizontally_packed_row<'a, P>(
        &'a self,
        r: usize,
    ) -> impl Iterator<Item = P> + Send + Sync
    where
        P: PackedValue<Value = T>,
        T: Clone + Default + 'a,
    {
        let mut row_iter = self.row(r).expect("Row index out of bounds.").into_iter();
        let num_elems = self.width().div_ceil(P::WIDTH);
        // array::from_fn is guaranteed to always call in order.
        (0..num_elems).map(move |_| P::from_fn(|_| row_iter.next().unwrap_or_default()))
    }

    /// Get a parallel iterator over all packed rows of the matrix.
    ///
    /// If the matrix width is not divisible by the packing width, the final elements
    /// of each row are returned as a base iterator with length `<= P::WIDTH - 1`.
    fn par_horizontally_packed_rows<'a, P>(
        &'a self,
    ) -> impl IndexedParallelIterator<
        Item = (
            impl Iterator<Item = P> + Send + Sync,
            impl Iterator<Item = T> + Send + Sync,
        ),
    >
    where
        P: PackedValue<Value = T>,
        T: Clone + 'a,
    {
        (0..self.height())
            .into_par_iter()
            .map(|r| self.horizontally_packed_row(r))
    }

    /// Get a parallel iterator over all packed rows of the matrix.
    ///
    /// If the matrix width is not divisible by the packing width, the final entry of each row will be zero-padded.
    fn par_padded_horizontally_packed_rows<'a, P>(
        &'a self,
    ) -> impl IndexedParallelIterator<Item = impl Iterator<Item = P> + Send + Sync>
    where
        P: PackedValue<Value = T>,
        T: Clone + Default + 'a,
    {
        (0..self.height())
            .into_par_iter()
            .map(|r| self.padded_horizontally_packed_row(r))
    }

    /// Pack together a collection of adjacent rows from the matrix.
    ///
    /// Returns an iterator whose i'th element is packing of the i'th element of the
    /// rows r through r + P::WIDTH - 1. If we exceed the height of the matrix,
    /// wrap around and include initial rows.
    #[inline]
    fn vertically_packed_row<P>(&self, r: usize) -> impl Iterator<Item = P>
    where
        T: Copy,
        P: PackedValue<Value = T>,
    {
        // Precompute row slices once to minimize redundant calls and improve performance.
        let rows = self.wrapping_row_slices(r, P::WIDTH);

        // Using precomputed rows avoids repeatedly calling `row_slice`, which is costly.
        (0..self.width()).map(move |c| P::from_fn(|i| rows[i][c]))
    }

    /// Pack together a collection of rows and "next" rows from the matrix.
    ///
    /// Returns a vector corresponding to 2 packed rows. The i'th element of the first
    /// row contains the packing of the i'th element of the rows r through r + P::WIDTH - 1.
    /// The i'th element of the second row contains the packing of the i'th element of the
    /// rows r + step through r + step + P::WIDTH - 1. If at some point we exceed the
    /// height of the matrix, wrap around and include initial rows.
    #[inline]
    fn vertically_packed_row_pair<P>(&self, r: usize, step: usize) -> Vec<P>
    where
        T: Copy,
        P: PackedValue<Value = T>,
    {
        // Whilst it would appear that this can be replaced by two calls to vertically_packed_row
        // tests seem to indicate that combining them in the same function is slightly faster.
        // It's probably allowing the compiler to make some optimizations on the fly.

        let rows = self.wrapping_row_slices(r, P::WIDTH);
        let next_rows = self.wrapping_row_slices(r + step, P::WIDTH);

        (0..self.width())
            .map(|c| P::from_fn(|i| rows[i][c]))
            .chain((0..self.width()).map(|c| P::from_fn(|i| next_rows[i][c])))
            .collect_vec()
    }

    /// Returns a view over a vertically strided submatrix.
    ///
    /// The view selects rows using `r = offset + i * stride` for each `i`.
    fn vertically_strided(self, stride: usize, offset: usize) -> VerticallyStridedMatrixView<Self>
    where
        Self: Sized,
    {
        VerticallyStridedRowIndexMap::new_view(self, stride, offset)
    }

    /// Compute Mᵀv, aka premultiply this matrix by the given vector,
    /// aka scale each row by the corresponding entry in `v` and take the sum across rows.
    /// `v` can be a vector of extension elements.
    #[instrument(level = "debug", skip_all, fields(dims = %self.dimensions()))]
    fn columnwise_dot_product<EF>(&self, v: &[EF]) -> Vec<EF>
    where
        T: Field,
        EF: ExtensionField<T>,
    {
        let packed_width = self.width().div_ceil(T::Packing::WIDTH);

        let packed_result: Vec<EF::ExtensionPacking> = self
            .par_padded_horizontally_packed_rows::<T::Packing>()
            .zip(v)
            .par_fold_reduce(
                || EF::ExtensionPacking::zero_vec(packed_width),
                |mut acc, (row, &scale)| {
                    let scale: EF::ExtensionPacking = scale.into();
                    acc.iter_mut().zip(row).for_each(|(l, r)| *l += scale * r);
                    acc
                },
                |mut acc_l, acc_r| {
                    acc_l.iter_mut().zip(&acc_r).for_each(|(l, r)| *l += *r);
                    acc_l
                },
            );

        EF::ExtensionPacking::to_ext_iter(packed_result.into_iter())
            .take(self.width())
            .collect()
    }

    /// Compute Mᵀ · [v₀, v₁, ..., vₙ₋₁] for N weight vectors simultaneously.
    ///
    /// Computes `result[col][j] = Σᵣ M[r, col] · vⱼ[r]` for all columns and all j ∈ [0, N).
    ///
    /// Batching N weight vectors reduces memory bandwidth: each matrix row is loaded once
    /// instead of N times. Uses SIMD packing (width W) to process W columns in parallel.
    #[instrument(level = "debug", skip_all, fields(dims = %self.dimensions()))]
    fn columnwise_dot_product_batched<EF, const N: usize>(
        &self,
        vs: &[FieldArray<EF, N>],
    ) -> Vec<FieldArray<EF, N>>
    where
        T: Field,
        EF: ExtensionField<T>,
    {
        let packed_width = self.width().div_ceil(T::Packing::WIDTH);

        let packed_results: Vec<EF::ExtensionPacking> = self
            .par_padded_horizontally_packed_rows::<T::Packing>()
            .zip(vs)
            .par_fold_reduce(
                || EF::ExtensionPacking::zero_vec(packed_width * N),
                |mut acc, (packed_row, scales)| {
                    let (acc_chunks, _) = acc.as_chunks_mut::<N>();
                    // Broadcast each scalar scale to all SIMD lanes
                    let packed_scales: [EF::ExtensionPacking; N] =
                        scales.map_array(EF::ExtensionPacking::from);

                    // acc[c][j] += scales[j] · row[c] for column batch c, point j
                    acc_chunks
                        .iter_mut()
                        .zip(packed_row)
                        .for_each(|(acc_c, row_c)| {
                            for j in 0..N {
                                acc_c[j] += packed_scales[j] * row_c;
                            }
                        });
                    acc
                },
                |mut acc_l, acc_r| {
                    acc_l.iter_mut().zip(&acc_r).for_each(|(lj, rj)| *lj += *rj);
                    acc_l
                },
            );

        // Unpack: chunk[j].lane(i) → result[c·W + i][j] for column batch c
        packed_results
            .chunks(N)
            .flat_map(|chunk| {
                (0..T::Packing::WIDTH)
                    .map(move |lane| FieldArray::from_fn(|j| chunk[j].extract_lane(lane)))
            })
            .take(self.width())
            .collect()
    }

    /// Compute the matrix vector product `M . vec`, aka take the dot product of each
    /// row of `M` by `vec`. If the length of `vec` is longer than the width of `M`,
    /// `vec` is truncated to the first `width()` elements.
    ///
    /// We make use of `PackedFieldExtension` to speed up computations. Thus `vec` is passed in as
    /// a slice of `PackedFieldExtension` elements.
    ///
    /// # Panics
    /// This function panics if the length of `vec` is less than `self.width().div_ceil(T::Packing::WIDTH)`.
    fn rowwise_packed_dot_product<EF>(
        &self,
        vec: &[EF::ExtensionPacking],
    ) -> impl IndexedParallelIterator<Item = EF>
    where
        T: Field,
        EF: ExtensionField<T>,
    {
        // The length of a `padded_horizontally_packed_row` is `self.width().div_ceil(T::Packing::WIDTH)`.
        assert!(vec.len() >= self.width().div_ceil(T::Packing::WIDTH));

        // TODO: This is a base - extension dot product and so it should
        // be possible to speed this up using ideas in `packed_linear_combination`.
        // TODO: Perhaps we should be packing rows vertically not horizontally.
        self.par_padded_horizontally_packed_rows::<T::Packing>()
            .map(move |row_packed| {
                let packed_sum_of_packed: EF::ExtensionPacking =
                    dot_product(vec.iter().copied(), row_packed);
                let sum_of_packed: EF = EF::from_basis_coefficients_fn(|i| {
                    packed_sum_of_packed.as_basis_coefficients_slice()[i]
                        .as_slice()
                        .iter()
                        .copied()
                        .sum()
                });
                sum_of_packed
            })
    }
}

// Allow creating matrix views over references by forwarding `Matrix` methods to the underlying
// matrix implementation. This enables using `&M` anywhere an `M: Matrix<T>` is expected,
// which is useful for lightweight views like lifted/strided without moving or cloning.
impl<T, M> Matrix<T> for &M
where
    T: Send + Sync + Clone,
    M: Matrix<T> + ?Sized,
{
    #[inline]
    fn width(&self) -> usize {
        (*self).width()
    }

    #[inline]
    fn height(&self) -> usize {
        (*self).height()
    }

    #[inline]
    fn get(&self, r: usize, c: usize) -> Option<T> {
        (*self).get(r, c)
    }

    #[inline]
    unsafe fn get_unchecked(&self, r: usize, c: usize) -> T {
        // Safety: Follows the contract of the underlying matrix method.
        unsafe { (*self).get_unchecked(r, c) }
    }

    #[inline]
    fn row(
        &self,
        r: usize,
    ) -> Option<impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync>> {
        (*self).row(r)
    }

    #[inline]
    unsafe fn row_unchecked(
        &self,
        r: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        // Safety: Follows the contract of the underlying matrix method.
        unsafe { (*self).row_unchecked(r) }
    }

    #[inline]
    unsafe fn row_subseq_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        // Safety: Follows the contract of the underlying matrix method.
        unsafe { (*self).row_subseq_unchecked(r, start, end) }
    }

    #[inline]
    fn row_slice(&self, r: usize) -> Option<impl Deref<Target = [T]>> {
        (*self).row_slice(r)
    }
    #[inline]
    unsafe fn row_slice_unchecked(&self, r: usize) -> impl Deref<Target = [T]> {
        // Safety: Follows the contract of the underlying matrix method.
        unsafe { (*self).row_slice_unchecked(r) }
    }

    #[inline]
    unsafe fn row_subslice_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl Deref<Target = [T]> {
        // Safety: Follows the contract of the underlying matrix method.
        unsafe { (*self).row_subslice_unchecked(r, start, end) }
    }

    #[inline]
    fn rows(&self) -> impl Iterator<Item = impl Iterator<Item = T>> + Send + Sync {
        (*self).rows()
    }

    #[inline]
    fn par_rows(
        &self,
    ) -> impl IndexedParallelIterator<Item = impl Iterator<Item = T>> + Send + Sync {
        (*self).par_rows()
    }

    #[inline]
    fn wrapping_row_slices(&self, r: usize, c: usize) -> Vec<impl Deref<Target = [T]>> {
        (*self).wrapping_row_slices(r, c)
    }

    #[inline]
    fn first_row(
        &self,
    ) -> Option<impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync>> {
        (*self).first_row()
    }

    #[inline]
    fn last_row(
        &self,
    ) -> Option<impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync>> {
        (*self).last_row()
    }

    #[inline]
    fn horizontally_packed_row<'b, P>(
        &'b self,
        r: usize,
    ) -> (
        impl Iterator<Item = P> + Send + Sync,
        impl Iterator<Item = T> + Send + Sync,
    )
    where
        P: PackedValue<Value = T>,
        T: Clone + 'b,
    {
        (*self).horizontally_packed_row::<P>(r)
    }

    #[inline]
    fn padded_horizontally_packed_row<'b, P>(
        &'b self,
        r: usize,
    ) -> impl Iterator<Item = P> + Send + Sync
    where
        P: PackedValue<Value = T>,
        T: Clone + Default + 'b,
    {
        (*self).padded_horizontally_packed_row::<P>(r)
    }

    #[inline]
    fn par_horizontally_packed_rows<'b, P>(
        &'b self,
    ) -> impl IndexedParallelIterator<
        Item = (
            impl Iterator<Item = P> + Send + Sync,
            impl Iterator<Item = T> + Send + Sync,
        ),
    >
    where
        P: PackedValue<Value = T>,
        T: Clone + 'b,
    {
        (*self).par_horizontally_packed_rows::<P>()
    }

    #[inline]
    fn par_padded_horizontally_packed_rows<'b, P>(
        &'b self,
    ) -> impl IndexedParallelIterator<Item = impl Iterator<Item = P> + Send + Sync>
    where
        P: PackedValue<Value = T>,
        T: Clone + Default + 'b,
    {
        (*self).par_padded_horizontally_packed_rows::<P>()
    }

    #[inline]
    fn vertically_packed_row<P>(&self, r: usize) -> impl Iterator<Item = P>
    where
        T: Copy,
        P: PackedValue<Value = T>,
    {
        (*self).vertically_packed_row::<P>(r)
    }

    #[inline]
    fn vertically_packed_row_pair<P>(&self, r: usize, step: usize) -> Vec<P>
    where
        T: Copy,
        P: PackedValue<Value = T>,
    {
        (*self).vertically_packed_row_pair::<P>(r, step)
    }

    #[inline]
    fn columnwise_dot_product<EF>(&self, v: &[EF]) -> Vec<EF>
    where
        T: Field,
        EF: ExtensionField<T>,
    {
        (*self).columnwise_dot_product::<EF>(v)
    }

    #[inline]
    fn columnwise_dot_product_batched<EF, const N: usize>(
        &self,
        vs: &[FieldArray<EF, N>],
    ) -> Vec<FieldArray<EF, N>>
    where
        T: Field,
        EF: ExtensionField<T>,
    {
        (*self).columnwise_dot_product_batched::<EF, N>(vs)
    }

    #[inline]
    fn rowwise_packed_dot_product<EF>(
        &self,
        vec: &[<EF as ExtensionField<T>>::ExtensionPacking],
    ) -> impl IndexedParallelIterator<Item = EF>
    where
        T: Field,
        EF: ExtensionField<T>,
    {
        (*self).rowwise_packed_dot_product::<EF>(vec)
    }
}
