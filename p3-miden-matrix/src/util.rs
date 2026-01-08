use core::borrow::BorrowMut;

use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, reverse_bits_len};
use tracing::instrument;

use crate::Matrix;
use crate::dense::{DenseMatrix, DenseStorage, RowMajorMatrix};

/// Reverse the order of matrix rows based on the bit-reversal of their indices.
///
/// Given a matrix `mat` of height `h = 2^k`, this function rearranges its rows by
/// reversing the binary representation of each row index. For example, if `h = 8` (i.e., 3 bits):
///
/// ```text
/// Original Index  Binary   Reversed   Target Index
/// --------------  -------  ---------  -------------
///      0          000      000        0
///      1          001      100        4
///      2          010      010        2
///      3          011      110        6
///      4          100      001        1
///      5          101      101        5
///      6          110      011        3
///      7          111      111        7
/// ```
///
/// The transformation is performed in-place.
///
/// # Panics
/// Panics if the height of the matrix is not a power of two.
///
/// # Arguments
/// - `mat`: The matrix whose rows should be reordered.
#[instrument(level = "debug", skip_all)]
pub fn reverse_matrix_index_bits<'a, F, S>(mat: &mut DenseMatrix<F, S>)
where
    F: Clone + Send + Sync + 'a,
    S: DenseStorage<F> + BorrowMut<[F]>,
{
    let w = mat.width();
    let h = mat.height();
    let log_h = log2_strict_usize(h);
    let values = mat.values.borrow_mut().as_mut_ptr() as usize;

    // SAFETY: Due to the i < j check, we are guaranteed that `swap_rows_raw
    // will never try and access a particular slice of data more than once
    // across all parallel threads. Hence the following code is safe and does
    // not trigger undefined behaviour.
    (0..h).into_par_iter().for_each(|i| {
        let values = values as *mut F;
        let j = reverse_bits_len(i, log_h);
        if i < j {
            unsafe { swap_rows_raw(values, w, i, j) };
        }
    });
}

/// Swap two rows `i` and `j` in a [`RowMajorMatrix`].
///
/// # Panics
/// Panics if the indices are out of bounds or not ordered as `i < j`.
///
/// # Arguments
/// - `mat`: The matrix to modify.
/// - `i`: The first row index (must be less than `j`).
/// - `j`: The second row index.
pub fn swap_rows<F: Clone + Send + Sync>(mat: &mut RowMajorMatrix<F>, i: usize, j: usize) {
    let w = mat.width();
    let (upper, lower) = mat.values.split_at_mut(j * w);
    let row_i = &mut upper[i * w..(i + 1) * w];
    let row_j = &mut lower[..w];
    row_i.swap_with_slice(row_j);
}

/// Swap two rows `i` and `j` in-place using raw pointer access.
///
/// This function is equivalent to [`swap_rows`] but uses unsafe raw pointer math for better performance.
///
/// # Safety
/// - The caller must ensure `i < j < h`, where `h` is the height of the matrix.
/// - The pointer must point to a vector corresponding to a matrix of width `w`.
///
/// # Arguments
/// - `mat`: A mutable pointer to the underlying matrix data.
/// - `w`: The matrix width (number of columns).
/// - `i`: The first row index.
/// - `j`: The second row index.
unsafe fn swap_rows_raw<F>(mat: *mut F, w: usize, i: usize, j: usize) {
    unsafe {
        let row_i = core::slice::from_raw_parts_mut(mat.add(i * w), w);
        let row_j = core::slice::from_raw_parts_mut(mat.add(j * w), w);
        row_i.swap_with_slice(row_j);
    }
}
