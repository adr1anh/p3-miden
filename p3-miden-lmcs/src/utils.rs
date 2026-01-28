//! Utility functions for LMCS operations.

use alloc::vec::Vec;
use core::array;

use p3_field::PackedValue;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_stateful_hasher::StatefulHasher;

/// Extension trait for `PackedValue` providing columnar pack/unpack operations.
///
/// These methods perform transpose operations on packed data, useful for
/// SIMD-parallelized Merkle tree construction.
pub trait PackedValueExt: PackedValue {
    /// Pack columns from `WIDTH` rows of scalar values.
    ///
    /// Given `WIDTH` rows of `N` scalar values, extract each column and pack it
    /// into a single packed value. This performs a transpose operation.
    #[inline]
    #[must_use]
    fn pack_columns<const N: usize>(rows: &[[Self::Value; N]]) -> [Self; N] {
        assert_eq!(rows.len(), Self::WIDTH);
        array::from_fn(|col| Self::from_fn(|lane| rows[lane][col]))
    }
}

// Blanket implementation for all PackedValue types
impl<T: PackedValue> PackedValueExt for T {}

/// Compute the aligned length for `len` given an alignment.
#[inline]
pub const fn aligned_len(len: usize, alignment: usize) -> usize {
    if alignment <= 1 {
        len
    } else {
        len.next_multiple_of(alignment)
    }
}

/// Return widths aligned to `alignment`.
pub fn aligned_widths(widths: impl IntoIterator<Item = usize>, alignment: usize) -> Vec<usize> {
    widths
        .into_iter()
        .map(|w| aligned_len(w, alignment))
        .collect()
}

/// Pad a row with `Default::default()` so its length is a multiple of `alignment`.
///
/// This is a formatting convention for transcript hints; LMCS does not enforce that
/// padded values are zero unless the caller checks them.
pub fn pad_row_to_alignment<F: Default>(mut row: Vec<F>, alignment: usize) -> Vec<F> {
    debug_assert!(alignment > 0, "alignment must be non-zero");
    let padded_len = aligned_len(row.len(), alignment);
    row.resize_with(padded_len, || F::default());
    row
}

/// Pad a vector of rows with `Default::default()` so each row length is a multiple of `alignment`.
///
/// This is a formatting convention for transcript hints; LMCS does not enforce that
/// padded values are zero unless the caller checks them.
pub fn pad_rows_to_alignment<F: Default>(mut rows: Vec<Vec<F>>, alignment: usize) -> Vec<Vec<F>> {
    debug_assert!(alignment > 0, "alignment must be non-zero");
    for row in &mut rows {
        let padded_len = aligned_len(row.len(), alignment);
        row.resize_with(padded_len, || F::default());
    }
    rows
}

/// Compute a leaf digest from row slices and a salt.
///
/// This is a zero-copy helper for callers that already have row slices
/// and don't want to materialize an owned opening.
///
/// Note: only the provided row elements are absorbed. Any alignment padding must be
/// included in `rows` by the caller. LMCS does not enforce that padded values are zero.
pub(crate) fn digest_rows_and_salt<'a, F, D, H, const WIDTH: usize, const DIGEST_ELEMS: usize>(
    sponge: &H,
    rows: impl IntoIterator<Item = &'a [F]>,
    salt: &[F],
) -> [D; DIGEST_ELEMS]
where
    F: Copy + 'a,
    D: Default + Copy,
    H: StatefulHasher<F, [D; DIGEST_ELEMS], State = [D; WIDTH]>,
{
    let mut state = [D::default(); WIDTH];
    for row in rows {
        sponge.absorb_into(&mut state, row.iter().copied());
    }
    if !salt.is_empty() {
        sponge.absorb_into(&mut state, salt.iter().copied());
    }
    sponge.squeeze(&state)
}

/// Upsample matrix to exactly `target_height` rows via nearest-neighbor repetition.
///
/// Each original row is repeated `target_height / height` times.
/// Requires `target_height >= height` and both be powers of two.
///
/// This is the explicit form of the "lifting" operation used in LMCS, where smaller
/// matrices are virtually extended to match the height of the tallest matrix.
pub fn upsample_matrix<F: Clone + Send + Sync>(
    matrix: &impl Matrix<F>,
    target_height: usize,
) -> RowMajorMatrix<F> {
    let height = matrix.height();
    debug_assert!(target_height >= height);
    debug_assert!(height.is_power_of_two() && target_height.is_power_of_two());

    let repeat_factor = target_height / height;
    let width = matrix.width();

    let mut values = Vec::with_capacity(target_height * width);
    for row in matrix.rows() {
        let row_vec: Vec<F> = row.collect();
        for _ in 0..repeat_factor {
            values.extend(row_vec.iter().cloned());
        }
    }

    RowMajorMatrix::new(values, width)
}
