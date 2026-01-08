use p3_util::{log2_strict_usize, reverse_bits_len};

use crate::Matrix;
use crate::dense::{DenseMatrix, DenseStorage, RowMajorMatrix};
use crate::row_index_mapped::{RowIndexMap, RowIndexMappedView};
use crate::util::reverse_matrix_index_bits;

/// A trait for matrices that support *bit-reversed row reordering*.
///
/// Implementers of this trait can switch between row-major order and bit-reversed
/// row order (i.e., reversing the binary representation of each row index).
///
/// This trait allows interoperability between regular matrices and views
/// that access their rows in a bit-reversed order.
pub trait BitReversibleMatrix<T: Send + Sync + Clone>: Matrix<T> {
    /// The type returned when this matrix is viewed in bit-reversed order.
    type BitRev: BitReversibleMatrix<T>;

    /// Return a version of the matrix with its row order reversed by bit index.
    fn bit_reverse_rows(self) -> Self::BitRev;
}

/// A row index permutation that reorders rows according to bit-reversed index.
///
/// Used internally to implement `BitReversedMatrixView`.
#[derive(Debug)]
pub struct BitReversalPerm {
    /// The logarithm (base 2) of the matrix height. For height `h`, this is `log2(h)`.
    ///
    /// This must be exact, so the height must be a power of two.
    log_height: usize,
}

impl BitReversalPerm {
    /// Create a new bit-reversal view over the given matrix.
    ///
    /// # Panics
    /// Panics if the height of the matrix is not a power of two.
    ///
    /// # Arguments
    /// - `inner`: The matrix to wrap in a bit-reversed row view.
    ///
    /// # Returns
    /// A `BitReversedMatrixView` that wraps the input with row reordering.
    pub fn new_view<T: Send + Sync + Clone, Inner: Matrix<T>>(
        inner: Inner,
    ) -> BitReversedMatrixView<Inner> {
        RowIndexMappedView {
            index_map: Self {
                log_height: log2_strict_usize(inner.height()),
            },
            inner,
        }
    }
}

impl RowIndexMap for BitReversalPerm {
    fn height(&self) -> usize {
        1 << self.log_height
    }

    fn map_row_index(&self, r: usize) -> usize {
        reverse_bits_len(r, self.log_height)
    }

    // This might not be more efficient than the lazy generic impl
    // if we have a nested view.
    fn to_row_major_matrix<T: Clone + Send + Sync, Inner: Matrix<T>>(
        &self,
        inner: Inner,
    ) -> RowMajorMatrix<T> {
        let mut inner = inner.to_row_major_matrix();
        reverse_matrix_index_bits(&mut inner);
        inner
    }
}

/// A matrix view that reorders its rows using bit-reversal.
///
/// This type is produced by applying `BitReversibleMatrix::bit_reverse_rows()`
/// to a `DenseMatrix`.
pub type BitReversedMatrixView<Inner> = RowIndexMappedView<BitReversalPerm, Inner>;

impl<T: Clone + Send + Sync, S: DenseStorage<T>> BitReversibleMatrix<T>
    for BitReversedMatrixView<DenseMatrix<T, S>>
{
    type BitRev = DenseMatrix<T, S>;

    fn bit_reverse_rows(self) -> Self::BitRev {
        self.inner
    }
}

impl<T: Clone + Send + Sync, S: DenseStorage<T>> BitReversibleMatrix<T> for DenseMatrix<T, S> {
    type BitRev = BitReversedMatrixView<Self>;

    fn bit_reverse_rows(self) -> Self::BitRev {
        BitReversalPerm::new_view(self)
    }
}
