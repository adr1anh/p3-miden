//! Trace commitment (LDE + bit-reverse + LMCS).
//!
//! This module provides types and functions for committing traces with lifting support:
//!
//! - [`commit_traces`]: Commit traces with lifting support (LDE → bit-reverse → LMCS)
//! - [`Committed`]: Wrapper around LMCS tree with domain metadata
//! - [`CommittedMatrixView`]: View into committed matrix with domain metadata

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_dft::TwoAdicSubgroupDft;
use p3_field::TwoAdicField;
use p3_matrix::Matrix;
use p3_matrix::bitrev::{BitReversedMatrixView, BitReversibleMatrix};
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix, RowMajorMatrixView};
use p3_miden_lmcs::{Lmcs, LmcsTree};
use p3_util::log2_strict_usize;

use crate::StarkConfig;
use p3_miden_lifted_stark::LiftedCoset;

// ============================================================================
// Committed
// ============================================================================

/// Committed polynomial evaluations with domain metadata.
///
/// Wraps an LMCS tree and stores the blowup factor used during commitment.
/// This enables reconstructing domain information for each committed matrix
/// without re-deriving it from the configuration.
///
/// # Type Parameters
///
/// - `F`: Scalar field element type
/// - `M`: Matrix type (e.g., `RowMajorMatrix<F>`)
/// - `L`: LMCS configuration type
///
/// # Usage
///
/// ```ignore
/// let committed = commit_traces(config, traces);
/// let root = committed.root();
/// let domain = committed.domain(0);  // Domain info for first matrix
/// let matrix = committed.matrix(0);  // Reference to first LDE matrix
/// ```
pub struct Committed<F, M, L>
where
    F: TwoAdicField,
    L: Lmcs<F = F>,
    M: Matrix<F>,
{
    /// The underlying LMCS tree.
    tree: L::Tree<M>,
    /// Log₂ of the blowup factor used during LDE.
    log_blowup: usize,
}

impl<F, M, L> Committed<F, M, L>
where
    F: TwoAdicField,
    L: Lmcs<F = F>,
    M: Matrix<F>,
{
    /// Create a new `Committed` wrapper.
    ///
    /// # Arguments
    ///
    /// - `tree`: The LMCS tree containing committed LDE matrices
    /// - `log_blowup`: Log₂ of the blowup factor used during LDE
    #[inline]
    pub fn new(tree: L::Tree<M>, log_blowup: usize) -> Self {
        Self { tree, log_blowup }
    }

    /// Get the commitment root.
    #[inline]
    pub fn root(&self) -> L::Commitment {
        self.tree.root()
    }

    /// Get a reference to the underlying tree.
    #[inline]
    pub fn tree(&self) -> &L::Tree<M> {
        &self.tree
    }

    /// Consume self and return the underlying tree.
    #[inline]
    pub fn into_tree(self) -> L::Tree<M> {
        self.tree
    }

    /// Get the number of committed matrices.
    #[inline]
    pub fn num_matrices(&self) -> usize {
        self.tree.leaves().len()
    }

    /// Get log₂ of the maximum LDE height across all matrices.
    ///
    /// This is the height of the tree (the largest matrix height).
    #[inline]
    pub fn log_max_lde_height(&self) -> usize {
        log2_strict_usize(self.tree.height())
    }

    /// Get log₂ of the blowup factor.
    #[inline]
    pub fn log_blowup(&self) -> usize {
        self.log_blowup
    }

    /// Get the domain information for the matrix at index `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= num_matrices()`.
    pub fn domain(&self, i: usize) -> LiftedCoset {
        let matrix = &self.tree.leaves()[i];
        let log_lde_height = log2_strict_usize(matrix.height());
        let log_trace_height = log_lde_height - self.log_blowup;
        let log_max_lde_height = self.log_max_lde_height();

        LiftedCoset::new(log_trace_height, log_lde_height, log_max_lde_height)
    }

    /// Get a reference to the matrix at index `i`.
    ///
    /// The matrix contains LDE evaluations in bit-reversed order.
    ///
    /// # Panics
    ///
    /// Panics if `i >= num_matrices()`.
    #[inline]
    pub fn matrix(&self, i: usize) -> &M {
        &self.tree.leaves()[i]
    }

    /// Get references to all committed matrices.
    #[inline]
    pub fn matrices(&self) -> &[M] {
        self.tree.leaves()
    }

    /// Get a view of the matrix at index `i` with its domain metadata.
    ///
    /// # Panics
    ///
    /// Panics if `i >= num_matrices()`.
    pub fn matrix_view(&self, i: usize) -> CommittedMatrixView<'_, F, M> {
        CommittedMatrixView {
            matrix: self.matrix(i),
            domain: self.domain(i),
            _phantom: PhantomData,
        }
    }

    /// Get views of all committed matrices with their domain metadata.
    pub fn matrix_views(&self) -> impl Iterator<Item = CommittedMatrixView<'_, F, M>> {
        (0..self.num_matrices()).map(move |i| self.matrix_view(i))
    }
}

// ============================================================================
// CommittedMatrixView
// ============================================================================

/// View into a committed matrix with domain metadata.
///
/// Provides access to the committed LDE evaluations (in bit-reversed order)
/// along with the domain information needed for operations like truncation.
///
/// # Type Parameters
///
/// - `F`: Scalar field element type
/// - `M`: Matrix type (e.g., `RowMajorMatrix<F>`)
pub struct CommittedMatrixView<'a, F, M>
where
    F: Clone + Send + Sync,
    M: Matrix<F>,
{
    matrix: &'a M,
    /// Domain information for this matrix.
    pub domain: LiftedCoset,
    _phantom: PhantomData<F>,
}

impl<'a, F, M> CommittedMatrixView<'a, F, M>
where
    F: Clone + Send + Sync,
    M: Matrix<F>,
{
    /// Get a reference to the underlying matrix (bit-reversed order).
    #[inline]
    pub fn as_bit_reversed(&self) -> &'a M {
        self.matrix
    }

    /// Get the height of the LDE matrix.
    #[inline]
    pub fn height(&self) -> usize {
        self.matrix.height()
    }

    /// Get the width of the matrix (number of columns).
    #[inline]
    pub fn width(&self) -> usize {
        self.matrix.width()
    }
}

impl<'a, F> CommittedMatrixView<'a, F, DenseMatrix<F>>
where
    F: Clone + Send + Sync,
{
    /// Truncate to the first `n` rows (in bit-reversed order).
    ///
    /// This is a zero-copy operation that returns a view into the first `n` rows.
    /// Use this for extracting nested cosets from a committed LDE.
    ///
    /// # Panics
    ///
    /// Panics if `n > self.height()`.
    #[inline]
    pub fn truncate(&self, n: usize) -> RowMajorMatrixView<'a, F> {
        self.matrix.split_rows(n).0
    }

    /// Truncate to the quotient domain `gJ` of size `trace_height × constraint_degree`.
    ///
    /// When constraint evaluation produces a polynomial of degree `N × D - 1`
    /// (where N is trace height and D is constraint degree), the quotient
    /// polynomial has degree at most `(N × D - 1) - (N - 1) = N × (D - 1)`.
    /// After dividing by the vanishing polynomial, evaluations on the first
    /// `N × D` points of the LDE domain suffice.
    ///
    /// # Arguments
    ///
    /// - `constraint_degree`: The maximum constraint degree `D` (typically 2-3)
    ///
    /// # Returns
    ///
    /// A view of the first `trace_height × constraint_degree` rows (bit-reversed).
    #[inline]
    pub fn truncate_to_quotient_domain(
        &self,
        constraint_degree: usize,
    ) -> RowMajorMatrixView<'a, F> {
        let quotient_height = self.domain.trace_height() * constraint_degree;
        self.truncate(quotient_height)
    }

    /// Get a natural-order view of the full LDE matrix.
    ///
    /// This is a zero-copy operation that wraps the bit-reversed matrix
    /// with an index transformation to provide natural-order access.
    #[inline]
    pub fn to_natural_order(&self) -> BitReversedMatrixView<RowMajorMatrixView<'a, F>> {
        self.matrix.as_view().bit_reverse_rows()
    }

    /// Truncate and convert to natural order in one operation.
    ///
    /// Equivalent to `truncate(n).bit_reverse_rows()` but expresses the
    /// intent more clearly: extract a nested coset and access it in natural order.
    ///
    /// # Panics
    ///
    /// Panics if `n > self.height()`.
    #[inline]
    pub fn truncate_natural(&self, n: usize) -> BitReversedMatrixView<RowMajorMatrixView<'a, F>> {
        self.truncate(n).bit_reverse_rows()
    }

    /// Truncate to quotient domain and convert to natural order.
    ///
    /// Returns a zero-copy view of the first `trace_height × constraint_degree` rows
    /// in natural order, suitable for constraint evaluation.
    #[inline]
    pub fn quotient_domain_natural(
        &self,
        constraint_degree: usize,
    ) -> BitReversedMatrixView<RowMajorMatrixView<'a, F>> {
        self.truncate_to_quotient_domain(constraint_degree)
            .bit_reverse_rows()
    }
}

// ============================================================================
// commit_traces
// ============================================================================

/// Commit multiple trace matrices with lifting: LDE → bit-reverse → LMCS tree.
///
/// Traces must be sorted by height in ascending order. Each trace is lifted to
/// the max LDE domain using the appropriate nested coset shift.
///
/// Returns a [`Committed`] wrapper providing:
/// - Access to the underlying LMCS tree via [`Committed::tree()`]
/// - Access to committed LDE matrices via [`Committed::matrix(i)`]
/// - Domain metadata via [`Committed::domain(i)`]
///
/// # Arguments
/// - `config`: STARK configuration containing PCS params, LMCS, and DFT
/// - `traces`: Trace matrices sorted by height (ascending)
///
/// # Panics
/// - If `traces` is empty
/// - If trace heights are not powers of two
/// - If traces are not sorted by height in ascending order
pub fn commit_traces<F, L, Dft>(
    config: &StarkConfig<L, Dft>,
    traces: Vec<RowMajorMatrix<F>>,
) -> Committed<F, RowMajorMatrix<F>, L>
where
    F: TwoAdicField,
    L: Lmcs<F = F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    assert!(!traces.is_empty(), "at least one trace required");

    // Validate traces are sorted by height
    assert!(
        traces.windows(2).all(|w| w[0].height() <= w[1].height()),
        "traces must be sorted by height in ascending order"
    );

    let log_blowup = config.pcs.fri.log_blowup;

    // Find max trace height and compute max LDE height
    let max_trace_height = traces.last().unwrap().height();
    let log_max_trace_height = log2_strict_usize(max_trace_height);
    let log_max_lde_height = log_max_trace_height + log_blowup;

    let ldes: Vec<_> = traces
        .into_iter()
        .enumerate()
        .map(|(idx, trace)| {
            let trace_height = trace.height();

            // Validate height is power of two
            assert!(
                trace_height.is_power_of_two(),
                "trace height must be power of two (index {idx})"
            );

            let log_trace_height = log2_strict_usize(trace_height);
            let log_lde_height = log_trace_height + log_blowup;

            // Compute lift ratio: how many times smaller this trace is vs max
            // r = max_height / trace_height = 2^(log_max - log_trace)
            let log_lift_ratio = log_max_lde_height - log_lde_height;

            // Coset shift for this trace: g^r where r = 2^log_lift_ratio
            // This places the LDE on the nested coset (gK)^r
            let coset_shift = F::GENERATOR.exp_power_of_2(log_lift_ratio);

            // Compute coset LDE and bit-reverse rows
            config
                .dft
                .coset_lde_batch(trace, log_blowup, coset_shift)
                .bit_reverse_rows()
                .to_row_major_matrix()
        })
        .collect();

    // Build aligned LMCS tree and wrap in Committed
    let tree = config.lmcs.build_aligned_tree(ldes);
    Committed::new(tree, log_blowup)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    type F = BabyBear;

    #[test]
    fn committed_matrix_view_truncate() {
        use p3_matrix::dense::RowMajorMatrix;

        // Create a 16x4 matrix (LDE height = 16, width = 4)
        let data: Vec<F> = (0u64..64).map(F::from_u64).collect();
        let matrix = RowMajorMatrix::new(data, 4);

        // Domain: trace_height = 4, lde_height = 16, max = 16 (blowup = 4)
        let domain = LiftedCoset::new(2, 4, 4);

        let view = CommittedMatrixView {
            matrix: &matrix,
            domain,
            _phantom: PhantomData,
        };

        assert_eq!(view.height(), 16);
        assert_eq!(view.width(), 4);

        // Truncate to 8 rows
        let truncated = view.truncate(8);
        assert_eq!(truncated.height(), 8);
        assert_eq!(truncated.width(), 4);

        // Verify first row is unchanged
        let row: Vec<F> = truncated.row(0).unwrap().into_iter().collect();
        assert_eq!(
            row,
            vec![
                F::from_u64(0),
                F::from_u64(1),
                F::from_u64(2),
                F::from_u64(3)
            ]
        );
    }

    #[test]
    fn committed_matrix_view_truncate_to_quotient_domain() {
        use p3_matrix::dense::RowMajorMatrix;

        // Create a 32x2 matrix (LDE height = 32)
        let data: Vec<F> = (0u64..64).map(F::from_u64).collect();
        let matrix = RowMajorMatrix::new(data, 2);

        // Domain: trace_height = 8, lde_height = 32 (blowup = 4)
        let domain = LiftedCoset::new(3, 5, 5);

        let view = CommittedMatrixView {
            matrix: &matrix,
            domain,
            _phantom: PhantomData,
        };

        // Truncate to quotient domain with constraint_degree = 2
        // Should give trace_height * 2 = 8 * 2 = 16 rows
        let quotient_view = view.truncate_to_quotient_domain(2);
        assert_eq!(quotient_view.height(), 16);
        assert_eq!(quotient_view.width(), 2);
    }

    #[test]
    fn committed_matrix_view_natural_order() {
        use p3_matrix::dense::RowMajorMatrix;
        use p3_util::reverse_bits_len;

        // Create an 8x2 matrix with values that let us verify bit-reversal
        // Row i (bit-reversed) contains [2*i, 2*i+1]
        let data: Vec<F> = (0u64..16).map(F::from_u64).collect();
        let matrix = RowMajorMatrix::new(data, 2);

        // Domain: trace_height = 2, lde_height = 8 (blowup = 4)
        let domain = LiftedCoset::new(1, 3, 3);

        let view = CommittedMatrixView {
            matrix: &matrix,
            domain,
            _phantom: PhantomData,
        };

        // Get natural-order view
        let natural = view.to_natural_order();
        assert_eq!(natural.height(), 8);
        assert_eq!(natural.width(), 2);

        // In bit-reversed storage, row 0 is at natural index 0
        // In bit-reversed storage, row 1 is at natural index 4 (reverse of 001 is 100)
        // In bit-reversed storage, row 2 is at natural index 2 (reverse of 010 is 010)
        // In bit-reversed storage, row 3 is at natural index 6 (reverse of 011 is 110)
        // etc.
        //
        // So natural row 0 should come from bit-reversed row 0: [0, 1]
        // Natural row 1 should come from bit-reversed row 4: [8, 9]
        // Natural row 2 should come from bit-reversed row 2: [4, 5]
        // Natural row 3 should come from bit-reversed row 6: [12, 13]
        // Natural row 4 should come from bit-reversed row 1: [2, 3]
        // etc.

        // Verify natural row 0 (from bit-reversed row 0)
        let row0: Vec<F> = natural.row(0).unwrap().into_iter().collect();
        assert_eq!(row0, vec![F::from_u64(0), F::from_u64(1)]);

        // Verify natural row 1 (from bit-reversed row 4)
        let row1: Vec<F> = natural.row(1).unwrap().into_iter().collect();
        assert_eq!(row1, vec![F::from_u64(8), F::from_u64(9)]);

        // Verify natural row 4 (from bit-reversed row 1)
        let row4: Vec<F> = natural.row(4).unwrap().into_iter().collect();
        assert_eq!(row4, vec![F::from_u64(2), F::from_u64(3)]);

        // General verification: natural row i should have values from bit-reversed row bitrev(i)
        for i in 0..8 {
            let br_i = reverse_bits_len(i, 3);
            let natural_row: Vec<F> = natural.row(i).unwrap().into_iter().collect();
            let expected: Vec<F> = vec![
                F::from_u64((br_i * 2) as u64),
                F::from_u64((br_i * 2 + 1) as u64),
            ];
            assert_eq!(natural_row, expected, "mismatch at natural row {i}");
        }
    }

    #[test]
    fn committed_matrix_view_truncate_natural() {
        use p3_matrix::dense::RowMajorMatrix;

        // Create a 16x2 matrix
        let data: Vec<F> = (0u64..32).map(F::from_u64).collect();
        let matrix = RowMajorMatrix::new(data, 2);

        // Domain: trace_height = 4, lde_height = 16 (blowup = 4)
        let domain = LiftedCoset::new(2, 4, 4);

        let view = CommittedMatrixView {
            matrix: &matrix,
            domain,
            _phantom: PhantomData,
        };

        // Truncate to 8 rows and convert to natural order
        let truncated_natural = view.truncate_natural(8);
        assert_eq!(truncated_natural.height(), 8);
        assert_eq!(truncated_natural.width(), 2);

        // This should be equivalent to truncate(8).bit_reverse_rows()
        let truncated = view.truncate(8);
        let expected = truncated.bit_reverse_rows();

        for i in 0..8 {
            let actual_row: Vec<F> = truncated_natural.row(i).unwrap().into_iter().collect();
            let expected_row: Vec<F> = expected.row(i).unwrap().into_iter().collect();
            assert_eq!(actual_row, expected_row, "mismatch at row {i}");
        }
    }
}
