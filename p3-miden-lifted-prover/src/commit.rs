//! Trace commitment (LDE + bit-reverse + LMCS).
//!
//! This module provides types and functions for committing traces with lifting support:
//!
//! - [`commit_traces`]: Commit traces with lifting support (LDE → bit-reverse → LMCS)
//! - [`Committed`]: Wrapper around LMCS tree with domain metadata

use alloc::vec::Vec;

use p3_dft::TwoAdicSubgroupDft;
use p3_field::TwoAdicField;
use p3_matrix::Matrix;
use p3_matrix::bitrev::{BitReversedMatrixView, BitReversibleMatrix};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
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
/// let view = committed.quotient_domain_natural(0, constraint_degree);
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

    /// Get log₂ of the maximum LDE height across all matrices.
    ///
    /// This is the height of the tree (the largest matrix height).
    #[inline]
    fn log_max_lde_height(&self) -> usize {
        log2_strict_usize(self.tree.height())
    }

    /// Get the domain information for the matrix at index `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= num_matrices()`.
    fn domain(&self, i: usize) -> LiftedCoset {
        let matrix = &self.tree.leaves()[i];
        let log_lde_height = log2_strict_usize(matrix.height());
        let log_trace_height = log_lde_height - self.log_blowup;
        let log_max_trace_height = self.log_max_lde_height() - self.log_blowup;

        LiftedCoset::new(log_trace_height, self.log_blowup, log_max_trace_height)
    }
}

impl<F, L> Committed<F, RowMajorMatrix<F>, L>
where
    F: TwoAdicField,
    L: Lmcs<F = F>,
{
    /// Truncate matrix `i` to the quotient domain and return in natural order.
    ///
    /// The quotient domain has size `trace_height × constraint_degree`. This is
    /// the first `N × D` rows of the LDE in bit-reversed storage, wrapped for
    /// natural-order access. Zero-copy.
    ///
    /// # Panics
    ///
    /// Panics if `i >= num_matrices()`.
    pub fn quotient_domain_natural(
        &self,
        i: usize,
        constraint_degree: usize,
    ) -> BitReversedMatrixView<RowMajorMatrixView<'_, F>> {
        let quotient_height = self.domain(i).trace_height() * constraint_degree;
        self.tree.leaves()[i]
            .split_rows(quotient_height)
            .0
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
/// - Commitment root via [`Committed::root()`]
/// - Underlying LMCS tree via [`Committed::tree()`]
/// - Quotient domain views via [`Committed::quotient_domain_natural()`]
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

    // Find max trace height
    let max_trace_height = traces.last().unwrap().height();
    let log_max_trace_height = log2_strict_usize(max_trace_height);

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

            // Use LiftedCoset to compute the coset shift
            let coset = LiftedCoset::new(log_trace_height, log_blowup, log_max_trace_height);
            let coset_shift = coset.lde_shift::<F>();

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

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::Matrix;
    use p3_matrix::bitrev::BitReversibleMatrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_util::reverse_bits_len;

    type F = BabyBear;

    #[test]
    fn split_rows_truncates_correctly() {
        // Create a 16x4 matrix (LDE height = 16, width = 4)
        let data: Vec<F> = (0u64..64).map(F::from_u64).collect();
        let matrix = RowMajorMatrix::new(data, 4);

        // Truncate to 8 rows via split_rows
        let truncated = matrix.split_rows(8).0;
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
    fn bit_reverse_rows_gives_natural_order() {
        // Create an 8x2 matrix with values that let us verify bit-reversal
        // Row i (bit-reversed) contains [2*i, 2*i+1]
        let data: Vec<F> = (0u64..16).map(F::from_u64).collect();
        let matrix = RowMajorMatrix::new(data, 2);

        let natural = matrix.as_view().bit_reverse_rows();
        assert_eq!(natural.height(), 8);
        assert_eq!(natural.width(), 2);

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
    fn truncate_then_bit_reverse() {
        // Create a 16x2 matrix
        let data: Vec<F> = (0u64..32).map(F::from_u64).collect();
        let matrix = RowMajorMatrix::new(data, 2);

        // Truncate to 8 rows and convert to natural order
        let truncated_natural = matrix.split_rows(8).0.bit_reverse_rows();
        assert_eq!(truncated_natural.height(), 8);
        assert_eq!(truncated_natural.width(), 2);

        for i in 0..8 {
            let row: Vec<F> = truncated_natural.row(i).unwrap().into_iter().collect();
            assert_eq!(row.len(), 2, "row {i} should have 2 elements");
        }
    }
}
