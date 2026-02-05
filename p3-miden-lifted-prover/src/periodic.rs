//! Prover-side periodic column handling.
//!
//! Periodic columns are stored as LDE values in a row-major matrix for efficient
//! constraint evaluation on the LDE domain. The key optimization is that a periodic
//! column with period `p` only needs `p * blowup` LDE values (not `trace_height * blowup`),
//! which are accessed via modular indexing.
//!
//! Uses NaiveDft since periodic column periods are typically small.

extern crate alloc;

use alloc::vec::Vec;

use p3_dft::{NaiveDft, TwoAdicSubgroupDft};
use p3_field::{PackedValue, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_lifted_stark::LiftedCoset;
use p3_util::log2_strict_usize;

/// Prover-side periodic LDE values for constraint evaluation.
///
/// Stores precomputed LDE values as a row-major matrix in natural order. The key insight
/// is that by repeating each column's values to the maximum period, we can use batch DFT
/// methods and store only `max_period * blowup` rows instead of `trace_height * blowup`.
#[derive(Clone, Debug)]
pub struct PeriodicLde<F: TwoAdicField> {
    /// LDE values in natural order (height = max_period * blowup).
    ldes: RowMajorMatrix<F>,
}

impl<F: TwoAdicField> PeriodicLde<F> {
    /// Build periodic LDEs from column evaluations (called by LiftedCoset::periodic_lde).
    ///
    /// Uses NaiveDft since periodic column periods are typically small.
    /// The coset shift is derived from the LiftedCoset's lde_shift().
    ///
    /// # Arguments
    /// - `coset`: The lifted coset providing domain information
    /// - `column_evals`: Periodic column evaluations, each on its respective subgroup
    ///
    /// # Returns
    /// `None` if any column has invalid length (zero or not power of two).
    pub fn build(coset: &LiftedCoset, column_evals: Vec<Vec<F>>) -> Option<Self> {
        if column_evals.is_empty() {
            return Some(Self {
                ldes: RowMajorMatrix::new(Vec::new(), 0),
            });
        }

        // Step 1: Find max period and validate all columns
        let mut max_period = 0;
        for column in &column_evals {
            let period = column.len();
            if period == 0 || !period.is_power_of_two() {
                return None;
            }
            max_period = max_period.max(period);
        }

        let num_columns = column_evals.len();
        let log_max_period = log2_strict_usize(max_period);
        let log_blowup = coset.log_blowup();

        // Step 2: Build matrix where each column is repeated to max_period
        // Row-major: values[row * num_columns + col]
        let mut repeated_values = Vec::with_capacity(max_period * num_columns);
        for row in 0..max_period {
            for column in &column_evals {
                repeated_values.push(column[row % column.len()]);
            }
        }
        let repeated_matrix = RowMajorMatrix::new(repeated_values, num_columns);

        // Step 3: Compute shift for max_period coset: g^r × (trace_height / max_period)
        // The coset shift is g^r. For a periodic column with period p,
        // we need shift (g^r)^(trace_height / p) = g^(r * trace_height / p).
        // Since lde_shift = g^r and trace_height = 2^log_trace_height:
        let log_ratio = coset.log_trace_height - log_max_period;
        let period_shift: F = coset.lde_shift::<F>().exp_power_of_2(log_ratio);

        // Step 4: Compute LDE using NaiveDft (periods are small)
        let ldes = NaiveDft
            .coset_lde_batch(repeated_matrix, log_blowup, period_shift)
            .to_row_major_matrix();

        Some(Self { ldes })
    }

    /// Get periodic values at natural index `i` (point g·ω^i).
    ///
    /// Returns F values only - conversion to EF happens in the folder if needed.
    /// Returns an empty slice iterator if there are no periodic columns.
    #[inline]
    pub fn values_at(&self, i: usize) -> core::iter::Copied<core::slice::Iter<'_, F>> {
        let height = self.ldes.height();
        let num_cols = self.ldes.width();
        // Handle empty case: return empty slice iterator
        if height == 0 || num_cols == 0 {
            return [].iter().copied();
        }
        let row = i % height;
        // Access values directly from row-major storage
        let start = row * num_cols;
        let end = start + num_cols;
        self.ldes.values[start..end].iter().copied()
    }

    /// Get packed values for consecutive natural indices [i, i+1, ..., i+WIDTH-1].
    #[inline]
    pub fn packed_values_at<P: PackedValue<Value = F>>(
        &self,
        i: usize,
    ) -> impl Iterator<Item = P> + '_ {
        let num_cols = self.ldes.width();
        let height = self.ldes.height();
        (0..num_cols).map(move |col| {
            P::from_fn(|k| {
                let row = (i + k) % height;
                self.ldes.values[row * num_cols + col]
            })
        })
    }

    /// Number of periodic columns.
    pub fn num_columns(&self) -> usize {
        self.ldes.width()
    }

    /// Check if there are no periodic columns.
    pub fn is_empty(&self) -> bool {
        self.ldes.width() == 0
    }

    /// Height of the LDE matrix (max_period * blowup).
    pub fn height(&self) -> usize {
        self.ldes.height()
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;

    use p3_dft::TwoAdicSubgroupDft;
    use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
    use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;

    type F = bb::F;

    /// Verify that `PeriodicLde::values_at()` matches the full LDE computation.
    ///
    /// Builds the full periodic column by repeating values to trace height,
    /// computes LDE via `coset_lde_batch`, and compares against `values_at()`.
    fn assert_periodic_lde_matches_full(
        columns: &[Vec<F>],
        log_trace_height: usize,
        log_blowup: usize,
    ) {
        let trace_height = 1 << log_trace_height;
        let lde_height = trace_height << log_blowup;
        let log_lde_height = log_trace_height + log_blowup;

        // Create a coset at max height (no lifting)
        let coset = LiftedCoset::new(log_trace_height, log_lde_height, log_lde_height);

        let periodic_lde =
            PeriodicLde::build(&coset, columns.to_vec()).expect("construction failed");

        // Compute expected LDE for each column via full expansion (natural order)
        let expected: Vec<Vec<F>> = columns
            .iter()
            .map(|col| {
                let full: Vec<F> = (0..trace_height).map(|i| col[i % col.len()]).collect();
                let matrix = RowMajorMatrix::new(full, 1);
                NaiveDft
                    .coset_lde_batch(matrix, log_blowup, F::GENERATOR)
                    .to_row_major_matrix()
                    .values
            })
            .collect();

        // Verify all LDE rows match (natural indices)
        for i in 0..lde_height {
            let actual: Vec<F> = periodic_lde.values_at(i).collect();
            for (col_idx, (&actual_val, expected_col)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    actual_val, expected_col[i],
                    "col {col_idx} mismatch at row {i}"
                );
            }
        }

        // Verify packed_values_at returns correct packed values
        type P = bb::P;
        let pack_width = P::WIDTH;
        for start in (0..lde_height).step_by(pack_width) {
            let packed: Vec<P> = periodic_lde.packed_values_at(start).collect();
            assert_eq!(packed.len(), columns.len());

            // Verify each lane matches values_at
            for k in 0..pack_width {
                let row = start + k;
                let scalar: Vec<F> = periodic_lde.values_at(row).collect();
                for (col_idx, (&packed_val, &scalar_val)) in packed.iter().zip(&scalar).enumerate()
                {
                    assert_eq!(
                        packed_val.as_slice()[k],
                        scalar_val,
                        "packed mismatch col {col_idx} row {row} lane {k}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_periodic_lde_matches_full_lde() {
        // Period 2, blowup 2
        assert_periodic_lde_matches_full(&[vec![F::ZERO, F::ONE]], 3, 1);

        // Period 4, blowup 2
        let col4: Vec<F> = [1, 2, 3, 4].into_iter().map(F::from_u64).collect();
        assert_periodic_lde_matches_full(&[col4], 3, 1);

        // Period 2, blowup 8 (higher blowup)
        let col2: Vec<F> = [5, 7].into_iter().map(F::from_u64).collect();
        assert_periodic_lde_matches_full(&[col2], 4, 3);

        // Multiple columns with different periods
        let col_p2: Vec<F> = [1, 2].into_iter().map(F::from_u64).collect();
        let col_p4: Vec<F> = [10, 20, 30, 40].into_iter().map(F::from_u64).collect();
        assert_periodic_lde_matches_full(&[col_p2, col_p4], 3, 2);
    }
}
