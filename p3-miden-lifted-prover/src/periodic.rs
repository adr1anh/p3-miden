//! Prover-side periodic column handling.
//!
//! Periodic columns are stored as LDE values in a row-major matrix for efficient
//! constraint evaluation on the LDE domain. The key optimization is that a periodic
//! column with period `p` only needs `p * blowup` LDE values (not `trace_height * blowup`),
//! which are accessed via modular indexing.

extern crate alloc;

use alloc::vec::Vec;

use p3_dft::TwoAdicSubgroupDft;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{ExtensionField, PackedValue, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_lmcs::Lmcs;
use p3_util::{log2_strict_usize, reverse_bits_len};

use p3_miden_lifted_stark::StarkConfig;

/// Prover-side periodic LDE values for constraint evaluation.
///
/// Stores precomputed LDE values as a row-major matrix. The key insight is that
/// by repeating each column's values to the maximum period, we can use batch DFT
/// methods and store only `max_period * blowup` rows instead of `trace_height * blowup`.
#[derive(Clone, Debug)]
pub struct PeriodicLde<F: TwoAdicField> {
    /// LDE values as a row-major matrix (height = max_period * blowup).
    ldes: RowMajorMatrix<F>,
    /// The target LDE coset we're lifting toward.
    target_coset: TwoAdicMultiplicativeCoset<F>,
}

impl<F: TwoAdicField> PeriodicLde<F> {
    /// Build periodic LDEs from column evaluations.
    ///
    /// Each column in `column_evals` contains evaluations on a subgroup of the given period.
    /// These are converted to LDE values on a coset of size `max_period * blowup`.
    ///
    /// # Arguments
    /// - `config`: STARK configuration providing DFT and blowup factor
    /// - `column_evals`: Periodic column evaluations, each on its respective subgroup
    /// - `target_lde_coset`: The LDE coset we're lifting toward (e.g., `gK` with size `trace_height * blowup`)
    ///
    /// # Returns
    /// `None` if any column has invalid length (zero or not power of two).
    pub fn new<L, Dft>(
        config: &StarkConfig<L, Dft>,
        column_evals: Vec<Vec<F>>,
        target_lde_coset: TwoAdicMultiplicativeCoset<F>,
    ) -> Option<Self>
    where
        L: Lmcs<F = F>,
        Dft: TwoAdicSubgroupDft<F>,
    {
        if column_evals.is_empty() {
            return Some(Self {
                ldes: RowMajorMatrix::new(Vec::new(), 0),
                target_coset: target_lde_coset,
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
        let log_blowup = config.pcs.fri.log_blowup;

        // Step 2: Build matrix where each column is repeated to max_period
        // Row-major: values[row * num_columns + col]
        let mut repeated_values = Vec::with_capacity(max_period * num_columns);
        for row in 0..max_period {
            for column in &column_evals {
                let period = column.len();
                repeated_values.push(column[row % period]);
            }
        }
        let repeated_matrix = RowMajorMatrix::new(repeated_values, num_columns);

        // Step 3: Compute shift for max_period coset: g^(trace_height / max_period)
        // The target coset is gK of size trace_height * blowup
        // We need a coset of size max_period * blowup with shift g^(trace_height / max_period)
        let log_lde_height = target_lde_coset.log_size();
        let log_trace_height = log_lde_height - log_blowup;
        let log_ratio = log_trace_height - log_max_period;
        let period_shift = target_lde_coset.shift().exp_power_of_2(log_ratio);

        // Step 4: Compute LDE on the period coset using coset_lde_batch
        // Store in natural order; values_at handles bit-reversed index conversion
        let ldes = config
            .dft
            .coset_lde_batch(repeated_matrix, log_blowup, period_shift)
            .to_row_major_matrix();

        Some(Self {
            ldes,
            target_coset: target_lde_coset,
        })
    }

    /// Get the periodic values at the given bit-reversed LDE row index.
    ///
    /// Index `i` corresponds to the evaluation point `g * w^{bitrev(i)}` on the target
    /// LDE coset, matching the convention used for trace LDEs. Internally converts
    /// from bit-reversed to natural order and applies modular wrapping.
    #[inline]
    pub fn values_at(&self, i: usize) -> impl Iterator<Item = F> + '_ {
        let row = self.to_natural_periodic_index(i);
        self.ldes.row(row).unwrap().into_iter()
    }

    /// Convert a bit-reversed LDE index to the corresponding natural-order periodic index.
    #[inline]
    fn to_natural_periodic_index(&self, i: usize) -> usize {
        let log_lde_height = self.target_coset.log_size();
        let natural_i = reverse_bits_len(i, log_lde_height);
        natural_i % self.ldes.height()
    }

    /// Get the periodic values at the given row as extension field elements.
    pub fn values_at_ext<EF>(&self, row: usize) -> Vec<EF>
    where
        EF: ExtensionField<F>,
    {
        if self.ldes.width() == 0 {
            return Vec::new();
        }
        self.values_at(row).map(EF::from).collect()
    }

    /// Get packed periodic values starting at the given bit-reversed index.
    ///
    /// Returns one packed value per column. Each packed value contains `P::WIDTH`
    /// consecutive values at bit-reversed indices `[i, i+1, ..., i+P::WIDTH-1]`,
    /// corresponding to evaluation points `g * w^{bitrev(i+k)}` for `k` in `0..P::WIDTH`.
    #[inline]
    pub fn packed_values_at<P>(&self, i: usize) -> impl Iterator<Item = P> + '_
    where
        P: PackedValue<Value = F>,
    {
        let num_cols = self.ldes.width();
        let height = self.ldes.height();
        let log_lde_height = self.target_coset.log_size();
        (0..num_cols).map(move |col| {
            P::from_fn(|k| {
                let natural_idx = reverse_bits_len(i + k, log_lde_height);
                let row = natural_idx % height;
                // Row-major: values[row * num_cols + col]
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

    /// The target LDE coset.
    pub fn target_coset(&self) -> &TwoAdicMultiplicativeCoset<F> {
        &self.target_coset
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;

    use p3_dft::{NaiveDft, TwoAdicSubgroupDft};
    use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
    use p3_matrix::bitrev::BitReversibleMatrix;
    use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
    use p3_miden_lifted_fri::PcsParams;
    use p3_miden_lifted_fri::deep::DeepParams;
    use p3_miden_lifted_fri::fri::{FriFold, FriParams};
    use p3_miden_lmcs::LmcsConfig;

    type F = bb::F;
    type TestLmcs =
        LmcsConfig<bb::P, bb::P, bb::Sponge, bb::Compress, { bb::WIDTH }, { bb::DIGEST }>;

    fn test_config(log_blowup: usize) -> StarkConfig<TestLmcs, NaiveDft> {
        let (_, sponge, compress) = bb::test_components();
        let lmcs: TestLmcs = LmcsConfig::new(sponge, compress);
        StarkConfig {
            pcs: PcsParams {
                fri: FriParams {
                    log_blowup,
                    fold: FriFold::ARITY_2,
                    log_final_degree: 0,
                    proof_of_work_bits: 0,
                },
                deep: DeepParams {
                    proof_of_work_bits: 0,
                },
                num_queries: 1,
                query_proof_of_work_bits: 0,
            },
            lmcs,
            dft: NaiveDft,
        }
    }

    /// Verify that `PeriodicLde::values_at()` matches the full LDE computation.
    ///
    /// Builds the full periodic column by repeating values to trace height,
    /// computes LDE via `coset_lde_batch`, and compares against `values_at()`.
    fn assert_periodic_lde_matches_full(
        columns: &[Vec<F>],
        log_trace_height: usize,
        log_blowup: usize,
    ) {
        let config = test_config(log_blowup);
        let trace_height = 1 << log_trace_height;
        let lde_height = trace_height << log_blowup;
        let target_coset =
            TwoAdicMultiplicativeCoset::new(F::GENERATOR, log_trace_height + log_blowup)
                .expect("valid coset");

        let periodic_lde =
            PeriodicLde::new(&config, columns.to_vec(), target_coset).expect("construction failed");

        // Compute expected LDE for each column via full expansion (bit-reversed)
        let expected: Vec<Vec<F>> = columns
            .iter()
            .map(|col| {
                let full: Vec<F> = (0..trace_height).map(|i| col[i % col.len()]).collect();
                let matrix = RowMajorMatrix::new(full, 1);
                NaiveDft
                    .coset_lde_batch(matrix, log_blowup, F::GENERATOR)
                    .bit_reverse_rows()
                    .to_row_major_matrix()
                    .values
            })
            .collect();

        // Verify all LDE rows match (bit-reversed indices)
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
