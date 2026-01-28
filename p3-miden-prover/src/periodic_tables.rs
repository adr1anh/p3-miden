//! Efficient evaluation of periodic columns in STARKs.
//!
//! # Overview
//!
//! Periodic columns are trace columns that repeat with a period dividing the trace length.
//! Instead of committing to these columns, both prover and verifier compute them independently.
//! This module provides efficient algorithms to evaluate periodic columns at:
//! - All points in the quotient domain (during proving, via compact LDE tables)
//! - A single out-of-domain challenge point (during verification)
//!
//! # Mathematical Foundation
//!
//! For a periodic column with period `P` and trace height `N` where `P | N`:
//!
//! The column repeats `N/P` times over the trace domain. Instead of interpolating over the
//! full trace height `N`, we leverage this periodicity to interpolate only over the minimal
//! repeating cycle of size `P`.
//!
//! ## Key Insight
//!
//! For a two-adic trace domain, we can evaluate periodic columns on the quotient domain
//! by doing an LDE on the *period* (not the full trace), padding all columns to the maximum
//! period and storing only `max_period × blowup` rows. The prover can then retrieve values
//! for any quotient index via modular indexing, avoiding per-point interpolation.
//!
//! # Functions
//!
//! - [`compute_periodic_on_quotient_eval_domain`]: Evaluates periodic columns over the quotient domain
//!   into a compact LDE table. Called by the prover during quotient polynomial computation.
//! - [`evaluate_periodic_at_point`]: Evaluates periodic columns at a single challenge point.
//!   Called by the verifier to check constraint satisfaction.

use alloc::vec;
use alloc::vec::Vec;

use p3_commit::PolynomialSpace;
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::{ExtensionField, TwoAdicField, batch_multiplicative_inverse};
use p3_interpolation::interpolate_coset_with_precomputation;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;
use tracing::{debug_span, info_span};

/// Compact storage for periodic column values on the quotient domain.
///
/// Instead of materializing the full LDE table, stores only `max_period × blowup` rows
/// and uses modular indexing to access values. This relies on `lde_idx % height`
/// indexing, which is valid because each column is padded to `max_period` before LDE.
#[derive(Clone, Debug)]
pub(crate) struct PeriodicLdeTable<F> {
    values: RowMajorMatrix<F>,
}

impl<F: Clone + Send + Sync> PeriodicLdeTable<F> {
    pub const fn new(values: RowMajorMatrix<F>) -> Self {
        Self { values }
    }

    pub fn empty() -> Self {
        Self {
            values: RowMajorMatrix::new(Vec::new(), 0),
        }
    }

    pub fn width(&self) -> usize {
        self.values.width
    }

    pub fn height(&self) -> usize {
        if self.values.width == 0 {
            0
        } else {
            self.values.values.len() / self.values.width
        }
    }

    #[inline]
    pub fn get(&self, lde_idx: usize, col_idx: usize) -> &F {
        let height = self.height();
        debug_assert!(height > 0, "cannot index into empty periodic table");
        let row_idx = lde_idx % height;
        &self.values.values[row_idx * self.values.width + col_idx]
    }
}

/// Computes periodic columns on the quotient domain into a compact LDE table.
///
/// Assumes the quotient domain is a blowup of the trace domain (i.e. `quotient_len` is a
/// multiple of `trace_len`) and all column periods are powers of two dividing `trace_len`.
pub(crate) fn compute_periodic_on_quotient_eval_domain<F>(
    periodic_table: &[Vec<F>],
    trace_domain: &impl PolynomialSpace<Val = F>,
    quotient_domain: &impl PolynomialSpace<Val = F>,
) -> PeriodicLdeTable<F>
where
    F: TwoAdicField + Ord,
{
    if periodic_table.is_empty() {
        return PeriodicLdeTable::empty();
    }

    let trace_len = trace_domain.size();
    let quotient_len = quotient_domain.size();
    let blowup = quotient_len / trace_len;
    let log_blowup = log2_strict_usize(blowup);

    let max_period = periodic_table
        .iter()
        .map(|col| {
            let period = col.len();
            debug_assert!(period.is_power_of_two(), "period must be a power of two");
            period
        })
        .max()
        .expect("non-empty periodic table");

    let extended_height = max_period * blowup;
    let lde_shift = quotient_domain.first_point();
    let periodic_shift = lde_shift.exp_u64((quotient_len / extended_height) as u64);

    let _span = info_span!(
        "periodic eval core",
        cols = periodic_table.len(),
        quotient_size = quotient_len,
        max_period,
        blowup
    )
    .entered();

    let dft = Radix2DitParallel::<F>::default();
    let num_cols = periodic_table.len();

    // Write column LDEs directly into the row-major buffer to avoid a full extra copy.
    let mut row_major_values = vec![F::ZERO; extended_height * num_cols];
    for (col_idx, col) in periodic_table.iter().enumerate() {
        let period = col.len();
        debug_assert!(period > 0, "periodic column cannot be empty");

        let padded: Vec<F> = if period == max_period {
            col.clone()
        } else {
            (0..max_period).map(|i| col[i % period]).collect()
        };

        let _col_span = debug_span!("periodic column lde", period, max_period).entered();
        let extended = dft.coset_lde(padded, log_blowup, periodic_shift);

        for (row_idx, value) in extended.into_iter().enumerate() {
            row_major_values[row_idx * num_cols + col_idx] = value;
        }
    }

    PeriodicLdeTable::new(RowMajorMatrix::new(row_major_values, num_cols))
}

/// Evaluates periodic columns at an out-of-domain challenge point `zeta`.
///
/// Used by the verifier to check constraint satisfaction. This function evaluates
/// all periodic columns at a single random challenge point.
///
/// # Implementation Details
///
/// For each periodic column with period `P` and trace height `N`:
/// 1. Shifts `zeta` by the trace domain's offset to get `unshifted_zeta`
/// 2. Computes `y = unshifted_zeta^(N/P)`, mapping `zeta` to its position within one period
/// 3. Interpolates the column over its minimal cycle (subgroup of size `P`)
///    using barycentric Lagrange interpolation to evaluate at `y`
///
/// # Arguments
///
/// * `periodic_table` - Vector of periodic columns, where each column is a vector
///   of length equal to its period (a power of 2 that divides trace height)
/// * `trace_domain` - The domain over which the trace is defined
/// * `zeta` - The out-of-domain challenge point at which to evaluate
///
/// # Returns
///
/// A vector containing the evaluation of each periodic column at `zeta`
pub(crate) fn evaluate_periodic_at_point<F, EF>(
    periodic_table: Vec<Vec<F>>,
    trace_domain: impl PolynomialSpace<Val = F>,
    zeta: EF,
) -> Vec<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    if periodic_table.is_empty() {
        return Vec::new();
    }

    let (trace_height, log_trace_height, shift_inv) = trace_context(&trace_domain);
    let unshifted_zeta = zeta * EF::from(shift_inv);

    periodic_table
        .into_iter()
        .map(|col| {
            if col.is_empty() {
                return EF::ZERO;
            }

            let (rate_bits, subgroup) =
                subgroup_data::<F>(trace_height, log_trace_height, col.len());

            // y = (zeta / shift)^{trace_height / period}
            let y = unshifted_zeta.exp_power_of_2(rate_bits);
            let diffs: Vec<_> = subgroup.iter().map(|&g| y - EF::from(g)).collect();
            let diff_invs = batch_multiplicative_inverse(&diffs);

            interpolate_coset_with_precomputation(
                &RowMajorMatrix::new(col, 1),
                F::ONE,
                y,
                &subgroup,
                &diff_invs,
            )
            .pop()
            .expect("single-column interpolation should return one value")
        })
        .collect()
}

/// Returns the trace height, its log2, and the inverse of the domain shift.
fn trace_context<F>(trace_domain: &impl PolynomialSpace<Val = F>) -> (usize, usize, F)
where
    F: TwoAdicField,
{
    let trace_height = trace_domain.size();
    let log_trace_height = log2_strict_usize(trace_height);
    let shift_inv = trace_domain.first_point().inverse();
    (trace_height, log_trace_height, shift_inv)
}

/// For a given period, returns the exponent needed to fold into the period and the subgroup elements.
fn subgroup_data<F>(trace_height: usize, log_trace_height: usize, period: usize) -> (usize, Vec<F>)
where
    F: TwoAdicField,
{
    debug_assert!(
        trace_height.is_multiple_of(period),
        "Periodic column length must divide trace length"
    );

    let log_period = log2_strict_usize(period);
    debug_assert!(
        log_trace_height >= log_period,
        "Periodic column period cannot exceed trace height"
    );
    // rate_bits = log2(trace_height / period); rate = 2^{rate_bits} so y = z^{rate}.
    let rate_bits = log_trace_height - log_period;
    let subgroup: Vec<_> = F::two_adic_generator(log_period)
        .powers()
        .take(period)
        .collect();

    (rate_bits, subgroup)
}

#[cfg(test)]
mod tests {
    use p3_field::coset::TwoAdicMultiplicativeCoset;
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_goldilocks::Goldilocks;
    use p3_interpolation::interpolate_coset;
    use p3_matrix::dense::RowMajorMatrix;

    use super::*;

    type Val = Goldilocks;

    /// Test that compute_periodic_on_quotient_eval_domain produces the same results as the naive method
    /// where we unpack the periodic table into a full column and do interpolation for the whole column
    #[test]
    fn test_compute_periodic_on_quotient_eval_domain_correctness() {
        // Test parameters
        let trace_height = 16; // Must be a power of 2
        let log_quotient_degree = 2;
        let quotient_size = trace_height << log_quotient_degree;

        // Create test periodic columns with different periods
        let periodic_table = vec![
            // Period 2: [10, 20]
            vec![Val::from_u32(10), Val::from_u32(20)],
            // Period 4: [1, 2, 3, 4]
            vec![
                Val::from_u32(1),
                Val::from_u32(2),
                Val::from_u32(3),
                Val::from_u32(4),
            ],
            // Period 8: [5, 6, 7, 8, 9, 10, 11, 12]
            vec![
                Val::from_u32(5),
                Val::from_u32(6),
                Val::from_u32(7),
                Val::from_u32(8),
                Val::from_u32(9),
                Val::from_u32(10),
                Val::from_u32(11),
                Val::from_u32(12),
            ],
        ];

        // Get the trace domain
        let log_trace_height = log2_strict_usize(trace_height);
        let trace_domain = TwoAdicMultiplicativeCoset::new(Val::GENERATOR, log_trace_height)
            .expect("valid trace domain");
        let quotient_domain = trace_domain.create_disjoint_domain(quotient_size);

        // Generate quotient points
        let quotient_points: Vec<Val> = {
            let mut pts = Vec::with_capacity(quotient_size);
            let mut point = quotient_domain.first_point();
            pts.push(point);
            for _ in 1..quotient_size {
                point = quotient_domain
                    .next_point(point)
                    .expect("quotient_domain should yield enough points");
                pts.push(point);
            }
            pts
        };

        // Method 1: Optimized method (compute_periodic_on_quotient_eval_domain)
        let optimized_table =
            compute_periodic_on_quotient_eval_domain(&periodic_table, &trace_domain, &quotient_domain);
        let optimized_result: Vec<Vec<Val>> = (0..periodic_table.len())
            .map(|col_idx| {
                (0..quotient_size)
                    .map(|i| optimized_table.get(i, col_idx).clone())
                    .collect()
            })
            .collect();

        // Method 2: Naive method - unpack each periodic column to full trace height and interpolate
        let shift = trace_domain.first_point();
        let naive_result: Vec<Vec<Val>> = periodic_table
            .iter()
            .map(|periodic_col| {
                let period = periodic_col.len();

                // Unpack: repeat the periodic column to fill the entire trace height
                let mut unpacked = Vec::with_capacity(trace_height);
                for i in 0..trace_height {
                    unpacked.push(periodic_col[i % period]);
                }

                // Create a matrix from the unpacked column
                let unpacked_matrix = RowMajorMatrix::new(unpacked, 1);

                // For each quotient point, interpolate the full column
                let mut evals = Vec::with_capacity(quotient_size);
                for &z in &quotient_points {
                    // Interpolate the full unpacked column at this point
                    let result = interpolate_coset(&unpacked_matrix, shift, z);
                    evals.push(result[0]);
                }

                evals
            })
            .collect();

        // Compare the results
        assert_eq!(optimized_result, naive_result);
    }

    /// Test with edge case: single period equals trace height
    #[test]
    fn test_compute_periodic_on_quotient_eval_domain_full_period() {
        let trace_height = 8;
        let log_quotient_degree = 1;
        let quotient_size = trace_height << log_quotient_degree;

        // Periodic column with period = trace_height (no repetition)
        let periodic_table = vec![vec![
            Val::from_u32(1),
            Val::from_u32(2),
            Val::from_u32(3),
            Val::from_u32(4),
            Val::from_u32(5),
            Val::from_u32(6),
            Val::from_u32(7),
            Val::from_u32(8),
        ]];

        let trace_domain =
            TwoAdicMultiplicativeCoset::new(Val::GENERATOR, log2_strict_usize(trace_height))
                .expect("valid trace domain");
        let quotient_domain = trace_domain.create_disjoint_domain(quotient_size);

        let quotient_points: Vec<Val> = {
            let mut pts = Vec::with_capacity(quotient_size);
            let mut point = quotient_domain.first_point();
            pts.push(point);
            for _ in 1..quotient_size {
                point = quotient_domain
                    .next_point(point)
                    .expect("quotient_domain should yield enough points");
                pts.push(point);
            }
            pts
        };

        let optimized_table =
            compute_periodic_on_quotient_eval_domain(&periodic_table, &trace_domain, &quotient_domain);
        let optimized_result: Vec<Vec<Val>> = (0..periodic_table.len())
            .map(|col_idx| {
                (0..quotient_size)
                    .map(|i| optimized_table.get(i, col_idx).clone())
                    .collect()
            })
            .collect();

        // Naive method
        let shift = trace_domain.first_point();
        let naive_result: Vec<Vec<Val>> = periodic_table
            .iter()
            .map(|periodic_col| {
                let unpacked_matrix = RowMajorMatrix::new(periodic_col.clone(), 1);
                let mut evals = Vec::with_capacity(quotient_size);
                for &z in &quotient_points {
                    let result = interpolate_coset(&unpacked_matrix, shift, z);
                    evals.push(result[0]);
                }
                evals
            })
            .collect();

        // Compare results
        assert_eq!(optimized_result, naive_result);
    }
}
