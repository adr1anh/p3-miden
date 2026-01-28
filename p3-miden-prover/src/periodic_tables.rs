//! Efficient evaluation of periodic columns in STARKs.
//!
//! # Overview
//!
//! Periodic columns are trace columns that repeat with a period dividing the trace length.
//! Instead of committing to these columns, both prover and verifier compute them
//! independently. This module provides efficient algorithms to evaluate periodic columns at:
//!
//! - All points in the evaluation (LDE/quotient) domain (during proving, via compact LDE tables)
//! - A single out-of-domain challenge point (during verification)
//!
//! # Mathematical Foundation
//!
//! ## Power-of-two requirement
//!
//! All period lengths are powers of two. The trace domain is a two-adic subgroup of size N
//! (a power of two), and the period P must divide N as a subgroup order. This forces P to
//! be a power of two as well.
//!
//! ## Mathematical background
//!
//! A periodic column with period P and trace length N repeats every P rows:
//! f(g * w^i) = f(g * w^{i + P}) for all i.
//!
//! The problem: f is a degree-(N-1) polynomial over the trace domain, but it only takes
//! P distinct values.
//!
//! We can leverage this structure to reduce work during the LDE phase.
//!
//! The key observation is that there exists a map pi that identifies points P apart.
//! Equivalently, pi is constant on cosets of the subgroup <w^P> of order N / P.
//!
//! More precisely, for cyclic groups of order N, x -> x^k is a homomorphism with kernel
//! size gcd(N, k). To get ker(pi) = <w^P> (order N / P), set pi(x) = x^(N / P).
//!
//! For a coset g * H, use pi(z) = (z / g)^(N / P).
//!
//! The image is H_p = {1, w^(N / P), w^(2N / P), ...}, a subgroup of order P.
//!
//! Thus f factors as f = f_period o pi, where f_period is a degree < P polynomial
//! interpolating the P periodic values.
//!
//! Using a group-theoretic perspective, pi: H -> H_p is surjective with kernel <w^P>.
//! Thus H / ker(pi) ~= H_p.
//!
//! The periodic column is constant on cosets of ker(pi), so it factors through pi.
//!
//! ## Evaluation (LDE/quotient) domain
//!
//! Let the evaluation domain be g_q * H', where H' = <w'> has size N * B (B = blowup).
//! Its points are z_i = g_q * (w')^i for i = 0..N*B-1. Plugging into the fold gives:
//!
//!   pi(z_i) = (g_q / g)^(N / P) * (w')^(i * N / P)
//!
//! The step factor (w')^(N / P) has order P * B, so {pi(z_i)} is a size-(P * B) coset
//! with shift (g_q / g)^(N / P).
//!
//! Thus evaluating the periodic column on the evaluation domain is exactly a coset LDE
//! of f_period of size P with blowup B and that shift.
//!
//! ## Evaluating at an out-of-domain point zeta
//!
//! Compute pi(zeta) = (zeta / g)^(N / P).
//!
//! Then evaluate f_period at pi(zeta) using interpolation over the size-P subgroup.
//!
//! ## Memory-efficient storage
//!
//! In practice we:
//! - Pad all columns to max_period.
//! - Run a coset LDE on that period (not the full trace).
//! - Store only max_period * blowup rows.
//!
//! Then lde_idx % height recovers the value for any evaluation-domain index.
//!
//! # Functions
//!
//! - [`compute_periodic_on_quotient_eval_domain`]: Evaluates periodic columns over the
//!   evaluation domain into a compact LDE table. Called by the prover during quotient
//!   polynomial computation.
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
use tracing::info_span;

/// Compact storage for periodic column values on the evaluation (LDE/quotient) domain.
///
/// Instead of materializing the full LDE table, stores only `max_period * blowup` rows
/// and uses modular indexing to access values.
///
/// Rows repeat with period `height` because the LDE depends on
/// (z / g)^(N / max_period), so `lde_idx % height` is sufficient.
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
        // lde_idx is an index into the evaluation (LDE/quotient) domain; rows repeat every
        // `height`.
        debug_assert!(height > 0, "cannot index into empty periodic table");
        let row_idx = lde_idx % height;
        &self.values.values[row_idx * self.values.width + col_idx]
    }
}

/// Computes periodic columns on the evaluation (LDE/quotient) domain into a compact LDE table.
///
/// Assumes the evaluation (LDE/quotient) domain is a blowup of the trace domain
/// (i.e. `quotient_len` is a multiple of `trace_len`) and all column periods are
/// powers of two dividing `trace_len`.
///
/// The LDE is performed on the period domain, using a shift derived from the ratio
/// `evaluation_shift / trace_shift` (where evaluation_shift is the quotient-domain shift).
/// Concretely:
///
///   periodic_shift = (quotient_shift / trace_shift)^(trace_len / max_period)
///
/// This matches the "unshift then fold" mapping above.
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
    let trace_shift = trace_domain.first_point();
    let lde_shift = quotient_domain.first_point() * trace_shift.inverse();
    // Fold the evaluation-domain (quotient) coset shift down to the period coset:
    // (quotient_shift / trace_shift)^(trace_len / max_period).
    let periodic_shift = lde_shift.exp_u64((trace_len / max_period) as u64);

    let _span = info_span!(
        "periodic columns LDE",
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

        let extended = dft.coset_lde(padded, log_blowup, periodic_shift);

        for (row_idx, value) in extended.into_iter().enumerate() {
            row_major_values[row_idx * num_cols + col_idx] = value;
        }
    }

    PeriodicLdeTable::new(RowMajorMatrix::new(row_major_values, num_cols))
}

/// Extract packed periodic values for a quotient chunk using modular indexing.
///
/// This always fills `out` with `PackedVal::WIDTH`-sized values; callers should handle
/// the tail chunk length separately.
pub(crate) fn fill_periodic_values<F, P>(
    periodic_table: &PeriodicLdeTable<F>,
    i_start: usize,
    out: &mut Vec<P>,
) where
    F: Clone + Send + Sync,
    P: p3_field::PackedValue<Value = F>,
{
    let num_cols = periodic_table.width();
    out.clear();
    if num_cols == 0 {
        return;
    }
    for col_idx in 0..num_cols {
        let packed = <P as p3_field::PackedValue>::from_fn(|j| {
            periodic_table.get(i_start + j, col_idx).clone()
        });
        out.push(packed);
    }
}

/// Evaluates periodic columns at an out-of-domain challenge point `zeta`.
///
/// Used by the verifier to check constraint satisfaction. This function evaluates all
/// periodic columns at a single random challenge point.
///
/// # Implementation Details
///
/// For each periodic column with period `P` and trace height `N`:
/// 1. Shift `zeta` by the trace domain's offset to get `unshifted_zeta`.
/// 2. Compute `y = unshifted_zeta^(N/P)` to map `zeta` into one period.
/// 3. Interpolate the column over its minimal cycle (subgroup of size `P`)
///    using barycentric Lagrange interpolation to evaluate at `y`.
///
/// # Arguments
///
/// * `periodic_table` - Vector of periodic columns, where each column is a vector
///   of length equal to its period (a power of 2 that divides trace height).
/// * `trace_domain` - The domain over which the trace is defined
/// * `zeta` - The out-of-domain challenge point at which to evaluate
///
/// # Returns
///
/// A vector containing the evaluation of each periodic column at `zeta`.
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

/// For a given period, returns the exponent needed to fold into the period and the
/// subgroup elements.
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

    /// Test that compute_periodic_on_quotient_eval_domain produces the same results as the
    /// naive method where we unpack the periodic table into a full column and do
    /// interpolation for the whole column
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
        let optimized_table = compute_periodic_on_quotient_eval_domain(
            &periodic_table,
            &trace_domain,
            &quotient_domain,
        );
        let optimized_result: Vec<Vec<Val>> = (0..periodic_table.len())
            .map(|col_idx| {
                (0..quotient_size)
                    .map(|i| *optimized_table.get(i, col_idx))
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

        let optimized_table = compute_periodic_on_quotient_eval_domain(
            &periodic_table,
            &trace_domain,
            &quotient_domain,
        );
        let optimized_result: Vec<Vec<Val>> = (0..periodic_table.len())
            .map(|col_idx| {
                (0..quotient_size)
                    .map(|i| *optimized_table.get(i, col_idx))
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
