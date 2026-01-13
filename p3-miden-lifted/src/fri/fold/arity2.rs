//! Arity-2 FRI folding using even-odd decomposition.
//!
//! Any polynomial `f(X)` can be uniquely decomposed into even and odd parts:
//!
//! ```text
//! f(X) = fₑ(X²) + X · fₒ(X²)
//! ```
//!
//! where `fₑ` contains the even-degree coefficients and `fₒ` the odd-degree coefficients.
//!
//! ## Key Identity
//!
//! From evaluations at `s` and `−s`, we can recover `fₑ(s²)` and `fₒ(s²)`:
//!
//! ```text
//! f(s)  = fₑ(s²) + s · fₒ(s²)
//! f(−s) = fₑ(s²) − s · fₒ(s²)
//! ```
//!
//! Solving:
//!
//! ```text
//! fₑ(s²) = (f(s) + f(−s)) / 2
//! fₒ(s²) = (f(s) − f(−s)) / (2s)
//! ```
//!
//! ## FRI Folding
//!
//! Given a challenge `β`, FRI computes:
//!
//! ```text
//! f(β) = fₑ(β²) + β · fₒ(β²)
//! ```
//!
//! Since we only have evaluations on the coset `{s, −s}`, we interpolate using the identity
//! above, noting that `fₑ` and `fₒ` are constant on this coset (they depend only on `s²`).

use p3_field::{Algebra, ExtensionField, PackedField, TwoAdicField};

use super::FriFold;

/// Marker type for arity-2 FRI folding.
///
/// Folds pairs of evaluations using the even-odd decomposition:
/// `f(β) = (f(s) + f(-s))/2 + β/s · (f(s) - f(-s))/2`
pub struct FriFold2;

impl FriFold<2> for FriFold2 {
    /// Evaluate `f(β)` from evaluations on a coset `{s, −s}`.
    ///
    /// ## Inputs
    ///
    /// - `evals`: evaluations `[f(s), f(−s)]` in bit-reversed order.
    /// - `s_inv`: the inverse of the coset generator `s`.
    /// - `beta`: the FRI folding challenge `β`.
    ///
    /// ## Algorithm
    ///
    /// Using the even-odd decomposition `f(X) = fₑ(X²) + X · fₒ(X²)`:
    ///
    /// 1. Compute `fₑ(s²) = (f(s) + f(−s)) / 2`
    /// 2. Compute `fₒ(s²) = (f(s) − f(−s)) / (2s)`
    /// 3. Return `f(β) = fₑ(s²) + β · fₒ(s²)` (valid since `β² = s²` in the folded domain)
    #[inline(always)]
    fn fold_evals<PF, EF, PEF>(evals: [PEF; 2], s_inv: PF, beta: EF) -> PEF
    where
        PF: PackedField,
        PF::Scalar: TwoAdicField,
        EF: ExtensionField<PF::Scalar>,
        PEF: Algebra<PF> + Algebra<EF>,
    {
        // y₀ = f(s), y₁ = f(−s)
        let [y0, y1] = evals;

        // Broadcast beta to PEF
        let beta_packed: PEF = beta.into();

        // f(β) = fₑ(s²) + β · fₒ(s²)
        // Even part: fₑ(s²) = (f(s) + f(−s)) / 2
        // Odd part: fₒ(s²) = (f(s) − f(−s)) / (2s)
        // Combined: ((y0 + y1) + (y0 - y1) * beta * s_inv) / 2
        let sum = y0.clone() + y1.clone();
        let diff = y0 - y1;
        let result = sum + diff * beta_packed * s_inv;

        // Divide by 2
        result.halve()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_util::reverse_slice_index_bits;
    use rand::distr::StandardUniform;
    use rand::prelude::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::fri::fold::tests::{EF, F, Pef, Pf, test_fold, test_fold_matrix_packed_equivalence};

    /// Test that fold_evals (arity 2) compiles with scalar types.
    #[test]
    fn test_fold_evals_arity2_scalar_types() {
        let evals: [EF; 2] = [EF::ZERO; 2];
        let s_inv = F::ONE;
        let beta = EF::ONE;
        let _result: EF = FriFold2::fold_evals::<F, EF, EF>(evals, s_inv, beta);
    }

    /// Test that fold_evals (arity 2) compiles with packed types.
    #[test]
    fn test_fold_evals_arity2_packed_types() {
        let evals: [Pef; 2] = [Pef::ZERO; 2];
        let s_inv = Pf::ZERO;
        let beta = EF::ONE;
        let _result: Pef = FriFold2::fold_evals::<Pf, EF, Pef>(evals, s_inv, beta);
    }

    #[test]
    fn test_arity_2_babybear() {
        test_fold::<F, EF, FriFold2, 2>();
    }

    #[test]
    fn test_fold_matrix_arity2_packed_equivalence() {
        test_fold_matrix_packed_equivalence::<FriFold2, 2>();
    }

    /// Test that our folding matches the reference implementation from two_adic_pcs.rs.
    #[test]
    fn test_fold_matches_reference() {
        let mut rng = SmallRng::seed_from_u64(99);

        let log_height = 4; // 16 evaluations
        let num_evals = 1 << log_height;
        let log_arity = 1; // arity 2
        let arity = 1 << log_arity;
        let log_num_cosets = log_height - log_arity;
        let num_cosets = 1 << log_num_cosets;

        // Generate random low-degree polynomial and compute evaluations
        let log_poly_degree = 3;
        let poly_degree = 1 << log_poly_degree;
        let coeffs: Vec<EF> = (0..poly_degree)
            .map(|_| rng.sample(StandardUniform))
            .collect();

        // DFT and bit-reverse to match FRI's expectation
        let mut full_coeffs = coeffs;
        full_coeffs.resize(num_evals, EF::ZERO);
        let dft = p3_dft::Radix2DFTSmallBatch::<EF>::default();
        let mut evals = p3_dft::TwoAdicSubgroupDft::dft_algebra(&dft, full_coeffs);
        reverse_slice_index_bits(&mut evals);

        let beta: EF = rng.sample(StandardUniform);

        // Our implementation
        let g_inv = F::two_adic_generator(log_num_cosets + log_arity).inverse();
        let mut s_invs: Vec<F> = g_inv.powers().take(num_cosets).collect();
        reverse_slice_index_bits(&mut s_invs);

        let matrix = RowMajorMatrix::new(evals.clone(), arity);
        let my_folded = FriFold2::fold_matrix_scalar::<F, EF>(matrix.as_view(), &s_invs, beta);

        // Reference implementation from two_adic_pcs.rs
        let ref_g_inv = F::two_adic_generator(log_num_cosets + 1).inverse();
        let mut ref_halve_inv_powers: Vec<F> = ref_g_inv
            .shifted_powers(F::ONE.halve())
            .take(num_cosets)
            .collect();
        reverse_slice_index_bits(&mut ref_halve_inv_powers);

        let ref_folded: Vec<EF> = evals
            .chunks(arity)
            .zip(ref_halve_inv_powers.iter())
            .map(|(row, &halve_inv_power)| {
                let lo = row[0];
                let hi = row[1];
                (lo + hi).halve() + (lo - hi) * beta * halve_inv_power
            })
            .collect();

        // Assert all values match
        assert_eq!(my_folded.len(), ref_folded.len());
        for (my_val, ref_val) in my_folded.iter().zip(ref_folded.iter()) {
            assert_eq!(my_val, ref_val, "Folded value mismatch");
        }
    }

    /// Test that FRI folding preserves the low-degree structure.
    ///
    /// After folding a degree-d polynomial, the result should have degree d/arity.
    /// This test verifies by checking that high coefficients are zero after IDFT.
    #[test]
    fn test_folding_preserves_low_degree() {
        let mut rng = SmallRng::seed_from_u64(42);

        let log_blowup = 2;
        let log_poly_degree = 4; // degree 16 polynomial
        let poly_degree = 1 << log_poly_degree;
        let log_lde_size = log_poly_degree + log_blowup;
        let lde_size = 1 << log_lde_size;
        let log_arity = 1; // arity 2
        let arity = 1 << log_arity;

        // Generate random low-degree polynomial
        let coeffs: Vec<EF> = (0..poly_degree)
            .map(|_| rng.sample(StandardUniform))
            .collect();

        // Compute LDE in bit-reversed order
        let mut full_coeffs = coeffs;
        full_coeffs.resize(lde_size, EF::ZERO);
        let dft = p3_dft::Radix2DFTSmallBatch::<EF>::default();
        let mut evals = p3_dft::TwoAdicSubgroupDft::dft_algebra(&dft, full_coeffs);
        reverse_slice_index_bits(&mut evals);

        // Compute s_invs
        let log_num_cosets = log_lde_size - log_arity;
        let num_cosets = 1 << log_num_cosets;
        let g_inv = F::two_adic_generator(log_lde_size).inverse();
        let mut s_invs: Vec<F> = g_inv.powers().take(num_cosets).collect();
        reverse_slice_index_bits(&mut s_invs);

        // Fold with random beta
        let beta: EF = rng.sample(StandardUniform);
        let matrix = RowMajorMatrix::new(evals, arity);
        let folded = FriFold2::fold_matrix_scalar::<F, EF>(matrix.as_view(), &s_invs, beta);

        // IDFT the result to get coefficients
        let mut folded_for_idft = folded;
        reverse_slice_index_bits(&mut folded_for_idft);
        let folded_coeffs = p3_dft::TwoAdicSubgroupDft::idft_algebra(&dft, folded_for_idft);

        // Check that all coefficients beyond degree/arity are zero
        let expected_degree = poly_degree / arity;
        for (i, coeff) in folded_coeffs.iter().enumerate().skip(expected_degree) {
            assert_eq!(
                *coeff,
                EF::ZERO,
                "High coefficient c[{i}] should be zero but was {:?}",
                coeff
            );
        }
    }
}
