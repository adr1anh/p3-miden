//! FRI Verifier
//!
//! Verifies that a committed polynomial is close to low-degree.
//!
//! # Domain Structure
//!
//! The prover commits to evaluations on domain D of size 2^log_domain_size in bit-reversed order.
//! Each folding round groups `arity` consecutive evaluations into cosets and folds them.
//!
//! For arity = 2:
//!   - Row i contains evaluations at coset {s, −s} where s = g^{bitrev(i)}
//!   - g is the generator of D (has order n)
//!
//! For arity = 4:
//!   - Row i contains evaluations at coset {s, −s, ωs, −ωs} where ω = √−1
//!
//! # Index Semantics
//!
//! The query `index` has two parts:
//!   - High bits: which row (coset) in the committed matrix
//!   - Low bits: which position within the coset
//!
//! After each fold, we shift off `log_arity` bits, moving to the parent coset.

use alloc::vec::Vec;
use core::iter::zip;
use core::{array, fmt};

use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{BatchOpening, Mmcs};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::Dimensions;
use p3_util::reverse_bits_len;

use crate::fri::fold::{FriFold, FriFold2, FriFold4};
use crate::fri::{FriError, FriParams};

/// Verifier's oracle for FRI query verification.
///
/// Created via [`FriOracle::new`], which samples folding challenges from
/// the Fiat-Shamir transcript. The oracle can then verify queries against
/// the committed polynomial.
pub struct FriOracle<EF: Field, FriMmcs: Mmcs<EF>> {
    /// Merkle commitments for each folding round.
    commitments: Vec<FriMmcs::Commitment>,
    /// Folding challenges β for each round.
    betas: Vec<EF>,
    /// Coefficients of the final low-degree polynomial.
    final_poly: Vec<EF>,
}

impl<EF: Field, FriMmcs: Mmcs<EF>> FriOracle<EF, FriMmcs> {
    /// Create an oracle by observing commitments and sampling challenges.
    ///
    /// This method enforces the correct Fiat-Shamir order:
    /// 1. For each round: observe commitment, sample beta
    /// 2. Observe final polynomial coefficients
    pub fn new<F, Challenger>(
        commitments: Vec<FriMmcs::Commitment>,
        final_poly: Vec<EF>,
        challenger: &mut Challenger,
    ) -> Self
    where
        F: Field,
        EF: ExtensionField<F>,
        FriMmcs::Commitment: Clone,
        Challenger: FieldChallenger<F> + CanObserve<FriMmcs::Commitment>,
    {
        // Observe each commitment and sample corresponding beta
        let betas: Vec<EF> = commitments
            .iter()
            .map(|commit| {
                challenger.observe(commit.clone());
                challenger.sample_algebra_element()
            })
            .collect();

        // Observe final polynomial coefficients
        for &coeff in &final_poly {
            challenger.observe_algebra_element(coeff);
        }

        Self {
            commitments,
            betas,
            final_poly,
        }
    }

    /// Returns the number of folding rounds.
    #[inline]
    pub fn num_rounds(&self) -> usize {
        self.commitments.len()
    }

    /// Derive the initial domain size from proof structure.
    ///
    /// `log_domain_size = num_rounds * log_folding_factor + log_final_poly_degree + log_blowup`
    #[inline]
    fn log_domain_size(&self, params: &FriParams) -> usize {
        let log_final_poly_degree = self.final_poly.len().trailing_zeros() as usize;
        let log_max_degree =
            self.commitments.len() * params.log_folding_factor + log_final_poly_degree;
        log_max_degree + params.log_blowup
    }

    /// Verify a FRI query by checking all folding rounds.
    ///
    /// Two-phase verification:
    /// 1. Verify all Merkle openings and collect opened rows
    /// 2. Process opened rows: check consistency and fold
    ///
    /// The `log_max_degree` is derived from `num_rounds * log_folding_factor + log_final_degree`
    /// where `log_final_degree = log(final_poly.len())`.
    ///
    /// ## Arguments
    ///
    /// - `mmcs`: The MMCS used for commitments
    /// - `params`: FRI parameters (blowup, folding factor)
    /// - `index`: Query index in the initial domain (bit-reversed)
    /// - `eval`: Initial evaluation f(x) at the queried point
    /// - `openings`: Merkle openings for each folding round
    ///
    /// ## Errors
    ///
    /// Returns `FriError` if any verification check fails.
    pub fn verify_query<F: TwoAdicField>(
        &self,
        mmcs: &FriMmcs,
        params: &FriParams,
        index: usize,
        eval: EF,
        openings: &[BatchOpening<EF, FriMmcs>],
    ) -> Result<(), FriError<FriMmcs::Error>>
    where
        EF: ExtensionField<F>,
        FriMmcs::Error: fmt::Debug,
    {
        // Verify the proof has the expected structure
        if openings.len() != self.commitments.len() {
            return Err(FriError::InvalidProofStructure);
        }

        // Verify all Merkle openings
        self.verify_openings(mmcs, params, index, openings)?;

        // Process opened rows and verify folding
        let rows = openings.iter().map(|o| o.opened_values[0].as_slice());
        self.verify_folding::<F, FriMmcs::Error>(params, index, eval, rows)?;

        Ok(())
    }

    /// Verify all Merkle openings without processing contents.
    ///
    /// Checks that each opening is valid against its commitment.
    fn verify_openings(
        &self,
        mmcs: &FriMmcs,
        params: &FriParams,
        mut index: usize,
        openings: &[BatchOpening<EF, FriMmcs>],
    ) -> Result<(), FriError<FriMmcs::Error>>
    where
        FriMmcs::Error: fmt::Debug,
    {
        let log_arity = params.log_folding_factor;
        let arity = 1 << log_arity;
        let mut num_rows = 1 << self.log_domain_size(params).saturating_sub(log_arity);

        for (round_idx, (commitment, opening)) in zip(&self.commitments, openings).enumerate() {
            let row_index = index >> log_arity;

            mmcs.verify_batch(
                commitment,
                &[Dimensions {
                    width: arity,
                    height: num_rows,
                }],
                row_index,
                opening.into(),
            )
            .map_err(|e| FriError::MmcsError(e, round_idx))?;

            index = row_index;
            num_rows >>= log_arity;
        }
        Ok(())
    }

    /// Verify folding consistency given opened rows.
    ///
    /// For each round:
    /// 1. Check that `eval` matches the expected position in the opened row
    /// 2. Compute coset generator inverse s⁻¹
    /// 3. Fold the coset evaluations: eval' = f(β) via interpolation
    ///
    /// Finally, check that the folded value matches the final polynomial.
    fn verify_folding<'a, F: TwoAdicField, MmcsError: fmt::Debug>(
        &self,
        params: &FriParams,
        mut index: usize,
        mut eval: EF,
        rows: impl Iterator<Item = &'a [EF]>,
    ) -> Result<(), FriError<MmcsError>>
    where
        EF: ExtensionField<F> + 'a,
    {
        let log_arity = params.log_folding_factor;
        let mut log_domain_size = self.log_domain_size(params);

        // Precompute g_inv once; we'll update it each round by raising to power arity
        let mut g_inv = F::two_adic_generator(log_domain_size).inverse();

        for (&beta, row) in zip(&self.betas, rows) {
            // index = (row_index × arity) + position_in_coset
            let position_in_coset = index & ((1 << log_arity) - 1);
            let row_index = index >> log_arity;

            // ─────────────────────────────────────────────────────────────────
            // Consistency check
            // ─────────────────────────────────────────────────────────────────
            // The evaluation we're carrying forward must match the opened value
            // at the corresponding position within the coset.
            if row[position_in_coset] != eval {
                return Err(FriError::EvaluationMismatch {
                    row_index,
                    position: position_in_coset,
                });
            }

            // ─────────────────────────────────────────────────────────────────
            // Compute coset generator inverse s⁻¹
            // ─────────────────────────────────────────────────────────────────
            // Row `row_index` contains coset s·⟨ω⟩ where:
            //   - ω is a primitive `arity`-th root of unity
            //   - s = g^{bitrev(row_index, log_num_rows)} with g having order 2^log_domain_size
            //
            // So s_inv = g_inv^{bitrev(row_index, log_num_rows)}
            let s_inv = {
                let log_num_rows = log_domain_size - log_arity;
                g_inv.exp_u64(reverse_bits_len(row_index, log_num_rows) as u64)
            };

            // ─────────────────────────────────────────────────────────────────
            // Fold: interpolate f on coset and evaluate at β
            // ─────────────────────────────────────────────────────────────────
            // Given evaluations [f(s), f(−s), ...] (bit-reversed), compute f(β).
            eval = match log_arity {
                1 => FriFold2::fold_evals::<F, EF, EF>(array::from_fn(|i| row[i]), s_inv, beta),
                2 => FriFold4::fold_evals::<F, EF, EF>(array::from_fn(|i| row[i]), s_inv, beta),
                _ => unreachable!("Unsupported folding arity"),
            };

            // Update for next round:
            // - index becomes row_index
            // - log_domain_size shrinks by log_arity
            // - g_inv needs order 2^(log_domain_size - log_arity) = g_inv^arity
            index = row_index;
            log_domain_size -= log_arity;
            g_inv = g_inv.exp_power_of_2(log_arity);
        }

        // ─────────────────────────────────────────────────────────────────────
        // Final polynomial check
        // ─────────────────────────────────────────────────────────────────────
        // After all folds, verify that `eval` equals p(x) where:
        //   - p is the final low-degree polynomial
        //   - x is the evaluation point corresponding to `index` in the FINAL domain
        //
        // The final domain has size 2^log_domain_size. The index refers to position
        // in bit-reversed storage, so the actual point is:
        //   x = g_final^{bitrev(index, log_domain_size)}
        // where g_final generates the final domain.
        let x_power = reverse_bits_len(index, log_domain_size) as u64;
        let x = F::two_adic_generator(log_domain_size).exp_u64(x_power);

        // Evaluate final polynomial via Horner's method: p(x) = Σᵢ cᵢ·xⁱ
        let final_eval = self
            .final_poly
            .iter()
            .rev()
            .fold(EF::ZERO, |acc, &coeff| acc * x + coeff);

        if final_eval != eval {
            return Err(FriError::FinalPolyMismatch);
        }

        Ok(())
    }
}
