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
use core::fmt;
use core::iter::zip;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::Dimensions;
use p3_util::reverse_bits_len;
use thiserror::Error;

use crate::fri::FriParams;
use crate::fri::proof::{FriProof, FriQuery};

/// Verifier's oracle for FRI query verification.
///
/// Created via [`FriOracle::new`], which samples folding challenges from
/// the Fiat-Shamir transcript. The oracle can then verify queries against
/// the committed polynomial.
///
/// Uses a single base-field MMCS. Opened base field values are reconstructed
/// to extension field for folding verification.
pub struct FriOracle<F, EF, Mmcs>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    Mmcs: p3_commit::Mmcs<F>,
{
    /// Merkle commitments for each folding round.
    commitments: Vec<Mmcs::Commitment>,
    /// Folding challenges β for each round.
    betas: Vec<EF>,
    /// Coefficients of the final low-degree polynomial.
    final_poly: Vec<EF>,
}

impl<F, EF, Mmcs> FriOracle<F, EF, Mmcs>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    Mmcs: p3_commit::Mmcs<F>,
{
    /// Create an oracle from FRI proof, checking per-round PoW witnesses.
    ///
    /// This method enforces the correct Fiat-Shamir order:
    /// 1. For each round: observe commitment, check PoW witness, sample beta
    /// 2. Observe final polynomial coefficients
    ///
    /// # Arguments
    /// - `proof`: FRI proof containing commitments, final polynomial, and per-round PoW witnesses
    /// - `challenger`: The Fiat-Shamir challenger
    /// - `proof_of_work_bits`: Number of bits for PoW verification (applied per round)
    ///
    /// # Errors
    /// Returns `FriError::InvalidPowWitness` if any round's witness verification fails.
    /// Returns `FriError::InvalidProofStructure` if witness count doesn't match commitment count.
    pub fn new<Challenger>(
        proof: &FriProof<F, EF, Mmcs, Challenger::Witness>,
        challenger: &mut Challenger,
        proof_of_work_bits: usize,
    ) -> Result<Self, FriError<Mmcs::Error>>
    where
        Mmcs::Commitment: Clone,
        Challenger: FieldChallenger<F> + CanObserve<Mmcs::Commitment> + GrindingChallenger,
    {
        // Validate structure: witnesses count must match commitments count
        if proof.pow_witnesses.len() != proof.commitments.len() {
            return Err(FriError::InvalidProofStructure);
        }

        // 1. For each round: observe commitment, check witness, sample beta
        let mut betas = Vec::with_capacity(proof.commitments.len());
        for (commit, &witness) in zip(&proof.commitments, &proof.pow_witnesses) {
            challenger.observe(commit.clone());

            // Check grinding witness for this round
            if !challenger.check_witness(proof_of_work_bits, witness) {
                return Err(FriError::InvalidPowWitness);
            }

            let beta: EF = challenger.sample_algebra_element();
            betas.push(beta);
        }

        // 2. Observe final polynomial coefficients
        for &coeff in &proof.final_poly {
            challenger.observe_algebra_element(coeff);
        }

        Ok(Self {
            commitments: proof.commitments.clone(),
            betas,
            final_poly: proof.final_poly.clone(),
        })
    }

    /// Verify a FRI query by checking all folding rounds.
    ///
    /// Two-phase verification:
    /// 1. Verify all Merkle openings (base field) and collect opened rows
    /// 2. Reconstruct extension field values and verify folding consistency
    ///
    /// The `log_max_degree` is derived from `num_rounds * log_folding_factor + log_final_degree`
    /// where `log_final_degree = log(final_poly.len())`.
    ///
    /// ## Arguments
    ///
    /// - `mmcs`: The base-field MMCS used for commitments
    /// - `params`: FRI parameters (blowup, folding factor)
    /// - `index`: Query index in the initial domain (bit-reversed)
    /// - `eval`: Initial evaluation f(x) at the queried point
    /// - `query`: Query proof containing Merkle openings (base field values)
    ///
    /// ## Errors
    ///
    /// Returns `FriError` if any verification check fails.
    pub fn verify_query(
        &self,
        params: &FriParams,
        mmcs: &Mmcs,
        index: usize,
        eval: EF,
        query: &FriQuery<F, Mmcs>,
    ) -> Result<(), FriError<Mmcs::Error>>
    where
        Mmcs::Error: fmt::Debug,
    {
        // Verify the proof has the expected structure
        if query.openings.len() != self.commitments.len() {
            return Err(FriError::InvalidProofStructure);
        }

        // Verify all Merkle openings (base field)
        self.verify_openings(mmcs, params, index, query)?;

        // Verify folding consistency (reconstructs EF from base field slices internally)
        let flat_rows = query.openings.iter().map(|o| o.opened_values[0].as_slice());
        self.verify_folding::<Mmcs::Error>(params, index, eval, flat_rows)?;

        Ok(())
    }

    /// Verify all Merkle openings without processing contents.
    ///
    /// Checks that each opening is valid against its commitment.
    /// Uses flattened dimensions (width * EF::DIMENSION) since data is stored as base field.
    fn verify_openings(
        &self,
        mmcs: &Mmcs,
        params: &FriParams,
        mut index: usize,
        query: &FriQuery<F, Mmcs>,
    ) -> Result<(), FriError<Mmcs::Error>>
    where
        Mmcs::Error: fmt::Debug,
    {
        let log_arity = params.fold.log_arity();
        let arity = params.fold.arity();
        let mut num_rows = 1 << self.log_domain_size(params).saturating_sub(log_arity);

        for (round_idx, (commitment, opening)) in
            zip(&self.commitments, &query.openings).enumerate()
        {
            let row_index = index >> log_arity;

            // Verify with flattened dimensions: each EF element is stored as EF::DIMENSION base elements
            mmcs.verify_batch(
                commitment,
                &[Dimensions {
                    width: arity * EF::DIMENSION,
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

    /// Verify folding consistency given flattened base field rows.
    ///
    /// For each round:
    /// 1. Reconstruct extension field values from base field slice
    /// 2. Check that `eval` matches the expected position in the opened row
    /// 3. Compute coset generator inverse s⁻¹
    /// 4. Fold the coset evaluations: eval' = f(β) via interpolation
    ///
    /// Finally, check that the folded value matches the final polynomial.
    fn verify_folding<'a, MmcsError: fmt::Debug>(
        &self,
        params: &FriParams,
        mut index: usize,
        mut eval: EF,
        flat_rows: impl Iterator<Item = &'a [F]>,
    ) -> Result<(), FriError<MmcsError>> {
        let log_arity = params.fold.log_arity();
        let mut log_domain_size = self.log_domain_size(params);

        // Precompute g_inv once; we'll update it each round by raising to power arity
        let mut g_inv = F::two_adic_generator(log_domain_size).inverse();

        for (&beta, flat_row) in zip(&self.betas, flat_rows) {
            // Reconstruct extension field values from flattened base field
            let row: Vec<EF> = EF::reconstitute_from_base(flat_row.to_vec());
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
            eval = params.fold.fold_evals(&row, s_inv, beta);

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

    /// Derive the initial domain size from proof structure.
    ///
    /// `log_domain_size = num_rounds * log_folding_factor + log_final_poly_degree + log_blowup`
    #[inline]
    fn log_domain_size(&self, params: &FriParams) -> usize {
        let log_final_poly_degree = self.final_poly.len().trailing_zeros() as usize;
        let log_max_degree =
            self.commitments.len() * params.fold.log_arity() + log_final_poly_degree;
        log_max_degree + params.log_blowup
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during FRI verification.
#[derive(Debug, Error)]
pub enum FriError<MmcsError> {
    /// Merkle verification failed.
    #[error("Merkle verification failed at round {1}: {0:?}")]
    MmcsError(MmcsError, usize),
    /// Proof structure doesn't match expected format.
    ///
    /// This includes wrong number of commitments, openings, betas, or final polynomial length.
    #[error("invalid proof structure")]
    InvalidProofStructure,
    /// Evaluation mismatch during folding.
    #[error("evaluation mismatch at row {row_index}, position {position}")]
    EvaluationMismatch { row_index: usize, position: usize },
    /// Final polynomial evaluation doesn't match folded value.
    #[error("final polynomial mismatch")]
    FinalPolyMismatch,
    /// Proof-of-work witness verification failed.
    #[error("invalid proof-of-work witness")]
    InvalidPowWitness,
}
