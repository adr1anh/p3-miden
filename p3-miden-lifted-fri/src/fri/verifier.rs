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

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, TwoAdicField};
use p3_miden_lmcs::{Lmcs, LmcsError};
use p3_util::reverse_bits_len;
use thiserror::Error;

use crate::fri::FriParams;
use crate::fri::proof::FriProof;

/// FRI low-degree test oracle.
///
/// Created via [`FriOracle::new`], which samples folding challenges from
/// the Fiat-Shamir transcript. The oracle tests that evaluations are close
/// to a low-degree polynomial.
///
/// Uses a single base-field LMCS. Opened base field values are reconstructed
/// to extension field for folding verification.
pub struct FriOracle<F, EF, L>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    L: Lmcs<F = F>,
{
    /// Merkle commitments for each folding round.
    commitments: Vec<L::Commitment>,
    /// Folding challenges β for each round.
    betas: Vec<EF>,
    /// Coefficients of the final low-degree polynomial.
    final_poly: Vec<EF>,
}

impl<F, EF, L> FriOracle<F, EF, L>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    L: Lmcs<F = F>,
{
    /// Create oracle from FRI proof, checking per-round PoW witnesses.
    pub fn new<Challenger>(
        proof: &FriProof<EF, L, Challenger::Witness>,
        challenger: &mut Challenger,
        proof_of_work_bits: usize,
    ) -> Result<Self, FriError>
    where
        L::Commitment: Clone,
        Challenger: FieldChallenger<F> + CanObserve<L::Commitment> + GrindingChallenger,
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

    /// Test low-degree proximity by checking all folding rounds in batch.
    pub fn test_low_degree(
        &self,
        lmcs: &L,
        params: &FriParams,
        indices: &[usize],
        initial_evals: &[EF],
        round_proofs: &[L::Proof],
    ) -> Result<(), FriError> {
        // Verify the proof has the expected structure
        if round_proofs.len() != self.commitments.len() {
            return Err(FriError::InvalidProofStructure);
        }

        if indices.len() != initial_evals.len() {
            return Err(FriError::InvalidProofStructure);
        }

        let log_arity = params.fold.log_arity();
        let arity = params.fold.arity();
        let mut log_domain_size = self.log_domain_size(params);

        // Track current indices and evals for each query
        let mut current_indices: Vec<usize> = indices.to_vec();
        let mut current_evals: Vec<EF> = initial_evals.to_vec();

        // Precompute g_inv once; we'll update it each round by raising to power arity
        let mut g_inv = F::two_adic_generator(log_domain_size).inverse();

        // Verify each round
        for (round_idx, (commitment, round_proof)) in
            zip(&self.commitments, round_proofs).enumerate()
        {
            let log_num_rows = log_domain_size.saturating_sub(log_arity);

            // Compute row indices for this round
            let row_indices: Vec<usize> = current_indices
                .iter()
                .map(|&idx| idx >> log_arity)
                .collect();

            // Verify LMCS opening - dimensions are flattened (EF elements stored as F)
            let widths = [arity * EF::DIMENSION];

            let opened_rows = lmcs
                .open_batch(commitment, &widths, log_num_rows, &row_indices, round_proof)
                .map_err(|e| FriError::LmcsError(e, round_idx))?;

            // Verify folding consistency for each query
            let beta = self.betas[round_idx];

            for (query_idx, ((&current_idx, &row_idx), current_eval)) in current_indices
                .iter()
                .zip(row_indices.iter())
                .zip(current_evals.iter_mut())
                .enumerate()
            {
                // Get the opened row for this query (first matrix, since FRI only has one)
                let flat_row = opened_rows[query_idx][0];

                // Reconstruct extension field values from flattened base field
                let row: Vec<EF> = EF::reconstitute_from_base(flat_row.to_vec());

                // position_in_coset = current_idx & (arity - 1)
                let position_in_coset = current_idx & ((1 << log_arity) - 1);

                // Check that the evaluation matches the opened value
                if row[position_in_coset] != *current_eval {
                    return Err(FriError::EvaluationMismatch {
                        row_index: row_idx,
                        position: position_in_coset,
                    });
                }

                // Compute coset generator inverse s⁻¹
                let s_inv = g_inv.exp_u64(reverse_bits_len(row_idx, log_num_rows) as u64);

                // Fold: interpolate f on coset and evaluate at β
                *current_eval = params.fold.fold_evals(&row, s_inv, beta);
            }

            // Update indices and domain size for next round
            current_indices = row_indices;
            log_domain_size -= log_arity;
            g_inv = g_inv.exp_power_of_2(log_arity);
        }

        // Final polynomial check for each query
        for (&index, &eval) in current_indices.iter().zip(current_evals.iter()) {
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
pub enum FriError {
    #[error("LMCS verification failed at round {1}: {0}")]
    LmcsError(LmcsError, usize),
    #[error("invalid proof structure")]
    InvalidProofStructure,
    #[error("evaluation mismatch at row {row_index}, position {position}")]
    EvaluationMismatch { row_index: usize, position: usize },
    #[error("final polynomial mismatch")]
    FinalPolyMismatch,
    #[error("invalid proof-of-work witness")]
    InvalidPowWitness,
}
