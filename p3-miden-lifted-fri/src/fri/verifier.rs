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

use p3_challenger::CanSample;
use p3_field::{BasedVectorSpace, ExtensionField, TwoAdicField};
use p3_miden_lmcs::{Lmcs, LmcsError};
use p3_miden_transcript::VerifierChannel;
use p3_util::reverse_bits_len;
use thiserror::Error;

use crate::fri::FriParams;

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
    /// Per-round commitment and folding challenge.
    rounds: Vec<FriRoundOracle<L::Commitment, EF>>,
    /// Coefficients of the final low-degree polynomial.
    final_poly: Vec<EF>,
}

struct FriRoundOracle<Commitment, EF> {
    commitment: Commitment,
    beta: EF,
}

impl<F, EF, L> FriOracle<F, EF, L>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + Clone,
    L: Lmcs<F = F>,
{
    /// Create oracle by reading from a verifier channel.
    pub fn new<Ch>(
        params: &FriParams,
        log_domain_size: usize,
        channel: &mut Ch,
    ) -> Result<Self, FriError>
    where
        Ch: VerifierChannel<F = F, Commitment = L::Commitment> + CanSample<F>,
    {
        let num_rounds = params.num_rounds(log_domain_size);
        let mut rounds = Vec::with_capacity(num_rounds);

        for _ in 0..num_rounds {
            let commitment = channel
                .receive_commitment()
                .ok_or(FriError::InvalidProofStructure)?
                .clone();

            if channel.grind(params.proof_of_work_bits).is_none() {
                return Err(FriError::InvalidPowWitness);
            }

            let beta: EF = channel.sample_algebra_element();
            rounds.push(FriRoundOracle { commitment, beta });
        }

        let final_degree = params.final_poly_degree(log_domain_size);
        let final_poly = channel
            .receive_algebra_slice(final_degree)
            .ok_or(FriError::InvalidProofStructure)?;

        Ok(Self { rounds, final_poly })
    }

    /// Test low-degree proximity by reading openings from a verifier channel.
    pub fn test_low_degree<Ch>(
        &self,
        lmcs: &L,
        params: &FriParams,
        indices: &[usize],
        initial_evals: &[EF],
        channel: &mut Ch,
    ) -> Result<(), FriError>
    where
        Ch: VerifierChannel<F = F, Commitment = L::Commitment>,
    {
        let log_arity = params.fold.log_arity();
        let arity = params.fold.arity();
        let mut log_domain_size = self.log_domain_size(params);

        if indices.len() != initial_evals.len() {
            return Err(FriError::InvalidProofStructure);
        }

        // Track current indices and evals for each query
        let mut current_indices: Vec<usize> = indices.to_vec();
        let mut current_evals: Vec<EF> = initial_evals.to_vec();

        // Precompute g_inv once; we'll update it each round by raising to power arity
        let mut g_inv = F::two_adic_generator(log_domain_size).inverse();

        // Verify each round
        for (round_idx, round) in self.rounds.iter().enumerate() {
            let log_num_rows = log_domain_size.saturating_sub(log_arity);

            // Compute row indices for this round
            let row_indices: Vec<usize> = current_indices
                .iter()
                .map(|&idx| idx >> log_arity)
                .collect();

            // Verify LMCS opening - dimensions are flattened (EF elements stored as F)
            let widths = [arity * <EF as BasedVectorSpace<F>>::DIMENSION];

            let opened_rows = lmcs
                .open_batch(
                    &round.commitment,
                    &widths,
                    log_num_rows,
                    &row_indices,
                    channel,
                )
                .map_err(|e| FriError::LmcsError(e, round_idx))?;

            for (query_idx, ((&current_idx, &row_idx), current_eval)) in current_indices
                .iter()
                .zip(row_indices.iter())
                .zip(current_evals.iter_mut())
                .enumerate()
            {
                // Get the opened row for this query (first matrix, since FRI only has one)
                let flat_row = &opened_rows[query_idx][0];

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
                *current_eval = params.fold.fold_evals(&row, s_inv, round.beta);
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
        let log_max_degree = self.rounds.len() * params.fold.log_arity() + log_final_poly_degree;
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
