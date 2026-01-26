//! PCS Verifier
//!
//! Verifies polynomial evaluation claims against commitments.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, TwoAdicField};
use p3_miden_lmcs::Lmcs;
use thiserror::Error;

use crate::PcsParams;
use crate::deep::DeepError;
use crate::deep::verifier::DeepOracle;
use crate::fri::FriError;
use crate::fri::verifier::FriOracle;
use crate::proof::Proof;
use crate::utils::MatrixGroupEvals;

/// Verify polynomial evaluation claims against commitments.
///
/// Returns `Ok(evals)` where `evals[point_idx][commit_idx]` contains the verified evaluations.
pub fn verify<F, EF, L, Challenger, const N: usize>(
    params: &PcsParams,
    lmcs: &L,
    commitments: &[(L::Commitment, Vec<usize>)],
    log_max_height: usize,
    eval_points: [EF; N],
    proof: &Proof<F, EF, L, Challenger::Witness>,
    challenger: &mut Challenger,
) -> Result<Vec<Vec<MatrixGroupEvals<EF>>>, PcsError>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    L: Lmcs<F = F>,
    Challenger: FieldChallenger<F> + CanObserve<L::Commitment> + GrindingChallenger,
{
    // Validate we have commitments
    if commitments.is_empty() {
        return Err(PcsError::NoCommitments);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Construct verifier's DEEP oracle (observes evals, checks PoW, samples α/β)
    // ─────────────────────────────────────────────────────────────────────────
    let deep_oracle = DeepOracle::<F, EF, L>::new(
        &params.deep,
        eval_points,
        &proof.evals,
        commitments.to_vec(),
        log_max_height,
        challenger,
        &proof.deep_proof,
    )?;

    // ─────────────────────────────────────────────────────────────────────────
    // Create FRI oracle (observes commitments + final poly, checks per-round PoW)
    // ─────────────────────────────────────────────────────────────────────────
    let fri_oracle = FriOracle::new(&proof.fri_proof, challenger, params.fri.proof_of_work_bits)?;

    // ─────────────────────────────────────────────────────────────────────────
    // Check query PoW witness and sample query indices
    // ─────────────────────────────────────────────────────────────────────────
    if !challenger.check_witness(params.query_proof_of_work_bits, proof.query_pow_witness) {
        return Err(PcsError::InvalidQueryPowWitness);
    }

    let query_indices: Vec<usize> = (0..params.num_queries)
        .map(|_| challenger.sample_bits(log_max_height))
        .collect();

    // ─────────────────────────────────────────────────────────────────────────
    // Verify DEEP openings for all queries at once
    // ─────────────────────────────────────────────────────────────────────────
    let deep_evals = deep_oracle.open_batch(lmcs, &query_indices, &proof.trace_query_proofs)?;

    // ─────────────────────────────────────────────────────────────────────────
    // Test low-degree proximity for all queries at once
    // ─────────────────────────────────────────────────────────────────────────
    fri_oracle.test_low_degree(
        lmcs,
        &params.fri,
        &query_indices,
        &deep_evals,
        &proof.fri_query_proofs,
    )?;

    // ─────────────────────────────────────────────────────────────────────────
    // Return verified evaluations
    // ─────────────────────────────────────────────────────────────────────────
    Ok(proof.evals.clone())
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during PCS verification.
#[derive(Debug, Error)]
pub enum PcsError {
    #[error("no commitments provided")]
    NoCommitments,
    #[error("DEEP error: {0}")]
    DeepError(#[from] DeepError),
    #[error("FRI error: {0}")]
    FriError(#[from] FriError),
    #[error("invalid query proof-of-work witness")]
    InvalidQueryPowWitness,
}
