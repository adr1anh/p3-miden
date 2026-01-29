//! PCS Verifier
//!
//! Verifies polynomial evaluation claims against commitments.

use alloc::vec::Vec;

use p3_challenger::{CanSample, CanSampleBits};
use p3_field::{ExtensionField, TwoAdicField};
use p3_miden_lmcs::Lmcs;
use p3_miden_transcript::VerifierChannel;
use thiserror::Error;

use crate::PcsParams;
use crate::deep::DeepError;
use crate::deep::DeepEvals;
use crate::deep::verifier::DeepOracle;
use crate::fri::FriError;
use crate::fri::verifier::FriOracle;

/// Verify polynomial evaluation claims against commitments using a verifier channel.
///
/// # Preconditions
/// - `eval_points` must be outside the evaluation domain `gK` (caller must ensure this).
/// - All commitments are expected to be lifted to the same max height `2^log_max_height`.
///
/// Commitment widths are expected to be aligned to the LMCS alignment.
///
/// Returns `Ok(evals)` where each group/matrix is a row-major matrix with one row per point.
pub fn verify_with_channel<F, EF, L, Ch, const N: usize>(
    params: &PcsParams,
    lmcs: &L,
    commitments: &[(L::Commitment, Vec<usize>)],
    log_max_height: usize,
    eval_points: [EF; N],
    channel: &mut Ch,
) -> Result<DeepEvals<EF>, PcsError>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + PartialEq + Clone,
    L: Lmcs<F = F>,
    Ch: VerifierChannel<F = F, Commitment = L::Commitment> + CanSample<F> + CanSampleBits<usize>,
{
    // Validate we have commitments
    if commitments.is_empty() {
        return Err(PcsError::NoCommitments);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Construct verifier's DEEP oracle (observes evals, checks PoW, samples α/β)
    // ─────────────────────────────────────────────────────────────────────────
    let (deep_oracle, evals) = DeepOracle::<F, EF, L>::new(
        &params.deep,
        &eval_points,
        commitments.to_vec(),
        log_max_height,
        channel,
    )?;

    // ─────────────────────────────────────────────────────────────────────────
    // Create FRI oracle (observes commitments + final poly, checks per-round PoW)
    // ─────────────────────────────────────────────────────────────────────────
    let fri_oracle = FriOracle::new(&params.fri, log_max_height, channel)?;

    // ─────────────────────────────────────────────────────────────────────────
    // Check query PoW witness and sample query indices
    // ─────────────────────────────────────────────────────────────────────────
    if channel.grind(params.query_proof_of_work_bits).is_none() {
        return Err(PcsError::InvalidQueryPowWitness);
    }

    let query_indices: Vec<usize> = (0..params.num_queries)
        .map(|_| channel.sample_bits(log_max_height))
        .collect();

    // ─────────────────────────────────────────────────────────────────────────
    // Verify DEEP openings for all queries at once
    // ─────────────────────────────────────────────────────────────────────────
    let deep_evals = deep_oracle.open_batch(lmcs, &query_indices, channel)?;

    // ─────────────────────────────────────────────────────────────────────────
    // Test low-degree proximity for all queries at once
    // ─────────────────────────────────────────────────────────────────────────
    fri_oracle.test_low_degree(lmcs, &params.fri, &query_indices, &deep_evals, channel)?;

    // ─────────────────────────────────────────────────────────────────────────
    // Return verified evaluations
    // ─────────────────────────────────────────────────────────────────────────
    Ok(evals)
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
