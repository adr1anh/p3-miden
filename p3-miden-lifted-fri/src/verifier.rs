//! PCS Verifier
//!
//! Verifies polynomial evaluation claims against commitments.

use alloc::vec::Vec;

use p3_challenger::{CanSample, CanSampleBits};
use p3_field::{ExtensionField, TwoAdicField};
use p3_miden_lmcs::Lmcs;
use p3_miden_transcript::{TranscriptError, VerifierChannel};
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
/// - `eval_points` must lie outside both the trace-domain subgroup `H` and the
///   LDE evaluation coset `gK` used by the PCS. If a point lies in either set,
///   denominators `(z_j - X)` in the DEEP quotient become zero for some domain element,
///   making the quotient undefined.
/// - All commitments are expected to be lifted to the same LDE height `2^log_lde_height`.
///
/// `log_lde_height` is the log₂ of the LDE evaluation domain height (i.e. the height of
/// the committed LDE matrices). When a trace degree is known, it is typically
/// `log_trace_height + params.fri.log_blowup` (plus any extension used by the caller).
/// In that common case, the trace subgroup `H` has size `2^(log_lde_height - params.fri.log_blowup)`,
/// while the LDE coset `gK` has size `2^log_lde_height`.
///
/// Trace commitment widths are expected to include any alignment padding
/// (i.e., trees built with `build_aligned_tree`).
///
/// Returns `Ok(evals)` where each group/matrix is a row-major matrix with one row per point.
pub fn verify_with_channel<F, EF, L, Ch, const N: usize>(
    params: &PcsParams,
    lmcs: &L,
    commitments: &[(L::Commitment, Vec<usize>)],
    log_lde_height: usize,
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
        log_lde_height,
        channel,
    )?;

    // ─────────────────────────────────────────────────────────────────────────
    // Create FRI oracle (observes commitments + final poly, checks per-round PoW)
    // ─────────────────────────────────────────────────────────────────────────
    let fri_oracle = FriOracle::new(&params.fri, log_lde_height, channel)?;

    // ─────────────────────────────────────────────────────────────────────────
    // Check query PoW witness and sample query indices
    // ─────────────────────────────────────────────────────────────────────────
    channel.grind(params.query_proof_of_work_bits)?;

    let query_indices: Vec<usize> = (0..params.num_queries)
        .map(|_| channel.sample_bits(log_lde_height))
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
    #[error("transcript error: {0}")]
    TranscriptError(#[from] TranscriptError),
}
