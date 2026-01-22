//! PCS Verifier
//!
//! Verifies polynomial evaluation claims against commitments.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::Dimensions;
use p3_miden_lmcs::Lmcs;
use p3_util::log2_strict_usize;
use thiserror::Error;

use super::config::PcsConfig;
use super::proof::Proof;
use crate::deep::DeepError;
use crate::deep::verifier::DeepOracle;
use crate::fri::FriError;
use crate::fri::verifier::FriOracle;
use crate::utils::MatrixGroupEvals;

/// Verify polynomial evaluation claims against commitments.
///
/// # Type Parameters
/// - `F`: Base field (must be two-adic for FRI)
/// - `EF`: Extension field for challenges and evaluations
/// - `L`: LMCS used for both input matrices and FRI round commitments
/// - `Challenger`: Fiat-Shamir challenger (must support grinding)
/// - `N`: Number of evaluation points (compile-time constant)
///
/// # Arguments
/// - `lmcs`: The LMCS instance used for all commitments
/// - `commitments`: The trace commitments with their dimensions (one per trace tree)
/// - `eval_points`: Array of N out-of-domain evaluation points
/// - `proof`: The proof to verify
/// - `challenger`: Mutable reference to the Fiat-Shamir challenger (must support grinding)
/// - `config`: PCS configuration (must match prover's)
///
/// # Returns
/// `Ok(evals)` where `evals[point_idx][commit_idx]` contains the verified evaluations,
/// or `Err` if verification fails.
pub fn verify<F, EF, L, Challenger, const N: usize>(
    lmcs: &L,
    commitments: &[(L::Commitment, Vec<Dimensions>)],
    eval_points: [EF; N],
    proof: &Proof<F, EF, L, Challenger::Witness>,
    challenger: &mut Challenger,
    config: &PcsConfig,
) -> Result<Vec<Vec<MatrixGroupEvals<EF>>>, PcsError>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    L: Lmcs<F = F>,
    Challenger: FieldChallenger<F> + CanObserve<L::Commitment> + GrindingChallenger,
{
    // ─────────────────────────────────────────────────────────────────────────
    // Extract dimensions for computing domain
    // ─────────────────────────────────────────────────────────────────────────
    let max_height = commitments
        .iter()
        .flat_map(|(_, dims)| dims.iter().map(|d| d.height))
        .max()
        .ok_or(PcsError::NoCommitments)?;
    let log_n = log2_strict_usize(max_height);

    // ─────────────────────────────────────────────────────────────────────────
    // Construct verifier's DEEP oracle (observes evals, checks PoW, samples α/β)
    // ─────────────────────────────────────────────────────────────────────────
    let deep_oracle = DeepOracle::<F, EF, L>::new(
        &config.deep,
        eval_points,
        &proof.evals,
        commitments.to_vec(),
        challenger,
        &proof.deep_proof,
    )?;

    // ─────────────────────────────────────────────────────────────────────────
    // Create FRI oracle (observes commitments + final poly, checks per-round PoW)
    // ─────────────────────────────────────────────────────────────────────────
    let fri_oracle = FriOracle::new(&proof.fri_proof, challenger, config.fri.proof_of_work_bits)?;

    // ─────────────────────────────────────────────────────────────────────────
    // Check query PoW witness and sample query indices
    // ─────────────────────────────────────────────────────────────────────────
    if !challenger.check_witness(config.query_proof_of_work_bits, proof.query_pow_witness) {
        return Err(PcsError::InvalidQueryPowWitness);
    }

    let query_indices: Vec<usize> = (0..config.num_queries)
        .map(|_| challenger.sample_bits(log_n))
        .collect();

    // ─────────────────────────────────────────────────────────────────────────
    // Verify DEEP openings for all queries at once
    // ─────────────────────────────────────────────────────────────────────────
    let deep_evals = deep_oracle.query(lmcs, &query_indices, &proof.trace_query_proofs)?;

    // ─────────────────────────────────────────────────────────────────────────
    // Verify FRI rounds for all queries at once
    // ─────────────────────────────────────────────────────────────────────────
    fri_oracle.verify_queries(
        lmcs,
        &config.fri,
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
    /// No commitments provided for verification.
    #[error("no commitments provided")]
    NoCommitments,
    /// DEEP oracle verification failed.
    #[error("DEEP error: {0}")]
    DeepError(#[from] DeepError),
    /// FRI verification failed.
    #[error("FRI error: {0}")]
    FriError(#[from] FriError),
    /// Query proof-of-work witness verification failed.
    #[error("invalid query proof-of-work witness")]
    InvalidQueryPowWitness,
}
