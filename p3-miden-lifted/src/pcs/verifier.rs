//! PCS Verifier
//!
//! Verifies polynomial evaluation claims against commitments.

use alloc::vec::Vec;
use core::iter::zip;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::Dimensions;
use p3_util::log2_strict_usize;
use thiserror::Error;

use super::config::PcsConfig;
use super::proof::Proof;
use crate::deep::DeepError;
use crate::deep::verifier::DeepOracle;
use crate::fri::FriError;
use crate::fri::verifier::FriOracle;
use crate::utils::MatrixGroupEvals;

/// Verify polynomial evaluation claims against a commitment.
///
/// # Type Parameters
/// Same as `open`
///
/// # Arguments
/// - `input_mmcs`: The MMCS instance used for initial commitment
/// - `commitments`: The commitments to verify against (with dimensions)
/// - `eval_points`: Array of N out-of-domain evaluation points
/// - `proof`: The proof to verify
/// - `challenger`: Mutable reference to the Fiat-Shamir challenger (must support grinding)
/// - `config`: PCS configuration (must match prover's)
/// - `fri_mmcs`: MMCS instance for FRI round commitments
///
/// # Returns
/// `Ok(evals)` where `evals[point_idx][commit_idx]` contains the verified evaluations,
/// or `Err` if verification fails.
#[allow(clippy::type_complexity)]
pub fn verify<F, EF, InputMmcs, FriMmcs, Challenger, const N: usize>(
    input_mmcs: &InputMmcs,
    commitments: &[(InputMmcs::Commitment, Vec<Dimensions>)],
    eval_points: [EF; N],
    proof: &Proof<F, EF, InputMmcs, FriMmcs, Challenger::Witness>,
    challenger: &mut Challenger,
    config: &PcsConfig,
    fri_mmcs: &FriMmcs,
) -> Result<Vec<Vec<MatrixGroupEvals<EF>>>, PcsError<InputMmcs::Error, FriMmcs::Error>>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    InputMmcs: Mmcs<F>,
    FriMmcs: Mmcs<EF>,
    Challenger: FieldChallenger<F> + CanObserve<FriMmcs::Commitment> + GrindingChallenger,
    InputMmcs::Error: core::fmt::Debug,
    FriMmcs::Error: core::fmt::Debug,
{
    // ─────────────────────────────────────────────────────────────────────────
    // Validate proof structure
    // ─────────────────────────────────────────────────────────────────────────
    if proof.query_proofs.len() != config.num_queries {
        return Err(PcsError::WrongNumQueries {
            expected: config.num_queries,
            actual: proof.query_proofs.len(),
        });
    }

    // Extract dimensions for computing domain
    let max_height = commitments
        .iter()
        .flat_map(|(_, dims)| dims.iter().map(|d| d.height))
        .max()
        .ok_or(PcsError::NoCommitments)?;
    let log_n = log2_strict_usize(max_height);

    // ─────────────────────────────────────────────────────────────────────────
    // Construct verifier's DEEP oracle (observes evals, checks PoW, samples α/β)
    // ─────────────────────────────────────────────────────────────────────────
    let deep_oracle = DeepOracle::new(
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
    // Verify each query
    // ─────────────────────────────────────────────────────────────────────────
    for (index, query_proof) in zip(query_indices, &proof.query_proofs) {
        // Verify input matrix openings and compute expected DeepPoly value
        let deep_eval = deep_oracle
            .query(input_mmcs, index, &query_proof.input_openings)
            .map_err(PcsError::InputMmcsError)?;

        // Verify FRI rounds
        fri_oracle.verify_query(
            &config.fri,
            fri_mmcs,
            index,
            deep_eval,
            &query_proof.fri_round_openings,
        )?;
    }

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
pub enum PcsError<InputMmcsError, FriMmcsError> {
    /// No commitments provided for verification.
    #[error("no commitments provided")]
    NoCommitments,
    /// Wrong number of queries in proof.
    #[error("wrong number of queries: expected {expected}, got {actual}")]
    WrongNumQueries { expected: usize, actual: usize },
    /// Input MMCS verification failed.
    #[error("input MMCS error: {0:?}")]
    InputMmcsError(InputMmcsError),
    /// DEEP oracle construction failed.
    #[error("DEEP error: {0}")]
    DeepError(#[from] DeepError),
    /// FRI verification failed.
    #[error("FRI error: {0}")]
    FriError(#[from] FriError<FriMmcsError>),
    /// Query proof-of-work witness verification failed.
    #[error("invalid query proof-of-work witness")]
    InvalidQueryPowWitness,
}
