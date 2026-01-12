//! Polynomial Commitment Scheme combining DEEP quotient and FRI.
//!
//! This module provides high-level `open` and `verify` functions that orchestrate
//! the DEEP quotient construction and FRI protocol into a complete PCS.
//!
//! # Overview
//!
//! The PCS operates in two phases:
//!
//! 1. **Opening (Prover)**: Given committed matrices and evaluation points,
//!    computes polynomial evaluations, constructs a DEEP quotient, and generates
//!    a FRI proof of low-degree.
//!
//! 2. **Verification (Verifier)**: Given commitments, evaluation points, and a proof,
//!    verifies the DEEP quotient and FRI queries to confirm the claimed evaluations.

mod config;
mod proof;

use alloc::vec::Vec;
use core::iter::zip;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, FieldArray, TwoAdicField};
use p3_matrix::{Dimensions, Matrix};
use p3_util::log2_strict_usize;

pub use self::config::PcsConfig;
pub use self::proof::{PcsError, Proof, QueryProof};
pub use crate::deep::DeepParams;
use crate::deep::MatrixGroupEvals;
use crate::deep::interpolate::PointQuotients;
use crate::deep::prover::DeepPoly;
use crate::deep::verifier::DeepOracle;
use crate::fri::FriError;
use crate::fri::prover::FriPolys;
use crate::fri::verifier::FriOracle;
use crate::utils::bit_reversed_coset_points;

/// Open committed matrices at N evaluation points.
///
/// # Type Parameters
/// - `F`: Base field (must be two-adic for FRI)
/// - `EF`: Extension field for challenges and evaluations
/// - `InputMmcs`: MMCS used to commit the input matrices
/// - `FriMmcs`: MMCS used for FRI round commitments (typically `ExtensionMmcs<F, EF, _>`)
/// - `M`: Matrix type for input matrices
/// - `Challenger`: Fiat-Shamir challenger (must support grinding)
/// - `N`: Number of evaluation points (compile-time constant)
///
/// # Arguments
/// - `input_mmcs`: The MMCS instance used for initial commitment
/// - `prover_data`: Prover data from the commitment phase (one per committed group)
/// - `eval_points`: Array of N out-of-domain evaluation points
/// - `challenger`: Mutable reference to the Fiat-Shamir challenger
/// - `config`: PCS configuration (FRI params + DEEP params)
/// - `fri_mmcs`: MMCS instance for FRI round commitments
///
/// # Returns
/// A `Proof` containing evaluations, grinding witnesses, and all opening proofs
pub fn open<F, EF, InputMmcs, FriMmcs, M, Challenger, const N: usize>(
    input_mmcs: &InputMmcs,
    prover_data: Vec<&InputMmcs::ProverData<M>>,
    eval_points: [EF; N],
    challenger: &mut Challenger,
    config: &PcsConfig,
    fri_mmcs: &FriMmcs,
) -> Proof<F, EF, InputMmcs, FriMmcs, Challenger::Witness>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    InputMmcs: Mmcs<F>,
    FriMmcs: Mmcs<EF>,
    M: Matrix<F>,
    Challenger: FieldChallenger<F> + CanObserve<FriMmcs::Commitment> + GrindingChallenger,
{
    // ─────────────────────────────────────────────────────────────────────────
    // Extract matrix structure from prover data
    // ─────────────────────────────────────────────────────────────────────────
    let matrices_groups: Vec<Vec<&M>> = prover_data
        .iter()
        .map(|pd| input_mmcs.get_matrices(*pd))
        .collect();

    // Determine LDE domain size from tallest matrix
    let max_height = matrices_groups
        .iter()
        .flat_map(|g| g.iter().map(|m| m.height()))
        .max()
        .expect("at least one matrix required");
    let log_n = log2_strict_usize(max_height);
    let coset_points = bit_reversed_coset_points::<F>(log_n);

    // ─────────────────────────────────────────────────────────────────────────
    // Compute evaluations at all N opening points (batched)
    // ─────────────────────────────────────────────────────────────────────────
    let quotient = PointQuotients::<F, EF, N>::new(FieldArray::from(eval_points), &coset_points);
    let batched_evals =
        quotient.batch_eval_lifted(&matrices_groups, &coset_points, config.fri.log_blowup);

    // Transpose batched evals: [group][matrix][col] of FieldArray<N> -> [point][group][matrix][col] of EF
    let evals: Vec<Vec<MatrixGroupEvals<EF>>> = (0..N)
        .map(|point_idx| {
            batched_evals
                .iter()
                .map(|group| group.map(|arr| arr[point_idx]))
                .collect()
        })
        .collect();

    // ─────────────────────────────────────────────────────────────────────────
    // Construct DEEP quotient (observes evals, grinds, samples α and β)
    // ─────────────────────────────────────────────────────────────────────────
    let (deep_poly, deep_proof) = DeepPoly::new(
        &config.deep,
        input_mmcs,
        prover_data,
        &evals,
        &batched_evals,
        &quotient,
        challenger,
    );

    // ─────────────────────────────────────────────────────────────────────────
    // FRI commit phase (observes commitments, grinds per-round, samples betas)
    // ─────────────────────────────────────────────────────────────────────────
    // The deep_poly contains evaluations on the LDE domain (size max_height).
    // FRI will prove that this polynomial is low-degree.
    let (fri_polys, fri_proof) =
        FriPolys::<F, EF, _>::new(&config.fri, fri_mmcs, &deep_poly.deep_evals, challenger);

    // ─────────────────────────────────────────────────────────────────────────
    // Grind for query sampling
    // ─────────────────────────────────────────────────────────────────────────
    let query_pow_witness = challenger.grind(config.query_proof_of_work_bits);

    // ─────────────────────────────────────────────────────────────────────────
    // Sample query indices
    // ─────────────────────────────────────────────────────────────────────────
    let query_indices: Vec<usize> = (0..config.num_queries)
        .map(|_| challenger.sample_bits(log_n))
        .collect();

    // ─────────────────────────────────────────────────────────────────────────
    // Generate query proofs
    // ─────────────────────────────────────────────────────────────────────────
    let query_proofs: Vec<QueryProof<F, EF, InputMmcs, FriMmcs>> = query_indices
        .iter()
        .map(|&index| {
            // Open DeepPoly at this index
            let deep_query = deep_poly.open(input_mmcs, index);

            // Open FRI rounds at this index
            let fri_round_openings = fri_polys.open_query(&config.fri, fri_mmcs, index);

            QueryProof::new(deep_query, fri_round_openings)
        })
        .collect();

    // ─────────────────────────────────────────────────────────────────────────
    // Assemble and return proof
    // ─────────────────────────────────────────────────────────────────────────
    Proof {
        evals,
        deep_proof,
        fri_proof,
        query_pow_witness,
        query_proofs,
    }
}

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
        .expect("at least one matrix required");
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
    let fri_oracle = FriOracle::new(&proof.fri_proof, challenger, config.fri.proof_of_work_bits)
        .map_err(|e| match e {
            FriError::InvalidPowWitness | FriError::InvalidProofStructure => {
                PcsError::InvalidFriPowWitness
            }
            FriError::MmcsError(err, _) => PcsError::FriMmcsError(err),
            _ => unreachable!("FriOracle::new only returns PoW, structure, or Mmcs errors"),
        })?;

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
        fri_oracle
            .verify_query::<F>(
                &config.fri,
                fri_mmcs,
                index,
                deep_eval,
                &query_proof.fri_round_openings,
            )
            .map_err(|e| match e {
                FriError::MmcsError(err, _) => PcsError::FriMmcsError(err),
                FriError::InvalidProofStructure | FriError::EvaluationMismatch { .. } => {
                    PcsError::FriFoldingError { query_index: index }
                }
                FriError::FinalPolyMismatch => PcsError::FinalPolyMismatch { query_index: index },
                FriError::InvalidPowWitness => unreachable!("PoW already checked"),
            })?;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Return verified evaluations
    // ─────────────────────────────────────────────────────────────────────────
    Ok(proof.evals.clone())
}

#[cfg(test)]
mod tests;
