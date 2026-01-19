//! PCS Prover
//!
//! Opens committed matrices at out-of-domain evaluation points.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, FieldArray, TwoAdicField};
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;

use super::config::PcsConfig;
use super::proof::{Proof, QueryProof};
use crate::deep::PointQuotients;
use crate::deep::prover::DeepPoly;
use crate::fri::prover::FriPolys;
use crate::utils::{MatrixGroupEvals, bit_reversed_coset_points};

/// Open committed matrices at N evaluation points.
///
/// # Type Parameters
/// - `F`: Base field (must be two-adic for FRI)
/// - `EF`: Extension field for challenges and evaluations
/// - `Mmcs`: MMCS used for both input matrices and FRI round commitments
/// - `M`: Matrix type for input matrices
/// - `Challenger`: Fiat-Shamir challenger (must support grinding)
/// - `N`: Number of evaluation points (compile-time constant)
///
/// # Arguments
/// - `mmcs`: The MMCS instance used for commitments
/// - `config`: PCS configuration (FRI params + DEEP params)
/// - `eval_points`: Array of N out-of-domain evaluation points
/// - `prover_data`: Prover data from the commitment phase (one per committed group)
/// - `challenger`: Mutable reference to the Fiat-Shamir challenger
///
/// # Returns
/// A `Proof` containing evaluations, grinding witnesses, and all opening proofs
pub fn open<F, EF, Mmcs, M, Challenger, const N: usize>(
    mmcs: &Mmcs,
    config: &PcsConfig,
    eval_points: [EF; N],
    prover_data: Vec<&Mmcs::ProverData<M>>,
    challenger: &mut Challenger,
) -> Proof<F, EF, Mmcs, Challenger::Witness>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    Mmcs: p3_commit::Mmcs<F>,
    M: Matrix<F>,
    Challenger: FieldChallenger<F> + CanObserve<Mmcs::Commitment> + GrindingChallenger,
{
    // ─────────────────────────────────────────────────────────────────────────
    // Extract matrix structure from prover data
    // ─────────────────────────────────────────────────────────────────────────
    let matrices_groups: Vec<Vec<&M>> = prover_data
        .iter()
        .map(|pd| mmcs.get_matrices(*pd))
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
        mmcs,
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
        FriPolys::<F, EF, _>::new(&config.fri, mmcs, &deep_poly.deep_evals, challenger);

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
    let query_proofs: Vec<QueryProof<F, Mmcs>> = query_indices
        .iter()
        .map(|&index| {
            // Open DeepPoly at this index
            let deep_query = deep_poly.open(mmcs, index);

            // Open FRI rounds at this index
            let fri_round_openings = fri_polys.open_query(&config.fri, mmcs, index);

            QueryProof {
                input_openings: deep_query,
                fri_round_openings,
            }
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
