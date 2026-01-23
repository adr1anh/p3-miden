//! PCS Prover
//!
//! Opens committed matrices at out-of-domain evaluation points.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, FieldArray, TwoAdicField};
use p3_matrix::Matrix;
use p3_miden_lmcs::{Lmcs, LmcsTree};
use p3_util::log2_strict_usize;

use super::config::PcsParams;
use super::proof::Proof;
use crate::deep::PointQuotients;
use crate::deep::prover::DeepPoly;
use crate::fri::prover::FriPolys;
use crate::utils::{MatrixGroupEvals, bit_reversed_coset_points};

/// Open committed matrices at N evaluation points.
pub fn open<F, EF, L, M, Challenger, const N: usize>(
    params: &PcsParams,
    lmcs: &L,
    eval_points: [EF; N],
    trace_trees: &[&L::Tree<M>],
    challenger: &mut Challenger,
) -> Proof<F, EF, L, Challenger::Witness>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    L: Lmcs<F = F>,
    M: Matrix<F>,
    Challenger: FieldChallenger<F> + CanObserve<L::Commitment> + GrindingChallenger,
{
    // ─────────────────────────────────────────────────────────────────────────
    // Extract matrix structure from trees (one group per trace tree)
    // ─────────────────────────────────────────────────────────────────────────
    let matrices_groups: Vec<Vec<&M>> = trace_trees
        .iter()
        .map(|tree| tree.leaves().iter().collect())
        .collect();

    // Determine LDE domain size from tallest matrix across all trees
    let max_height = trace_trees
        .iter()
        .flat_map(|tree| tree.leaves().iter().map(|m| m.height()))
        .max()
        .expect("at least one matrix required");
    let log_n = log2_strict_usize(max_height);
    let coset_points = bit_reversed_coset_points::<F>(log_n);

    // ─────────────────────────────────────────────────────────────────────────
    // Compute evaluations at all N opening points (batched)
    // ─────────────────────────────────────────────────────────────────────────
    let quotient = PointQuotients::<F, EF, N>::new(FieldArray::from(eval_points), &coset_points);
    let batched_evals =
        quotient.batch_eval_lifted(&matrices_groups, &coset_points, params.fri.log_blowup);

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
        &params.deep,
        &matrices_groups,
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
        FriPolys::<F, EF, L>::new(&params.fri, lmcs, deep_poly.deep_evals, challenger);

    // ─────────────────────────────────────────────────────────────────────────
    // Grind for query sampling
    // ─────────────────────────────────────────────────────────────────────────
    let query_pow_witness = challenger.grind(params.query_proof_of_work_bits);

    // ─────────────────────────────────────────────────────────────────────────
    // Sample query indices
    // ─────────────────────────────────────────────────────────────────────────
    let query_indices: Vec<usize> = (0..params.num_queries)
        .map(|_| challenger.sample_bits(log_n))
        .collect();

    // ─────────────────────────────────────────────────────────────────────────
    // Generate query proofs
    // ─────────────────────────────────────────────────────────────────────────
    // Open input trees at all query indices at once (one proof per tree)
    let trace_query_proofs: Vec<L::Proof> = trace_trees
        .iter()
        .map(|tree| tree.prove_batch(&query_indices))
        .collect();

    // Open all FRI rounds at all query indices at once (one proof per round)
    let fri_query_proofs = fri_polys.prove_queries(&params.fri, &query_indices);

    // ─────────────────────────────────────────────────────────────────────────
    // Assemble and return proof
    // ─────────────────────────────────────────────────────────────────────────
    Proof {
        evals,
        deep_proof,
        fri_proof,
        query_pow_witness,
        trace_query_proofs,
        fri_query_proofs,
    }
}
