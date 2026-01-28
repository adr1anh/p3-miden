//! PCS Prover
//!
//! Opens committed matrices at out-of-domain evaluation points.

use alloc::vec::Vec;

use p3_challenger::{CanSample, CanSampleBits};
use p3_field::{ExtensionField, FieldArray, TwoAdicField};
use p3_matrix::Matrix;
use p3_miden_lmcs::{Lmcs, LmcsTree};
use p3_miden_transcript::ProverChannel;
use p3_util::log2_strict_usize;

use crate::PcsParams;
use crate::deep::PointQuotients;
use crate::deep::prover::DeepPoly;
use crate::fri::prover::FriPolys;
use crate::utils::bit_reversed_coset_points;

/// Open committed matrices at N evaluation points, writing to a prover channel.
///
/// Alignment is derived from the LMCS instance to pad DEEP evaluations consistently.
pub fn open_with_channel<F, EF, L, M, Ch, const N: usize>(
    params: &PcsParams,
    lmcs: &L,
    eval_points: [EF; N],
    trace_trees: &[&L::Tree<M>],
    channel: &mut Ch,
) where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    L: Lmcs<F = F>,
    M: Matrix<F>,
    Ch: ProverChannel<F = F, Commitment = L::Commitment> + CanSample<F> + CanSampleBits<usize>,
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
    let alignment = lmcs.alignment();

    // ─────────────────────────────────────────────────────────────────────────
    // Construct DEEP quotient (observes evals, grinds, samples α and β)
    // ─────────────────────────────────────────────────────────────────────────
    let deep_poly = DeepPoly::new(
        &params.deep,
        &matrices_groups,
        batched_evals,
        &quotient,
        alignment,
        channel,
    );

    // ─────────────────────────────────────────────────────────────────────────
    // FRI commit phase (observes commitments, grinds per-round, samples betas)
    // ─────────────────────────────────────────────────────────────────────────
    // The deep_poly contains evaluations on the LDE domain (size max_height).
    // FRI will prove that this polynomial is low-degree.
    let fri_polys = FriPolys::<F, EF, L>::new(&params.fri, lmcs, deep_poly.deep_evals, channel);

    // ─────────────────────────────────────────────────────────────────────────
    // Grind for query sampling
    // ─────────────────────────────────────────────────────────────────────────
    let _query_pow_witness = channel.grind(params.query_proof_of_work_bits);

    // ─────────────────────────────────────────────────────────────────────────
    // Sample query indices
    // ─────────────────────────────────────────────────────────────────────────
    let query_indices: Vec<usize> = (0..params.num_queries)
        .map(|_| channel.sample_bits(log_n))
        .collect();

    // ─────────────────────────────────────────────────────────────────────────
    // Generate query proofs
    // ─────────────────────────────────────────────────────────────────────────
    // Open input trees at all query indices at once (one proof per tree)
    for tree in trace_trees {
        tree.prove_batch(&query_indices, channel);
    }

    // Open all FRI rounds at all query indices at once (one proof per round)
    fri_polys.prove_queries(&params.fri, &query_indices, channel);
}
