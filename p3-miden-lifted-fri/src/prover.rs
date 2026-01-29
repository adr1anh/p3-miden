//! PCS Prover
//!
//! Opens committed matrices at out-of-domain evaluation points.

use alloc::vec::Vec;

use p3_challenger::{CanSample, CanSampleBits};
use p3_field::{ExtensionField, FieldArray, TwoAdicField};
use p3_matrix::Matrix;
use p3_miden_lmcs::{Lmcs, LmcsTree};
use p3_miden_transcript::ProverChannel;

use crate::PcsParams;
use crate::deep::PointQuotients;
use crate::deep::prover::DeepPoly;
use crate::fri::prover::FriPolys;
use crate::utils::bit_reversed_coset_points;

/// Open committed matrices at N evaluation points, writing to a prover channel.
///
/// # Preconditions
/// - `eval_points` must lie outside both the trace-domain subgroup `H` and the
///   LDE evaluation coset `gK` used by the PCS. If a point lies in either set,
///   denominators `(z_j - X)` in the DEEP quotient become zero for some domain element,
///   making the quotient undefined.
/// - All trace trees must be built at the same LDE height `2^log_lde_height`.
///   Multiple LDE heights are not supported yet and will panic.
///
/// `log_lde_height` is the log₂ of the LDE evaluation domain height (i.e. the height of
/// the committed LDE matrices). When a trace degree is known, it is typically
/// `log_trace_height + params.fri.log_blowup` (plus any extension used by the caller).
/// In that common case, the trace subgroup `H` has size `2^(log_lde_height - params.fri.log_blowup)`,
/// while the LDE coset `gK` has size `2^log_lde_height`.
///
/// Alignment is derived from the LMCS instance to pad DEEP evaluations consistently.
pub fn open_with_channel<F, EF, L, M, Ch, const N: usize>(
    params: &PcsParams,
    lmcs: &L,
    log_lde_height: usize,
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

    // Determine LDE domain size from the supplied LDE height.
    // For now, all trace trees must share this height; mixed LDE heights are not supported yet.
    assert!(!trace_trees.is_empty(), "at least one trace tree required");
    let expected_height = 1usize << log_lde_height;
    assert!(
        trace_trees
            .iter()
            .all(|tree| tree.height() == expected_height),
        "mixed LDE heights are not supported yet",
    );
    let coset_points = bit_reversed_coset_points::<F>(log_lde_height);

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
    // The deep_poly contains evaluations on the LDE domain (size 2^log_lde_height).
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
        .map(|_| channel.sample_bits(log_lde_height))
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
