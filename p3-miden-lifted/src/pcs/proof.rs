use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};
use p3_miden_lmcs::Lmcs;

use crate::deep::DeepProof;
use crate::fri::FriProof;
use crate::utils::MatrixGroupEvals;

/// Complete PCS opening proof.
///
/// Contains all information needed by the verifier to check polynomial
/// evaluation claims against a commitment:
/// - Evaluations at opening points
/// - DEEP grinding witness
/// - FRI proof with commitments, final polynomial, and per-round grinding witnesses
/// - Query grinding witness
/// - Query proofs: compact multi-opening proofs for trace trees and FRI rounds
///
/// Uses a single LMCS for both trace commitments and FRI round commitments.
pub struct Proof<F, EF, L, Witness>
where
    F: Field,
    EF: ExtensionField<F>,
    L: Lmcs<F = F>,
{
    /// Claimed evaluations at each opening point.
    /// Structure: `evals[point_idx][commit_idx][matrix_idx][col_idx]`
    pub(super) evals: Vec<Vec<MatrixGroupEvals<EF>>>,

    /// DEEP proof containing grinding witness.
    pub(super) deep_proof: DeepProof<Witness>,

    /// FRI proof containing commitments, final polynomial, and per-round grinding witnesses.
    pub(super) fri_proof: FriProof<EF, L, Witness>,

    /// Proof-of-work witness for query sampling grinding.
    pub(super) query_pow_witness: Witness,

    /// Compact multi-opening proofs for trace matrices at all query indices.
    /// One proof per trace tree, each covering all query indices.
    pub(super) trace_query_proofs: Vec<L::Proof>,

    /// Compact multi-opening proofs for FRI rounds at all query indices.
    /// One proof per FRI round, each covering all query indices for that round.
    pub(super) fri_query_proofs: Vec<L::Proof>,
}

impl<F, EF, L, Witness> Proof<F, EF, L, Witness>
where
    F: Field,
    EF: ExtensionField<F>,
    L: Lmcs<F = F>,
{
    /// Returns the claimed evaluations at each opening point.
    ///
    /// Structure: `evals[point_idx][commit_idx]` yields a [`MatrixGroupEvals`]
    /// containing `[matrix_idx][col_idx]` evaluations.
    pub fn evals(&self) -> &[Vec<MatrixGroupEvals<EF>>] {
        &self.evals
    }

    /// Returns the DEEP proof containing the grinding witness.
    pub fn deep_proof(&self) -> &DeepProof<Witness> {
        &self.deep_proof
    }

    /// Returns the FRI proof with commitments, final polynomial, and per-round witnesses.
    pub fn fri_proof(&self) -> &FriProof<EF, L, Witness> {
        &self.fri_proof
    }

    /// Returns the proof-of-work witness for query sampling grinding.
    pub fn query_pow_witness(&self) -> &Witness {
        &self.query_pow_witness
    }

    /// Returns the compact multi-opening proofs for trace matrices.
    /// One proof per trace tree, each covering all query indices.
    pub fn trace_query_proofs(&self) -> &[L::Proof] {
        &self.trace_query_proofs
    }

    /// Returns the compact multi-opening proofs for FRI rounds.
    /// One proof per FRI round, each covering all query indices for that round.
    pub fn fri_query_proofs(&self) -> &[L::Proof] {
        &self.fri_query_proofs
    }
}
