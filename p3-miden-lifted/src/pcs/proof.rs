use alloc::vec::Vec;

use crate::deep::{DeepProof, DeepQuery};
use crate::fri::{FriProof, FriQuery};
use crate::utils::MatrixGroupEvals;
use p3_field::{ExtensionField, Field};

/// Complete PCS opening proof.
///
/// Contains all information needed by the verifier to check polynomial
/// evaluation claims against a commitment:
/// - Evaluations at opening points
/// - DEEP grinding witness
/// - FRI proof with commitments, final polynomial, and per-round grinding witnesses
/// - Query grinding witness
/// - Query proofs with Merkle openings
///
/// Uses a single base-field MMCS for both input matrices and FRI round commitments.
pub struct Proof<F, EF, Mmcs, Witness>
where
    F: Field,
    EF: ExtensionField<F>,
    Mmcs: p3_commit::Mmcs<F>,
{
    /// Claimed evaluations at each opening point.
    /// Structure: `evals[point_idx][commit_idx][matrix_idx][col_idx]`
    pub(super) evals: Vec<Vec<MatrixGroupEvals<EF>>>,

    /// DEEP proof containing grinding witness.
    pub(super) deep_proof: DeepProof<Witness>,

    /// FRI proof containing commitments, final polynomial, and per-round grinding witnesses.
    pub(super) fri_proof: FriProof<F, EF, Mmcs, Witness>,

    /// Proof-of-work witness for query sampling grinding.
    pub(super) query_pow_witness: Witness,

    /// Query phase proofs, one per query index.
    pub(super) query_proofs: Vec<QueryProof<F, Mmcs>>,
}

impl<F, EF, Mmcs, Witness> Proof<F, EF, Mmcs, Witness>
where
    F: Field,
    EF: ExtensionField<F>,
    Mmcs: p3_commit::Mmcs<F>,
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
    pub fn fri_proof(&self) -> &FriProof<F, EF, Mmcs, Witness> {
        &self.fri_proof
    }

    /// Returns the proof-of-work witness for query sampling grinding.
    pub fn query_pow_witness(&self) -> &Witness {
        &self.query_pow_witness
    }

    /// Returns the query phase proofs, one per query index.
    pub fn query_proofs(&self) -> &[QueryProof<F, Mmcs>] {
        &self.query_proofs
    }
}

/// Proof for a single FRI query index.
///
/// Contains Merkle openings for both the input matrices (via DEEP)
/// and each FRI folding round, allowing the verifier to check consistency.
pub struct QueryProof<F: Field, Mmcs: p3_commit::Mmcs<F>> {
    /// Openings of the input matrices at this query index
    /// (one BatchOpening per committed matrix group)
    pub(super) input_openings: DeepQuery<F, Mmcs>,

    /// Openings for each FRI folding round
    pub(super) fri_round_openings: FriQuery<F, Mmcs>,
}

impl<F: Field, Mmcs: p3_commit::Mmcs<F>> QueryProof<F, Mmcs> {
    /// Returns the DEEP query containing input matrix openings.
    pub fn input_openings(&self) -> &DeepQuery<F, Mmcs> {
        &self.input_openings
    }

    /// Returns the FRI query containing folding round openings.
    pub fn fri_round_openings(&self) -> &FriQuery<F, Mmcs> {
        &self.fri_round_openings
    }
}
