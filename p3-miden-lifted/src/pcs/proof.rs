use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};
use p3_miden_lmcs::Lmcs;

use crate::deep::DeepProof;
use crate::fri::FriProof;
use crate::utils::MatrixGroupEvals;

/// Complete PCS opening proof.
pub struct Proof<F, EF, L, Witness>
where
    F: Field,
    EF: ExtensionField<F>,
    L: Lmcs<F = F>,
{
    /// `evals[point_idx][commit_idx][matrix_idx][col_idx]`
    pub(super) evals: Vec<Vec<MatrixGroupEvals<EF>>>,
    pub(super) deep_proof: DeepProof<Witness>,
    pub(super) fri_proof: FriProof<EF, L, Witness>,
    pub(super) query_pow_witness: Witness,
    /// One proof per trace tree.
    pub(super) trace_query_proofs: Vec<L::Proof>,
    /// One proof per FRI round.
    pub(super) fri_query_proofs: Vec<L::Proof>,
}

impl<F, EF, L, Witness> Proof<F, EF, L, Witness>
where
    F: Field,
    EF: ExtensionField<F>,
    L: Lmcs<F = F>,
{
    pub fn evals(&self) -> &[Vec<MatrixGroupEvals<EF>>] {
        &self.evals
    }

    pub fn deep_proof(&self) -> &DeepProof<Witness> {
        &self.deep_proof
    }

    pub fn fri_proof(&self) -> &FriProof<EF, L, Witness> {
        &self.fri_proof
    }

    pub fn query_pow_witness(&self) -> &Witness {
        &self.query_pow_witness
    }

    pub fn trace_query_proofs(&self) -> &[L::Proof] {
        &self.trace_query_proofs
    }

    pub fn fri_query_proofs(&self) -> &[L::Proof] {
        &self.fri_query_proofs
    }
}
