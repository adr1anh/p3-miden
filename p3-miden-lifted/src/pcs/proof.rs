use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_commit::{BatchOpening, Mmcs};
use p3_field::{ExtensionField, Field};

use crate::deep::{DeepProof, DeepQuery, MatrixGroupEvals};
use crate::fri::FriProof;

/// Complete PCS opening proof.
///
/// Contains all information needed by the verifier to check polynomial
/// evaluation claims against a commitment:
/// - Evaluations at opening points
/// - DEEP grinding witness
/// - FRI proof with commitments, final polynomial, and per-round grinding witnesses
/// - Query grinding witness
/// - Query proofs with Merkle openings
pub struct Proof<F: Field, EF: ExtensionField<F>, InputMmcs: Mmcs<F>, FriMmcs: Mmcs<EF>, Witness> {
    /// Claimed evaluations at each opening point.
    /// Structure: `evals[point_idx][commit_idx][matrix_idx][col_idx]`
    pub(crate) evals: Vec<Vec<MatrixGroupEvals<EF>>>,

    /// DEEP proof containing grinding witness.
    pub(crate) deep_proof: DeepProof<Witness>,

    /// FRI proof containing commitments, final polynomial, and per-round grinding witnesses.
    pub(crate) fri_proof: FriProof<EF, FriMmcs, Witness>,

    /// Proof-of-work witness for query sampling grinding.
    pub(crate) query_pow_witness: Witness,

    /// Query phase proofs, one per query index.
    pub(crate) query_proofs: Vec<QueryProof<F, EF, InputMmcs, FriMmcs>>,
}

/// Proof for a single FRI query index.
///
/// Contains Merkle openings for both the input matrices (via DEEP)
/// and each FRI folding round, allowing the verifier to check consistency.
pub struct QueryProof<F: Field, EF: ExtensionField<F>, InputMmcs: Mmcs<F>, FriMmcs: Mmcs<EF>> {
    /// Openings of the input matrices at this query index
    /// (one BatchOpening per committed matrix group)
    pub(crate) input_openings: DeepQuery<F, InputMmcs>,

    /// Openings for each FRI folding round
    pub(crate) fri_round_openings: Vec<BatchOpening<EF, FriMmcs>>,

    _marker: PhantomData<F>,
}

impl<F: Field, EF: ExtensionField<F>, InputMmcs: Mmcs<F>, FriMmcs: Mmcs<EF>>
    QueryProof<F, EF, InputMmcs, FriMmcs>
{
    /// Create a new query proof from input and FRI round openings.
    pub const fn new(
        input_openings: DeepQuery<F, InputMmcs>,
        fri_round_openings: Vec<BatchOpening<EF, FriMmcs>>,
    ) -> Self {
        Self {
            input_openings,
            fri_round_openings,
            _marker: PhantomData,
        }
    }
}
