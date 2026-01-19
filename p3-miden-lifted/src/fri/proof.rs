//! FRI proof data structures.

use alloc::vec::Vec;

use p3_commit::BatchOpening;
use p3_field::{ExtensionField, Field};

/// FRI proof data including per-round grinding witnesses.
///
/// Contains the FRI round commitments, final polynomial, and the proof-of-work
/// witnesses for each folding round's beta challenge.
///
/// Uses a single base-field MMCS for commitments. Extension field evaluations
/// are flattened to base field before commitment and reconstructed after opening.
pub struct FriProof<F, EF, Mmcs, Witness>
where
    F: Field,
    EF: ExtensionField<F>,
    Mmcs: p3_commit::Mmcs<F>,
{
    /// Merkle commitments for each folding round.
    pub(super) commitments: Vec<Mmcs::Commitment>,

    /// Coefficients of the final low-degree polynomial.
    pub(super) final_poly: Vec<EF>,

    /// Proof-of-work witnesses for each round's beta challenge grinding.
    pub(super) pow_witnesses: Vec<Witness>,
}

impl<F, EF, Mmcs, Witness> FriProof<F, EF, Mmcs, Witness>
where
    F: Field,
    EF: ExtensionField<F>,
    Mmcs: p3_commit::Mmcs<F>,
{
    /// Returns the Merkle commitments for each folding round.
    pub fn commitments(&self) -> &[Mmcs::Commitment] {
        &self.commitments
    }

    /// Returns the coefficients of the final low-degree polynomial.
    pub fn final_poly(&self) -> &[EF] {
        &self.final_poly
    }

    /// Returns the proof-of-work witnesses for each round's beta challenge grinding.
    pub fn pow_witnesses(&self) -> &[Witness] {
        &self.pow_witnesses
    }
}

/// Query proof containing Merkle openings for FRI folding verification.
///
/// Holds the batch openings for each FRI folding round that the verifier
/// needs to check consistency during query verification.
///
/// Openings contain base field values. The verifier reconstructs extension
/// field values after Merkle verification succeeds.
pub struct FriQuery<F: Field, Mmcs: p3_commit::Mmcs<F>> {
    pub(super) openings: Vec<BatchOpening<F, Mmcs>>,
}

impl<F: Field, Mmcs: p3_commit::Mmcs<F>> FriQuery<F, Mmcs> {
    /// Returns the batch openings for each FRI folding round.
    pub fn openings(&self) -> &[BatchOpening<F, Mmcs>] {
        &self.openings
    }
}
