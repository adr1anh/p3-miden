//! FRI proof data structures.

use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};
use p3_miden_lmcs::Lmcs;

/// FRI proof data including per-round grinding witnesses.
///
/// Contains the FRI round commitments, final polynomial, and the proof-of-work
/// witnesses for each folding round's beta challenge.
///
/// Uses LMCS for commitments. Extension field evaluations are flattened to base
/// field before commitment and reconstructed after opening.
pub struct FriProof<EF, L: Lmcs, Witness>
where
    L::F: Field,
{
    /// Merkle commitments for each folding round.
    pub(super) commitments: Vec<L::Commitment>,

    /// Coefficients of the final low-degree polynomial.
    pub(super) final_poly: Vec<EF>,

    /// Proof-of-work witnesses for each round's beta challenge grinding.
    pub(super) pow_witnesses: Vec<Witness>,
}

impl<EF, L: Lmcs, Witness> FriProof<EF, L, Witness>
where
    L::F: Field,
    EF: ExtensionField<L::F>,
{
    pub fn commitments(&self) -> &[L::Commitment] {
        &self.commitments
    }

    pub fn final_poly(&self) -> &[EF] {
        &self.final_poly
    }

    pub fn pow_witnesses(&self) -> &[Witness] {
        &self.pow_witnesses
    }
}
