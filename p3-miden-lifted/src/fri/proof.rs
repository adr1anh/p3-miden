//! FRI proof data structures.

use alloc::vec::Vec;

use p3_commit::{BatchOpening, Mmcs};
use p3_field::Field;

/// FRI proof data including per-round grinding witnesses.
///
/// Contains the FRI round commitments, final polynomial, and the proof-of-work
/// witnesses for each folding round's beta challenge.
pub struct FriProof<EF: Field, FriMmcs: Mmcs<EF>, Witness> {
    /// Merkle commitments for each folding round.
    pub(super) commitments: Vec<FriMmcs::Commitment>,

    /// Coefficients of the final low-degree polynomial.
    pub(super) final_poly: Vec<EF>,

    /// Proof-of-work witnesses for each round's beta challenge grinding.
    pub(super) pow_witnesses: Vec<Witness>,
}

impl<EF: Field, FriMmcs: Mmcs<EF>, Witness> FriProof<EF, FriMmcs, Witness> {
    /// Returns the Merkle commitments for each folding round.
    pub fn commitments(&self) -> &[FriMmcs::Commitment] {
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
pub struct FriQuery<EF: Field, FriMmcs: Mmcs<EF>> {
    pub(super) openings: Vec<BatchOpening<EF, FriMmcs>>,
}

impl<EF: Field, FriMmcs: Mmcs<EF>> FriQuery<EF, FriMmcs> {
    /// Returns the batch openings for each FRI folding round.
    pub fn openings(&self) -> &[BatchOpening<EF, FriMmcs>] {
        &self.openings
    }
}
