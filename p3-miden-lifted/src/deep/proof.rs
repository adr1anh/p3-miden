//! DEEP proof data structures.

use alloc::vec::Vec;

use p3_commit::{BatchOpening, Mmcs};
use p3_field::Field;

/// DEEP proof-of-work witness.
///
/// The evaluations are stored in the PCS `Proof` struct.
/// This struct only contains the grinding witness for DEEP challenge sampling.
pub struct DeepProof<Witness> {
    /// Proof-of-work witness for DEEP challenge grinding.
    pub(super) pow_witness: Witness,
}

impl<Witness> DeepProof<Witness> {
    /// Returns the proof-of-work witness for DEEP challenge grinding.
    pub fn pow_witness(&self) -> &Witness {
        &self.pow_witness
    }
}

/// Query proof containing Merkle openings for DEEP quotient verification.
///
/// Holds the batch openings from the input commitment that the verifier
/// needs to reconstruct `f_reduced(X)` at the queried point.
pub struct DeepQuery<F: Field, Commit: Mmcs<F>> {
    pub(super) openings: Vec<BatchOpening<F, Commit>>,
}

impl<F: Field, Commit: Mmcs<F>> DeepQuery<F, Commit> {
    /// Returns the batch openings from the input commitment.
    pub fn openings(&self) -> &[BatchOpening<F, Commit>] {
        &self.openings
    }
}
