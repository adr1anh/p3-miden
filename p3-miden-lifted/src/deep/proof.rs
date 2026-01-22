//! DEEP proof data structures.

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
