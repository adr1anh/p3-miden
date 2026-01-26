//! DEEP proof data structures.

/// DEEP proof-of-work witness.
pub struct DeepProof<Witness> {
    pub(super) pow_witness: Witness,
}

impl<Witness> DeepProof<Witness> {
    pub fn pow_witness(&self) -> &Witness {
        &self.pow_witness
    }
}
