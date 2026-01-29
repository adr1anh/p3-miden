//! Lifted STARK prover notes (subset from root prover.md).
//!
//! Prover flow (minimal, current scaffold):
//! 1. Compute trace degree and LDE domain (blowup from params).
//! 2. Commit main trace LDE on nested coset (bit-reversed), observe commitment.
//! 3. Sample aux randomness, build aux trace (required), commit aux LDE.
//! 4. Sample per-AIR alpha and a shared beta.
//! 5. Compute folded constraints per AIR on (gK)^r, lift by upsampling into gK.
//! 6. Combine across AIRs via Horner in permutation order; divide by X^N - 1 once.
//! 7. Commit combined quotient, sample zeta, derive zeta_next.
//! 8. Call lifted FRI `open_with_channel` for [zeta, zeta_next].
//!
//! Quotient combination (lifting):
//! - C_i evaluated on (gK)^r; bit-reverse then repeat values r times to lift.
//! - Combine across AIRs with challenge beta; divide by X^N - 1 once.
//!
//! Simplifications enforced in the scaffold:
//! - No preprocessed trace.
//! - Aux trace always present.
//! - ZK out of scope.
