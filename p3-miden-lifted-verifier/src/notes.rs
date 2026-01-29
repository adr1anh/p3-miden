//! Lifted STARK verifier notes (subset from root prover.md).
//!
//! Verifier flow (minimal, current scaffold):
//! 1. Replay transcript: params snapshot, layout, periodic tables, commitments.
//! 2. Sample aux randomness, alphas, beta, zeta (derive zeta_next).
//! 3. Verify openings via lifted FRI `verify_with_channel` at [zeta, zeta_next].
//! 4. Recompose quotient(zeta) from opened base coefficients.
//! 5. Evaluate AIR constraints at zeta^r using the EF constraint folder.
//! 6. Check folded numerator * inv_vanishing == quotient(zeta).
