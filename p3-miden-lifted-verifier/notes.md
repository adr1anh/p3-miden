# p3-miden-lifted-verifier notes

This crate hosts the end-to-end verification flow for the lifted STARK protocol.

## Highlights
- Replays the transcript to reconstruct params, layout, periodic tables, and commitments.
- Uses `p3-miden-lifted-fri::verifier::verify_with_channel` to open [zeta, zeta_next].
- Recomputes folded constraints at zeta^r per AIR and checks the combined quotient.

## TODO / follow-ups
- Consider a dedicated error type for periodic table validation failures.
- Decide how to expose `StarkTranscript` (debug-only vs public API).

## Verifier flow notes (subset from root prover.md)

Verifier flow (minimal, current scaffold):
1. Replay transcript: params snapshot, layout, periodic tables, commitments.
2. Sample aux randomness, alphas, beta, then rejection-sample zeta outside H and gK (loop ~1x, derive zeta_next).
3. Verify openings via lifted FRI `verify_with_channel` at [zeta, zeta_next].
4. Recompose quotient(zeta) from opened base coefficients.
5. Evaluate AIR constraints at zeta^r using the EF constraint folder.
6. Check folded numerator * inv_vanishing == quotient(zeta).
