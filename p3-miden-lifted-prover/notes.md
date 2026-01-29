# p3-miden-lifted-prover notes

This crate hosts the end-to-end proving flow for the lifted STARK protocol.

## Highlights
- Uses LMCS to commit LDEs over nested cosets ((gK)^r) for each AIR trace.
- Aux trace is required; preprocessed trace is ignored in this scaffold.
- Combines per-AIR constraint numerators on the max domain via upsample + Horner.
- Opens all committed trees via `p3-miden-lifted-fri::prover::open_with_channel`.
- Channel-first API: `prove_with_channel` accepts a `ProverChannel`; `prove` is a wrapper.
- Transcript parameters are encoded as checked `u32` values (writes return `Option`).

## TODO / follow-ups
- Move quotient helpers into a shared `quotient.rs` once the API stabilizes.
- Decide whether to remove `alignment` from config in favor of `Lmcs::alignment()`.
- Clarify periodic column encoding (currently LDE on nested coset + index mod).

## Prover flow notes (subset from root prover.md)

Prover flow (minimal, current scaffold):
1. Compute trace degree and LDE domain (blowup from params).
2. Commit main trace LDE on nested coset (bit-reversed), observe commitment.
3. Sample aux randomness, build aux trace (required), commit aux LDE.
4. Sample per-AIR alpha and a shared beta.
5. Compute folded constraints per AIR on (gK)^r, lift by upsample into gK.
6. Combine across AIRs via Horner in permutation order; divide by X^N - 1 once.
7. Commit combined quotient, rejection-sample zeta outside H and gK (loop ~1x), derive zeta_next.
8. Call lifted FRI `open_with_channel` for [zeta, zeta_next].

### Quotient combination (lifting)
- C_i evaluated on (gK)^r; bit-reverse then repeat values r times to lift.
- Combine across AIRs with challenge beta; divide by X^N - 1 once.

### Simplifications enforced in the scaffold
- No preprocessed trace.
- Aux trace always present.
- ZK out of scope.
