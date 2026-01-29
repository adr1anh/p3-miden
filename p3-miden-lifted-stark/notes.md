# p3-miden-lifted-stark notes

Purpose: shared scaffolding for the lifted STARK prover/verifier (LMCS-based).

## Module map
- `config.rs`: `LiftedStarkConfig` + `ParamsSnapshot` (public params serialized in transcript).
- `layout.rs`: prover `TraceLayout` and serialized `LayoutSnapshot`.
- `transcript.rs`: read/write helpers for layout + periodic tables.
- `periodic.rs`: periodic column LDE builder (prover) and evaluation (verifier).
- `selectors.rs`: two-adic selector polynomials.
- `folder.rs`: EF-only `ConstraintFolder` implementing `MidenAirBuilder`.
- `utils.rs`: LDE/padding/upsample helpers and row conversions.

## Design notes (carried over from prover.md)
- Multi-AIR support with per-AIR trace heights, nested coset domains, and a permutation order.
- Layout snapshot is currently verbose; eventually split protocol-level params vs instance data.
- Periodic tables are serialized per AIR, per column, and evaluated at `(zeta^r)^(n/p)`.
- Quotient lifting combines per-AIR numerators via Horner + single division by `X^N - 1`.

## Follow-ups / suggested changes
- Consider deriving `alignment` from `Lmcs::alignment()` and remove it from the transcript.
- Add a helper in `p3-miden-lifted-fri` (or here) to commit a trace by LDE + LMCS tree build.
- Revisit `p3-miden-air` API to make periodic/aux semantics explicit for lifted STARK needs.
- Reduce transcript data once `PublicParams` vs instance layout is clarified.

## Lifted STARK (LMCS) exploration notes (full copy from root prover.md)

Date: 2026-01-28
Scope: new lifted prover/verifier using LMCS + MidenAir, minimal/generic-light.

### Constraints from request (initial; see Updates for current state)
- Use new `Lmcs` trait (p3-miden-lmcs). Use LMCS for commitments.
- Use MidenAir (p3-miden-air) for AIR eval.
- Initially single AIR only; current implementation supports multiple AIRs.
- Ignore preprocessed trace entirely.
- Assume aux trace always exists (build_aux_trace required).
- Periodic columns are now included in the transcript/eval flow (initially "ignore").
- Keep generics minimal; avoid Plonky3-style generic overload (mersenne support).
- Quotient splitting should live in prover/verifier (not PCS); currently a single combined quotient is committed.
- Need a helper in PCS/FRI side: commit a trace over H by building LMCS tree over its LDE on gK.

### Existing code references
- Current Miden STARK: p3-miden-prover (MidenAir based), p3-miden-uni-stark (Air based).
- Lifted PCS (LMCS + DEEP + FRI): p3-miden-lifted-fri.
- LMCS trait and tree API: p3-miden-lmcs.
- The old generic STARK flow is in p3-miden-uni-stark; heavy generics and PCS-managed quotient split.

#### Key files inspected
- `p3-miden-prover/src/prover.rs`, `p3-miden-prover/src/verifier.rs`, `folder.rs`
- `p3-miden-uni-stark/src/prover.rs`, `verifier.rs`
- `p3-miden-lifted-fri/src/prover.rs`, `verifier.rs`, `deep/*`, `fri/*`, `utils.rs`
- `p3-miden-lmcs/src/lib.rs`, `lmcs.rs`, `lifted_tree.rs`
- `p3-miden-air/src/air.rs`
- `p3-miden-fri/src/two_adic_pcs.rs` (for LDE/bit-reversal logic)

### Observed architecture (today)
- p3-miden-prover uses PCS (`Pcs`) and commits to LDEs via PCS `commit`, and PCS handles quotient splitting via `commit_quotient`.
- p3-miden-lifted-fri is LMCS-based and channel-driven, not a `Pcs` implementation. It expects the caller to:
  - build LMCS trees for input matrices (LDEs over gK)
  - open these via `open_with_channel` at evaluation points
  - verify with `verify_with_channel` on a verifier channel
- LMCS interface supports building a tree from matrices and opening at query indices via transcript hints.

### Current prover/verifier structure
Crates now live at `p3-miden-lifted-{stark,prover,verifier}` with minimal modules:

- `config.rs`
  - Minimal config wrapper: PCS params (DEEP+FRI), LMCS instance, DFT, alignment.
  - Challenger is supplied to `prove`/`verify` or via channel-first APIs.
  - Avoid `StarkGenericConfig` to reduce generics.

- `proof.rs`
  - Proof container with transcript data only; commitments are inside the transcript.

- `folder.rs`
  - MidenAir constraint folders (prover+verifier), simplified:
    - no preprocessed
    - aux always present
    - periodic values evaluated via periodic tables (see below)

- `prover.rs`
  - end-to-end proving flow (multi-AIR, aux required)

- `verifier.rs`
  - end-to-end verification flow (multi-AIR, aux required)

- (Planned) `quotient.rs`
  - extract quotient lifting/recompose helpers from prover/verifier (todo)

### Prover flow (minimal)
1. Build layout from AIRs/traces: per-AIR log_degrees, permutation order, ratios r, log_max_degree/log_max_height.
2. Write params snapshot and layout snapshot into the transcript.
3. Write periodic tables (per AIR, per column) into the transcript.
4. Commit main trace LDEs over nested cosets (gK)^r in bit-reversed order, observe commitment.
5. Sample aux randomness (AIR order), build aux traces, commit aux LDEs (permutation order).
6. Sample per-AIR alphas (AIR order) and a shared beta.
7. Compute per-AIR folded constraint numerators on (gK)^r, lift by upsampling into gK, Horner-combine with beta, divide once by X^N - 1.
8. Commit the combined quotient as a single EF column split into base-field coefficients.
9. Rejection-sample `zeta` outside H and gK (loop ~1x), derive `zeta_next`.
10. Call `p3-miden-lifted-fri::prover::open_with_channel` with eval points `[zeta, zeta_next]`.
11. Output proof: transcript data only (commitments are inside).

### Verifier flow (minimal)
1. Replay transcript: params snapshot, layout snapshot, periodic tables, commitments.
2. Sample aux randomness and per-AIR alphas (AIR order), then beta, then rejection-sample `zeta` outside H and gK (derive `zeta_next`).
3. Verify openings via lifted FRI `verify_with_channel` at `[zeta, zeta_next]`.
4. Recompose quotient(zeta) from the single quotient matrix (EF coefficients in base field).
5. Evaluate AIR constraints at `zeta^r` using the EF folder and periodic values.
6. Check folded constraints * inv_vanishing == quotient(zeta) and ensure the transcript is consumed.

### Quotient handling (current)
- The prover computes a single combined quotient on the max domain and commits it as
  one EF column split into base-field coefficients (no chunking yet).
- If/when splitting is added, it should live in prover/verifier (not PCS), with the
  verifier recomposing quotient(zeta) from chunk openings.

### Helper needed in PCS/FRI layer
Add a commitment helper to the lifted FRI PCS or the new STARK crate:
- Input: evaluations on H (RowMajorMatrix<F>), blowup factor, domain shift
- Output: LDE on gK in bit-reversed order + LMCS tree (and tree metadata)
- Similar to `TwoAdicFriPcs::commit` in `p3-miden-fri`, but LMCS-based.
- Today, the prover does the LDE + `Lmcs::build_tree` inline; this helper would reduce duplication.

### Periodic columns
- Periodic tables are written into the transcript (per AIR, per column).
- Prover builds periodic LDEs per column on a coset of size p*b with shift g^(N/p),
  then uses `i % (p*b)` indexing to supply periodic values during constraint eval.
- Verifier treats each periodic column as evaluations of a polynomial over a
  subgroup of size p and evaluates at y = (zeta^r)^(n/p), where n is the trace size.

### Simplifications to enforce now
- No preprocessed trace handling.
- Aux trace always present; fail/panic if build_aux_trace is None.
- No ZK for now (assume `is_zk = 0`).
- Single combined quotient (no chunk splitting yet).
- Minimal generics: only F, EF, Lmcs, Dft; avoid `StarkGenericConfig`.

### Open questions / confirmations
1) New crate name/location? Resolved: `p3-miden-lifted-{stark,prover,verifier}`.
2) Commitments: 3 separate LMCS trees (main, aux, quotient) vs combine? Current: 3 separate.
3) Where to place the LDE+commit helper (lifted-fri vs new STARK crate)?
4) ZK explicitly out-of-scope for now? Still yes.
5) LMCS alignment: `Lmcs` trait doesn't currently expose it; should we add it or pass as config?

### Updates (2026-01-28, latest decisions + prototype)
- Multi-AIR support: prover/verifier take slices of AIRs and traces (aux trace per AIR, same height as its main trace).
- Layout metadata is explicit and written into the transcript:
  - num_airs, log_degrees (per air), permutation (sorted by trace height),
    log_max_degree, log_max_height, num_randomness (per air),
    widths for main/aux (aligned), quotient widths (single combined),
    and the permutation itself (as field elements).
- Proof = transcript data only; commitments are inside the transcript.
  - A replayable `StarkTranscript` can be reconstructed from a verifier channel and
    includes params snapshot, layout, periodic tables, commitments, alphas, beta, zeta,
    randomness, and the PCS transcript.
- Channel-first API: `prove_with_channel`/`verify_with_channel` take transcript channels;
  `prove`/`verify` accept a `GrindingChallenger` and config no longer stores a challenger.
- Combined quotient is committed as a single EF column split into base-field coefficients (no chunk splitting yet).
- Zeta-next is no longer sampled; it is derived as `zeta_next = zeta * h_max`
  where `h_max` is the generator of H_max. Zeta is rejection-sampled so
  `zeta^N != 1` and zeta not in gK (loop ~1x). For an AIR with ratio r, use `zeta^r`
  for local evaluation and `(zeta_next)^r` for "next" evaluations.

#### Domains and ratios
- Let N = max trace height (power-of-two), H = subgroup of size N, and K the
  subgroup of size N*b (b = blowup). Largest trace is evaluated on H and LDEs to gK.
- A trace of size n = N / r is defined over H^r, and its LDE domain is (gK)^r,
  with shift `g^r`. Selectors are computed over the nested coset (gK)^r.
- For OOD checks and periodicity, evaluate at `zeta^r` (and `(zeta_next)^r`).

#### Quotient combination (lifting)
- For each AIR, compute folded constraint numerators over its (gK)^r domain.
- Convert to bit-reversed order, then upsample by repeating each value r times
  to lift onto the max domain gK.
- Combine across AIRs in permutation order using a single challenge beta
  (Horner folding).
- Divide by X^N - 1 only once on the combined polynomial.

#### Periodic columns (now included)
- Periodic tables are written into the transcript (per AIR, per column).
- Verifier treats each periodic column as evaluations of a polynomial over a
  subgroup of size p and evaluates at y = (zeta^r)^(n/p), where n is the trace size.
- Prover builds periodic LDEs per column on a coset of size p*b with shift g^(N/p),
  then uses `i % (p*b)` indexing to supply periodic values during constraint eval.
- This keeps periodic handling simple while aligning with nested coset domains.

#### Prototype files (not wired into workspace)
- `prover.rs`: standalone lifted prover prototype with LMCS + MidenAir,
  multi-AIR layout, transcript-driven flow, and periodic handling.
- `verifier.rs`: standalone verifier prototype with transcript replay,
  periodic interpolation at `(zeta^r)^(n/p)`, and combined quotient check.

