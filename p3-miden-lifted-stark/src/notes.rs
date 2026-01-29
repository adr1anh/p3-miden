//! Lifted STARK (LMCS) exploration notes (source: root prover.md).
//!
//! # Lifted STARK (LMCS) exploration notes
//! 
//! Date: 2026-01-28
//! Scope: new lifted prover/verifier using LMCS + MidenAir, minimal/generic-light.
//! 
//! ## Constraints from request
//! - Use new `Lmcs` trait (p3-miden-lmcs). Use LMCS for commitments.
//! - Use MidenAir (p3-miden-air) for AIR eval.
//! - Single AIR only.
//! - Ignore preprocessed trace entirely.
//! - Assume aux trace always exists (build_aux_trace required).
//! - Ignore periodic columns for now, but keep a placeholder hook.
//! - Keep generics minimal; avoid Plonky3-style generic overload (mersenne support).
//! - Quotient splitting should live in prover/verifier (not PCS).
//! - Need a helper in PCS/FRI side: commit a trace over H by building LMCS tree over its LDE on gK.
//! 
//! ## Existing code references
//! - Current Miden STARK: p3-miden-prover (MidenAir based), p3-miden-uni-stark (Air based).
//! - Lifted PCS (LMCS + DEEP + FRI): p3-miden-lifted-fri.
//! - LMCS trait and tree API: p3-miden-lmcs.
//! - The old generic STARK flow is in p3-miden-uni-stark; heavy generics and PCS-managed quotient split.
//! 
//! ### Key files inspected
//! - `p3-miden-prover/src/prover.rs`, `p3-miden-prover/src/verifier.rs`, `folder.rs`
//! - `p3-miden-uni-stark/src/prover.rs`, `verifier.rs`
//! - `p3-miden-lifted-fri/src/prover.rs`, `verifier.rs`, `deep/*`, `fri/*`, `utils.rs`
//! - `p3-miden-lmcs/src/lib.rs`, `lmcs.rs`, `lifted_tree.rs`
//! - `p3-miden-air/src/air.rs`
//! - `p3-miden-fri/src/two_adic_pcs.rs` (for LDE/bit-reversal logic)
//! 
//! ## Observed architecture (today)
//! - p3-miden-prover uses PCS (`Pcs`) and commits to LDEs via PCS `commit`, and PCS handles quotient splitting via `commit_quotient`.
//! - p3-miden-lifted-fri is LMCS-based and channel-driven, not a `Pcs` implementation. It expects the caller to:
//!   - build LMCS trees for input matrices (LDEs over gK)
//!   - open these via `open_with_channel` at evaluation points
//!   - verify with `verify_with_channel` on a verifier channel
//! - LMCS interface supports building a tree from matrices and opening at query indices via transcript hints.
//! 
//! ## Proposed new prover/verifier structure
//! Create a new crate (e.g., `p3-miden-lifted-prover`) or reuse a suitable existing crate, with minimal modules:
//! 
//! - `config.rs`
//!   - Minimal config wrapper: PCS params (DEEP+FRI), LMCS instance, challenger factory.
//!   - Avoid `StarkGenericConfig` to reduce generics.
//! 
//! - `proof.rs`
//!   - Proof container with:
//!     - commitments (main, aux, quotient)
//!     - transcript data (from ProverTranscript)
//!     - degree bits / domain size
//! 
//! - `folder.rs`
//!   - MidenAir constraint folders (prover+verifier), simplified:
//!     - no preprocessed
//!     - aux always present
//!     - periodic values: placeholder API, for now empty/zero
//! 
//! - `quotient.rs`
//!   - quotient splitting (power-of-two chunks)
//!   - quotient recomposition at a point (verifier)
//! 
//! - `prover.rs`
//!   - end-to-end proving flow (single AIR, aux required)
//! 
//! - `verifier.rs`
//!   - end-to-end verification flow (single AIR, aux required)
//! 
//! ## Prover flow (minimal)
//! 1. Compute `degree`, `log_degree` from trace height.
//! 2. Build trace domain H and LDE domain gK (blowup from params.fri.log_blowup).
//! 3. Commit main trace:
//!    - LDE evaluations over gK in bit-reversed order.
//!    - Build LMCS tree from that matrix, observe commitment into transcript.
//! 4. Sample aux randomness, build aux trace (must exist), commit aux tree, observe commitment.
//! 5. Build symbolic constraints, compute `alpha`.
//! 6. Compute quotient values over gK.
//! 7. Split quotient into chunks (power-of-two), LDE each chunk, commit quotient tree, observe commitment.
//! 8. Sample `zeta`, `zeta_next`.
//! 9. Call `p3-miden-lifted-fri::prover::open_with_channel` with eval points `[zeta, zeta_next]` and the three trees.
//! 10. Output proof: commitments + transcript data + degree bits.
//! 
//! ## Verifier flow (minimal)
//! 1. Observe commitments in same order (main, aux, quotient) into challenger.
//! 2. Sample aux randomness, `alpha`.
//! 3. Sample `zeta`, `zeta_next`.
//! 4. Use `p3-miden-lifted-fri::verifier::verify_with_channel` to open at `[zeta, zeta_next]`.
//! 5. Recompose quotient at `zeta` from chunks.
//! 6. Evaluate AIR constraints at `zeta` using a VerifierConstraintFolder.
//! 7. Check folded constraints * inv_vanishing == quotient(zeta).
//! 
//! ## Quotient splitting
//! - Move quotient splitting into prover/verifier instead of PCS:
//!   - Prover splits quotient polynomial into chunks and commits.
//!   - Verifier recomposes quotient(zeta) from chunk openings.
//! - Reuse existing recomposition logic from current verifier.
//! 
//! ## Helper needed in PCS/FRI layer
//! Add a commitment helper to the lifted FRI PCS or the new STARK crate:
//! - Input: evaluations on H (RowMajorMatrix<F>), blowup factor, domain shift
//! - Output: LDE on gK in bit-reversed order + LMCS tree (and tree metadata)
//! - Similar to `TwoAdicFriPcs::commit` in `p3-miden-fri`, but LMCS-based.
//! 
//! ## Periodic columns
//! - MidenAir exposes `periodic_table()`, currently used in p3-miden-prover.
//! - For now: ignore periodic columns but keep a placeholder in folders and constraint evaluation.
//! 
//! ## Simplifications to enforce now
//! - No preprocessed trace handling.
//! - Aux trace always present; fail/panic if build_aux_trace is None.
//! - Single AIR only.
//! - No ZK for now (assume `is_zk = 0`).
//! - Minimal generics: only F, EF, Lmcs, Challenger; avoid `StarkGenericConfig`.
//! 
//! ## Open questions / confirmations
//! 1) New crate name/location? (`p3-miden-lifted-prover` vs existing crate)
//! 2) Commitments: 3 separate LMCS trees (main, aux, quotient) vs combine?
//! 3) Where to place the LDE+commit helper (lifted-fri vs new STARK crate)?
//! 4) ZK explicitly out-of-scope for now?
//! 5) LMCS alignment: `Lmcs` trait doesn't currently expose it; should we add it or pass as config?
//! 
//! ## Updates (2026-01-28, latest decisions + prototype)
//! - Multi-AIR support: prover/verifier take slices of AIRs and traces (aux trace per AIR, same height as its main trace).
//! - Layout metadata is explicit and written into the transcript:
//!   - num_airs, log_degrees (per air), permutation (sorted by trace height),
//!     log_max_degree, log_max_height, num_randomness (per air),
//!     widths for main/aux (aligned), quotient widths (single combined),
//!     and the permutation itself (as field elements).
//! - Proof = transcript data only; commitments are inside the transcript.
//!   - A replayable `StarkTranscript` can be reconstructed from a verifier channel and
//!     includes params snapshot, layout, periodic tables, commitments, alphas, beta, zeta,
//!     randomness, and the PCS transcript.
//! - Zeta-next is no longer sampled; it is derived as `zeta_next = zeta * h_max`
//!   where `h_max` is the generator of H_max. For an AIR with ratio r, use
//!   `zeta^r` for local evaluation and `(zeta_next)^r` for "next" evaluations.
//! 
//! ### Domains and ratios
//! - Let N = max trace height (power-of-two), H = subgroup of size N, and K the
//!   subgroup of size N*b (b = blowup). Largest trace is evaluated on H and LDEs to gK.
//! - A trace of size n = N / r is defined over H^r, and its LDE domain is (gK)^r,
//!   with shift `g^r`. Selectors are computed over the nested coset (gK)^r.
//! - For OOD checks and periodicity, evaluate at `zeta^r` (and `(zeta_next)^r`).
//! 
//! ### Quotient combination (lifting)
//! - For each AIR, compute folded constraint numerators over its (gK)^r domain.
//! - Convert to bit-reversed order, then upsample by repeating each value r times
//!   to lift onto the max domain gK.
//! - Combine across AIRs in permutation order using a single challenge beta
//!   (Horner folding).
//! - Divide by X^N - 1 only once on the combined polynomial.
//! 
//! ### Periodic columns (now included)
//! - Periodic tables are written into the transcript (per AIR, per column).
//! - Verifier treats each periodic column as evaluations of a polynomial over a
//!   subgroup of size p and evaluates at y = (zeta^r)^(n/p), where n is the trace size.
//! - Prover builds periodic LDEs per column on a coset of size p*b with shift g^(N/p),
//!   then uses `i % (p*b)` indexing to supply periodic values during constraint eval.
//! - This keeps periodic handling simple while aligning with nested coset domains.
//! 
//! ### Prototype files (not wired into workspace)
//! - `prover.rs`: standalone lifted prover prototype with LMCS + MidenAir,
//!   multi-AIR layout, transcript-driven flow, and periodic handling.
//! - `verifier.rs`: standalone verifier prototype with transcript replay,
//!   periodic interpolation at `(zeta^r)^(n/p)`, and combined quotient check.
