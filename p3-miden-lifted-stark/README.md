# Lifted STARK Protocol

Multi-trace STARK prover and verifier using LMCS commitments, DEEP quotient
batching, and lifted FRI for low-degree testing.

## Overview

The lifted STARK protocol is split across three crates:

| Crate | Role |
|-------|------|
| `p3-miden-lifted-stark` | Shared types: `StarkConfig`, `LiftedCoset`, `Selectors` |
| `p3-miden-lifted-prover` | Proving: trace commitment, constraint evaluation, quotient construction |
| `p3-miden-lifted-verifier` | Verification: OOD check, quotient reconstruction, transcript canonicality |

All three compose with the lower layers:

```
p3-miden-lifted-{prover,verifier}
└── p3-miden-lifted-stark          ← this crate
    ├── p3-miden-lifted-fri        ← PCS (DEEP + FRI)
    │   └── p3-miden-lmcs          ← Merkle commitments with lifting
    └── p3-miden-air               ← AIR traits (aux columns, periodic columns)
```

The system supports **multiple traces of different heights** (power-of-two,
ascending order). Shorter traces are virtually lifted to the maximum height
via LMCS upsampling, so the PCS and verifier operate on a single uniform view.

## Protocol Summary

### Prover (`prove_multi`)

1. **Commit main traces** — LDE each trace on its lifted coset, bit-reverse
   rows, build LMCS tree. Send root.
2. **Sample randomness** — Squeeze auxiliary randomness from the Fiat-Shamir
   channel. Build and commit auxiliary traces.
3. **Sample challenges** — `alpha` (constraint folding) and `beta`
   (cross-trace accumulation).
4. **Evaluate constraints** — For each trace in ascending height order,
   evaluate AIR constraints on the quotient domain using SIMD-packed
   arithmetic. Produces a numerator N_j per trace (no vanishing division).
5. **Accumulate numerators** — Fold across traces:
   `acc = cyclic_extend(acc) * beta + N_j`.
6. **Divide by vanishing polynomial** — One pass on the full quotient domain,
   exploiting Z_H periodicity for batch inverse.
7. **Commit quotient** — Decompose Q into D chunks via fused iDFT + coefficient
   scaling + flatten + DFT pipeline. Commit via LMCS.
8. **Sample OOD point zeta** — Rejection-sampled to lie outside H and the LDE
   coset.
9. **Open via PCS** — Delegate to `p3-miden-lifted-fri`.

### Verifier (`verify_multi`)

1. **Receive commitments** — Main, auxiliary, and quotient roots from transcript.
2. **Re-derive challenges** — Same `alpha`, `beta`, `zeta` via Fiat-Shamir.
3. **Verify PCS openings** — At `[zeta, zeta * g_max]`.
4. **Reconstruct Q(zeta)** — Barycentric interpolation over the D quotient
   chunks.
5. **Evaluate constraints at OOD** — For each AIR at the lifted OOD point
   `y_j = zeta^{r_j}`: compute selectors, evaluate periodic polynomials,
   fold constraints with alpha, accumulate with beta.
6. **Check identity** — `accumulated == Q(zeta) * Z_H(zeta)`.
7. **Ensure transcript is fully consumed** — Canonicality enforcement.

## Math Sketch

### Multi-Trace Lifting

Each trace j has height `n_j = n_max / r_j` where `r_j` is a power-of-two
lift ratio. The committed polynomial is `p_j(X^{r_j})`, so opening the LMCS
commitment at `zeta` yields `p_j(zeta^{r_j})`. The coset shift for trace j
is `g^{r_j}` where g is the multiplicative generator.

### Constraint Folding

For a single trace, constraints `C_0, C_1, ...` are folded via Horner
accumulation:

```
folded = (...((C_0 * alpha + C_1) * alpha + C_2)...) * alpha + C_k
```

This avoids precomputing alpha powers and does not require knowing the
constraint count ahead of time.

### Cross-Trace Accumulation

Numerators from traces of increasing height are combined:

```
acc = cyclic_extend(acc) * beta + N_j
```

where `cyclic_extend` repeats the accumulator via modular indexing
(`i & (len - 1)`) to match the next trace's quotient domain size.
This works because:

```
Z_H(x) = Z_{H^r}(x) * Phi_r(x)
```

so cyclic extension of a polynomial divisible by `Z_{H^r}` preserves
divisibility by `Z_H`.

### Vanishing Division

After accumulation, the combined numerator is divided by `Z_H(x) = x^N - 1`
once on the full quotient domain. Z_H has only `2^rate_bits` distinct values
on the coset, so batch inverse computes only those distinct inverses and
accesses them via modular indexing.

### Quotient Decomposition

The quotient polynomial Q of degree `N * D - 1` is decomposed into D chunks
`q_0, ..., q_{D-1}` of degree `N - 1`:

```
Q(X) = q_0(X^D) + X * q_1(X^D) + ... + X^{D-1} * q_{D-1}(X^D)
```

The prover commits evaluations of each `q_t` over the LDE domain. The
verifier reconstructs `Q(zeta)` from `q_t(zeta)` via barycentric
interpolation:

```
Q(zeta) = sum_t w_t * q_t(zeta) / sum_t w_t
    where w_t = omega^t / (u - omega^t),  u = (zeta/g)^N
```

### Virtual OOD Point

For a trace with lift ratio `r_j`, the effective OOD evaluation point is
`y_j = zeta^{r_j}`. The verifier evaluates selectors and periodic polynomials
at `y_j`, and the opened trace values already correspond to `p_j(y_j)`.

## Optimizations

- **SIMD constraint evaluation** — Constraints are evaluated on `PackedVal::WIDTH`
  points simultaneously. Main trace stays in base field; only auxiliary columns
  use extension field arithmetic.
- **Horner folding** — Constraint accumulation via `acc = acc * alpha + C_i`
  avoids precomputing and storing alpha powers.
- **Fused quotient pipeline** — iDFT, coefficient scaling by `(omega^t)^{-k}`,
  flatten to base field, zero-pad, forward DFT — all in one pass, no redundant
  coset operations.
- **Periodic vanishing exploit** — Z_H has only `2^rate_bits` distinct values
  on the quotient coset; batch inverse computes those once.
- **Zero-copy quotient domain** — `split_rows().bit_reverse_rows()` gives a
  natural-order view of committed LDE data without copying.
- **Efficient periodic columns** — Only `max_period * blowup` LDE values
  stored per periodic table; accessed via modular indexing.
- **Cyclic extension** — Cross-trace accumulation uses bitwise AND for
  modular indexing (power-of-two sizes).
- **Parallel execution** — Rayon parallelism throughout constraint evaluation
  and vanishing division (gated by `parallel` feature).

## Entry Points

| Item | Crate | Purpose |
|------|-------|---------|
| `prove_single` | `lifted-prover` | Prove a single-AIR STARK |
| `prove_multi` | `lifted-prover` | Prove a multi-trace STARK (ascending heights) |
| `AirWithTrace` | `lifted-prover` | Bundle an AIR with its trace and public values |
| `verify_single` | `lifted-verifier` | Verify a single-AIR proof |
| `verify_multi` | `lifted-verifier` | Verify a multi-trace proof |
| `AirWithLogHeight` | `lifted-verifier` | Bundle an AIR with its log height and public values |
| `Proof` | `lifted-verifier` | Raw transcript data (the proof artifact) |
| `StarkConfig` | `lifted-stark` | PCS params + LMCS + DFT configuration |
| `LiftedCoset` | `lifted-stark` | Domain operations: selectors, vanishing, coset shifts |
| `Selectors` | `lifted-stark` | `is_first_row`, `is_last_row`, `is_transition` |

## Modules

### `p3-miden-lifted-stark`

| Path | Purpose |
|------|---------|
| `src/config.rs` | `StarkConfig` — wraps `PcsParams`, LMCS, and DFT |
| `src/coset.rs` | `LiftedCoset` — domain queries, selector computation, vanishing |
| `src/selectors.rs` | `Selectors<T>` — generic container for row selectors |

### `p3-miden-lifted-prover`

| Path | Purpose |
|------|---------|
| `src/prover.rs` | `prove_single`, `prove_multi` — orchestration and protocol flow |
| `src/commit.rs` | `Committed` — LDE, bit-reverse, LMCS tree construction |
| `src/constraints.rs` | Constraint evaluation (SIMD), quotient commitment pipeline |
| `src/periodic.rs` | `PeriodicLde` — precomputed periodic column LDEs |

### `p3-miden-lifted-verifier`

| Path | Purpose |
|------|---------|
| `src/verifier.rs` | `verify_single`, `verify_multi` — orchestration and identity check |
| `src/constraints.rs` | `ConstraintFolder` — OOD constraint evaluation, quotient reconstruction |
| `src/periodic.rs` | `PeriodicPolys` — polynomial coefficients for OOD evaluation |
| `src/proof.rs` | `Proof` — raw transcript wrapper |

## Conventions & Assumptions

- **Ascending height order** — Traces must be supplied in ascending height
  order (shortest first). The prover and verifier both validate this.
- **Power-of-two heights** — All trace heights are powers of two.
- **Bit-reversed storage** — All evaluation matrices are in bit-reversed order.
- **Constraint degree** — Fixed at `D = 4` (`LOG_CONSTRAINT_DEGREE = 2`).
  Both prover and verifier must agree on this constant.
- **Transcript ordering** — The Fiat-Shamir transcript follows a strict
  observe/squeeze protocol. Prover and verifier must process commitments and
  challenges in identical order. This is security-critical.
- **Extension field discipline** — Main trace and preprocessed data stay in
  the base field. Only auxiliary columns, challenges, alpha powers, and the
  accumulator use the extension field.
- **Periodic columns** — Column periods must be powers of two and divide the
  trace height. Columns are grouped by period for batch interpolation.

## Tests

The end-to-end test suite lives in `p3-miden-lifted-prover/tests/`:

- **`tiny_air.rs`** — `TinyAir` exercising single-trace, multi-trace
  (same and different heights), periodic columns, and malformed transcript
  rejection.
- **`aux_shape.rs`** — Validates that mismatched auxiliary trace dimensions
  are caught.

Run with:
```bash
cargo test -p p3-miden-lifted-prover
```

## Security

Audits should start with `SECURITY.md` at the workspace root for transcript
ordering, lifting correctness, constraint identity, and critical paths.

## License

Dual-licensed under MIT and Apache-2.0 at the workspace root.
