# p3-miden-lifted (Lifted FRI PCS)

Lifted Polynomial Commitment Scheme (PCS) combining DEEP quotient construction with FRI.

**Implementation note**: The code lives in the `p3-miden-lifted-fri` crate. This README documents that implementation.

## Overview

This crate provides a complete PCS for proving polynomial evaluations:

1. **Commit**: Use LMCS to commit to polynomial evaluation matrices
2. **Open**: Prove evaluations at out-of-domain points using DEEP quotient + FRI
3. **Verify**: Check claimed evaluations against commitments

The "lifted" design here is about committing to **evaluation matrices of varying heights** on a single, verifier-uniform domain. Shorter matrices are committed via LMCS by exposing a deterministic *lifted view* (nearest-neighbor repetition in bit-reversed order).

## Contract (Read This First)

### Verifier Indistinguishability of Lifting

LMCS lifting is **not** an additional algebraic claim. From the verifier’s perspective, every opened column is just a length-`N` evaluation table on the max domain.

Concretely, the verifier cannot distinguish:
- a committed matrix that was “lifted” (rows repeated because the original matrix had smaller height), from
- a committed matrix of full height `N` with no repetition at all.

This is intentional: lifting is a prover optimization and a verifier-uniformization mechanism. Any higher-level meaning of “this came from a smaller object”, “this is periodic”, or “these rows must repeat” must be enforced upstream (e.g. by a STARK/AIR protocol) or be irrelevant to soundness in that context.

### Trusted Inputs (Upstream Responsibilities)

This PCS verifies openings *relative to* metadata supplied by the higher-level protocol. In particular, the verifier is expected to treat the following as part of the statement/instance (i.e. fixed before Fiat–Shamir challenges are sampled):

- **Commitment structure**: for each commitment, the ordered list of committed matrices and their widths (in commitment order).
- **Universal domain size**: a shared `log_max_height` (Merkle tree height / evaluation domain size) used for query sampling and for interpreting indices.
- **Domain conventions**: evaluations are over a multiplicative coset `gK` in **bit-reversed order** (see below); the shift `g` is fixed to `F::GENERATOR`, and `log_n = log_max_height`.
- **Evaluation points**: all `z_j` must be outside `gK`. Callers must enforce this (typically by sampling). The failure probability for a uniform sample is at most `|gK| / |EF|`, which is overwhelmingly small for typical parameters.
- **Protocol parameters**: `PcsParams` (FRI params, DEEP params, query counts, PoW bits).

This PCS does not attempt to “discover” or “authenticate” those parameters on its own; it assumes they are bound by the higher-level protocol (e.g. by transcript ordering / instance commitment).

### Alignment / “Virtual Zeros” vs “Extra Columns”

DEEP batching uses an alignment parameter (from LMCS) to make transcript observation and Horner reduction behave like rows were padded with zeros up to a multiple of the alignment. This padding is **virtual**:
- it does not create additional committed columns, and
- the padded values are semantic zeros (not prover-controlled data).

If a higher-level protocol instead pads by adding *actual columns* to the committed matrices, those columns become real committed polynomials: the PCS will enforce their low-degreeness, but the AIR must constrain them if they must be zero/unused.

### Current API Assumption: Shared Max Height

Query indices are sampled once using `log_max_height` and then opened against every commitment. In the current API, this requires that every committed tree is built at the same max height `N`, so that all sampled indices are in range for all commitments. **Mixed max heights are not supported yet**; the prover will panic if this precondition is violated.

## Architecture

```
                    ┌─────────────────────────────────┐
                    │         PCS Interface           │
                    │ open_with_channel / verify_with_channel │
                    └───────────────┬─────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              │                                           │
              ▼                                           ▼
    ┌─────────────────────┐                 ┌─────────────────────┐
    │   DEEP Quotient     │                 │        FRI          │
    │                     │                 │                     │
    │ Batches evaluation  │────────────────▶│ Proves low-degree   │
    │ claims into single  │  Q(X) evals     │ via iterative       │
    │ low-degree poly Q   │                 │ folding + queries   │
    └─────────────────────┘                 └─────────────────────┘
              │                                           │
              │                                           │
              ▼                                           ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    Query Phase                              │
    │  • Open trace matrices via LMCS at query indices            │
    │  • Open FRI folded matrices at progressively shifted indices│
    │  • All openings use LMCS batch hints for compactness        │
    └─────────────────────────────────────────────────────────────┘
```

## Modules (p3-miden-lifted-fri)

### `deep` - DEEP Quotient Construction

DEEP (Dimension Extension of Evaluation Protocol) converts polynomial evaluation claims into a low-degree test.

Given committed polynomials $\{f_i\}$ and claimed evaluations $f_i(z_j) = v_{ij}$:

$$Q(X) = \sum_j \beta^j \sum_i \alpha^i \cdot \frac{v_{ij} - f_i(X)}{z_j - X}$$

**Key property**: $Q(X)$ is low-degree if and only if all claims are correct. A false claim creates a pole at some $z_j$.

**Files**:
- `deep/prover.rs`: `DeepPoly` construction from committed matrices
- `deep/verifier.rs`: `DeepOracle` for query-time verification
- `deep/interpolate.rs`: Barycentric evaluation with weight folding for lifted polynomials

### `fri` - Fast Reed-Solomon IOP

FRI proves that a committed polynomial has degree below a target bound through iterative folding.

**Files**:
- `fri/fold/arity2.rs`, `arity4.rs`, `arity8.rs`: Folding implementations
- `fri/prover.rs`: `FriPolys` commit phase (folding + commitments)
- `fri/verifier.rs`: `FriOracle` query verification

### Top-Level PCS (channel-driven)

**Files**:
- `params.rs`: `PcsParams` aggregates all parameters
- `prover.rs`: `open_with_channel()`
- `verifier.rs`: `verify_with_channel()`
- `proof.rs`: `PcsTranscript` (structured transcript view)

## Configuration

```rust
pub struct PcsParams {
    pub deep: DeepParams,
    pub fri: FriParams,
    pub num_queries: usize,
    pub query_proof_of_work_bits: usize,
}

pub struct DeepParams {
    pub proof_of_work_bits: usize,
}

pub struct FriParams {
    pub log_blowup: usize,
    pub fold: FriFold,
    pub log_final_degree: usize,
    pub proof_of_work_bits: usize,
}
```

## Usage (Channel-Based)

### Prover

```rust
use p3_miden_lifted_fri::{PcsParams, deep::DeepParams, fri::FriParams};
use p3_miden_lifted_fri::prover::open_with_channel;
use p3_miden_transcript::ProverTranscript;
use p3_util::log2_strict_usize;

let params = PcsParams {
    deep: DeepParams { proof_of_work_bits: 16 },
    fri: FriParams { log_blowup: 3, fold, log_final_degree: 4, proof_of_work_bits: 8 },
    num_queries: 30,
    query_proof_of_work_bits: 8,
};

let log_max_height = log2_strict_usize(main_trace_tree.height());
let mut channel = ProverTranscript::new(challenger);
open_with_channel(
    &params,
    &lmcs,
    log_max_height,
    [z, z_next],
    &[&main_trace_tree, &aux_trace_tree],
    &mut channel,
);
let proof_data = channel.into_data();
```

### Verifier

```rust
use p3_miden_lifted_fri::verifier::verify_with_channel;
use p3_miden_transcript::VerifierTranscript;

let mut channel = VerifierTranscript::from_data(challenger, &proof_data);
let evals = verify_with_channel(
    &params,
    &lmcs,
    &[(main_commit, main_widths), (aux_commit, aux_widths)],
    log_max_height,
    [z, z_next],
    &mut channel,
)?;
```

## Security Analysis

### Protocol Flow and Trust Boundaries

```
Prover                                          Verifier
───────                                         ────────
Commit trace matrices ─── commitment ──────────▶ Store commitment

                      ◀── z (eval point) ─────── Sample via Fiat-Shamir

Compute evaluations f_i(z) ── evals ───────────▶ Observe evals

Compute DEEP quotient Q(X) evals ─────────────▶ (FRI input)

                      ◀── α, β (DEEP challs) ── Grind + sample

FRI commit phase ──────── FRI commits ─────────▶ Observe commits

                      ◀── β_i (FRI challs) ──── Per-round: grind + sample

                      ◀── query indices ─────── Grind + sample

Open at query indices ──── query proofs ───────▶ Verify openings + folding
```

### Critical Security Properties

#### 1. Fiat-Shamir Binding

All challenges must be sampled **after** observing the data they depend on.

#### 2. Proof-of-Work Grinding

Grinding prevents the prover from trying many transcripts to find favorable challenges.

**Locations**:
- `DeepPoly::new` (before $\alpha, \beta$)
- `FriPolys::new` (before each folding $\beta_i$)
- `open_with_channel` / `verify_with_channel` (before query indices)

#### 3. Domain Separation

Evaluation points $z_j$ must be **outside** the evaluation domain $gK$. Callers must enforce this; if $z_j$ is sampled uniformly from the extension field, the failure probability is at most $|gK| / |EF|$, which is overwhelmingly small in typical settings.

#### 4. Lifting Consistency

The verifier must reconstruct domain points exactly as the prover does:

```rust
let index_bit_rev = reverse_bits_len(index, log_max_height);
let row_point = shift * generator.exp_u64(index_bit_rev as u64);
```

#### 5. FRI Folding Correctness

The verifier must check that opened evaluations are consistent with the folded value.

## Testing

Key test cases to review:

1. **Folding correctness**: `p3-miden-lifted-fri/src/fri/fold/mod.rs::tests`
2. **Barycentric evaluation**: `p3-miden-lifted-fri/src/deep/interpolate.rs::tests`
3. **Full PCS round-trip**: `p3-miden-lifted-fri/src/tests.rs`
