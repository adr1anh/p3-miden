# p3-miden-lmcs

Lifted Matrix Commitment Scheme (LMCS) for matrices with power-of-two heights.

## Overview

LMCS is a Merkle tree commitment scheme optimized for committing to multiple matrices of **varying heights** that store polynomial evaluations in **bit-reversed order** over multiplicative cosets. Unlike standard Merkle commitments that require all matrices to have the same height, LMCS "lifts" smaller matrices to the maximum height via virtual upsampling.

## Contract (Read This First)

LMCS is a *commitment format* for a **uniform-height view** of a list of matrices.

### What LMCS Proves

Given:
- a commitment (Merkle root), and
- a set of query indices chosen by the verifier,

LMCS verification proves that the opened rows are consistent with *some* committed leaf preimages, under LMCS’s deterministic row-selection rule (lifting), and under the configured hash/compression functions.

### What LMCS Does *Not* Prove

- **No “unlifted” semantics.** LMCS does not prove that a matrix “really had height `n`” in any semantic sense beyond the supplied `Dimensions`. Concretely, the verifier cannot distinguish:
  - committing to an explicit height-`N` matrix, vs
  - committing to a height-`n` matrix “lifted” to `N` by repetition.
- **No periodicity/structure checks.** If an upstream protocol interprets some columns as periodic, or interprets lifting as a prover optimization, those properties must be enforced upstream (e.g. by a STARK/AIR protocol) or be irrelevant to soundness in that context.

### Trusted Inputs (Upstream Responsibilities)

LMCS verification takes `(commitment, widths, log_max_height)` as statement/instance data. LMCS assumes:
- `widths` (the **matrix widths** in commitment order) are part of the *statement/instance* fixed by the higher-level protocol, not prover-controlled metadata.
- `log_max_height` (the Merkle tree height / max evaluation domain size) is fixed by the higher-level protocol (typically derived from trace degree and blowup), not prover-controlled metadata.
- matrices are *meaningfully identified* by their position in this ordered list; LMCS itself assigns no semantics (e.g. “main trace”, “aux trace”) to an index.

## Key Concepts

### Bit-Reversed Evaluation Order

Polynomial evaluations over a coset $gK$ (where $K = \langle\omega\rangle$ is a multiplicative subgroup of order $n$) can be stored in two orderings:

- **Canonical order**: $[f(g\omega^0), f(g\omega^1), \ldots, f(g\omega^{n-1})]$
- **Bit-reversed order**: $[f(g\omega^{\text{bitrev}(0)}), f(g\omega^{\text{bitrev}(1)}), \ldots, f(g\omega^{\text{bitrev}(n-1)})]$

LMCS operates on bit-reversed data, which is the natural output of radix-2 FFT algorithms.

### Lifting via Upsampling

When committing to matrices of heights $n_0 \leq n_1 \leq \cdots \leq n_{t-1}$ (each a power of two), smaller matrices are "lifted" to height $N = n_{t-1}$ using **nearest-neighbor upsampling**: each row is repeated contiguously $r = N/n$ times.

```
Original (n=4):     [row0, row1, row2, row3]
Upsampled (N=8):    [row0, row0, row1, row1, row2, row2, row3, row3]
```

### Why Upsampling Works

For bit-reversed polynomial evaluations, upsampling is **mathematically equivalent** to evaluating the polynomial $f'(X) = f(X^r)$ on the larger domain. This means:

- The verifier sees all polynomials as living on the same (max) domain
- Opening at index $i$ retrieves $f'(g\omega_N^{\text{bitrev}_N(i)})$
- Upsampling is done **virtually** without data movement

**Formal statement**: If `data` contains bit-reversed evaluations of $f(X)$ on coset $gK$ of size $n$, then upsampling to size $N = rn$ produces bit-reversed evaluations of $f(X^r)$ on coset $gK'$ of size $N$.

**Verifier indistinguishability**: LMCS commits to the *lifted view*. The verifier does not (and cannot) check whether this lifted view “came from” a smaller matrix; it only checks Merkle consistency of the view that was committed.

## API

### Configuration Types

```rust
// Non-hiding commitment
let config = LmcsConfig::<PF, PD, H, C, WIDTH, DIGEST>::new(sponge, compress);

// Hiding commitment with salt
let hiding_config = HidingLmcsConfig::<PF, PD, H, C, R, WIDTH, DIGEST, SALT>::new(sponge, compress, rng);
```

### Traits

```rust
/// Trait for LMCS configurations
pub trait Lmcs: Clone {
    type F: Clone + Send + Sync;           // Scalar field element
    type Commitment: Clone;                 // Root hash
    type BatchProof: Clone;                 // Parsed batch opening
    type Tree<M>: LmcsTree<...>;           // Built tree type

    fn build_tree<M: Matrix<Self::F>>(&self, leaves: Vec<M>) -> Self::Tree<M>;
    fn open_batch(..., indices: &[usize], channel: &mut Ch) -> Result<...>;
}

/// Trait for built LMCS trees
pub trait LmcsTree<F, Commitment, M> {
    fn root(&self) -> Commitment;
    fn height(&self) -> usize;
    fn leaves(&self) -> &[M];
    fn rows(&self, index: usize) -> Vec<Vec<F>>;
    fn widths(&self) -> Vec<usize>;
    fn prove_batch(&self, indices: &[usize], channel: &mut Ch);
}
```

### Basic Usage

```rust
use p3_miden_lmcs::{LmcsConfig, Lmcs, LmcsTree};
use p3_miden_transcript::{ProverTranscript, VerifierTranscript};
use p3_util::log2_strict_usize;

// Build tree from matrices (must be sorted by height!)
let tree = config.build_tree(vec![small_matrix, large_matrix]);
let commitment = tree.root();

// Open at multiple indices (prover)
let indices = [0, 42, 100];
let mut prover_channel = ProverTranscript::new(challenger);

// Stream LMCS batch hints into the transcript
let _ = tree.prove_batch(&indices, &mut prover_channel);
let transcript = prover_channel.into_data();

// Verify by reading hints and recomputing the root
let mut verifier_channel = VerifierTranscript::from_data(challenger, &transcript);
let widths = tree.widths();
let log_max_height = log2_strict_usize(tree.height());
let rows = config.open_batch(
    &commitment,
    &widths,
    log_max_height,
    &indices,
    &mut verifier_channel,
)?;
```

### Commitment Preimage (Order Matters)

Each leaf corresponds to a single row index `i` in `[0, N)` where `N = max_height` is the Merkle tree height.
For each committed matrix `m_j`, LMCS selects a row via nearest-neighbor lifting and then hashes the
sequence of selected rows in **matrix order**:

- `leaf(i) = sponge( row_0(i) || row_1(i) || ... || row_{t-1}(i) || salt? )`

This means permuting the matrices changes the commitment. LMCS does not interpret the permutation;
the higher-level protocol must define what each position means, and must bind that ordering in the
statement/transcript.

## Proof Types

### `BatchProof<F, D, DIGEST_ELEMS, SALT_ELEMS>`

Parsed batch opening read from transcript hints. Stores:
- `openings`: map from leaf index to opened rows + salt
- `siblings`: map from `(depth, index)` to sibling digests

You can parse without hashing via `BatchProof::read_from_channel`, then reconstruct per-index proofs
using `BatchProof::single_proofs` once the hashing context is available.

### `Proof<F, D, DIGEST_ELEMS, SALT_ELEMS>`

Single-opening proof for one query index. Contains:
- `rows`: opened rows
- `salt`: optional salt
- `siblings`: authentication path (bottom-to-top)

Single proofs can be reconstructed from a `BatchProof` using `single_proofs`.

## Security Analysis

### Trust Boundaries

#### 1. Matrix Ordering Invariant

**Requirement**: Matrices must be sorted by height (shortest to tallest).

**Enforced by**: `validate_heights()` in `lifted_tree.rs`

**Consequence if violated**: Upsampling logic in `build_leaf_states_upsampled` will produce incorrect digests. States accumulated for smaller matrices would not align correctly with larger matrices.

**Auditor action**: Verify that all call sites pass matrices sorted by height.

#### 2. Index Bounds

**Requirement**: Query indices must be less than tree height.

**Enforced by**: Assertions in `prove_batch()` and bounds checks in `open_batch()`.

**Consequence if violated**: `InvalidProof` error returned during verification.

#### 3. Proof Canonicality

**Requirement**: Batch proofs must contain exactly the siblings needed for verification—no more, no less.

**Enforced by**: The hint format is canonical and `open_batch` consumes exactly the required siblings. Extra hints are ignored and left unread.

**Why this matters**: Prevents ambiguity in proof generation; callers can enforce transcript exhaustion if strict canonicality is required.

### Cryptographic Assumptions

1. **Collision-resistant hashing**
   - The `StatefulHasher` (sponge) must be collision-resistant
   - The `PseudoCompressionFunction` must be collision-resistant
   - **Auditor action**: Verify that concrete instantiations use secure primitives (e.g., Poseidon2)

2. **Correct sponge semantics**
   - `StatefulHasher::absorb_into` must properly mix input into state
   - `StatefulHasher::squeeze` must produce a binding commitment to absorbed data
   - **Auditor action**: Review `p3-miden-stateful-hasher` implementations

### Implementation Details Requiring Audit

| File | Function | Security Relevance |
|------|----------|-------------------|
| `lmcs.rs` | `open_batch` | Core Merkle verification; canonical sibling consumption |
| `lifted_tree.rs` | `build_leaf_states_upsampled` | Upsampling correctness; state maintenance |
| `lifted_tree.rs` | `prove_batch` | Canonical sibling emission |
| `utils.rs` | `digest_rows_and_salt` | Leaf hashing; width validation |

### Non-Security-Critical Code

- `mmcs.rs`: Thin wrapper over LMCS traits; delegates all verification
- `utils.rs`: SIMD packing helpers are correctness/perf only; note `digest_rows_and_salt` is security-critical (see table above)
- SIMD paths in `lifted_tree.rs`: Must match scalar paths (tested)

## Differences from Plonky3 `Mmcs`

| Aspect | Plonky3 `Mmcs` | LMCS |
|--------|---------------|------|
| Matrix heights | All same | Variable via lifting |
| Leaf hashing | `H::hash_iter_slices` (one-shot) | `StatefulHasher::absorb_into` (incremental) |
| Proof format | `Vec<[D; N]>` per query | `BatchProof` with deduplicated siblings |
| Multi-opening | Not native | First-class `prove_batch` API |
| Hiding support | Via separate type | Unified via `SALT_ELEMS` generic |

## Error Types

```rust
pub enum LmcsError {
    InvalidProof,
    RootMismatch,
}
```

## Performance Considerations

- **SIMD parallelization**: Tree construction uses packed operations when matrix height >= `PF::WIDTH`
- **Parallel iteration**: Uses `rayon` for row-wise operations
- **Zero-copy upsampling**: Virtual upsampling via index arithmetic, no data movement
- **Batch inversion**: Openings use Montgomery's trick for efficient multi-inversion
