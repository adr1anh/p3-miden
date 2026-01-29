# Security Audit Guide

This document provides a focused guide for security auditors reviewing the Lifted FRI PCS implementation.

## Scope Overview

| Crate | Lines of Code | Security Criticality |
|-------|---------------|---------------------|
| `p3-miden-lmcs` | ~800 | **High** - Merkle commitment verification |
| `p3-miden-lifted-fri` | ~1500 | **High** - DEEP quotient + FRI protocol |
| `p3-miden-stateful-hasher` | ~200 | **Medium** - Sponge construction |

## Protocol Contracts (High Level)

### LMCS (p3-miden-lmcs)

**What LMCS proves**:
- Given a commitment (root) and query indices, the opened rows are consistent
  with some committed leaf preimages, under the configured hash and compression
  functions.
- The openings are for the lifted, uniform-height view of the input matrices.

**What LMCS does not prove**:
- It does not prove that any matrix "really had height n" beyond the supplied
  dimensions. A lifted height-n matrix is indistinguishable from an explicit
  height-N matrix that repeats rows.
- It does not enforce periodicity or other structure on columns; upstream
  protocols must enforce such semantics if required.
- It assigns no meaning to matrix positions; the higher-level protocol must
  bind the ordering.

**Trusted inputs / statement data**:
- `widths` (matrix widths, in commitment order).
- `log_max_height` (Merkle tree height / max domain size).
- The ordered list of matrices; permutation changes the commitment.

**Commitment preimage**:
- Each leaf hashes the concatenation of lifted rows in matrix order, with an
  optional salt appended:
  `leaf(i) = H(row_0(i) || row_1(i) || ... || row_{t-1}(i) || salt?)`.

### Lifted FRI PCS (p3-miden-lifted-fri)

**What the PCS proves**:
- Given commitments and transcript data, the claimed evaluations are consistent
  with a low-degree polynomial that matches the committed evaluations, as
  enforced by the DEEP quotient and FRI checks.

**What the PCS does not prove**:
- It does not validate that domain sizes, blowup factors, or folding arities are
  appropriate for a specific security level; these are parameter choices.
- It does not reinterpret LMCS lifting; smaller matrices are treated as lifted
  evaluations on the max domain.

**Trusted inputs / statement data**:
- Domain sizes, folding arities, and other `PcsParams`.
- Transcript ordering and challenge sampling must follow the prescribed flow.

## Security-Critical Code Paths

### 1. Merkle Verification (`p3-miden-lmcs`)

**File**: `p3-miden-lmcs/src/lmcs.rs`

| Function | What to Check |
|----------|---------------|
| `LmcsConfig::open_batch` | Root recomputation, sibling consumption order, duplicate indices |
| `LmcsConfig::read_batch_proof_from_channel` | Transcript parsing shape |

**File**: `p3-miden-lmcs/src/lifted_tree.rs`

| Function | What to Check |
|----------|---------------|
| `LiftedMerkleTree::prove_batch` | Canonical sibling emission order |

**File**: `p3-miden-lmcs/src/proof.rs`

| Function | What to Check |
|----------|---------------|
| `BatchProof::read_from_channel` | Canonical sibling parsing; no out-of-range indices |
| `BatchProof::single_proofs` | Deterministic path reconstruction |

**Key invariants to verify**:
- [ ] Siblings are consumed in canonical order (left-to-right, bottom-to-top)
- [ ] Missing siblings return `InvalidProof`
- [ ] Duplicate indices require identical rows/salt
- [ ] Out-of-range indices return `InvalidProof`
- [ ] Extra hints are ignored and left unread (callers can enforce transcript exhaustion)

### 2. Leaf Digest Computation (`p3-miden-lmcs`)

**File**: `p3-miden-lmcs/src/utils.rs`

| Function | What to Check |
|----------|---------------|
| `digest_rows_and_salt` | Width validation (caller), salt absorption order |

**File**: `p3-miden-lmcs/src/lifted_tree.rs`

| Function | What to Check |
|----------|---------------|
| `build_leaf_states_upsampled` | Upsampling correctness, state maintenance |
| `absorb_matrix` | Correct row absorption into states |
| `validate_heights` | Height ordering enforcement |

**Key invariants to verify**:
- [ ] Matrices absorbed in height-sorted order (enforced by panic)
- [ ] Upsampling duplicates states correctly when height increases
- [ ] Salt absorbed after all matrix rows
- [ ] SIMD path produces identical results to scalar path

### 3. DEEP Quotient Construction (`p3-miden-lifted-fri`)

**File**: `p3-miden-lifted-fri/src/deep/prover.rs`

| Function | What to Check |
|----------|---------------|
| `DeepPoly::new` | Coefficient derivation, accumulation order |
| `accumulate_matrices` | Virtual upsampling during accumulation |
| `derive_coeffs_from_challenge` | Alignment padding, reversed coefficient order |

**Key invariants to verify**:
- [ ] Evaluations observed into transcript before grinding
- [ ] Grinding witness requested before challenge sampling
- [ ] Column coefficients $\alpha^i$ computed in correct order for Horner
- [ ] Point coefficients $\beta^j$ applied correctly
- [ ] Upsampling during accumulation matches lifting semantics

### 4. DEEP Quotient Verification (`p3-miden-lifted-fri`)

**File**: `p3-miden-lifted-fri/src/deep/verifier.rs`

| Function | What to Check |
|----------|---------------|
| `DeepOracle::new` | Transcript order, PoW check, structure validation |
| `DeepOracle::open_batch` | Domain point reconstruction, Horner reduction |

**Key invariants to verify**:
- [ ] `channel.grind()` is called before sampling challenges
- [ ] Domain point $X = g \cdot \omega^{\text{bitrev}(i)}$ computed identically to prover
- [ ] `horner_acc` uses same alignment/padding as prover

### 5. FRI Commit Phase (`p3-miden-lifted-fri`)

**File**: `p3-miden-lifted-fri/src/fri/prover.rs`

| Function | What to Check |
|----------|---------------|
| `FriPolys::new` | Loop termination, s_inv computation, final poly extraction |
| `FriPolys::prove_queries` | Index shifting by `log_arity * (round + 1)` |

**Key invariants to verify**:
- [ ] Folding continues until `domain_size <= final_domain_size`
- [ ] `s_inv` values computed correctly for each round (bit-reversed)
- [ ] Final polynomial extracted via IDFT of truncated, de-bit-reversed evals
- [ ] Query indices shifted correctly for each round

### 6. FRI Query Verification (`p3-miden-lifted-fri`)

**File**: `p3-miden-lifted-fri/src/fri/verifier.rs`

| Function | What to Check |
|----------|---------------|
| `FriOracle::new` | Per-round PoW verification |
| `FriOracle::test_low_degree` | Evaluation consistency, final poly check |

**Key invariants to verify**:
- [ ] Per-round PoW witnesses verified in order
- [ ] Opened value at `position_in_coset` matches `current_eval`
- [ ] Coset generator `s_inv` computed identically to prover
- [ ] Final polynomial evaluated at correct point

### 7. FRI Folding (`p3-miden-lifted-fri`)

**Files**: `p3-miden-lifted-fri/src/fri/fold/arity2.rs`, `arity4.rs`, `arity8.rs`

| Function | What to Check |
|----------|---------------|
| `fold_evals` | Interpolation formula matches standard FRI derivation |

**Key invariants to verify**:
- [ ] Arity-2: $f(\beta) = \frac{f(s) + f(-s)}{2} + \frac{f(s) - f(-s)}{2s} \cdot \beta$
- [ ] Arity-4/8: Inverse FFT applied correctly
- [ ] `s_inv` used correctly (not `s`)

### 8. Barycentric Interpolation (`p3-miden-lifted-fri`)

**File**: `p3-miden-lifted-fri/src/deep/interpolate.rs`

| Function | What to Check |
|----------|---------------|
| `PointQuotients::new` | Batch inversion correctness |
| `PointQuotients::batch_eval_lifted` | Weight folding for lifted polynomials |

**Key invariants to verify**:
- [ ] Differences `z_j - x_i` all nonzero (evaluation points outside domain)
- [ ] Barycentric scaling factor computed correctly
- [ ] Weight folding sums adjacent weights correctly
- [ ] Lifted evaluation uses $z^r$ where $r = \text{domain\_size} / \text{matrix\_height}$

## Fiat-Shamir Transcript Order

Required order for the channel-driven PCS:

```
1. [Prover] Observe trace commitments
2. [Both]   Sample evaluation point z

3. [Prover] Compute evaluations f_i(z_j)
4. [Both]   Observe evaluations (with alignment padding)
5. [Both]   Grind for DEEP PoW
6. [Both]   Sample DEEP challenges α, β

7. [Prover] Compute DEEP quotient Q(X)
   For each FRI round i:
8.    [Both]   Observe round commitment
9.    [Both]   Grind for round PoW
10.   [Both]   Sample folding challenge β_i

11. [Both]   Observe final polynomial coefficients
12. [Both]   Grind for query PoW
13. [Both]   Sample query indices

14. [Prover] Generate query proofs (LMCS batch hints)
15. [Verifier] Verify query proofs and FRI folding
```

**Auditor action**: Trace through `p3-miden-lifted-fri/src/prover.rs` and `verifier.rs` to verify this order.

## Trust Assumptions

### External Dependencies

| Dependency | Assumption | Where Verified |
|------------|------------|----------------|
| `p3-field` | Field arithmetic correct | Upstream Plonky3 |
| `p3-symmetric` | `PseudoCompressionFunction` collision-resistant | Upstream Plonky3 |
| `p3-challenger` | Fiat-Shamir challenger is sound | Upstream Plonky3 |
| `p3-dft` | FFT/IFFT correct | Upstream Plonky3 |

### Internal Assumptions

| Assumption | Where Relied Upon | Consequence if False |
|------------|-------------------|---------------------|
| Matrices sorted by height | `build_leaf_states_upsampled` | Wrong leaf digests |
| Evaluation points outside domain | `batch_eval_lifted`, `DeepOracle::open_batch` | Division by zero |
| `SALT_ELEMS` matches actual salt | `HidingLmcsConfig` | Verification failure |
| Packed/scalar paths equivalent | SIMD code paths | Wrong outputs |

## Checklist for Complete Audit

### Merkle Tree (`p3-miden-lmcs`)
- [ ] `open_batch` consumes exactly the required siblings in canonical order
- [ ] `validate_heights` rejects non-sorted input
- [ ] `build_leaf_states_upsampled` upsamples correctly
- [ ] `digest_rows_and_salt` absorbs rows and salt in correct order
- [ ] `prove_batch` emits siblings in canonical order

### DEEP Quotient (`p3-miden-lifted-fri/deep`)
- [ ] Prover and verifier use identical Horner reduction
- [ ] Domain points reconstructed identically
- [ ] Grinding before challenge sampling

### FRI Protocol (`p3-miden-lifted-fri/fri`)
- [ ] Folding formulas match standard FRI
- [ ] `s_inv` computed correctly at each round
- [ ] Final polynomial check uses correct domain point
- [ ] Per-round grinding before folding challenge

### PCS Orchestration (`p3-miden-lifted-fri`)
- [ ] Transcript operations in correct order
- [ ] All proofs verified (none skipped)
- [ ] Error types propagated correctly

## Known Non-Issues

These are intentional design decisions, not bugs:

1. **Panics on invalid input**: Functions like `validate_heights` panic rather than return errors. This is intentional—these are programmer errors, not verifier-controlled input.

2. **Unbounded proof sizes**: LMCS batch hints do not enforce a maximum size. The protocol inherently bounds proof size via parameter choices.

3. **No constant-time operations**: This is not designed for side-channel resistance. Polynomial arithmetic is inherently variable-time.

4. **Parallel iteration order**: `rayon` parallel iterators may process in non-deterministic order, but results are accumulated in deterministic order.
