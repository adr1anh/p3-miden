# p3-miden-lmcs

Lifted Matrix Commitment Scheme (LMCS) for committing to multiple evaluation
matrices with power-of-two heights, using a uniform lifted view in bit-reversed
order.

## Overview

- Commit to matrices of different heights via virtual upsampling to a shared max.
- Hash rows with a configurable stateful hasher; internal nodes use a 2-to-1
  compression function.
- Support batch openings with canonical sibling emission.

## Lifting & Commitment Sketch

Let a polynomial f be evaluated over a coset gK with |K| = n, stored in
bit-reversed order. Let N = r * n be the max height. Lifting repeats each row r
times, which corresponds to evaluating f'(X) = f(X^r) over a larger coset of
size N. The commitment is the Merkle root of leaf hashes computed by absorbing
lifted rows (in matrix order) into the configured `StatefulHasher`, optionally
absorbing a salt, and then squeezing a hash:

```
leaf(i) = squeeze(absorb(row_0(i) || row_1(i) || ... || row_{t-1}(i) || salt?))
```

where row_j(i) is the lifted row for matrix j at index i. Matrix order is
binding.

## Protocol Contract (High Level)

**What LMCS proves**:
- Opened rows are consistent with some committed leaf preimages under the
  configured hash and compression functions.
- Openings refer to the lifted, uniform-height view of the input matrices.

**What LMCS does not prove**:
- It does not prove a matrix "really had height n" beyond the supplied
  dimensions; lifting is indistinguishable from explicit repetition.
- It does not enforce periodicity or other structure; upstream protocols must.

**Trusted inputs / statement data**:
- `widths` (matrix widths, in commitment order).
- `log_max_height` (tree height / max domain size).
- Matrix ordering; permutation changes the commitment.

## Entry Points

| Item | Purpose |
|------|---------|
| `LmcsConfig` / `HidingLmcsConfig` | Build non-hiding or salted commitments |
| `Lmcs::build_tree` | Build a commitment tree with no transcript padding |
| `Lmcs::build_aligned_tree` | Build a tree using the hasher alignment for transcript padding |
| `LmcsTree::prove_batch` | Prove openings at multiple indices |
| `Lmcs::open_batch` | Verify batch openings |
| `BatchProof` / `Proof` | Parsed proof formats |

## Assumptions & Invariants

- Matrices are sorted by height (shortest to tallest).
- `widths` and `log_max_height` are statement data chosen by the protocol.
- Evaluation rows are in bit-reversed order.
- Query indices are in range of the max height.
- Duplicate indices are coalesced in the transcript (first-occurrence order); verifiers
  return the same opening for each occurrence.
- If `build_aligned_tree` is used, alignment padding (as recorded on the tree) must be
  included in absorbed rows; LMCS does not enforce zero-padding.
- Hash and compression functions are collision-resistant.

## Code Map

| Path | Purpose |
|------|---------|
| `src/lmcs.rs` | Public API and verification |
| `src/lifted_tree.rs` | Tree construction and lifting |
| `src/utils.rs` | Leaf hashing helpers |

## Security

Audits should start with `SECURITY.md` at the workspace root for canonical proof
parsing, lifting correctness, and critical paths.

## License

Dual-licensed under MIT and Apache-2.0 at the workspace root.
