# p3-miden-lifted-fri

Lifted Polynomial Commitment Scheme (PCS) built from a DEEP quotient and FRI,
using LMCS for commitments over evaluation matrices.

## Protocol Summary

This is a polynomial commitment scheme where the verifier requests evaluations
of all committed matrices at the same two points (derived via Fiat-Shamir).
All matrices appear at a uniform height via LMCS upsampling, which:
- Eliminates height-dependent code paths in the verifier.
- Simplifies DEEP quotient construction (same domain points for all matrices).
- Enables uniform recursive verifier implementations.

**Protocol flow**:
1. **Commit**: LMCS commits to (possibly upsampled) evaluation matrices.
2. **Open**: DEEP batches evaluation claims into one low-degree polynomial.
3. **Verify**: FRI checks low degree; LMCS verifies consistency with commitments.

FRI folding is configurable (arity 2, 4, or 8).

## Math Sketch

Given evaluation claims f_i(z_j), DEEP forms a quotient polynomial

```
Q(X) = ∑_i ∑_j α^i · β^j · (f_i(X) - f_i(z_j)) / (X - z_j)
```

so Q is low-degree iff all claims are consistent. Evaluations are stored in
bit-reversed order. **Lifting** treats shorter matrices as evaluations of
f_i(X^r) on the LDE domain (r = lde_height / height_i), achieved via
**upsampling** (row repetition in bit-reversed order). FRI then repeatedly
folds evaluations by the chosen arity to test low degree.

### What a query opening means for lifted traces

When the verifier opens tree index `i` from a commitment of height `N`, the
LMCS returns one value per committed matrix. For a matrix of height `n = N/r`:

- The commitment stores `f(g^r · (ω_n)^{bitrev_n(j)})` at original index `j`.
- After nearest-neighbor upsampling to height `N`, index `i` maps to original
  index `j = i >> log₂(r)`.
- So the opened value is `f( (g · (ω_N)^{bitrev_N(i)})^r )`: the original
  polynomial `f` evaluated at the *r*-th power of the large-domain point.

Equivalently, defining the lifted polynomial `f'(X) = f(X^r)`, the opened
value is `f'(g · (ω_N)^{bitrev_N(i)})` — the same domain point used for
full-height matrices. This is what makes the DEEP quotient uniform: both the
out-of-domain evaluations (`f(z^r)`) and the query openings (`f(X^r)`) use
the lifted polynomial, so all columns appear to live on the same domain
regardless of their original height.

## Optimizations

- **Aligned openings**: Rows are padded to a multiple of the sponge rate,
  ensuring a fixed number of permutations per leaf hash. Simplifies verifier
  parsing.
- **Uniform opening points**: All matrices (regardless of original height) are
  opened at the same domain points. DEEP batching uses a single set of
  z-values; no per-matrix domain reconstruction.
- **Recursive verifier benefits**: Uniform height eliminates branching on
  matrix dimensions. The verifier logic is a single code path.

## Entry Points

| Path | Purpose |
|------|---------|
| `src/prover.rs` | Prover flow and transcript emission |
| `src/verifier.rs` | Verifier flow and transcript consumption |
| `params.rs` | `PcsParams` configuration |
| `proof.rs` | `PcsTranscript` typed view of proof data |

## Modules

| Module | Purpose |
|--------|---------|
| `deep/` | DEEP quotient batching and interpolation |
| `fri/` | FRI folding and query logic |
| `utils.rs` | Domain mapping and bit-reversed helpers |

## Conventions & Assumptions

- Evaluation rows are in bit-reversed order.
- Domain sizes and folding factors come from `PcsParams`.
- The PCS APIs use `log_lde_height` to denote the LDE evaluation domain height (the
  height of committed LDE matrices). When a trace degree is known, the typical mapping is:
  `log_lde_height = log_trace_height + log_blowup` (plus any caller-chosen extension).
- **Upsampling** (row repetition in bit-reversed order) semantics follow `p3-miden-lmcs`.
- Alignment padding is a transcript-formatting convenience for trace commitments built with
  `build_aligned_tree` (alignment is recorded on the tree). FRI round commitments are unaligned. Padding keeps verifier parsing
  uniform because the underlying sponge pads with zeros after absorbing a list of elements.
  The padded columns are treated as additional polynomials and are checked for low degree by
  the PCS, but they are not forced to be zero unless the caller/AIR enforces that.
- Transcript ordering (observe → grind → sample) is security-critical; see
  `SECURITY.md`.

## Transcript Model

Channels produce compact proof buffers suitable for transmission:
- `ProverTranscript` records field elements, commitments, and LMCS hints.
- `VerifierTranscript` consumes the buffer while verifying.

Structured types (`PcsTranscript`, `DeepTranscript`, `FriTranscript`) exist for
**export and debugging**. They reconstruct the full prover-verifier interaction
(including challenges and PoW witnesses) without re-running verification. These
types validate parsing shape only—they do **not** guarantee proof validity.

**Future**: A transcript fingerprint (hash of the fully consumed transcript) may
be added to check implementation equivalence across verifier implementations.

## Tests to Start With

- `src/fri/fold/mod.rs`
- `src/deep/interpolate.rs`
- `src/tests.rs`

## Security

Audits should start with `SECURITY.md` at the workspace root for transcript
ordering, DEEP/FRI invariants, and critical paths.

## License

Dual-licensed under MIT and Apache-2.0 at the workspace root.
