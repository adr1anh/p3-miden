# p3-miden-lifted-fri

Lifted Polynomial Commitment Scheme (PCS) built from a DEEP quotient and FRI,
using LMCS for commitments over evaluation matrices.

## Protocol Summary

1. **Commit**: LMCS commits to (possibly lifted) evaluation matrices.
2. **Open**: DEEP batches evaluation claims into one low-degree polynomial.
3. **Verify**: FRI checks low degree and LMCS openings.

FRI folding is configurable (arity 2/4/8).

## Math Sketch

Given evaluation claims f_i(z_j), DEEP forms a quotient polynomial

```
Q(X) = sum_i sum_j alpha^i * beta^j * (f_i(X) - f_i(z_j)) / (X - z_j)
```

so Q is low-degree iff all claims are consistent. Evaluations are stored in
bit-reversed order. Lifting treats shorter matrices as evaluations of
f_i(X^r) on the LDE domain (r = lde_height / height_i). FRI then repeatedly
folds evaluations by the chosen arity to test low degree.

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
- Lifting semantics follow `p3-miden-lmcs`.
- Alignment padding is a transcript-formatting convenience: openings are padded to the
  sponge's alignment (rate), which keeps verifier implementations uniform. This works because
  the underlying sponge pads with zeros after absorbing a list of elements. The padded columns
  are treated as additional polynomials and are checked for low degree by the PCS, but they are
  not forced to be zero unless the caller/AIR enforces that.
- Transcript ordering (observe → grind → sample) is security-critical; see
  `SECURITY.md`.

## Tests to Start With

- `src/fri/fold/mod.rs`
- `src/deep/interpolate.rs`
- `src/tests.rs`

## Security

Audits should start with `SECURITY.md` at the workspace root for transcript
ordering, DEEP/FRI invariants, and critical paths.

## License

Dual-licensed under MIT and Apache-2.0 at the workspace root.
