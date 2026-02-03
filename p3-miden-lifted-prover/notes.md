# p3-miden-lifted-prover notes

This crate hosts the end-to-end proving flow for the lifted STARK protocol.

## Highlights
- Uses LMCS to commit LDEs over nested cosets for the AIR trace.
- Aux trace is required; preprocessed trace is ignored in this scaffold.
- Computes constraint numerators on the LDE domain and divides by vanishing polynomial.
- Opens all committed trees via `p3-miden-lifted-fri::prover::open_with_channel`.
- Channel-first API: `prove_single` accepts a `ProverChannel` for transcript operations.

## API

### `prove_single`

```rust
pub fn prove_single<F, EF, A, L, Dft, Ch>(
    config: &StarkConfig<L, Dft>,
    air: &A,
    trace: &RowMajorMatrix<F>,
    public_values: &[F],
    channel: &mut Ch,
) -> Result<(), ProverError>
```

**Arguments:**
- `config`: STARK configuration (PCS params, LMCS, DFT)
- `air`: The AIR definition implementing `MidenAir<F, EF>`
- `trace`: Main trace matrix
- `public_values`: Public values for this AIR
- `channel`: Prover channel for transcript (implements `ProverChannel`)

The channel should be initialized with domain separator and public values before calling.

## Prover flow

1. Validate trace dimensions against AIR definition.
2. Commit main trace LDE on nested coset (bit-reversed), observe commitment.
3. Sample aux randomness, build aux trace (required), commit aux LDE.
4. Sample constraint folding challenge alpha.
5. Build periodic LDEs for periodic columns.
6. Compute folded constraint numerator on natural-order LDE domain.
7. Convert to bit-reversed order, divide by vanishing polynomial.
8. Commit quotient polynomial.
9. Sample OOD point zeta (rejection-sampled outside trace domain), derive zeta_next.
10. Open via PCS at [zeta, zeta_next] for main, aux, and quotient trees.

## TODO / follow-ups
- Move quotient helpers into a shared `quotient.rs` once the API stabilizes.
- Clarify periodic column encoding (currently LDE on nested coset + index mod).
