# p3-miden-lifted-verifier notes

This crate hosts the end-to-end verification flow for the lifted STARK protocol.

## Highlights
- Replays the transcript to receive commitments and openings.
- Uses `p3-miden-lifted-fri::verifier::verify_with_channel` to open [zeta, zeta_next].
- Recomputes folded constraints at zeta and checks the quotient identity.
- Channel-first API: `verify_single` accepts a `VerifierChannel` for transcript operations.

## API

### `verify_single`

```rust
pub fn verify_single<F, EF, A, L, Dft, Ch>(
    config: &StarkConfig<L, Dft>,
    air: &A,
    log_trace_height: usize,
    public_values: &[F],
    channel: &mut Ch,
) -> Result<(), VerifierError>
```

**Arguments:**
- `config`: STARK configuration (PCS params, LMCS, DFT)
- `air`: The AIR definition implementing `MidenAir<F, EF>`
- `log_trace_height`: Log2 of the trace height
- `public_values`: Public values for this AIR
- `channel`: Verifier channel for transcript (implements `VerifierChannel`)

The channel should be initialized with domain separator and public values before calling.

## Verifier flow

1. Receive main trace commitment.
2. Sample aux randomness.
3. Receive aux trace commitment.
4. Sample constraint folding challenge alpha.
5. Receive quotient commitment.
6. Sample OOD point zeta (rejection-sampled outside trace domain), derive zeta_next.
7. Verify PCS openings at [zeta, zeta_next] for main, aux, and quotient.
8. Extract opened values (main local/next, aux local/next, quotient).
9. Evaluate periodic polynomials at zeta.
10. Evaluate constraints at zeta using the constraint folder.
11. Check quotient identity: folded * inv_vanishing == quotient(zeta).
12. Ensure transcript is fully consumed.

## TODO / follow-ups
- Consider a dedicated error type for periodic table validation failures.
- Decide how to expose `StarkTranscript` (debug-only vs public API).
