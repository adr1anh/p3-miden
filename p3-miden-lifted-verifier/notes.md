# p3-miden-lifted-verifier notes

This crate hosts the end-to-end verification flow for the lifted STARK protocol.

## Highlights
- Replays the transcript to receive commitments and openings.
- Uses `p3-miden-lifted-fri::verifier::verify_with_channel` to open [zeta, zeta_next].
- Recomputes folded constraints at projected points `y_j = zeta^{r_j}` and checks the quotient identity at zeta.
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
4. Sample constraint folding challenge alpha and cross-trace accumulator beta.
5. Receive quotient commitment.
6. Sample OOD point zeta (rejection-sampled outside trace domain), derive zeta_next.
7. Verify PCS openings at [zeta, zeta_next] for main, aux, and quotient.
8. Reconstruct `Q(zeta)` from the opened quotient chunks.
9. For each trace instance j, set `y_j = zeta^{r_j}` and evaluate folded constraints at `y_j`.
10. Horner-accumulate across traces with beta.
11. Check quotient identity: accumulated == Q(zeta) * (zeta^N - 1).
12. Ensure transcript is fully consumed.

## Mathematical background (lifted STARK, verifier view)

Audience: you already know STARK verifiers; this section explains how *lifting* lets us
verify mixed-height traces using a single uniform opening point and a single quotient
identity.

We use the notation from `coset.md` (workspace root). All sizes are powers of two.

### Domains and cosets

Let:

- $N = 2^n$ be the **maximum** trace height across all traces.
- $D = 2^d$ be the **constraint degree** (quotient-domain blowup).
- $B = 2^b$ be the **PCS/Fri blowup**, with $D \le B$.
- $g$ be the fixed multiplicative shift (`F::GENERATOR`).

Define subgroups:

$$
H = \langle \omega_H \rangle,\ |H| = N
\qquad
J = \langle \omega_J \rangle,\ |J| = N\,D
\qquad
K = \langle \omega_K \rangle,\ |K| = N\,B
$$

and shifted cosets $gH, gJ, gK$.

### What “lifted traces” mean to the verifier

Suppose instance $j$ has trace height

$$
n_j = N / r_j
\qquad\text{with } r_j = 2^{\ell_j}.
$$

Let $t_j(X)$ be the degree-$<n_j$ interpolant of the trace column (over $H^{r_j}$),
and define the lifted polynomial

$$
t_j^*(X) := t_j(X^{r_j}).
$$

Key facts:

1. $\deg(t_j^*) < N$, so it “fits” the max-degree regime.
2. For any point $x$,
   $$
   t_j^*(x) = t_j(x^{r_j}).
   $$
3. Choosing the commitment coset shift as $g^{r_j}$ makes the evaluation domains line up:
   $$
   (gK)^{r_j} = \{ (g\,k)^{r_j} : k \in K \} = g^{r_j} K^{r_j}.
   $$

#### Uniform-height view

Conceptually, you can think of each short trace as being “stretched” to height $N$
by repeating it $r_j$ times (stacking copies). From the verifier’s perspective this
means each trace behaves like a single height-$N$ object:

- there is one global out-of-domain point $\zeta$,
- one global “next-row” multiplier $\omega_H$ (the generator of the max trace domain),
- and each trace instance just uses a different *projection* of that point.

### How openings at $[\zeta,\ \zeta\cdot\omega_H]$ give per-trace local/next pairs

The verifier samples a single $\zeta$ outside both the max trace domain $H$ and the max
LDE coset $gK$, and sets:

$$
\zeta_{\mathrm{next}} = \zeta \cdot \omega_H.
$$

For instance $j$, define the **virtual** evaluation point

$$
y_j := \pi_{r_j}(\zeta) = \zeta^{r_j}.
$$

Then:

$$
t_j^*(\zeta) = t_j(\zeta^{r_j}) = t_j(y_j),
$$

and likewise for the next-row point:

$$
t_j^*(\zeta_{\mathrm{next}})
  = t_j\big((\zeta\cdot\omega_H)^{r_j}\big)
  = t_j\big(\zeta^{r_j}\cdot\omega_H^{r_j}\big)
  = t_j\big(y_j\cdot\omega_{H^{r_j}}\big).
$$

Since $\omega_{H^{r_j}} = \omega_H^{r_j}$ is the generator of the smaller trace domain,
the pair opened at $[\zeta,\zeta_{\mathrm{next}}]$ is exactly the local/next pair needed
to evaluate AIR transition constraints for that trace.

This is why verifier code computes `y_j = zeta^{r_j}` and evaluates selectors/periodics
at $y_j$, while still requesting PCS openings only at the global points
$[\zeta,\zeta_{\mathrm{next}}]$.

### Constraint folding at the lifted OOD point

For each instance $j$, the verifier:

1. Interprets the opened main/aux values as $(T_j(y_j), T_j(y_j\cdot\omega_{H^{r_j}}))$.
2. Computes the usual row selectors at $y_j$ using:

$$
Z_{H^{r_j}}(y_j) = y_j^{n_j} - 1,
$$

and the unnormalized selector formulas (matching `LiftedCoset::selectors_at`):

$$
\mathrm{is\_first}(y) = \frac{Z_{H^{r_j}}(y)}{y-1},
\quad
\mathrm{is\_last}(y) = \frac{Z_{H^{r_j}}(y)}{y-\omega_{H^{r_j}}^{-1}},
\quad
\mathrm{is\_transition}(y) = y-\omega_{H^{r_j}}^{-1}.
$$

3. Evaluates periodic columns at $y_j$ (each period-$p$ column is evaluated at
   $y_j^{n_j/p}$).
4. Folds constraints with the per-proof challenge $\alpha$ using Horner accumulation:

$$
\mathrm{folded}_j
  = (((c_0\cdot\alpha + c_1)\cdot\alpha + c_2)\cdots)\cdot\alpha + c_k.
$$

5. Accumulates across instances using the per-proof challenge $\beta$:

$$
\mathrm{acc} \leftarrow \mathrm{acc}\cdot\beta + \mathrm{folded}_j.
$$

Because lifting is just composition by $X^{r_j}$, “evaluate then lift” matches
“lift then evaluate”: $N_j^*(\zeta) = N_j(\zeta^{r_j})$. So the verifier’s accumulation
matches the prover’s accumulation at the max point $\zeta$.

### Quotient reconstruction at $\zeta$

The prover commits to a single quotient object that represents $Q$ on the max quotient
domain $gJ$ and is sent as $D$ “chunk” polynomials $q_0,\dots,q_{D-1}$ of degree $<N$.

At verification time we open each $q_t$ at $\zeta$ and reconstruct $Q(\zeta)$ via the
barycentric formula used in `reconstruct_quotient`:

- Let $\omega_S := \omega_J^N$ be the $D$-th root of unity.
- Let $u := (\zeta/g)^N$.
- Define weights
$$
  w_t := \frac{\omega_S^t}{u - \omega_S^t}.
$$

Then:

$$
Q(\zeta) = \frac{\sum_{t=0}^{D-1} w_t\,q_t(\zeta)}{\sum_{t=0}^{D-1} w_t}.
$$

### The single identity the verifier checks

After accumulating all folded constraint evaluations, the verifier checks:

$$
\mathrm{acc} = Q(\zeta)\cdot Z_H(\zeta),
\qquad
Z_H(\zeta) = \zeta^N - 1.
$$

This is the lifted analogue of the classic STARK quotient identity, but it works for
mixed-height traces because each instance’s constraints were evaluated at its projected
point $y_j = \zeta^{r_j}$.

## TODO / follow-ups
- Consider a dedicated error type for periodic table validation failures.
- Decide how to expose `StarkTranscript` (debug-only vs public API).
