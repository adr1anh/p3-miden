# Mathematical Description of LMCS Lifting with Target Height

## Setup

We have $t$ matrices $M_0, M_1, \ldots, M_{t-1}$ with heights $n_0 \leq n_1 \leq \cdots \leq n_{t-1}$, all powers of two. Each matrix $M_i$ has width $w_i$ and stores evaluations of polynomials in **bit-reversed order** over a multiplicative coset $g K_i$ where $|K_i| = n_i$.

The **natural height** is $N_{\text{nat}} = n_{t-1}$ (the tallest matrix). With the new parameter, we specify a **target height** $N = 2^{\ell}$ where $N \geq N_{\text{nat}}$.

## Upsampling as Polynomial Lifting

For matrix $M_i$ of height $n_i$, the entry at index $j$ stores:

$$M_i[j] = f_i(g \cdot \omega_{n_i}^{\text{bitrev}_{n_i}(j)})$$

where $f_i$ is a polynomial of degree $< n_i$ and $\omega_{n_i}$ is a primitive $n_i$-th root of unity.

**Nearest-neighbor upsampling** from height $n_i$ to height $N$ maps index $k \in [0, N)$ to:

$$\widetilde{M}_i[k] = M_i[\lfloor k / r_i \rfloor] = f_i(g \cdot \omega_{n_i}^{\text{bitrev}_{n_i}(\lfloor k/r_i \rfloor)})$$

where $r_i = N / n_i$ is the repetition factor.

Using the bit-reversal identity $\text{bitrev}_N(k) \bmod n_i = \text{bitrev}_{n_i}(k \gg \log r_i)$, this equals:

$$\widetilde{M}_i[k] = f_i'(g \cdot \omega_N^{\text{bitrev}_N(k)})$$

where $f_i'(X) = f_i(X^{r_i})$ is the **lifted polynomial** of degree $< n_i \cdot r_i$ (but with the same number of non-zero coefficients). In other words, upsampling in bit-reversed order is equivalent to evaluating $f_i(X^{r_i})$ over the larger coset $g K$ with $|K| = N$.

## Leaf Construction (Sponge Absorption)

At each leaf index $k \in [0, N)$, we incrementally absorb the upsampled rows into a sponge state:

$$\sigma_k = \text{Absorb}(\sigma_0,\; \widetilde{M}_0[k],\; \widetilde{M}_1[k],\; \ldots,\; \widetilde{M}_{t-1}[k])$$

where $\sigma_0$ is the initial (zero) sponge state. In the implementation, this is done efficiently: matrices are processed shortest-to-tallest, and when transitioning from height $n_i$ to $n_{i+1}$, the intermediate states are duplicated via nearest-neighbor before absorbing the next matrix.

## Target Height Extension

When $N > N_{\text{nat}}$, we have $N_{\text{nat}}$ sponge states after absorbing all matrices. These must be extended to $N$ leaves. **The strategy depends on whether salt is present.**

### Case 1: With Salt (Hiding)

Upsample the state vector **before** finalization:

$$\hat{\sigma}_k = \sigma_{\lfloor k / r \rfloor}, \quad r = N / N_{\text{nat}}, \quad k \in [0, N)$$

Then absorb independent salt $s_k \in \mathbb{F}^{\text{SALT}}$ (sampled uniformly) and squeeze:

$$d_k = \text{Squeeze}(\text{Absorb}(\hat{\sigma}_k, s_k))$$

Since $s_k$ is independent for each $k$, duplicated states produce **distinct** leaf digests even when $\hat{\sigma}_k = \hat{\sigma}_{k+1}$. This preserves the hiding property.

**Algorithm (salt path):**

```
1. Build N_nat sponge states by absorbing all matrices (with internal upsampling).
2. Upsample: repeat each state r = N / N_nat times contiguously.
      states = [σ_0, ..., σ_0, σ_1, ..., σ_1, ..., σ_{N_nat-1}, ..., σ_{N_nat-1}]
                 \___ r ___/    \___ r ___/         \_________ r _________/
3. For each k in [0, N): absorb salt s_k into states[k].
4. For each k in [0, N): squeeze states[k] to get leaf digest d_k.
5. Build Merkle tree from [d_0, d_1, ..., d_{N-1}].
```

### Case 2: Without Salt (Non-Hiding)

Squeeze **before** upsampling:

$$d_j = \text{Squeeze}(\sigma_j), \quad j \in [0, N_{\text{nat}})$$

Then upsample the digest vector:

$$\hat{d}_k = d_{\lfloor k / r \rfloor}, \quad k \in [0, N)$$

This is valid because without salt, squeezing duplicated states would yield identical digests anyway: $\text{Squeeze}(\hat{\sigma}_k) = \text{Squeeze}(\sigma_{\lfloor k/r \rfloor}) = d_{\lfloor k/r \rfloor}$. Upsampling post-squeeze avoids redundant squeeze operations.

**Algorithm (no-salt path):**

```
1. Build N_nat sponge states by absorbing all matrices (with internal upsampling).
2. Squeeze each state to get N_nat digests: [d_0, d_1, ..., d_{N_nat-1}].
3. Upsample: repeat each digest r = N / N_nat times contiguously.
      digests = [d_0, ..., d_0, d_1, ..., d_1, ..., d_{N_nat-1}, ..., d_{N_nat-1}]
                  \___ r ___/    \___ r ___/         \__________ r __________/
4. Build Merkle tree from the N upsampled digests.
```

### Why the Two Cases Differ

The ordering (upsample-then-squeeze vs. squeeze-then-upsample) matters because:

- **With salt**, upsampling states first and *then* absorbing independent salt per leaf ensures each duplicate gets a unique hash. If we squeezed first, the salt would never enter the sponge.
- **Without salt**, there is nothing to differentiate duplicated states, so squeezing them would produce identical digests regardless of order. Squeezing first is strictly more efficient since we only squeeze $N_{\text{nat}}$ states instead of $N$.

## Merkle Tree Structure with Self-Compression

The $N$ leaf digests $\hat{d}_0, \hat{d}_1, \ldots, \hat{d}_{N-1}$ form the bottom layer of a binary Merkle tree. In the non-salted case, consecutive blocks of $r$ leaves are identical:

$$\hat{d}_{k} = \hat{d}_{k'} \quad \text{whenever} \quad \lfloor k/r \rfloor = \lfloor k'/r \rfloor$$

This creates a characteristic pattern in the lower $\log_2 r$ levels of the tree. At the leaf level, sibling pairs are identical, producing **self-compressions**:

$$\text{Compress}(\hat{d}_{2i}, \hat{d}_{2i+1}) = \text{Compress}(d_j, d_j) \quad \text{for } j = \lfloor i / (r/2) \rfloor$$

This pattern propagates upward. Defining the iterated self-compression:

$$C^{(0)}(d) = d, \quad C^{(\lambda+1)}(d) = \text{Compress}(C^{(\lambda)}(d),\; C^{(\lambda)}(d))$$

at level $\lambda < \log_2 r$, every node equals $C^{(\lambda)}(d_j)$ for the appropriate $j$. At level $\log_2 r$, we recover the tree that would have been built from the $N_{\text{nat}}$ original digests. Above that level, the tree is identical to the natural-height tree.

**Example** ($N_{\text{nat}} = 4$, $N = 16$, $r = 4$):

```
Leaf digests (level 0):
  [d0, d0, d0, d0, d1, d1, d1, d1, d2, d2, d2, d2, d3, d3, d3, d3]

Level 1 (8 nodes, pairwise compress):
  [C(d0,d0), C(d0,d0), C(d1,d1), C(d1,d1), C(d2,d2), C(d2,d2), C(d3,d3), C(d3,d3)]

Level 2 (4 nodes) — equivalent to natural-height leaf layer:
  [C(C(d0,d0), C(d0,d0)),  C(C(d1,d1), C(d1,d1)),  C(C(d2,d2), C(d2,d2)),  C(C(d3,d3), C(d3,d3))]

Levels 3+ proceed as the natural-height tree would from its 4 leaves.
```

## Opening Semantics

When opening leaf $k$ in a target-height tree, the verifier receives the rows of each matrix at their upsampled index:

$$\text{row}_i = M_i[\lfloor k / (N/n_i) \rfloor]$$

The verifier can reconstruct the leaf hash and verify the authentication path of depth $\log_2 N$ as usual. The verifier does not need to know that a target height was used — it simply sees a Merkle tree of height $N$ with standard openings.
