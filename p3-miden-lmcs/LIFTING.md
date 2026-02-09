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

### Case 2: Without Salt (Non-Hiding) — Optimized Self-Compression

Squeeze **before** extending:

$$d_j = \text{Squeeze}(\sigma_j), \quad j \in [0, N_{\text{nat}})$$

This is valid because without salt, squeezing duplicated states would yield identical digests anyway: $\text{Squeeze}(\hat{\sigma}_k) = \text{Squeeze}(\sigma_{\lfloor k/r \rfloor}) = d_{\lfloor k/r \rfloor}$. Squeezing first avoids redundant operations.

Rather than materializing all $N$ upsampled leaf digests and building a full $N$-leaf Merkle tree (which would require $N - 1$ compression calls), we exploit the self-compression structure. Define the iterated self-compression:

$$C^{(0)}(d) = d, \quad C^{(\lambda+1)}(d) = \text{Compress}(C^{(\lambda)}(d),\; C^{(\lambda)}(d))$$

Let $k = \log_2(N / N_{\text{nat}})$. The key observation is that in a conceptual $N$-leaf tree built from upsampled digests, the bottom $k$ levels are entirely determined by self-compressions. Level $\lambda$ (for $0 \leq \lambda < k$) has $N / 2^\lambda$ nodes, but only $N_{\text{nat}}$ distinct values — each is $C^{(\lambda)}(d_j)$ for some $j$. At level $k$, we reach $N_{\text{nat}}$ distinct nodes: $C^{(k)}(d_j)$ for $j \in [0, N_{\text{nat}})$.

The optimization: instead of building a tree with $N$ leaves, we:
1. Compute $C^{(k)}(d_j)$ for each of the $N_{\text{nat}}$ original digests.
2. Build a standard Merkle tree from these $N_{\text{nat}}$ nodes.
3. Store the intermediate self-compression layers ($C^{(\lambda)}(d_j)$ for $\lambda \in [0, k)$) as **virtual layers** for proof generation.

This reduces the tree construction cost from $O(N)$ to $O(N_{\text{nat}})$ compressions.

**Algorithm (no-salt path):**

```
1. Build N_nat sponge states by absorbing all matrices (with internal upsampling).
2. Squeeze each state to get N_nat digests: [d_0, d_1, ..., d_{N_nat-1}].
3. Let k = log2(N / N_nat).
4. Build k virtual layers:
      virtual[0] = [d_0, d_1, ..., d_{N_nat-1}]            (original digests)
      virtual[λ] = [C(v, v) for v in virtual[λ-1]]         for λ = 1, ..., k-1
5. Compute the stored tree base:
      base[j] = C(virtual[k-1][j], virtual[k-1][j])        for j = 0, ..., N_nat-1
             = C^(k)(d_j)
6. Build Merkle tree from base[0..N_nat] (standard tree construction).
7. Store virtual layers for proving.
```

**Example** ($N_{\text{nat}} = 4$, $N = 16$, $k = 2$):

```
Squeeze: [d0, d1, d2, d3]

virtual[0] = [d0, d1, d2, d3]                                 (original digests)
virtual[1] = [C(d0,d0), C(d1,d1), C(d2,d2), C(d3,d3)]        (one self-compression)

Stored tree base (= C^2):
  [C(C(d0,d0),C(d0,d0)), C(C(d1,d1),C(d1,d1)), C(C(d2,d2),C(d2,d2)), C(C(d3,d3),C(d3,d3))]

Stored tree levels 1+: standard Merkle compression of 4 nodes up to root.
```

This produces the **same root** as a full 16-leaf tree built from the upsampled digests, but only computes $4 + 4 + 3 = 11$ compressions instead of $15$.

### Why the Two Cases Differ

The ordering (upsample-then-squeeze vs. squeeze-then-self-compress) matters because:

- **With salt**, upsampling states first and *then* absorbing independent salt per leaf ensures each duplicate gets a unique hash. If we squeezed first, the salt would never enter the sponge.
- **Without salt**, there is nothing to differentiate duplicated states, so squeezing them would produce identical digests regardless of order. Self-compression avoids materializing the full $N$-leaf tree while producing an identical root.

## Virtual Levels and Proof Generation

The conceptual $N$-leaf tree has $\log_2 N$ levels. The bottom $k$ levels are **virtual** — their structure is fully determined by the self-compression property. During proof generation for query index $i$:

**At virtual level $\lambda$ (for $0 \leq \lambda < k$):**
- The node at position $p$ has hash $C^{(\lambda)}(d_j)$ where $j = p \gg (k - \lambda)$.
- The sibling at position $p \oplus 1$ maps to the same original index $j$ (since flipping bit 0 does not affect bits $\geq 1$, and at level $\lambda$ positions within groups of $2^{k-\lambda}$ share the same value).
- Therefore, the **sibling hash equals the node hash**: both are $C^{(\lambda)}(d_j)$.
- The sibling hash is read from $\text{virtual}[\lambda][j]$.

**At stored level $r$ (for $r \geq 0$, corresponding to conceptual level $k + r$):**
- Standard Merkle sibling lookup from `digest_layers[r]`.

The prover emits virtual-level siblings as ordinary commitment hashes in the transcript. The **verifier does not need to know** about the virtual/stored distinction — it simply reads $\log_2 N$ sibling hashes and walks up to the root as usual.

**Example proof** for query $i = 5$ in the $N_{\text{nat}} = 4$, $N = 16$ tree ($k = 2$):

```
Original index j = i >> k = 5 >> 2 = 1

Virtual level 0: sibling = virtual[0][1] = d1            (same as leaf hash)
Virtual level 1: sibling = virtual[1][1] = C(d1, d1)     (same as running hash)
Stored level 0:  sibling = digest_layers[0][1 ^ 1] = digest_layers[0][0] = C^2(d0)
Stored level 1:  sibling = digest_layers[1][0 ^ 1] = digest_layers[1][1]
...up to root
```

## Opening Semantics

When opening leaf $k$ in a target-height tree, the verifier receives the rows of each matrix at their upsampled index:

$$\text{row}_i = M_i[\lfloor k / (N/n_i) \rfloor]$$

The verifier reconstructs the leaf hash and verifies the authentication path of depth $\log_2 N$. The verifier does not need to know that a target height was used or that virtual levels exist — it simply sees a Merkle tree of height $N$ with standard openings and standard sibling hashes.
