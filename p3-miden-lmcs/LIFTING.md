# Mathematical Description of LMCS Lifting with Target Height

## Setup

We have $t$ matrices $M_0, M_1, \ldots, M_{t-1}$ with heights $n_0 \leq n_1 \leq \cdots \leq n_{t-1}$, all powers of two. Each matrix $M_i$ has width $w_i$ and stores evaluations of polynomials in **bit-reversed order** over a multiplicative coset $g K_i$ where $|K_i| = n_i$.

The **natural height** is $N_{\text{nat}} = n_{t-1}$ (the tallest matrix). With the new parameter, we specify a **target height** $N = 2^{\ell}$ where $N \geq N_{\text{nat}}$.

## Upsampling as Polynomial Lifting

For matrix $M_i$ of height $n_i$, the entry at index $j$ stores:

$$M_i[j] = f_i\!\bigl(g \cdot \omega_{n_i}^{\operatorname{bitrev}_{n_i}(j)}\bigr)$$

where $f_i$ is a polynomial of degree $< n_i$ and $\omega_{n_i}$ is a primitive $n_i$-th root of unity.

**Nearest-neighbor upsampling** from height $n_i$ to height $N$ maps index $k \in [0, N)$ to:

$$\widetilde{M}_i[k] = M_i\!\bigl[\lfloor k / r_i \rfloor\bigr] = f_i\!\bigl(g \cdot \omega_{n_i}^{\operatorname{bitrev}_{n_i}(\lfloor k/r_i \rfloor)}\bigr)$$

where $r_i = N / n_i$ is the repetition factor.

Using the bit-reversal identity $\operatorname{bitrev}_N(k) \bmod n_i = \operatorname{bitrev}_{n_i}(k \gg \log r_i)$, this equals:

$$\widetilde{M}_i[k] = f_i'\!\bigl(g \cdot \omega_N^{\operatorname{bitrev}_N(k)}\bigr)$$

where $f_i'(X) = f_i(X^{r_i})$ is the **lifted polynomial** of degree $< n_i \cdot r_i$ (but with the same number of non-zero coefficients). In other words, upsampling in bit-reversed order is equivalent to evaluating $f_i(X^{r_i})$ over the larger coset $g K$ with $|K| = N$.

## Leaf Construction (Sponge Absorption)

At each leaf index $k \in [0, N)$, we incrementally absorb the upsampled rows into a sponge state:

$$\sigma_k = \operatorname{Absorb}\!\bigl(\sigma_0,\; \widetilde{M}_0[k],\; \widetilde{M}_1[k],\; \ldots,\; \widetilde{M}_{t-1}[k]\bigr)$$

where $\sigma_0$ is the initial (zero) sponge state. In the implementation, this is done efficiently: matrices are processed shortest-to-tallest, and when transitioning from height $n_i$ to $n_{i+1}$, the intermediate states are duplicated via nearest-neighbor before absorbing the next matrix.

## Target Height Extension

When $N > N_{\text{nat}}$, we have $N_{\text{nat}}$ sponge states after absorbing all matrices. These must be extended to $N$ leaves. **The strategy depends on whether salt is present:**

### Case 1: With Salt (Hiding)

Upsample the state vector **before** finalization:

$$\hat{\sigma}_k = \sigma_{\lfloor k / r \rfloor}, \quad r = N / N_{\text{nat}}, \quad k \in [0, N)$$

Then absorb independent salt $s_k \in \mathbb{F}^{\text{SALT}}$ (sampled uniformly) and squeeze:

$$d_k = \operatorname{Squeeze}\!\bigl(\operatorname{Absorb}(\hat{\sigma}_k, s_k)\bigr)$$

Since $s_k$ is independent for each $k$, duplicated states produce **distinct** leaf digests even when $\hat{\sigma}_k = \hat{\sigma}_{k+1}$. This preserves the hiding property.

### Case 2: Without Salt (Non-Hiding)

Squeeze **before** upsampling:

$$d_j = \operatorname{Squeeze}(\sigma_j), \quad j \in [0, N_{\text{nat}})$$

Then upsample the digest vector:

$$\hat{d}_k = d_{\lfloor k / r \rfloor}, \quad k \in [0, N)$$

This is valid because without salt, $\operatorname{Squeeze}(\hat{\sigma}_k) = \operatorname{Squeeze}(\sigma_{\lfloor k/r \rfloor}) = d_{\lfloor k/r \rfloor}$ — squeezing duplicated states would yield identical digests anyway. Upsampling post-squeeze avoids redundant squeeze operations.

## Merkle Tree Structure with Self-Compression

The $N$ leaf digests $\hat{d}_0, \hat{d}_1, \ldots, \hat{d}_{N-1}$ form the bottom layer of a binary Merkle tree. In the non-salted case, consecutive blocks of $r$ leaves are identical:

$$\hat{d}_{k} = \hat{d}_{k'} \quad \text{whenever} \quad \lfloor k/r \rfloor = \lfloor k'/r \rfloor$$

This creates a characteristic pattern in the lower $\log_2 r$ levels of the tree. At the leaf level, sibling pairs are identical, producing **self-compressions**:

$$\operatorname{Compress}(\hat{d}_{2i}, \hat{d}_{2i+1}) = \operatorname{Compress}(d_j, d_j) \quad \text{for } j = \lfloor i / (r/2) \rfloor$$

This pattern propagates upward: at level $\lambda < \log_2 r$, every node equals $\operatorname{Compress}^{(\lambda)}(d_j, d_j)$ for the appropriate $j$. At level $\log_2 r$, we recover the tree that would have been built from the $N_{\text{nat}}$ original digests. Above that level, the tree is identical to the natural-height tree.

## Opening Semantics

When opening leaf $k$ in a target-height tree, the verifier receives the rows of each matrix at their upsampled index:

$$\text{row}_i = M_i\!\bigl[\lfloor k / (N/n_i) \rfloor\bigr]$$

The verifier can reconstruct the leaf hash and verify the authentication path of depth $\log_2 N$ as usual. The verifier does not need to know that a target height was used — it simply sees a Merkle tree of height $N$ with standard openings.
