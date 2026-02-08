//! # DEEP Quotient for Lifted FRI
//!
//! DEEP converts evaluation claims into a low-degree test. Given committed polynomials
//! `{fᵢ}` and claimed evaluations `fᵢ(zⱼ) = vᵢⱼ`, the quotient
//!
//! ```text
//! Q(X) = Σⱼ βʲ · Σᵢ αⁱ · (vᵢⱼ - fᵢ(X)) / (zⱼ - X)
//! ```
//!
//! is low-degree iff all claims are correct. A false claim creates a pole, detectable by FRI.
//!
//! ## Design Choices
//!
//! **Uniform opening points.** All columns share the same opening points `{zⱼ}`. This enables
//! factoring out `f_reduced(X) = Σᵢ αⁱ·fᵢ(X)`, so the verifier computes one inner product
//! per query rather than one per column per point.
//!
//! **Two challenges.** Separating α (columns) from β (points) improves soundness. With a
//! single challenge, a cheating prover must avoid collisions among k·m terms; with two,
//! only k+m terms matter. This costs one extra field element in the transcript.
//!
//! **Lifting.** The prover evaluates shorter polynomials on the r-th power of the LDE
//! coset: `f` on `g^r·K_n` where `r = N/n`. This coset is exactly `{(g·ω_N^j)^r : j}`,
//! so after LMCS upsampling the committed values are evaluations of `f'(X) = f(Xʳ)` on
//! the full LDE coset `g·K_N`. In bit-reversed order, this means each evaluation repeats
//! r times consecutively — implemented by virtual upsampling without data movement.
//!
//! **Why lifting is transparent.** Because the prover works over the powered coset, the
//! verifier sees all polynomials on the same domain. The prover provides `fᵢ(zʳ)` as the
//! out-of-domain evaluation (= `fᵢ'(z)`), and LMCS query openings return `fᵢ(Xʳ)` at
//! domain points (= `fᵢ'(X)`). Both use the same lifted polynomial `fᵢ'`, so the DEEP
//! quotient works uniformly and enables the `f_reduced` factorization.
//!
//! ## Preconditions (caller responsibility)
//!
//! The DEEP constructors assume all opening points are valid: distinct and outside the
//! trace subgroup `H` and the LDE evaluation coset `gK`. Invalid points can trigger
//! division by zero in the barycentric weights. In practice, the outer STARK protocol
//! is expected to enforce this before invoking DEEP.

mod evals;
mod interpolate;
mod proof;
pub(crate) mod prover;
pub(crate) mod verifier;

pub use evals::DeepEvals;
pub use interpolate::PointQuotients;
pub use proof::DeepTranscript;
pub use verifier::DeepError;

/// DEEP quotient parameters.
///
/// Controls proof-of-work grinding for DEEP challenge sampling.
/// Column alignment is handled at the LMCS layer and by padding evaluations.
#[derive(Clone, Copy, Debug)]
pub struct DeepParams {
    /// Proof-of-work grinding bits before DEEP challenge sampling.
    pub proof_of_work_bits: usize,
}

#[cfg(test)]
mod tests;
