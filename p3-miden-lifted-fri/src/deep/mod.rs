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
//! **Lifting.** Polynomials of degree d on domain D embed into a larger domain D* via
//! `f(X) ↦ f(Xʳ)` where r = |D*|/|D|. In bit-reversed order, this means each evaluation
//! repeats r times consecutively—implemented by virtual upsampling without data movement.
//!
//! **Verifier's view of lifting.** From the verifier's perspective, all polynomials
//! appear to be evaluated at the same point z on the same domain. The prover computes
//! `fᵢ(zʳ)` for degree-d polynomials, but this equals `fᵢ'(z)` where `fᵢ'(X) = fᵢ(Xʳ)`
//! is the lifted polynomial. This uniformity enables the `f_reduced` factorization.
//!
//! ## Preconditions (caller responsibility)
//!
//! The DEEP constructors assume all opening points are valid: distinct and outside the
//! trace subgroup `H` and the LDE evaluation coset `gK`. Invalid points can trigger
//! division by zero in the barycentric weights. In practice, the outer STARK protocol
//! is expected to enforce this before invoking DEEP.
//!
//! ## Random Linear Combination Convention
//!
//! The batching challenge `α` reduces all columns via Horner evaluation:
//! `f_reduced = horner(α, [f₀, f₁, ..., fₙ₋₁])`, where columns are flattened
//! across groups and matrices in commitment order. The `horner` function assigns
//! the highest power to the first element: `f₀·αⁿ⁻¹ + f₁·αⁿ⁻² + ... + fₙ₋₁·α⁰`.
//!
//! This convention is shared by:
//! - Prover OOD reduction ([`evals::BatchedEvals::reduce`])
//! - Verifier OOD reduction ([`evals::DeepEvals::reduce_point`])
//! - Verifier query-time row reduction ([`verifier::DeepOracle::open_batch`] via `horner_acc`)
//! - Prover LDE evaluation ([`prover::DeepPoly::from_evals`] via explicit dot-product
//!   with reversed negated coefficients — see comments there)

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
    /// Grinding bits before DEEP challenge sampling.
    pub deep_pow_bits: usize,
}

#[cfg(test)]
mod tests;
