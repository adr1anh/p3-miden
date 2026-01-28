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

mod evals;
mod interpolate;
mod proof;
pub(crate) mod prover;
mod utils;
pub(crate) mod verifier;

pub use evals::{BatchedEvals, BatchedGroupEvals, DeepEvals};
pub use interpolate::PointQuotients;
pub use proof::DeepTranscript;
pub use verifier::DeepError;

/// DEEP quotient parameters.
///
/// Controls batching alignment and proof-of-work grinding for DEEP challenge sampling.
#[derive(Clone, Copy, Debug)]
pub struct DeepParams {
    /// Column alignment for batching in DEEP quotient construction.
    ///
    /// Typically set to the hasher's rate (e.g., 8 for Poseidon2 with WIDTH=16, RATE=8).
    /// Ensures coefficients are aligned for efficient hashing.
    pub alignment: usize,

    /// Number of bits for proof-of-work grinding before DEEP challenge sampling.
    ///
    /// Set to 0 to disable grinding. Higher values increase prover work but improve
    /// soundness by preventing grinding attacks on DEEP challenges (α, β).
    pub proof_of_work_bits: usize,
}

#[cfg(test)]
mod tests;
