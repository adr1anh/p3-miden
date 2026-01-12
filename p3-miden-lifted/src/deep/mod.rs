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

pub(crate) mod interpolate;
pub(crate) mod prover;
pub(crate) mod verifier;

use alloc::vec::Vec;

use p3_challenger::FieldChallenger;
use p3_commit::{BatchOpening, Mmcs};
use p3_field::{ExtensionField, Field};

use thiserror::Error;

use crate::utils::alignment_padding;

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

/// DEEP proof-of-work witness.
///
/// The evaluations are stored in the PCS `Proof` struct.
/// This struct only contains the grinding witness for DEEP challenge sampling.
pub struct DeepProof<Witness> {
    /// Proof-of-work witness for DEEP challenge grinding.
    pub(crate) pow_witness: Witness,
}

/// Query proof containing Merkle openings for DEEP quotient verification.
///
/// Holds the batch openings from the input commitment that the verifier
/// needs to reconstruct `f_reduced(X)` at the queried point.
pub struct DeepQuery<F: Field, Commit: Mmcs<F>> {
    pub(crate) openings: Vec<BatchOpening<F, Commit>>,
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during DEEP oracle construction or verification.
#[derive(Debug, Error)]
pub enum DeepError {
    /// Claimed evaluations don't match commitment structure.
    ///
    /// This can mean wrong number of evaluation groups, matrices, or columns.
    #[error("evaluation structure doesn't match commitment")]
    StructureMismatch,
    /// Proof-of-work witness verification failed.
    #[error("invalid proof-of-work witness")]
    InvalidPowWitness,
}

/// Observe evaluations into the Fiat-Shamir transcript.
///
/// Each matrix's columns are observed with alignment padding to match
/// the coefficient derivation in the DEEP quotient.
fn observe_evals<F, EF, Challenger>(
    evals: &[Vec<MatrixGroupEvals<EF>>],
    challenger: &mut Challenger,
    alignment: usize,
) where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    for point_evals in evals {
        for group_evals in point_evals {
            for matrix_evals in group_evals.iter_matrices() {
                for val in matrix_evals {
                    challenger.observe_algebra_element(*val);
                }
                // Pad to alignment with zeros (must match coefficient alignment)
                for _ in 0..alignment_padding(matrix_evals.len(), alignment) {
                    challenger.observe_algebra_element(EF::ZERO);
                }
            }
        }
    }
}

/// Evaluations of polynomial columns at an out-of-domain point, organized by matrix.
///
/// Structure: `evals[matrix_idx][column_idx]` holds `f_{matrix,col}(z)`.
///
/// The grouping by matrix preserves the structure needed for batched reduction,
/// where matrices are processed in height order and each matrix's columns are
/// reduced with consecutive challenge powers.
#[derive(Clone, Debug)]
pub struct MatrixGroupEvals<T>(pub(crate) Vec<Vec<T>>);

impl<T> MatrixGroupEvals<T> {
    /// Create a new `MatrixGroupEvals` from nested vectors.
    ///
    /// Structure: `evals[matrix_idx][column_idx]` for each matrix in a commitment group.
    pub const fn new(evals: Vec<Vec<T>>) -> Self {
        Self(evals)
    }

    /// Returns the number of matrices in this group.
    pub const fn num_matrices(&self) -> usize {
        self.0.len()
    }

    /// Iterate over matrices, yielding the column evaluations for each.
    pub fn iter_matrices(&self) -> impl Iterator<Item = &[T]> {
        self.0.iter().map(|v| v.as_slice())
    }

    /// Iterate over all column evaluations across all matrices.
    ///
    /// Yields evaluations in order: all columns of matrix 0, then matrix 1, etc.
    pub fn iter_evals(&self) -> impl Iterator<Item = &T> {
        self.0.iter().flatten()
    }

    /// Transform each evaluation using the provided closure.
    ///
    /// Preserves the matrix/column structure while mapping `T -> U`.
    pub fn map<U, F: FnMut(&T) -> U>(&self, mut f: F) -> MatrixGroupEvals<U> {
        MatrixGroupEvals::new(
            self.0
                .iter()
                .map(|matrix| matrix.iter().map(&mut f).collect())
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests;
