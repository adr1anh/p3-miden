//! DEEP quotient helper utilities.

use alloc::vec::Vec;

use crate::utils::MatrixGroupEvals;
use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field};

/// Observe evaluations into the Fiat-Shamir transcript.
///
/// Each matrix's columns are observed with alignment padding to match
/// the coefficient derivation in the DEEP quotient.
pub fn observe_evals<F, EF, Challenger>(
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

/// Compute padding needed to align `len` to `alignment`.
///
/// Returns the number of zeros to add so that `len + padding` is a multiple of `alignment`.
#[inline]
pub const fn alignment_padding(len: usize, alignment: usize) -> usize {
    len.next_multiple_of(alignment) - len
}

/// Horner reduction: computes `Σᵢ αⁿ⁻¹⁻ⁱ · vᵢ` via left-to-right accumulation.
///
/// For each value v, computes `acc = α·acc + v`. The reversed coefficient order
/// (from [`super::prover::derive_coeffs_from_challenge`]) makes this produce the
/// same result as explicit `Σᵢ coeffs[i] · vals[i]`.
///
/// # Alignment
///
/// After each slice, multiplies by `α^gap` where `gap` pads the slice length to
/// the next multiple of `alignment`. This is equivalent to:
/// - Padding each row with zeros to the alignment width
/// - Including those zeros in the Horner accumulation
///
/// An alternative implementation could materialize zero-padded rows; this approach
/// achieves the same result without allocating the padding.
pub fn reduce_with_powers<'a, F, EF>(
    slices: impl IntoIterator<Item = &'a [F]>,
    challenge: EF,
    alignment: usize,
) -> EF
where
    F: Field + 'a,
    EF: ExtensionField<F>,
{
    slices.into_iter().fold(EF::ZERO, |acc, slice| {
        // Horner's method on this slice: acc = α·acc + v for each v
        let acc = slice.iter().fold(acc, |a, &val| a * challenge + val);
        // Skip alignment gap: equivalent to processing implicit zeros
        acc * challenge.exp_u64(alignment_padding(slice.len(), alignment) as u64)
    })
}
