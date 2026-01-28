//! DEEP quotient helper utilities.

use p3_field::{ExtensionField, Field};

/// Horner reduction starting from an accumulator over multiple slices.
///
/// Slices should already include any padding values that must be absorbed.
pub fn reduce_with_powers_from<'a, F, EF>(
    acc_prev: EF,
    slices: impl IntoIterator<Item = &'a [F]>,
    challenge: EF,
) -> EF
where
    F: Field + 'a,
    EF: ExtensionField<F>,
{
    slices
        .into_iter()
        .flatten()
        .fold(acc_prev, |acc, val| acc * challenge + *val)
}
