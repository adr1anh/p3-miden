//! DEEP quotient helper utilities.

use p3_field::{ExtensionField, Field};

/// Compute padding needed to align `len` to `alignment`.
///
/// Returns the number of zeros to add so that `len + padding` is a multiple of `alignment`.
#[inline]
pub const fn alignment_padding(len: usize, alignment: usize) -> usize {
    len.next_multiple_of(alignment) - len
}

/// Horner reduction starting from an accumulator over multiple slices.
pub fn reduce_with_powers_from<'a, F, EF>(
    acc: EF,
    slices: impl IntoIterator<Item = &'a [F]>,
    challenge: EF,
    alignment: usize,
) -> EF
where
    F: Field + 'a,
    EF: ExtensionField<F>,
{
    slices.into_iter().fold(acc, |acc, slice| {
        let acc = slice.iter().fold(acc, |a, &val| a * challenge + val);
        acc * challenge.exp_u64(alignment_padding(slice.len(), alignment) as u64)
    })
}
