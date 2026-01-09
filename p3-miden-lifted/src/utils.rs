use alloc::vec::Vec;
use core::array;

use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, TwoAdicField};
use p3_util::reverse_slice_index_bits;

// ============================================================================
// Extension trait for PackedFieldExtension methods not in upstream
// ============================================================================

/// Extension trait adding `pack_ext_columns` and `to_ext_slice` methods.
///
/// These methods enable efficient SIMD operations on arrays of extension field elements
/// by providing column-wise packing and unpacking utilities.
pub trait PackedFieldExtensionExt<
    BaseField: Field,
    ExtField: ExtensionField<BaseField, ExtensionPacking = Self>,
>: PackedFieldExtension<BaseField, ExtField>
{
    /// Pack N columns from WIDTH rows into N packed extension field elements.
    ///
    /// Input: `rows[lane][col]` - WIDTH rows, each with N extension field elements.
    /// Output: `result[col]` - N packed values, where each packs WIDTH lanes.
    fn pack_ext_columns<const N: usize>(rows: &[[ExtField; N]]) -> [Self; N] {
        let width = BaseField::Packing::WIDTH;
        debug_assert_eq!(rows.len(), width);
        array::from_fn(|col| {
            let col_elems: Vec<ExtField> = (0..width).map(|lane| rows[lane][col]).collect();
            Self::from_ext_slice(&col_elems)
        })
    }

    /// Extract all lanes to an output slice.
    fn to_ext_slice(&self, out: &mut [ExtField]) {
        let width = BaseField::Packing::WIDTH;
        for (lane, slot) in out.iter_mut().enumerate().take(width) {
            *slot = self.extract(lane);
        }
    }
}

impl<
    BaseField: Field,
    ExtField: ExtensionField<BaseField, ExtensionPacking = P>,
    P: PackedFieldExtension<BaseField, ExtField>,
> PackedFieldExtensionExt<BaseField, ExtField> for P
{
}

/// Compute padding needed to align `len` to `alignment`.
///
/// Returns the number of zeros to add so that `len + padding` is a multiple of `alignment`.
#[inline]
pub const fn alignment_padding(len: usize, alignment: usize) -> usize {
    len.next_multiple_of(alignment) - len
}

/// Coset points `gK` in bit-reversed order.
///
/// Bit-reversal gives two properties essential for lifting:
/// - **Adjacent negation**: `gK[2i+1] = -gK[2i]`, so both square to the same value
/// - **Prefix nesting**: `gK[0..n/r]` equals the r-th power coset `(gK)ʳ`
///
/// Together these enable iterative weight folding in barycentric evaluation.
pub fn bit_reversed_coset_points<F: TwoAdicField>(log_n: usize) -> Vec<F> {
    let coset = TwoAdicMultiplicativeCoset::new(F::GENERATOR, log_n).unwrap();
    let mut pts: Vec<F> = coset.iter().collect();
    reverse_slice_index_bits(&mut pts);
    pts
}
