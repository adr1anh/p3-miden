//! FRI folding via polynomial interpolation.
//!
//! FRI (Fast Reed-Solomon IOP of Proximity) requires computing `f(β)` from evaluations
//! of a polynomial `f` on a coset. This module provides a trait-based abstraction for
//! FRI folding at different arities.
//!
//! ## Arity
//!
//! The **arity** determines how many evaluations are folded together in each round:
//! - **Arity 2**: Fold pairs `{f(s), f(-s)}` using even-odd decomposition
//! - **Arity 4**: Fold quadruples `{f(s), f(-s), f(is), f(-is)}` using inverse FFT
//!
//! Higher arity reduces the number of FRI rounds but increases per-round work.

mod arity2;
mod arity4;
mod arity8;

pub use arity2::FriFold2;
pub use arity4::FriFold4;
pub use arity8::FriFold8;

use alloc::vec::Vec;

use p3_field::{Algebra, ExtensionField, PackedField, PackedValue, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrixView;
use p3_maybe_rayon::prelude::*;

use crate::utils::PackedFieldExtensionExt;

// ============================================================================
// Trait Definition
// ============================================================================

/// FRI folding strategy for evaluating `f(β)` from coset evaluations.
///
/// Given evaluations of a polynomial `f` on a coset of size `ARITY`, this trait
/// provides a method to recover `f(β)` for an arbitrary challenge point `β`.
pub trait FriFold<const ARITY: usize> {
    /// Evaluate `f(β)` from evaluations on a coset.
    ///
    /// ## Inputs
    ///
    /// - `evals`: evaluations in bit-reversed order
    /// - `s_inv`: inverse of the coset generator `s`
    /// - `beta`: the FRI folding challenge `β`
    fn fold_evals<PF, EF, PEF>(evals: [PEF; ARITY], s_inv: PF, beta: EF) -> PEF
    where
        PF: PackedField,
        PF::Scalar: TwoAdicField,
        EF: ExtensionField<PF::Scalar>,
        PEF: Algebra<PF> + Algebra<EF>;

    fn fold_matrix<F: TwoAdicField, EF: ExtensionField<F>>(
        input: RowMajorMatrixView<'_, EF>,
        s_invs: &[F],
        beta: EF,
    ) -> Vec<EF> {
        let width = F::Packing::WIDTH;
        if input.height() < width || width == 1 {
            Self::fold_matrix_scalar(input, s_invs, beta)
        } else {
            Self::fold_matrix_packed(input, s_invs, beta)
        }
    }

    /// Fold a matrix of coset evaluations using the challenge `beta`.
    ///
    /// Each row contains evaluations on a coset `s·⟨ω⟩`. Returns folded
    /// evaluations, one per row, maintaining bit-reversed order.
    fn fold_matrix_scalar<F: TwoAdicField, EF: ExtensionField<F>>(
        input: RowMajorMatrixView<'_, EF>,
        s_invs: &[F],
        beta: EF,
    ) -> Vec<EF> {
        assert_eq!(input.width, ARITY);
        let (evals, _) = input.values.as_chunks::<ARITY>();

        evals
            .par_iter()
            .zip(s_invs.par_iter())
            .map(|(evals, s_inv)| {
                // Scalar mode: PF=F, EF=EF, PEF=EF
                Self::fold_evals::<F, EF, EF>(*evals, *s_inv, beta)
            })
            .collect()
    }

    /// SIMD-optimized matrix folding using packed field operations.
    ///
    /// Processes multiple rows in parallel using horizontal SIMD packing.
    /// Equivalent to [`Self::fold_matrix_scalar`] but faster for large matrices.
    fn fold_matrix_packed<F: TwoAdicField, EF: ExtensionField<F>>(
        input: RowMajorMatrixView<'_, EF>,
        s_invs: &[F],
        beta: EF,
    ) -> Vec<EF> {
        assert_eq!(input.width, ARITY);
        let (evals, _) = input.values.as_chunks::<ARITY>();
        let width = F::Packing::WIDTH;
        assert!(evals.len().is_multiple_of(width));

        let mut new_evals = EF::zero_vec(evals.len());

        new_evals
            .par_chunks_exact_mut(width)
            .zip(evals.par_chunks_exact(width))
            .zip(s_invs.par_chunks_exact(width))
            .for_each(|((new_evals_chunk, evals_chunk), s_inv_chunk)| {
                let evals_packed =
                    <EF::ExtensionPacking as PackedFieldExtensionExt<F, EF>>::pack_ext_columns(
                        evals_chunk,
                    );
                let s_invs_packed = F::Packing::from_slice(s_inv_chunk);
                let new_evals_packed = Self::fold_evals::<F::Packing, EF, EF::ExtensionPacking>(
                    evals_packed,
                    *s_invs_packed,
                    beta,
                );
                <EF::ExtensionPacking as PackedFieldExtensionExt<F, EF>>::to_ext_slice(
                    &new_evals_packed,
                    new_evals_chunk,
                );
            });
        new_evals
    }
}

// ============================================================================
// Butterfly Helpers
// ============================================================================

/// DIT butterfly: `(x1 + twiddle * x2, x1 - twiddle * x2)`
///
/// See [`p3_dft::DitButterfly`] for the standard implementation.
/// This version supports mixed-type operations where values are extension field
/// elements and twiddles are base field elements.
#[inline(always)]
pub(super) fn dit_butterfly<PF: PackedField, PEF: Algebra<PF>>(
    x1: PEF,
    x2: PEF,
    twiddle: PF,
) -> (PEF, PEF) {
    let x2_tw = x2 * twiddle;
    (x1.clone() + x2_tw.clone(), x1 - x2_tw)
}

/// Twiddle-free butterfly: `(x1 + x2, x1 - x2)`
///
/// See [`p3_dft::TwiddleFreeButterfly`] for the standard implementation.
#[inline(always)]
pub(super) fn twiddle_free_butterfly<
    PEF: Clone + core::ops::Add<Output = PEF> + core::ops::Sub<Output = PEF>,
>(
    x1: PEF,
    x2: PEF,
) -> (PEF, PEF) {
    (x1.clone() + x2.clone(), x1 - x2)
}

// ============================================================================
// Shared Test Utilities
// ============================================================================

#[cfg(test)]
pub(super) mod tests {
    use core::array;

    use alloc::vec::Vec;

    use p3_field::{ExtensionField, Field, TwoAdicField};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_util::reverse_slice_index_bits;
    use rand::distr::{Distribution, StandardUniform};
    use rand::prelude::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    pub(super) use crate::tests::{EF, F};

    pub type Pf = <F as Field>::Packing;
    pub type Pef = <EF as ExtensionField<F>>::ExtensionPacking;

    /// Evaluate polynomial using Horner's method.
    pub fn horner<F: Field, EF: ExtensionField<F>>(coeffs: &[EF], x: F) -> EF {
        coeffs
            .iter()
            .rev()
            .copied()
            .reduce(|acc, c| acc * x + c)
            .unwrap_or(EF::ZERO)
    }

    /// Generic test for FRI folding at any arity.
    ///
    /// Creates a random polynomial of degree `ARITY - 1`, evaluates it on a coset
    /// of size `ARITY`, then verifies that `fold_evals` correctly recovers `f(β)`.
    pub fn test_fold<Fo, EFo, FF: FriFold<ARITY>, const ARITY: usize>()
    where
        Fo: TwoAdicField,
        EFo: ExtensionField<Fo>,
        StandardUniform: Distribution<EFo> + Distribution<Fo>,
    {
        let rng = &mut SmallRng::seed_from_u64(1);
        let beta: EFo = rng.sample(StandardUniform);

        // Random polynomial of degree ARITY - 1
        let poly: [EFo; ARITY] = array::from_fn(|_| rng.sample(StandardUniform));

        // Compute roots of unity in bit-reversed order for this arity
        // For ARITY=2: [1, -1]
        // For ARITY=4: [1, -1, w, -w] = [w^0, w^2, w^1, w^3]
        let roots: [Fo; ARITY] = {
            let log_arity = ARITY.ilog2() as usize;
            let mut points = Fo::two_adic_generator(log_arity).powers().collect_n(ARITY);
            reverse_slice_index_bits(&mut points);
            points.try_into().unwrap()
        };

        let s: Fo = rng.sample(StandardUniform);
        let s_inv = s.inverse();

        // Evaluate polynomial at coset points: [f(s·root) for root in roots]
        let evals: [EFo; ARITY] = roots.map(|root| horner(&poly, root * s));

        // Expected: f(beta)
        let expected = horner::<EFo, EFo>(&poly, beta);

        // Test fold_evals with scalar types: PF=F, EF=EF, PEF=EF
        let result = FF::fold_evals::<Fo, EFo, EFo>(evals, s_inv, beta);
        assert_eq!(result, expected);
    }

    /// Test that `fold_matrix` and `fold_matrix_packed` produce identical results.
    pub fn test_fold_matrix_packed_equivalence<FF: FriFold<ARITY>, const ARITY: usize>() {
        let rng = &mut SmallRng::seed_from_u64(42);

        // Create input matrix with height = multiple of packing width
        let height = Pf::WIDTH * 4; // 4 packed rows worth
        let width = ARITY;
        let values: Vec<EF> = (0..height * width)
            .map(|_| rng.sample(StandardUniform))
            .collect();
        let input = RowMajorMatrix::new(values, width);

        // Generate s_invs (one per row)
        let s_invs: Vec<F> = (0..height)
            .map(|_| rng.sample::<F, _>(StandardUniform).inverse())
            .collect();

        let beta: EF = rng.sample(StandardUniform);

        // Call both implementations
        let result_scalar = FF::fold_matrix_scalar::<F, EF>(input.as_view(), &s_invs, beta);
        let result_packed = FF::fold_matrix_packed::<F, EF>(input.as_view(), &s_invs, beta);

        // They should be identical
        assert_eq!(result_scalar, result_packed);
    }
}
