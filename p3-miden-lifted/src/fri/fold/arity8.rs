//! Arity-8 FRI folding using inverse FFT.
//!
//! Given evaluations of a polynomial `f` on a coset `s·⟨ω⟩` where `ω` is a primitive
//! 8th root of unity, we recover `f(β)` for an arbitrary challenge point `β`.
//!
//! ## Algorithm
//!
//! 1. **Inverse FFT**: Recover coefficients of `f(sX)` from evaluations on `⟨ω⟩`.
//! 2. **Evaluate**: Compute `f(sX)` at `X = β/s`, yielding `f(β)`.
//!
//! The inverse FFT uses a 3-stage Cooley-Tukey DIT butterfly structure.

use p3_field::{Algebra, ExtensionField, PackedField, TwoAdicField};

use super::{FriFold, dit_butterfly, twiddle_free_butterfly};

/// Marker type for arity-8 FRI folding.
///
/// Folds octuples of evaluations via size-8 inverse FFT followed by polynomial evaluation.
pub struct FriFold8;

/// Size-8 inverse FFT (unscaled), input in bit-reversed order.
///
/// Returns coefficients `[c₀, c₁, ..., c₇]` of `8·f(sX)`.
///
/// Uses DIT butterfly operations following the pattern from [`p3_dft::DitButterfly`].
#[inline(always)]
fn ifft8<PF, PEF>(evals: [PEF; 8]) -> [PEF; 8]
where
    PF: PackedField,
    PF::Scalar: TwoAdicField,
    PEF: Algebra<PF>,
{
    // Roots of unity (as packed base field for efficient multiplication)
    let w4: PF = PF::Scalar::two_adic_generator(2).into();
    let w8: PF = PF::Scalar::two_adic_generator(3).into();

    // Precompute inverse twiddles for IDFT: ω⁻ᵏ = ω^(N-k)
    let w4_inv = w4.cube(); // ω₄³ = ω₄⁻¹
    let w8_2 = w8.square(); // ω₈²
    let w8_3 = w8_2 * w8; // ω₈³
    let w8_5 = w8_3 * w8_2; // ω₈⁵ = ω₈⁻³
    let w8_6 = w8_3.square(); // ω₈⁶ = ω₈⁻²
    let w8_7 = w8_6 * w8; // ω₈⁷ = ω₈⁻¹

    // Bit-reversed input: [y₀, y₄, y₂, y₆, y₁, y₅, y₃, y₇]
    let [y0, y4, y2, y6, y1, y5, y3, y7] = evals;

    // -------------------------------------------------------------------------
    // Stage 0: 4 twiddle-free butterflies
    // -------------------------------------------------------------------------
    let (a0, a1) = twiddle_free_butterfly(y0, y4);
    let (a2, a3) = twiddle_free_butterfly(y2, y6);
    let (a4, a5) = twiddle_free_butterfly(y1, y5);
    let (a6, a7) = twiddle_free_butterfly(y3, y7);

    // -------------------------------------------------------------------------
    // Stage 1: length-4 butterflies with twiddle ω₄⁻¹
    // -------------------------------------------------------------------------
    let (b0, b2) = twiddle_free_butterfly(a0, a2);
    let (b1, b3) = dit_butterfly(a1, a3, w4_inv);

    let (b4, b6) = twiddle_free_butterfly(a4, a6);
    let (b5, b7) = dit_butterfly(a5, a7, w4_inv);

    // -------------------------------------------------------------------------
    // Stage 2: length-8 butterflies with twiddles ω₈⁻ᵏ
    // -------------------------------------------------------------------------
    let (c0, c4) = twiddle_free_butterfly(b0, b4);
    let (c1, c5) = dit_butterfly(b1, b5, w8_7); // ω₈⁻¹ = ω₈⁷
    let (c2, c6) = dit_butterfly(b2, b6, w8_6); // ω₈⁻² = ω₈⁶
    let (c3, c7) = dit_butterfly(b3, b7, w8_5); // ω₈⁻³ = ω₈⁵

    [c0, c1, c2, c3, c4, c5, c6, c7]
}

impl FriFold<8> for FriFold8 {
    /// Evaluate `f(β)` from evaluations on a coset.
    ///
    /// ## Inputs
    ///
    /// - `evals`: evaluations `[f(s), f(ω⁴s), f(ω²s), f(ω⁶s), f(ωs), f(ω⁵s), f(ω³s), f(ω⁷s)]`
    ///   in bit-reversed order, where `ω` is the primitive 8th root of unity.
    /// - `s_inv`: the inverse of the coset generator `s`.
    /// - `beta`: the FRI folding challenge `β`.
    #[inline(always)]
    fn fold_evals<PF, EF, PEF>(evals: [PEF; 8], s_inv: PF, beta: EF) -> PEF
    where
        PF: PackedField,
        PF::Scalar: TwoAdicField,
        EF: ExtensionField<PF::Scalar>,
        PEF: Algebra<PF> + Algebra<EF>,
    {
        // Recover coefficients [c₀, ..., c₇] of 8·f(sX) via inverse FFT.
        let coeffs = ifft8::<PF, PEF>(evals);

        // f(β) = f(s · β/s) = (1/8) · Σᵢ cᵢ · xⁱ  where x = β/s.
        let x = PEF::from(beta) * s_inv;

        // Compute powers of x efficiently
        let x2 = x.square();
        let x3 = x2.clone() * x.clone();
        let x4 = x2.clone().square();
        let x5 = x4.clone() * x.clone();
        let x6 = x4.clone() * x2.clone();
        let x7 = x4.clone() * x3.clone();

        let terms = [
            coeffs[0].clone(),
            coeffs[1].clone() * x,
            coeffs[2].clone() * x2,
            coeffs[3].clone() * x3,
            coeffs[4].clone() * x4,
            coeffs[5].clone() * x5,
            coeffs[6].clone() * x6,
            coeffs[7].clone() * x7,
        ];

        // Divide by 8
        PEF::sum_array::<8>(&terms).halve().halve().halve()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::fri::fold::tests::{EF, F, Pef, Pf, test_fold, test_fold_matrix_packed_equivalence};

    /// Test that ifft8 compiles with scalar types (F, EF).
    #[test]
    fn test_ifft8_scalar_types() {
        let evals: [EF; 8] = [EF::ZERO; 8];
        let _coeffs: [EF; 8] = ifft8::<F, EF>(evals);
    }

    /// Test that ifft8 compiles with packed types (Pf, Pef).
    #[test]
    fn test_ifft8_packed_types() {
        let evals: [Pef; 8] = [Pef::ZERO; 8];
        let _coeffs: [Pef; 8] = ifft8::<Pf, Pef>(evals);
    }

    /// Test that fold_evals (arity 8) compiles with scalar types.
    #[test]
    fn test_fold_evals_arity8_scalar_types() {
        let evals: [EF; 8] = [EF::ZERO; 8];
        let s_inv = F::ONE;
        let beta = EF::ONE;
        let _result: EF = FriFold8::fold_evals::<F, EF, EF>(evals, s_inv, beta);
    }

    /// Test that fold_evals (arity 8) compiles with packed types.
    #[test]
    fn test_fold_evals_arity8_packed_types() {
        let evals: [Pef; 8] = [Pef::ZERO; 8];
        let s_inv = Pf::ZERO;
        let beta = EF::ONE;
        let _result: Pef = FriFold8::fold_evals::<Pf, EF, Pef>(evals, s_inv, beta);
    }

    #[test]
    fn test_arity_8_babybear() {
        test_fold::<F, EF, FriFold8, 8>();
    }

    #[test]
    fn test_fold_matrix_arity8_packed_equivalence() {
        test_fold_matrix_packed_equivalence::<FriFold8, 8>();
    }
}
