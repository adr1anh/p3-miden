//! Arity-4 FRI folding using inverse FFT.
//!
//! Given evaluations of a polynomial `f` on a coset `s·⟨ω⟩` where `ω = i` is a primitive
//! 4th root of unity, we recover `f(β)` for an arbitrary challenge point `β`.
//!
//! ## Setup
//!
//! Let `f(X) = c₀ + c₁X + c₂X² + c₃X³` with evaluations on the coset `s·⟨ω⟩`:
//!
//! ```text
//! y₀ = f(s),   y₁ = f(ωs),   y₂ = f(ω²s),   y₃ = f(ω³s)
//! ```
//!
//! We store these in **bit-reversed order**: `[y₀, y₂, y₁, y₃]`.
//!
//! ## Algorithm
//!
//! 1. **Inverse FFT**: Recover coefficients of `f(sX)` from evaluations on `⟨ω⟩`.
//! 2. **Evaluate**: Compute `f(sX)` at `X = β/s`, yielding `f(β)`.

use p3_field::{Algebra, ExtensionField, PackedField, TwoAdicField};

use super::FriFold;

/// Marker type for arity-4 FRI folding.
///
/// Folds quadruples of evaluations via size-4 inverse FFT followed by Horner evaluation.
/// More efficient than two rounds of arity-2 folding.
pub struct FriFold4;

/// Size-4 inverse FFT (unscaled), input in bit-reversed order.
///
/// Returns coefficients `[c₀, c₁, c₂, c₃]` of `4·f(sX) = c₀ + c₁X + c₂X² + c₃X³`.
///
/// ## Type Parameters
///
/// - `PF`: Packed base field (can be `F` for scalar or `F::Packing` for SIMD)
/// - `PEF`: Packed extension field (can be `EF` for scalar or `EF::ExtensionPacking` for SIMD)
///
/// The caller chooses whether to operate on scalars or packed values by selecting
/// the appropriate type parameters.
#[inline(always)]
fn ifft4<PF, PEF>(evals: [PEF; 4]) -> [PEF; 4]
where
    PF: PackedField,
    PF::Scalar: TwoAdicField,
    PEF: Algebra<PF>,
{
    // ω = i, primitive 4th root of unity
    let w: PF = PF::Scalar::two_adic_generator(2).into();

    // Input (bit-reversed): [y₀, y₂, y₁, y₃]
    let [y0, y2, y1, y3] = evals;

    // Inverse DFT formula (without 1/N normalization):
    //   4cⱼ = Σₖ yₖ · ω^(−jk)
    //
    // Expanded for each coefficient (i = imaginary unit):
    //   4c₀ = y₀ + y₁ + y₂ + y₃
    //   4c₁ = y₀ − i·y₁ − y₂ + i·y₃
    //   4c₂ = y₀ − y₁ + y₂ − y₃
    //   4c₃ = y₀ + i·y₁ − y₂ − i·y₃

    // -------------------------------------------------------------------------
    // Stage 0: length-2 butterflies on bit-reversed pairs
    // -------------------------------------------------------------------------
    let s02 = y0.clone() + y2.clone(); // y₀ + y₂  (used in c₀, c₂)
    let d02 = y0 - y2; // y₀ − y₂  (used in c₁, c₃)
    let s13 = y1.clone() + y3.clone(); // y₁ + y₃  (used in c₀, c₂)
    let d31 = y3 - y1; // y₃ − y₁  (note: negated so we can multiply by ω instead of ω⁻¹)

    // -------------------------------------------------------------------------
    // Stage 1: combine via length-4 butterflies
    //
    // Rewriting the target formulas using stage 0 results:
    //   4c₀ = (y₀ + y₂) + (y₁ + y₃)           = s02 + s13
    //   4c₂ = (y₀ + y₂) − (y₁ + y₃)           = s02 − s13
    //   4c₁ = (y₀ − y₂) + i(y₃ − y₁)          = d02 + i·d31
    //   4c₃ = (y₀ − y₂) − i(y₃ − y₁)          = d02 − i·d31
    // -------------------------------------------------------------------------
    let d31_w = d31 * w; // i · (y₃ − y₁)

    [
        s02.clone() + s13.clone(),   // 4c₀
        d02.clone() + d31_w.clone(), // 4c₁
        s02 - s13,                   // 4c₂
        d02 - d31_w,                 // 4c₃
    ]
}

impl FriFold<4> for FriFold4 {
    /// Evaluate `f(β)` from evaluations on a coset.
    ///
    /// ## Inputs
    ///
    /// - `evals`: evaluations `[f(s), f(ω²s), f(ωs), f(ω³s)]` in bit-reversed order,
    ///   equivalently `[f(s), f(−s), f(is), f(−is)]` since `ω = i`.
    /// - `s_inv`: the inverse of the coset generator `s`.
    /// - `beta`: the FRI folding challenge `β`.
    ///
    /// ## FRI Context
    ///
    /// In arity-4 FRI, the polynomial `f` is evaluated on cosets of the form `s·⟨ω⟩`.
    /// The verifier needs to check that `f(β)` equals the claimed folded value.
    /// This function recovers `f(β)` from the four coset evaluations via interpolation.
    #[inline(always)]
    fn fold_evals<PF, EF, PEF>(evals: [PEF; 4], s_inv: PF, beta: EF) -> PEF
    where
        PF: PackedField,
        PF::Scalar: TwoAdicField,
        EF: ExtensionField<PF::Scalar>,
        PEF: Algebra<PF> + Algebra<EF>,
    {
        // Recover coefficients [c₀, c₁, c₂, c₃] of 4·f(sX) via inverse FFT.
        let [c0, c1, c2, c3] = ifft4::<PF, PEF>(evals);

        // f(β) = f(s · β/s) = (1/4) · (c₀ + c₁·x + c₂·x² + c₃·x³)  where x = β/s.
        let x = PEF::from(beta) * s_inv;
        let terms = [
            c0,              // c₀
            c1 * x.clone(),  // c₁ · x
            c2 * x.square(), // c₂ · x²
            c3 * x.cube(),   // c₃ · x³
        ];

        // Divide by 4: use base field 1/4 since EF×F is cheaper than EF.halve().halve().
        PEF::sum_array::<4>(&terms).halve().halve()
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

    /// Test that ifft4 compiles with scalar types (F, EF).
    #[test]
    fn test_ifft4_scalar_types() {
        let evals: [EF; 4] = [EF::ZERO; 4];
        let _coeffs: [EF; 4] = ifft4::<F, EF>(evals);
    }

    /// Test that ifft4 compiles with packed types (Pf, Pef).
    #[test]
    fn test_ifft4_packed_types() {
        let evals: [Pef; 4] = [Pef::ZERO; 4];
        let _coeffs: [Pef; 4] = ifft4::<Pf, Pef>(evals);
    }

    /// Test that fold_evals (arity 4) compiles with scalar types.
    #[test]
    fn test_fold_evals_arity4_scalar_types() {
        let evals: [EF; 4] = [EF::ZERO; 4];
        let s_inv = F::ONE;
        let beta = EF::ONE;
        let _result: EF = FriFold4::fold_evals::<F, EF, EF>(evals, s_inv, beta);
    }

    /// Test that fold_evals (arity 4) compiles with packed types.
    #[test]
    fn test_fold_evals_arity4_packed_types() {
        let evals: [Pef; 4] = [Pef::ZERO; 4];
        let s_inv = Pf::ZERO;
        let beta = EF::ONE;
        let _result: Pef = FriFold4::fold_evals::<Pf, EF, Pef>(evals, s_inv, beta);
    }

    #[test]
    fn test_arity_4_babybear() {
        test_fold::<F, EF, FriFold4, 4>();
    }

    #[test]
    fn test_fold_matrix_arity4_packed_equivalence() {
        test_fold_matrix_packed_equivalence::<FriFold4, 4>();
    }
}
