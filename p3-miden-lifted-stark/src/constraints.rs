//! Constraint evaluation helpers.
//!
//! Provides selector computation for constraint evaluation at OOD points.

use p3_field::{ExtensionField, PrimeCharacteristicRing, TwoAdicField};
use p3_util::log2_strict_usize;

/// Selector values at a single evaluation point.
///
/// These selectors are used to conditionally enable constraints based on row position:
/// - `is_first_row`: Non-zero only for the first row of the trace
/// - `is_last_row`: Non-zero only for the last row of the trace
/// - `is_transition`: Non-zero for all rows except the last (enables transition constraints)
/// - `inv_vanishing`: Inverse of the vanishing polynomial at this point
#[derive(Clone, Copy, Debug)]
pub struct Selectors<EF> {
    pub is_first_row: EF,
    pub is_last_row: EF,
    pub is_transition: EF,
    pub inv_vanishing: EF,
}

/// Compute selectors at an evaluation point for a trace of given height.
///
/// Uses the standard two-adic formulas for Lagrange basis polynomials:
/// - `is_first_row = (x^n - 1) / (n * (x - 1))`
/// - `is_last_row = (x^n - 1) / (n * (x - ω^{-1}) * ω)`
/// - `is_transition = 1 - is_last_row`
///
/// # Arguments
/// - `x`: The evaluation point (typically the OOD point zeta)
/// - `n`: The trace height (must be a power of two)
pub fn selectors_at<F, EF>(x: EF, n: usize) -> Selectors<EF>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F>,
{
    let n_f = EF::from(F::from_usize(n));
    let h = F::two_adic_generator(log2_strict_usize(n));
    let h_inv = h.inverse();

    let x_n = x.exp_u64(n as u64);
    let vanishing = x_n - EF::ONE;
    let inv_vanishing = vanishing.inverse();

    // is_first_row = vanishing / (n * (x - 1))
    let is_first = vanishing * (n_f * (x - EF::ONE)).inverse();

    // is_last_row = vanishing / (n * (x - ω^{-1}) * ω)
    let denom_last = n_f * (x - EF::from(h_inv)) * EF::from(h);
    let is_last = vanishing * denom_last.inverse();

    // is_transition = 1 - is_last_row
    let is_transition = EF::ONE - is_last;

    Selectors {
        is_first_row: is_first,
        is_last_row: is_last,
        is_transition,
        inv_vanishing,
    }
}
