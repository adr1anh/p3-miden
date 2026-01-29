//! Selector polynomials for two-adic domains.

use p3_field::{ExtensionField, PrimeCharacteristicRing, TwoAdicField};
use p3_util::log2_strict_usize;
#[derive(Clone, Debug)]
pub struct Selectors<EF> {
    pub is_first_row: EF,
    pub is_last_row: EF,
    pub is_transition: EF,
    pub inv_vanishing: EF,
}

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

    let is_first = vanishing * (n_f * (x - EF::ONE)).inverse();
    let denom_last = n_f * (x - EF::from(h_inv)) * EF::from(h);
    let is_last = vanishing * denom_last.inverse();
    let is_transition = EF::ONE - is_last;

    Selectors {
        is_first_row: is_first,
        is_last_row: is_last,
        is_transition,
        inv_vanishing,
    }
}
