//! Shared utilities for the lifted STARK prover and verifier.

use p3_field::{ExtensionField, TwoAdicField};
use p3_miden_transcript::Channel;

use crate::LiftedCoset;

/// Sample an OOD evaluation point from the channel that lies outside both the
/// trace-domain subgroup `H` and the LDE evaluation coset `gK`.
///
/// Repeatedly draws `sample_algebra_element` candidates until one satisfies
/// both exclusion tests. This terminates with overwhelming probability because
/// `|H ∪ gK|` is negligible relative to the extension field size.
pub fn sample_ood_point<F, EF>(channel: &mut impl Channel<F = F>, coset: &LiftedCoset) -> EF
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    loop {
        let candidate: EF = channel.sample_algebra_element();
        if !coset.is_in_trace_domain::<F, _>(candidate) && !coset.is_in_lde_coset::<F, _>(candidate)
        {
            break candidate;
        }
    }
}
