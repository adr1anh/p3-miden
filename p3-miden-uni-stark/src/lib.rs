//! A minimal univariate STARK framework.

#![no_std]

extern crate alloc;

use p3_field::{Algebra, Field, PrimeCharacteristicRing};

mod check_constraints;
mod config;
mod folder;
mod preprocessed;
mod proof;
mod prover;
mod sub_builder;
mod symbolic_builder;
mod symbolic_expression;
mod symbolic_variable;
mod verifier;

pub use check_constraints::*;
pub use config::*;
pub use folder::*;
pub use preprocessed::*;
pub use proof::*;
pub use prover::*;
pub use sub_builder::*;
pub use symbolic_builder::*;
pub use symbolic_expression::*;
pub use symbolic_variable::*;
pub use verifier::*;

/// Helper for packed EF linear combinations with scalar EF coefficients.
///
/// This mirrors `PackedField::packed_linear_combination` (base-field case),
/// but for extension-field scalars. It avoids packing EF coefficients into `PackedChallenge`
/// vectors just to compute dot products.
pub trait PackedChallengeLinearCombination<ExtField: Field>:
    PrimeCharacteristicRing + Algebra<ExtField> + Copy
{
    fn packed_linear_combination<const N: usize>(coeffs: &[ExtField], vecs: &[Self]) -> Self {
        assert_eq!(coeffs.len(), N);
        assert_eq!(vecs.len(), N);
        let combined: [Self; N] = core::array::from_fn(|i| vecs[i] * coeffs[i]);
        Self::sum_array::<N>(&combined)
    }
}

impl<ExtField: Field, T> PackedChallengeLinearCombination<ExtField> for T where
    T: PrimeCharacteristicRing + Algebra<ExtField> + Copy
{
}
