//! The `LiftedAirBuilder` super-trait for constraint evaluation builders.

use p3_air::{AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PermutationAirBuilder};

use crate::PeriodicAirBuilder;

/// Super-trait bundling all builder capabilities needed by the lifted STARK system.
///
/// Instead of re-declaring methods from upstream traits, this inherits from all of them.
/// The only Miden-specific addition is [`aux_values`], which provides auxiliary boundary
/// values for the lifting protocol.
pub trait LiftedAirBuilder:
    AirBuilder
    + AirBuilderWithPublicValues
    + ExtensionBuilder
    + PermutationAirBuilder
    + PeriodicAirBuilder
{
    /// Auxiliary boundary values (extension field) carried in the proof.
    fn aux_values(&self) -> &[Self::VarEF] {
        &[]
    }
}
