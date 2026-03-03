//! The `LiftedAirBuilder` super-trait for constraint evaluation builders.

use crate::{AirBuilder, ExtensionBuilder, PeriodicAirBuilder, PermutationAirBuilder};

/// Super-trait bundling all builder capabilities needed by the lifted STARK system.
///
/// Instead of re-declaring methods from upstream traits, this inherits from all of them.
/// The only Miden-specific addition is [`LiftedAirBuilder::aux_values`], which provides auxiliary boundary
/// values for the lifting protocol.
pub trait LiftedAirBuilder:
    AirBuilder + ExtensionBuilder + PermutationAirBuilder + PeriodicAirBuilder
{
    /// Auxiliary boundary values (extension field) carried in the proof.
    fn aux_values(&self) -> &[Self::VarEF] {
        &[]
    }
}
