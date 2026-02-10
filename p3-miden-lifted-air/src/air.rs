//! The `LiftedAir` super-trait for AIR definitions in the lifted STARK system.

use p3_air::{BaseAir, BaseAirWithPublicValues};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use crate::{AirWithPeriodicColumns, LiftedAirBuilder};

/// Super-trait for AIR definitions used by the lifted STARK prover/verifier.
///
/// Inherits from upstream traits for width, public values, and periodic columns.
/// Adds Miden-specific auxiliary trace support.
///
/// # Type Parameters
/// - `F`: Base field
/// - `EF`: Extension field (for aux trace challenges)
pub trait LiftedAir<F: Field, EF>:
    Sync + BaseAir<F> + BaseAirWithPublicValues<F> + AirWithPeriodicColumns<F>
{
    /// Number of extension-field challenges required for the auxiliary trace.
    fn num_randomness(&self) -> usize {
        0
    }

    /// Number of extension-field columns in the auxiliary trace.
    fn aux_width(&self) -> usize {
        0
    }

    /// Build the auxiliary trace given the main trace and EF challenges.
    /// Returns `None` if no auxiliary trace is needed.
    fn build_aux_trace(
        &self,
        _main: &RowMajorMatrix<F>,
        _challenges: &[EF],
    ) -> Option<RowMajorMatrix<EF>> {
        None
    }

    /// Evaluate all AIR constraints using the provided builder.
    fn eval<AB: LiftedAirBuilder<F = F>>(&self, builder: &mut AB);
}
