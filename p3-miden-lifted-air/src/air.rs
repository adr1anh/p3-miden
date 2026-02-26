//! The `LiftedAir` super-trait for AIR definitions in the lifted STARK system.

use p3_air::BaseAir;
use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_ceil_usize;

use crate::{AirWithPeriodicColumns, LiftedAirBuilder, SymbolicAirBuilder};

/// Super-trait for AIR definitions used by the lifted STARK prover/verifier.
///
/// Inherits from upstream traits for width, public values, and periodic columns.
/// Adds Miden-specific auxiliary trace support.
///
/// # Type Parameters
/// - `F`: Base field
/// - `EF`: Extension field (for aux trace challenges)
pub trait LiftedAir<F: Field, EF>: Sync + BaseAir<F> + AirWithPeriodicColumns<F> {
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

    /// Log₂ of the number of quotient chunks, inferred from symbolic constraint analysis.
    ///
    /// Evaluates the AIR on a [`SymbolicAirBuilder`](crate::SymbolicAirBuilder) to determine
    /// the maximum constraint degree M, then returns `log2_ceil(M - 1)` (padded so M ≥ 2).
    ///
    /// Uses `SymbolicAirBuilder<F, F>` (base field only) which is sufficient for degree
    /// computation since extension-field operations have the same degree structure.
    fn log_quotient_degree(&self) -> usize
    where
        Self: Sized,
    {
        let preprocessed_width = self.preprocessed_trace().map_or(0, |t| t.width());
        let mut builder = SymbolicAirBuilder::<F>::new(
            preprocessed_width,
            self.width(),
            self.num_public_values(),
            self.aux_width(),
            self.num_randomness(),
            self.periodic_columns().len(),
        );
        self.eval(&mut builder);

        let base_degree = builder
            .base_constraints()
            .iter()
            .map(|c| c.degree_multiple())
            .max()
            .unwrap_or(0);
        let ext_degree = builder
            .extension_constraints()
            .iter()
            .map(|c| c.degree_multiple())
            .max()
            .unwrap_or(0);
        let constraint_degree = base_degree.max(ext_degree).max(2);

        log2_ceil_usize(constraint_degree - 1)
    }

    /// Number of quotient chunks: `2^log_quotient_degree()`.
    fn constraint_degree(&self) -> usize
    where
        Self: Sized,
    {
        1 << self.log_quotient_degree()
    }
}
