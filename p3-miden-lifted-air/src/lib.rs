//! AIR traits for the Miden lifted STARK protocol.
//!
//! This crate provides:
//! - [`LiftedAir`]: Super-trait for AIR definitions (inherits upstream + adds aux trace support)
//! - [`LiftedAirBuilder`]: Super-trait for constraint builders
//! - [`AirWithPeriodicColumns`]: Periodic column data trait (column periods and evaluations)
//! - [`auxiliary`]: Auxiliary trace types (builder, cross-AIR identity checking).

#![no_std]

extern crate alloc;

mod air;
pub mod auxiliary;
mod builder;
mod instance;
mod periodic;

pub use air::{AirValidationError, BuilderMismatchError, LiftedAir};
pub use auxiliary::{AuxBuilder, ReducedAuxValues, ReductionError, VarLenPublicInputs};
pub use builder::LiftedAirBuilder;
pub use instance::{AirInstance, AirWitness, validate_instances};
pub use periodic::AirWithPeriodicColumns;

pub use p3_air::{
    Air, AirBuilder, AirBuilderWithContext, BaseAir, ExtensionBuilder, FilteredAirBuilder,
    PeriodicAirBuilder, PermutationAirBuilder, RowWindow, WindowAccess,
};

pub use p3_air::symbolic::{
    AirLayout, BaseEntry, ConstraintLayout, ExtEntry, SymbolicAirBuilder, SymbolicExpression,
    SymbolicExpressionExt, SymbolicVariable, SymbolicVariableExt, get_all_symbolic_constraints,
    get_constraint_layout, get_max_constraint_degree, get_max_constraint_degree_extension,
    get_symbolic_constraints, get_symbolic_constraints_extension,
};

// Re-export commonly used field/matrix types.
pub use p3_field::{ExtensionField, Field};
pub use p3_matrix::dense::RowMajorMatrix;
