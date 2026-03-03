//! AIR traits for the Miden lifted STARK protocol.
//!
//! This crate provides:
//! - [`LiftedAir`]: Super-trait for AIR definitions (inherits upstream + adds aux trace support)
//! - [`LiftedAirBuilder`]: Super-trait for constraint builders (blanket impl over upstream traits)
//! - [`PeriodicAirBuilder`] / [`AirWithPeriodicColumns`]: Periodic column support (copied from
//!   upstream, not yet in published p3-air 0.4.2)
//! - [`symbolic`]: Symbolic constraint analysis (expression trees, degree computation)

#![no_std]

extern crate alloc;

mod air;
mod builder;
mod periodic;
pub mod symbolic;

pub use air::LiftedAir;
pub use builder::LiftedAirBuilder;
pub use periodic::{AirWithPeriodicColumns, PeriodicAirBuilder};

pub use p3_air::{
    Air, AirBuilder, BaseAir, ExtensionBuilder, FilteredAirBuilder, PermutationAirBuilder,
};

// Re-export key symbolic types at crate root for ergonomic imports.
pub use symbolic::{
    ConstraintLayout, Entry, SymbolicAirBuilder, SymbolicExpression, SymbolicVariable,
    get_constraint_layout,
};

// Re-export commonly used field/matrix types.
pub use p3_field::{ExtensionField, Field};
pub use p3_matrix::dense::RowMajorMatrix;
