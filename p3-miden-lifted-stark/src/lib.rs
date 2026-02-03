//! Lifted STARK shared scaffolding (LMCS-based).
//!
//! This crate contains shared types and utilities used by the lifted STARK
//! prover and verifier crates:
//!
//! - [`StarkConfig`]: Minimal configuration wrapping PCS params, LMCS, and DFT
//! - [`commit_traces`]: Helper for committing traces (LDE → bit-reverse → LMCS)
//! - [`Selectors`] and [`selectors_at`]: Constraint selectors for OOD evaluation
//! - [`ConstraintFolder`]: Constraint evaluation machinery for MidenAir
//! - [`Proof`]: Proof container type

#![no_std]

extern crate alloc;

mod commit;
mod config;
mod constraints;
mod folder;
mod proof;
mod utils;

pub use commit::*;
pub use config::*;
pub use constraints::*;
pub use folder::*;
pub use proof::*;
pub use utils::*;
