//! Lifted STARK shared scaffolding (LMCS-based).
//!
//! This crate contains shared types and utilities used by the lifted STARK
//! prover and verifier crates:
//!
//! - [`LiftedCoset`]: Central abstraction for domain operations (selectors, vanishing, etc.)
//! - [`StarkConfig`]: Minimal configuration wrapping PCS params, LMCS, and DFT
//! - [`Selectors`]: Constraint selectors for OOD and coset evaluation
//! - [`Proof`]: Proof container type

#![no_std]

extern crate alloc;

mod config;
mod coset;
mod proof;
mod selectors;
mod utils;

pub use config::*;
pub use coset::*;
pub use proof::*;
pub use selectors::*;
pub use utils::*;
