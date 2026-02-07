//! Lifted STARK prover scaffold.
//!
//! This crate provides the prover for the lifted STARK protocol using LMCS
//! commitments and the lifted FRI PCS.

#![no_std]

extern crate alloc;

mod commit;
mod constraints;
mod periodic;
mod prover;

// Re-exports from dependencies
pub use p3_miden_lifted_stark::StarkConfig;
pub use p3_miden_lifted_verifier::Proof;

// Public API
pub use prover::{AirWithTrace, ProverError, prove_multi, prove_single};
