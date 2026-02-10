#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

mod commit;
mod constraints;
mod periodic;
mod prover;

// Re-exports from dependencies
pub use p3_miden_lifted_stark::{AirWitness, StarkConfig};
pub use p3_miden_lifted_verifier::Proof;

// Public API
pub use prover::{ProverError, prove_multi, prove_single};
