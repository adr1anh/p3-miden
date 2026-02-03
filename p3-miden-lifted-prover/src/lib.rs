//! Lifted STARK prover scaffold.
//!
//! This crate provides the prover for the lifted STARK protocol using LMCS
//! commitments and the lifted FRI PCS.

#![no_std]

extern crate alloc;

mod periodic;
mod prover;

pub use p3_miden_lifted_stark::{Proof, StarkConfig};
pub use periodic::*;
pub use prover::*;
