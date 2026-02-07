//! Lifted STARK verifier scaffold.
//!
//! This crate provides the verifier for the lifted STARK protocol using LMCS
//! commitments and the lifted FRI PCS.

#![no_std]

extern crate alloc;

mod constraints;
mod periodic;
mod proof;
mod verifier;

pub use p3_miden_lifted_stark::StarkConfig;
pub use proof::Proof;
pub use verifier::{AirWithLogHeight, VerifierError, verify_multi, verify_single};
