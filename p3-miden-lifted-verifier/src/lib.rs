//! Lifted STARK verifier scaffold.
//!
//! This crate provides the verifier for the lifted STARK protocol using LMCS
//! commitments and the lifted FRI PCS.

#![no_std]

extern crate alloc;

mod periodic;
mod verifier;

pub use p3_miden_lifted_stark::{Proof, StarkConfig};
pub use periodic::*;
pub use verifier::*;
