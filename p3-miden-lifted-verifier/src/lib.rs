//! Lifted STARK verifier scaffold.
//!
//! This crate wires the end-to-end verification flow using the shared lifted
//! STARK helpers in `p3-miden-lifted-stark` and the LMCS-based lifted FRI PCS.

#![no_std]
#![allow(dead_code, unused_imports)]

extern crate alloc;

mod notes;
mod verifier;

pub use p3_miden_lifted_stark::{LiftedStarkConfig, Proof};
pub use verifier::*;
