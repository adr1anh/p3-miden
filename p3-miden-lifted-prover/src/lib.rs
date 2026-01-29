//! Lifted STARK prover scaffold.
//!
//! This crate wires the end-to-end proving flow using the shared lifted STARK
//! helpers in `p3-miden-lifted-stark` and the LMCS-based lifted FRI PCS.

#![no_std]
#![allow(dead_code, unused_imports)]

extern crate alloc;

mod notes;
mod prover;

pub use p3_miden_lifted_stark::{LiftedStarkConfig, Proof};
pub use prover::*;
