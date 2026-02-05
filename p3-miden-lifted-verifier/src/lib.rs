//! Lifted STARK verifier scaffold.
//!
//! This crate provides the verifier for the lifted STARK protocol using LMCS
//! commitments and the lifted FRI PCS.

#![no_std]

extern crate alloc;

mod constraints;
mod periodic;
mod verifier;

pub use constraints::{
    CONSTRAINT_DEGREE, ConstraintFolder, LOG_CONSTRAINT_DEGREE, align_width,
    extract_quotient_chunks, reconstruct_quotient, row_to_packed_ext,
};
pub use p3_miden_lifted_stark::{Proof, StarkConfig};
pub use periodic::*;
pub use verifier::*;
