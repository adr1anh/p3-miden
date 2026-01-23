//! Polynomial Commitment Scheme combining DEEP quotient and FRI.
//!
//! This module provides high-level `open` and `verify` functions that orchestrate
//! the DEEP quotient construction and FRI protocol into a complete PCS.
//!
//! # Overview
//!
//! The PCS operates in two phases:
//!
//! 1. **Opening (Prover)**: Given committed matrices and evaluation points,
//!    computes polynomial evaluations, constructs a DEEP quotient, and generates
//!    a FRI proof of low-degree.
//!
//! 2. **Verification (Verifier)**: Given commitments, evaluation points, and a proof,
//!    verifies the DEEP quotient and FRI queries to confirm the claimed evaluations.

mod config;
mod proof;
pub mod prover;
pub mod verifier;

pub use config::PcsParams;
pub use proof::Proof;
pub use verifier::PcsError;

#[cfg(test)]
mod tests;
