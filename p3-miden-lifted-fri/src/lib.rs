//! # Lifted FRI PCS
//!
//! A polynomial commitment scheme (PCS) combining DEEP quotient construction with FRI
//! for efficient low-degree testing over two-adic fields.
//!
//! ## Overview
//!
//! This crate provides:
//!
//! - **[`deep`]**: DEEP (Dimension Extension of Evaluation Protocol) quotient construction
//!   for batching polynomial evaluation claims into a single low-degree polynomial.
//!
//! - **[`fri`]**: FRI (Fast Reed-Solomon IOP) protocol for low-degree testing, with
//!   configurable folding arities and final polynomial degree.
//!
//! - **PCS API (crate root)**: complete PCS implementation combining DEEP quotient and
//!   FRI via `prover::open_with_channel` and `verifier::verify_with_channel`, plus `PcsParams`.
//!
//! For the Lifted Matrix Commitment Scheme (LMCS), see the [`p3_miden_lmcs`] crate.

#![no_std]

extern crate alloc;

/// DEEP quotient construction for batched polynomial evaluation.
pub mod deep;

/// FRI protocol for low-degree testing.
pub mod fri;

/// LDE + LMCS helpers for lifted commitments.
pub mod lde;

mod params;
mod proof;
pub mod prover;
pub mod verifier;

pub mod utils;

pub use params::PcsParams;
pub use proof::PcsTranscript;
pub use verifier::PcsError;

#[cfg(test)]
mod tests;
