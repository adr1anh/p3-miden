//! # Lifted PCS
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
//! - **[`pcs`]**: Complete PCS implementation combining DEEP quotient and FRI into
//!   high-level `open` and `verify` functions.
//!
//! For the Lifted Matrix Commitment Scheme (LMCS), see the [`p3_miden_lmcs`] crate.

#![no_std]

extern crate alloc;

/// DEEP quotient construction for batched polynomial evaluation.
pub mod deep;

/// FRI protocol for low-degree testing.
pub mod fri;

/// Polynomial Commitment Scheme combining DEEP quotient and FRI.
pub mod pcs;

pub mod utils;

#[cfg(test)]
mod tests;
