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
//! - **[`merkle_tree`]**: Lifted Merkle tree commitments supporting matrices of varying
//!   heights via upsampling lifting strategies.
//!
//! - **[`pcs`]**: Complete PCS implementation combining DEEP quotient and FRI into
//!   high-level `open` and `verify` functions.

#![no_std]

extern crate alloc;

/// DEEP quotient construction for batched polynomial evaluation.
pub mod deep;

/// FRI protocol for low-degree testing.
pub mod fri;

/// Lifted Merkle tree commitments for matrices of varying heights.
pub mod merkle_tree;

/// Polynomial Commitment Scheme combining DEEP quotient and FRI.
pub mod pcs;

pub(crate) mod utils;

#[cfg(test)]
mod tests;
