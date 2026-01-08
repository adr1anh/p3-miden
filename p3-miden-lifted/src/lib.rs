//! # Lifted PCS
//!
//! A polynomial commitment scheme (PCS) combining DEEP quotient construction with FRI
//! for efficient low-degree testing over two-adic fields.
//!
//! ## Overview
//!
//! This crate provides:
//!
//! - **[`merkle_tree`]**: Lifted Merkle tree commitments supporting matrices of varying
//!   heights via upsampling lifting strategies.

#![no_std]

extern crate alloc;

/// Lifted Merkle tree commitments for matrices of varying heights.
pub mod merkle_tree;

pub mod utils;

#[cfg(test)]
mod tests;
