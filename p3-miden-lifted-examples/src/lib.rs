//! Example AIRs wrapped for the lifted STARK prover.
//!
//! Each module adapts an upstream Plonky3 AIR into a `LiftedAir` so it can be proven
//! and verified with the lifted STARK protocol.

#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod blake3;
pub mod keccak;
pub mod miden;
pub mod poseidon2;

#[cfg(feature = "std")]
pub mod stats;
