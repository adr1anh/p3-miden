//! A framework for symmetric cryptography primitives.
//!
//! This crate extends `p3-symmetric` with `StatefulHasher` support for
//! incremental/streaming hash operations needed by lifted Merkle trees.

#![no_std]

extern crate alloc;

mod compression;
mod hash;
mod hasher;
mod serializing_hasher;
mod sponge;
mod stateful;

#[cfg(test)]
mod testing;

pub use compression::*;
pub use hash::*;
pub use hasher::*;
pub use serializing_hasher::*;
pub use sponge::*;
pub use stateful::*;

// Re-export upstream permutation traits for convenience
pub use p3_symmetric::{CryptographicPermutation, Permutation};
