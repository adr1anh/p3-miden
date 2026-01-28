//! Lifted Matrix Commitment Scheme (LMCS) for matrices with power-of-two heights.
//!
//! This crate provides a Merkle tree commitment scheme optimized for matrices that store
//! polynomial evaluations in **bit-reversed order** over multiplicative cosets.
//!
//! # Main Types
//!
//! - [`LmcsConfig`]: Configuration holding cryptographic primitives (sponge + compression)
//!   with packed types for SIMD parallelization.
//! - [`Lmcs`]: Trait for LMCS configurations, providing type-erased access to commitment operations.
//! - [`LmcsTree`]: Trait for built LMCS trees, providing opening operations.
//! - [`LiftedMerkleTree`]: The underlying Merkle tree data structure.
//! - [`Proof`]: Single-opening proof with rows, optional salt, and authentication path.
//!
//! # API Overview
//!
//! ## Direct Usage (Simple)
//!
//! ```ignore
//! use p3_miden_lmcs::{HidingLmcsConfig, Lmcs, LmcsConfig, LmcsTree};
//! use p3_miden_transcript::{ProverTranscript, VerifierTranscript};
//!
//! // Config captures PF, PD (packed types), H, C, WIDTH, DIGEST
//! // F = PF::Value and D = PD::Value are derived
//! let config = LmcsConfig::<PF, PD, _, _, WIDTH, DIGEST>::new(sponge, compress);
//! let challenger = /* ... */;
//!
//! // Build tree - no turbofish needed, packed types are known from config
//! let tree = config.build_tree(matrices);
//! let root = tree.root();
//! let mut prover_channel = ProverTranscript::new(challenger);
//! tree.prove_batch(&indices, &mut prover_channel);
//! let transcript = prover_channel.into_data();
//!
//! let mut verifier_channel = VerifierTranscript::from_data(challenger, &transcript);
//! let rows = config.open_batch(&root, &widths, log_max_height, &indices, &mut verifier_channel)?;
//!
//! // For hiding commitment with salt, use HidingLmcsConfig with RNG
//! let hiding_config = HidingLmcsConfig::<PF, PD, _, _, _, WIDTH, DIGEST, 4>::new(sponge, compress, rng);
//! let tree = hiding_config.build_tree(matrices);
//! ```
//!
//! ## Trait-Based Usage (for generic code like FRI)
//!
//! ```ignore
//! use p3_miden_lmcs::{Lmcs, LmcsTree};
//! use p3_miden_transcript::ProverTranscript;
//!
//! fn commit_and_open<L: Lmcs>(lmcs: &L, matrices: Vec<impl Matrix<L::F>>) {
//!     let tree = lmcs.build_tree(matrices);
//!     let commitment = tree.root();
//!     let challenger = /* ... */;
//!     let mut channel = ProverTranscript::new(challenger);
//!     tree.prove_batch(&[0, 1, 2], &mut channel);
//!     // ...
//! }
//! ```
//!
//! ## Transcript Hints
//!
//! Batch openings are streamed as transcript hints: `prove_batch` writes rows/salt and
//! sibling hashes without observing them into the Fiat-Shamir challenger, and
//! `open_batch` reads and verifies those hints against a commitment. Use
//! `read_batch_from_channel` if you need per-index [`Proof`] objects for inspection.
//!
//! # Mathematical Foundation
//!
//! Consider a polynomial `f(X)` of degree less than `d`, and let `g` be the coset generator and
//! `K` a subgroup of order `n ≥ d` with primitive root `ω`. The coset evaluations
//! `{f(g·ω^j) : j ∈ [0, n)}` can be stored in two orderings:
//!
//! - **Canonical order**: `[f(g·ω^0), f(g·ω^1), ..., f(g·ω^{n-1})]`
//! - **Bit-reversed order**: `[f(g·ω^{bitrev(0)}), f(g·ω^{bitrev(1)}), ..., f(g·ω^{bitrev(n-1)})]`
//!
//! where `bitrev(i)` is the bit-reversal of index `i` within `log2(n)` bits.
//!
//! # Lifting by Upsampling
//!
//! When we have matrices with different heights `n_0 ≤ n_1 ≤ ... ≤ n_{t-1}` (each a power of two),
//! we "lift" smaller matrices to the maximum height `N = n_{t-1}` using **nearest-neighbor
//! upsampling**: each row is repeated contiguously `r = N/n` times.
//!
//! For a matrix of height `n` lifted to `N`, the index map is: `i ↦ floor(i / r) = i >> log2(r)`
//!
//! **Example** (`n=4`, `N=8`):
//! - Original rows: `[row0, row1, row2, row3]`
//! - Upsampled: `[row0, row0, row1, row1, row2, row2, row3, row3]` (blocks of 2)
//!
//! # Why Upsampling for Bit-Reversed Data
//!
//! Given bit-reversed evaluations of `f(X)` over a coset `gK` where `|K| = n`, upsampling to
//! height `N = n · r` (where `r = 2^k`) produces the bit-reversed evaluations of `f'(X) = f(X^r)`
//! over the coset `gK'` where `|K'| = N`.
//!
//! Mathematically, if the input contains `f(g·(ω_n)^{bitrev_n(j)})` at index `j`, then after
//! upsampling, each index `i` in `[0, N)` maps to the original index `j = i >> k`, giving:
//!
//! ```text
//! upsampled[i] = f(g·(ω_n)^{bitrev_n(i >> k)}) = f'(g·(ω_N)^{bitrev_N(i)})
//! ```
//!
//! where `f'(X) = f(X^r)`. This is exactly the bit-reversed evaluation of `f'` over `gK'`.
//!
//! # Opening Semantics
//!
//! When opening at index `i`, we retrieve the value at position `i` in the bit-reversed list.
//! For the lifted polynomial `f'(X) = f(X^r)`, this gives `f'(g·(ω_N)^{bitrev_N(i)})`.
//!
//! Equivalently, this is `f'(g·ξ^i)` where `ξ = (ω_N)^{bitrev_N(i)}` is the `i`-th element
//! when iterating over `K'` in the order induced by bit-reversed indices.
//!
//! # Equivalence to Cyclic Lifting
//!
//! Upsampling bit-reversed data is equivalent to cyclically repeating canonically-ordered data:
//!
//! ```text
//! Upsample(BitReverse(data)) = BitReverse(Cyclic(data))
//! ```
//!
//! where cyclic repetition tiles the original `n` rows periodically: `[row0, row1, ..., row_{n-1}, row0, ...]`.
//!
//! This equivalence follows from the bit-reversal identity: for `r = N/n = 2^k`,
//! `bitrev_N(i) mod n = bitrev_n(i >> k)`.

#![no_std]

extern crate alloc;

mod hiding_lmcs;
mod lifted_tree;
mod lmcs;
pub mod mmcs;
pub mod proof;
#[cfg(test)]
mod tests;
pub mod utils;

use alloc::vec::Vec;

use p3_matrix::Matrix;
use p3_miden_transcript::{ProverChannel, VerifierChannel};
use thiserror::Error;

// ============================================================================
// Public Re-exports
// ============================================================================

pub use hiding_lmcs::HidingLmcsConfig;
pub use lifted_tree::LiftedMerkleTree;
pub use lmcs::LmcsConfig;
pub use proof::Proof;

// ============================================================================
// Traits
// ============================================================================

/// Trait for LMCS configurations.
pub trait Lmcs: Clone {
    /// Scalar field element type for matrix data.
    ///
    /// `Send + Sync` bounds required by [`Matrix<F>`].
    type F: Clone + Send + Sync;
    /// Commitment type (root hash).
    type Commitment: Clone;
    /// Single-opening proof type.
    type SingleProof;
    /// Tree type (prover data), parameterized by matrix type.
    type Tree<M: Matrix<Self::F>>: LmcsTree<Self::F, Self::Commitment, M>;

    /// Build a tree from matrices.
    ///
    /// The packed types are known from `self`, so no turbofish is needed.
    fn build_tree<M: Matrix<Self::F>>(&self, leaves: Vec<M>) -> Self::Tree<M>;

    /// Open a batch proof by reading it from a transcript channel.
    fn open_batch<Ch>(
        &self,
        commitment: &Self::Commitment,
        widths: &[usize],
        log_max_height: usize,
        indices: &[usize],
        channel: &mut Ch,
    ) -> Result<Vec<Vec<Vec<Self::F>>>, LmcsError>
    where
        Ch: VerifierChannel<F = Self::F, Commitment = Self::Commitment>;

    /// Read a batch opening from a transcript channel and reconstruct per-index proofs.
    ///
    /// This only parses hints; it does not verify against a commitment.
    fn read_batch_from_channel<Ch>(
        &self,
        widths: &[usize],
        log_max_height: usize,
        indices: &[usize],
        channel: &mut Ch,
    ) -> Result<Vec<Self::SingleProof>, LmcsError>
    where
        Ch: VerifierChannel<F = Self::F, Commitment = Self::Commitment>;
}

/// Trait for built LMCS trees.
///
/// Provides methods for accessing tree data and generating proofs.
pub trait LmcsTree<F, Commitment, M> {
    /// Get the tree root (commitment).
    fn root(&self) -> Commitment;

    /// Get the tree height (number of leaves).
    fn height(&self) -> usize;

    /// Get references to the committed matrices.
    fn leaves(&self) -> &[M];

    /// Get the opened rows for a given leaf index.
    fn rows(&self, index: usize) -> Vec<Vec<F>>;

    /// Prove a batch opening and stream it into a transcript channel.
    fn prove_batch<Ch>(&self, indices: &[usize], channel: &mut Ch)
    where
        Ch: ProverChannel<F = F, Commitment = Commitment>;
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during LMCS operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LmcsError {
    #[error("invalid proof")]
    InvalidProof,
    #[error("root mismatch")]
    RootMismatch,
}
