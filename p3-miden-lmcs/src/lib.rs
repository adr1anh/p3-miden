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
//! - [`Proof`]: Multi-opening proof with opened rows and compact Merkle siblings.
//!
//! # API Overview
//!
//! ## Direct Usage (Simple)
//!
//! ```ignore
//! use p3_miden_lmcs::{LmcsConfig, HidingLmcsConfig, Lmcs, LmcsTree};
//!
//! // Config captures PF, PD (packed types), H, C, WIDTH, DIGEST
//! // F = PF::Value and D = PD::Value are derived
//! let config = LmcsConfig::<PF, PD, _, _, WIDTH, DIGEST>::new(sponge, compress);
//!
//! // Build tree - no turbofish needed, packed types are known from config
//! let tree = config.build_tree(matrices);
//! let root = tree.root();
//! let proof = tree.open_multi(&indices);
//! let rows = config.verify(&root, &dims, &indices, &proof)?;
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
//!
//! fn commit_and_open<L: Lmcs>(lmcs: &L, matrices: Vec<impl Matrix<L::F>>) {
//!     let tree = lmcs.build_tree(matrices);
//!     let commitment = tree.root();
//!     let proof = tree.open_multi(&[0, 1, 2]);
//!     // ...
//! }
//! ```
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

mod lifted_tree;
pub mod mmcs;
pub mod proof;
pub mod utils;

use alloc::vec::Vec;
use core::cell::RefCell;
use core::fmt::Debug;
use core::iter::zip;
use core::marker::PhantomData;

use p3_field::PackedValue;
use p3_matrix::{Dimensions, Matrix};
use p3_miden_stateful_hasher::StatefulHasher;
use p3_symmetric::{Hash, PseudoCompressionFunction};
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use thiserror::Error;

// ============================================================================
// Public Re-exports
// ============================================================================

pub use lifted_tree::LiftedMerkleTree;
pub use proof::{Opening, Proof};

// ============================================================================
// Traits
// ============================================================================

/// Trait for LMCS configurations.
///
/// Provides a unified interface for building trees and verifying proofs.
/// The packed types are captured at the type level, so method calls don't
/// need turbofish syntax.
///
/// This trait enables generic code to work with different LMCS configurations
/// without knowing the specific packed types.
pub trait Lmcs: Clone {
    /// Scalar field element type for matrix data.
    ///
    /// `Send + Sync` bounds required by [`Matrix<F>`].
    type F: Clone + Send + Sync;
    /// Commitment type (root hash).
    type Commitment: Clone;
    /// Multi-opening proof type.
    type Proof: Clone;
    /// Tree type (prover data), parameterized by matrix type.
    type Tree<M: Matrix<Self::F>>: LmcsTree<Self::F, Self::Commitment, Self::Proof, M>;

    /// Build a tree from matrices.
    ///
    /// The packed types are known from `self`, so no turbofish is needed.
    fn build_tree<M: Matrix<Self::F>>(&self, leaves: Vec<M>) -> Self::Tree<M>;

    /// Verify a multi-opening proof.
    ///
    /// Returns references to opened rows on success.
    fn verify<'a>(
        &self,
        commitment: &Self::Commitment,
        dimensions: &[Dimensions],
        indices: &[usize],
        proof: &'a Self::Proof,
    ) -> Result<Vec<Vec<&'a [Self::F]>>, LmcsError>;
}

/// Trait for built LMCS trees.
///
/// Provides methods for accessing tree data and generating proofs.
pub trait LmcsTree<F, Commitment, Proof, M> {
    /// Get the tree root (commitment).
    fn root(&self) -> Commitment;

    /// Get the tree height (number of leaves).
    fn height(&self) -> usize;

    /// Get references to the committed matrices.
    fn leaves(&self) -> &[M];

    /// Get the opened rows for a given leaf index.
    fn rows(&self, index: usize) -> Vec<Vec<F>>;

    /// Open multiple indices at once, returning a compact proof.
    fn open_multi(&self, indices: &[usize]) -> Proof;
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for Lifted Matrix Commitment Scheme (LMCS).
///
/// Holds the cryptographic primitives needed for committing to matrices
/// and verifying openings. This type captures packed types at the type level,
/// so method calls don't need turbofish syntax.
///
/// For hiding commitments with salt, use [`HidingLmcsConfig`] instead.
///
/// # Type Parameters
///
/// - `PF`: Packed field element type for SIMD operations. `PF::Value` is the scalar field.
/// - `PD`: Packed digest element type. `PD::Value` is the scalar digest element type.
/// - `H`: Stateful hasher/sponge type for hashing matrix rows.
/// - `C`: 2-to-1 compression function type for building tree nodes.
/// - `WIDTH`: State width for the hasher.
/// - `DIGEST`: Number of elements in a digest.
///
/// # Example
///
/// ```ignore
/// use p3_miden_lmcs::{LmcsConfig, Lmcs, LmcsTree};
///
/// // PF and PD are packed types; F = PF::Value, D = PD::Value
/// let config = LmcsConfig::<PF, PD, _, _, WIDTH, DIGEST>::new(sponge, compress);
///
/// // Build tree - no turbofish needed
/// let tree = config.build_tree(matrices);
/// let root = tree.root();
/// let proof = tree.open_multi(&indices);
/// ```
#[derive(Clone, Debug)]
pub struct LmcsConfig<PF, PD, H, C, const WIDTH: usize, const DIGEST: usize> {
    /// Stateful sponge for hashing matrix rows into leaf digests.
    pub sponge: H,
    /// 2-to-1 compression function for building internal tree nodes.
    pub compress: C,
    _phantom: PhantomData<(PF, PD)>,
}

impl<PF, PD, H, C, const WIDTH: usize, const DIGEST: usize>
    LmcsConfig<PF, PD, H, C, WIDTH, DIGEST>
{
    /// Create a new LMCS configuration.
    #[inline]
    pub const fn new(sponge: H, compress: C) -> Self {
        Self {
            sponge,
            compress,
            _phantom: PhantomData,
        }
    }
}

impl<PF, PD, H, C, const WIDTH: usize, const DIGEST: usize> Lmcs
    for LmcsConfig<PF, PD, H, C, WIDTH, DIGEST>
where
    PF: PackedValue + Default,
    PD: PackedValue + Default,
    H: StatefulHasher<PF::Value, [PD::Value; DIGEST], State = [PD::Value; WIDTH]>
        + StatefulHasher<PF, [PD; DIGEST], State = [PD; WIDTH]>
        + Sync,
    C: PseudoCompressionFunction<[PD::Value; DIGEST], 2>
        + PseudoCompressionFunction<[PD; DIGEST], 2>
        + Sync,
{
    type F = PF::Value;
    type Commitment = Hash<PF::Value, PD::Value, DIGEST>;
    type Proof = Proof<PF::Value, PD::Value, DIGEST, 0>;
    type Tree<M: Matrix<PF::Value>> = LiftedMerkleTree<PF::Value, PD::Value, M, DIGEST, 0>;

    fn build_tree<M: Matrix<Self::F>>(&self, leaves: Vec<M>) -> Self::Tree<M> {
        LiftedMerkleTree::build::<PF, PD, H, C, WIDTH>(&self.sponge, &self.compress, leaves, None)
    }

    fn verify<'a>(
        &self,
        commitment: &Self::Commitment,
        dimensions: &[Dimensions],
        indices: &[usize],
        proof: &'a Self::Proof,
    ) -> Result<Vec<Vec<&'a [Self::F]>>, LmcsError> {
        proof.verify::<H, C, WIDTH>(
            &self.sponge,
            &self.compress,
            commitment,
            dimensions,
            indices,
        )
    }
}

// ============================================================================
// Hiding Configuration
// ============================================================================

/// Configuration for hiding LMCS with random salt.
///
/// This type wraps a [`LmcsConfig`] and adds an RNG for generating salt
/// during tree construction. The RNG is stored in a `RefCell` to allow
/// salt generation without `&mut self` (required by `Mmcs::commit`).
///
/// # Type Parameters
///
/// - `PF`: Packed field element type for SIMD operations.
/// - `PD`: Packed digest element type.
/// - `H`: Stateful hasher/sponge type.
/// - `C`: 2-to-1 compression function type.
/// - `R`: Random number generator type.
/// - `WIDTH`: State width for the hasher.
/// - `DIGEST`: Number of elements in a digest.
/// - `SALT`: Number of salt elements per leaf (must be > 0).
///
/// # Example
///
/// ```ignore
/// use p3_miden_lmcs::{HidingLmcsConfig, Lmcs, LmcsTree};
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// let rng = StdRng::seed_from_u64(42);
/// let config = HidingLmcsConfig::<PF, PD, _, _, _, WIDTH, DIGEST, 4>::new(sponge, compress, rng);
///
/// let tree = config.build_tree(matrices);
/// let root = tree.root();
/// ```
#[derive(Clone, Debug)]
pub struct HidingLmcsConfig<
    PF,
    PD,
    H,
    C,
    R,
    const WIDTH: usize,
    const DIGEST: usize,
    const SALT: usize,
> {
    /// Inner non-hiding config with sponge and compression.
    pub inner: LmcsConfig<PF, PD, H, C, WIDTH, DIGEST>,
    /// RNG for salt generation. Uses `RefCell` for interior mutability.
    rng: RefCell<R>,
}

impl<PF, PD, H, C, R, const WIDTH: usize, const DIGEST: usize, const SALT: usize>
    HidingLmcsConfig<PF, PD, H, C, R, WIDTH, DIGEST, SALT>
{
    /// Create a new hiding LMCS configuration.
    ///
    /// # Compile-time Error
    ///
    /// Fails to compile if `SALT == 0`. Use [`LmcsConfig`] for non-hiding commitments.
    #[inline]
    pub fn new(sponge: H, compress: C, rng: R) -> Self {
        const { assert!(SALT > 0) }
        Self {
            inner: LmcsConfig::new(sponge, compress),
            rng: RefCell::new(rng),
        }
    }
}

impl<PF, PD, H, C, R, const WIDTH: usize, const DIGEST: usize, const SALT: usize> Lmcs
    for HidingLmcsConfig<PF, PD, H, C, R, WIDTH, DIGEST, SALT>
where
    PF: PackedValue + Default,
    PD: PackedValue + Default,
    R: Rng + Clone,
    StandardUniform: Distribution<PF::Value>,
    H: StatefulHasher<PF::Value, [PD::Value; DIGEST], State = [PD::Value; WIDTH]>
        + StatefulHasher<PF, [PD; DIGEST], State = [PD; WIDTH]>
        + Sync,
    C: PseudoCompressionFunction<[PD::Value; DIGEST], 2>
        + PseudoCompressionFunction<[PD; DIGEST], 2>
        + Sync,
{
    type F = PF::Value;
    type Commitment = Hash<PF::Value, PD::Value, DIGEST>;
    type Proof = Proof<PF::Value, PD::Value, DIGEST, SALT>;
    type Tree<M: Matrix<PF::Value>> = LiftedMerkleTree<PF::Value, PD::Value, M, DIGEST, SALT>;

    fn build_tree<M: Matrix<Self::F>>(&self, leaves: Vec<M>) -> Self::Tree<M> {
        use p3_matrix::dense::RowMajorMatrix;

        let tree_height = leaves.last().map(|m| m.height()).unwrap_or(0);
        let salt = RowMajorMatrix::rand(&mut *self.rng.borrow_mut(), tree_height, SALT);

        LiftedMerkleTree::build::<PF, PD, H, C, WIDTH>(
            &self.inner.sponge,
            &self.inner.compress,
            leaves,
            Some(salt),
        )
    }

    fn verify<'a>(
        &self,
        commitment: &Self::Commitment,
        dimensions: &[Dimensions],
        indices: &[usize],
        proof: &'a Self::Proof,
    ) -> Result<Vec<Vec<&'a [Self::F]>>, LmcsError> {
        proof.verify::<H, C, WIDTH>(
            &self.inner.sponge,
            &self.inner.compress,
            commitment,
            dimensions,
            indices,
        )
    }
}

impl<F, D, M, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    LmcsTree<F, Hash<F, D, DIGEST_ELEMS>, Proof<F, D, DIGEST_ELEMS, SALT_ELEMS>, M>
    for LiftedMerkleTree<F, D, M, DIGEST_ELEMS, SALT_ELEMS>
where
    F: Copy + Default + Send + Sync,
    D: Copy + PartialEq + Send + Sync,
    M: Matrix<F>,
{
    fn root(&self) -> Hash<F, D, DIGEST_ELEMS> {
        LiftedMerkleTree::root(self)
    }

    fn height(&self) -> usize {
        LiftedMerkleTree::height(self)
    }

    fn leaves(&self) -> &[M] {
        &self.leaves
    }

    fn rows(&self, index: usize) -> Vec<Vec<F>> {
        LiftedMerkleTree::rows(self, index)
    }

    fn open_multi(&self, indices: &[usize]) -> Proof<F, D, DIGEST_ELEMS, SALT_ELEMS> {
        LiftedMerkleTree::open_multi(self, indices)
    }
}

// ============================================================================
// Verification Helpers
// ============================================================================

/// Compute leaf digest from rows and salt.
///
/// # Arguments
///
/// - `sponge`: The stateful hasher for absorbing row data.
/// - `rows`: Opened rows (one per committed matrix).
/// - `widths`: Expected width for each row.
/// - `salt`: Salt elements (empty slice for non-hiding).
///
/// # Errors
///
/// Returns `Err` if any row width doesn't match the corresponding expected width.
pub(crate) fn compute_leaf_digest<F, D, H, const WIDTH: usize, const DIGEST_ELEMS: usize>(
    sponge: &H,
    rows: &[Vec<F>],
    widths: impl IntoIterator<Item = usize>,
    salt: &[F],
) -> Result<[D; DIGEST_ELEMS], LmcsError>
where
    F: Default + Copy,
    D: Default + Copy,
    H: StatefulHasher<F, [D; DIGEST_ELEMS], State = [D; WIDTH]>,
{
    let mut state = [D::default(); WIDTH];
    for (idx, (row, width)) in zip(rows, widths).enumerate() {
        if row.len() != width {
            return Err(LmcsError::WrongWidth { matrix: idx });
        }
        sponge.absorb_into(&mut state, row.iter().copied());
    }

    // Absorb salt
    if !salt.is_empty() {
        sponge.absorb_into(&mut state, salt.iter().copied());
    }

    Ok(sponge.squeeze(&state))
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during LMCS operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LmcsError {
    /// Number of opened rows doesn't match number of committed matrices.
    #[error("wrong batch size")]
    WrongBatchSize,
    /// Opened row width doesn't match the committed matrix width.
    #[error("wrong width at matrix {matrix}")]
    WrongWidth { matrix: usize },
    /// Salt row length doesn't match expected width.
    #[error("wrong salt width")]
    WrongSalt,
    /// Authentication path length doesn't match tree height.
    #[error("wrong proof length")]
    WrongProofLen,
    /// Query index exceeds tree height.
    #[error("index out of bounds")]
    IndexOutOfBounds,
    /// Recomputed root doesn't match the commitment.
    #[error("root mismatch")]
    RootMismatch,
    /// Proof structure doesn't match expected siblings for given leaves.
    #[error("invalid proof")]
    InvalidProof,
    /// Same leaf index provided with different hashes.
    #[error("conflicting leaf")]
    ConflictingLeaf,
}

#[cfg(test)]
mod tests;
