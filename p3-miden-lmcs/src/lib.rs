//! Lifted Matrix Commitment Scheme (LMCS) for matrices with power-of-two heights.
//!
//! This crate provides a Merkle tree commitment scheme optimized for matrices that store
//! polynomial evaluations in **bit-reversed order** over multiplicative cosets.
//!
//! # Main Types
//!
//! - [`LmcsConfig`]: Simple configuration holding cryptographic primitives (sponge + compression)
//! - [`LmcsMmcs`]: MMCS-compatible wrapper with full type parameters (for FRI integration)
//! - [`LiftedMerkleTree`]: The underlying Merkle tree data structure
//! - [`Proof`]: Multi-opening proof with opened rows and compact Merkle siblings
//!
//! # API Overview
//!
//! ## Direct Usage (Simple)
//!
//! ```ignore
//! use p3_miden_lmcs::LmcsConfig;
//!
//! // Config captures F, D, H, C, WIDTH, DIGEST, SALT (default 0 for non-hiding)
//! let config = LmcsConfig::<F, D, _, _, WIDTH, DIGEST>::new(sponge, compress);
//!
//! // Build tree - only packed types needed in turbofish
//! let tree = config.build_tree::<PF, PD, _>(matrices);
//! let root = tree.root();
//! let proof = tree.open_multi(&indices);
//! let rows = config.verify(&root, &dims, &indices, &proof)?;
//!
//! // For hiding commitment with salt, specify SALT in the config type
//! let hiding_config = LmcsConfig::<F, D, _, _, WIDTH, DIGEST, 4>::new(sponge, compress);
//! let tree = hiding_config.build_tree_hiding::<PF, PD, _>(matrices, &mut rng);
//! ```
//!
//! ## MMCS Integration (for FRI)
//!
//! ```ignore
//! use p3_miden_lmcs::LmcsMmcs;
//! use p3_commit::Mmcs;
//!
//! let mmcs = LmcsMmcs::<P, P, Sponge, Compress, WIDTH, DIGEST>::new(sponge, compress);
//! let (commitment, prover_data) = mmcs.commit(matrices);
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
mod mmcs;
pub mod proof;
pub mod utils;

use alloc::vec::Vec;
use core::iter::zip;
use core::marker::PhantomData;

use p3_field::PackedValue;
use p3_matrix::{Dimensions, Matrix};
use p3_miden_stateful_hasher::StatefulHasher;
use p3_symmetric::{Hash, PseudoCompressionFunction};
use p3_util::log2_strict_usize;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use thiserror::Error;

// ============================================================================
// Public Re-exports
// ============================================================================

pub use lifted_tree::{LiftedHidingMerkleTree, LiftedMerkleTree};
pub use mmcs::{HidingLmcsMmcs, LmcsMmcs};
pub use proof::{CompactProof, CompactProofError, IndexedPath, NodeIndex, Opening, Proof};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for Lifted Matrix Commitment Scheme (LMCS).
///
/// Holds the cryptographic primitives needed for committing to matrices
/// and verifying openings. This type is typically constructed once and
/// reused for multiple commit/verify operations.
///
/// # Type Parameters
///
/// - `F`: Field element type (scalar) for matrix data.
/// - `D`: Digest element type (scalar) for hash outputs.
/// - `H`: Stateful hasher/sponge type for hashing matrix rows.
/// - `C`: 2-to-1 compression function type for building tree nodes.
/// - `WIDTH`: State width for the hasher.
/// - `DIGEST`: Number of elements in a digest.
///
/// # Example
///
/// ```ignore
/// use p3_miden_lmcs::LmcsConfig;
///
/// let config = LmcsConfig::<F, D, _, _, WIDTH, DIGEST>::new(sponge, compress);
///
/// // Build tree using config builder (only packed types needed)
/// let tree = config.build_tree::<PF, PD, _>(matrices);
/// let root = tree.root();
/// let proof = tree.open_multi(&indices);
/// ```
///
/// For MMCS trait integration, use [`LmcsMmcs`](crate::LmcsMmcs) which wraps
/// this config with the additional type parameters required by the trait.
#[derive(Copy, Clone, Debug)]
pub struct LmcsConfig<F, D, H, C, const WIDTH: usize, const DIGEST: usize, const SALT: usize = 0> {
    /// Stateful sponge for hashing matrix rows into leaf digests.
    pub sponge: H,
    /// 2-to-1 compression function for building internal tree nodes.
    pub compress: C,
    _phantom: PhantomData<(F, D)>,
}

impl<F, D, H, C, const WIDTH: usize, const DIGEST: usize, const SALT: usize>
    LmcsConfig<F, D, H, C, WIDTH, DIGEST, SALT>
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

impl<F, D, H, C, const WIDTH: usize, const DIGEST: usize, const SALT: usize>
    LmcsConfig<F, D, H, C, WIDTH, DIGEST, SALT>
where
    F: Default + Copy + PartialEq,
    D: Default + Copy + PartialEq + Send + Sync,
    H: StatefulHasher<F, [D; DIGEST], State = [D; WIDTH]>,
    C: PseudoCompressionFunction<[D; DIGEST], 2>,
{
    /// Verify a multi-opening proof and return references to the opened rows.
    ///
    /// This unified verify function handles both non-hiding (`SALT = 0`) and
    /// hiding (`SALT > 0`) proofs based on the struct's `SALT` parameter.
    ///
    /// # Arguments
    ///
    /// - `commitment`: The commitment (root hash) to verify against.
    /// - `dimensions`: Dimensions of each committed matrix.
    /// - `indices`: The indices being opened (supplied by verifier).
    /// - `proof`: The multi-opening proof.
    ///
    /// # Returns
    ///
    /// On success, returns `Ok(rows)` where `rows[query_idx][matrix_idx]` is a
    /// slice of the opened row data.
    ///
    /// # Errors
    ///
    /// Returns `Err` if verification fails (root mismatch, wrong dimensions, etc.).
    pub fn verify<'a>(
        &self,
        commitment: &Hash<F, D, DIGEST>,
        dimensions: &[Dimensions],
        indices: &[usize],
        proof: &'a Proof<F, D, DIGEST, [F; SALT]>,
    ) -> Result<Vec<Vec<&'a [F]>>, LmcsError> {
        // Validate proof structure
        if indices.len() != proof.num_queries() {
            return Err(LmcsError::WrongBatchSize);
        }

        if dimensions.is_empty() {
            return Err(LmcsError::WrongBatchSize);
        }

        let final_height = dimensions.last().unwrap().height;
        let depth = log2_strict_usize(final_height);

        // Validate all indices are in bounds
        for &index in indices {
            if index >= final_height {
                return Err(LmcsError::IndexOutOfBounds);
            }
        }

        // Compute leaf digests for each opened index
        let leaves: Vec<(usize, [D; DIGEST])> = indices
            .iter()
            .zip(proof.openings().iter())
            .map(|(&index, opening)| {
                let digest = compute_leaf_digest(
                    &self.sponge,
                    opening.rows(),
                    dimensions.iter().map(|d| d.width),
                    opening.salt(),
                )?;
                Ok((index, digest))
            })
            .collect::<Result<_, LmcsError>>()?;

        // Recompute root from leaves and siblings
        let computed_root = proof
            .recompute_root(depth, &leaves, &self.compress)
            .map_err(|_| LmcsError::RootMismatch)?;

        // Compare against commitment
        if Hash::from(computed_root) != *commitment {
            return Err(LmcsError::RootMismatch);
        }

        // Return references to opened rows
        let result = proof
            .openings()
            .iter()
            .map(|opening| opening.rows().iter().map(|r| r.as_slice()).collect())
            .collect();

        Ok(result)
    }

    /// Build a lifted Merkle tree from matrices.
    ///
    /// This is a convenience method that encapsulates the hasher, compression function,
    /// and const parameters, requiring only the packed value types and matrix type.
    ///
    /// # Type Parameters
    ///
    /// - `PF`: Packed field element type (use `F` for scalar, `F::Packing` for SIMD).
    /// - `PD`: Packed digest element type (use `D` for scalar, `D::Packing` for SIMD).
    /// - `M`: Matrix type implementing `Matrix<F>`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = LmcsConfig::<F, D, _, _, WIDTH, DIGEST>::new(sponge, compress);
    /// let tree = config.build_tree::<PF, PD, _>(matrices);
    /// let root = tree.root();
    /// ```
    pub fn build_tree<PF, PD, M>(
        &self,
        leaves: Vec<M>,
    ) -> LiftedHidingMerkleTree<F, D, M, DIGEST, SALT>
    where
        PF: PackedValue<Value = F> + Default,
        PD: PackedValue<Value = D> + Default,
        F: Send + Sync,
        M: Matrix<F>,
        H: StatefulHasher<PF, [PD; DIGEST], State = [PD; WIDTH]> + Sync,
        C: PseudoCompressionFunction<[PD; DIGEST], 2> + Sync,
    {
        const { assert!(SALT == 0, "SALT must be 0; use `build_tree_hiding` instead") }

        LiftedHidingMerkleTree::build::<PF, PD, H, C, WIDTH>(
            &self.sponge,
            &self.compress,
            leaves,
            None,
        )
    }

    /// Build a hiding lifted Merkle tree with randomly generated salt.
    ///
    /// This is a convenience method that encapsulates the hasher, compression function,
    /// and const parameters, requiring only the packed value types and matrix type.
    /// The salt size is determined by the config's `SALT` parameter.
    ///
    /// # Type Parameters
    ///
    /// - `PF`: Packed field element type (use `F` for scalar, `F::Packing` for SIMD).
    /// - `PD`: Packed digest element type (use `D` for scalar, `D::Packing` for SIMD).
    /// - `M`: Matrix type implementing `Matrix<F>`.
    /// - `R`: Random number generator type.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = LmcsConfig::<F, D, _, _, WIDTH, DIGEST, 4>::new(sponge, compress);
    /// let tree = config.build_tree_hiding::<PF, PD, _>(matrices, &mut rng);
    /// let root = tree.root();
    /// ```
    pub fn build_tree_hiding<PF, PD, M>(
        &self,
        leaves: Vec<M>,
        rng: &mut impl Rng,
    ) -> LiftedHidingMerkleTree<F, D, M, DIGEST, SALT>
    where
        StandardUniform: Distribution<F>,
        PF: PackedValue<Value = F> + Default,
        PD: PackedValue<Value = D> + Default,
        F: Send + Sync,
        M: Matrix<F>,
        H: StatefulHasher<PF, [PD; DIGEST], State = [PD; WIDTH]> + Sync,
        C: PseudoCompressionFunction<[PD; DIGEST], 2> + Sync,
    {
        use p3_matrix::dense::RowMajorMatrix;

        const { assert!(SALT > 0, "SALT must be > 0; use `build_tree` instead") }

        // Determine tree height from the tallest matrix
        let tree_height = leaves.last().map(|m| m.height()).unwrap_or(0);

        // Generate salt matrix only when SALT > 0
        let salt = RowMajorMatrix::rand(rng, tree_height, SALT);

        LiftedHidingMerkleTree::build::<PF, PD, H, C, WIDTH>(
            &self.sponge,
            &self.compress,
            leaves,
            Some(salt),
        )
    }
}

// ============================================================================
// Verification Helpers
// ============================================================================

/// Compute leaf digest from rows and salt.
///
/// This is the shared implementation used by both [`LmcsConfig::verify`]
/// and [`LmcsMmcs::verify_batch`].
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
fn compute_leaf_digest<F, D, H, const WIDTH: usize, const DIGEST_ELEMS: usize>(
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

    // Absorb salt (no-op when salt is empty)
    sponge.absorb_into(&mut state, salt.iter().copied());

    Ok(sponge.squeeze(&state))
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during LMCS operations.
#[derive(Debug, Error)]
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
}

#[cfg(test)]
mod tests;
