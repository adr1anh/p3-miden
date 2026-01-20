//! MMCS trait implementation for LMCS.
//!
//! This module provides [`LmcsMmcs`] and [`HidingLmcsMmcs`], wrapper types that implement
//! the [`Mmcs`] trait for integration with FRI and other proof systems.
//!
//! The wrappers are needed because the `Mmcs` trait requires associated types that depend
//! on field types and digest sizes, which aren't part of the simpler [`LmcsConfig`].
//!
//! # Hiding MMCS
//!
//! [`HidingLmcsMmcs`] provides a hiding commitment scheme that generates random salt
//! during commit and includes it in proofs. This follows the pattern from plonky3's
//! `MerkleTreeHidingMmcs`.

use alloc::vec::Vec;
use core::cell::RefCell;
use core::marker::PhantomData;

use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_field::PackedValue;
use p3_matrix::{Dimensions, Matrix};
use p3_miden_stateful_hasher::StatefulHasher;
use p3_symmetric::{Hash, PseudoCompressionFunction};
use p3_util::log2_strict_usize;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Serialize};

use crate::lifted_tree::LiftedHidingMerkleTree;
use crate::{LmcsConfig, LmcsError};

/// Non-hiding MMCS wrapper for LMCS configuration.
///
/// This wrapper adds the type parameters needed for the [`Mmcs`] trait implementation.
/// Use this type when you need MMCS compatibility (e.g., for FRI).
///
/// For hiding commitments with salt, use [`HidingLmcsMmcs`] instead.
///
/// # Type Parameters
///
/// - `PF`: Packed field type (for SIMD operations). Use `F` (scalar field) if no SIMD.
/// - `PD`: Packed digest type. Use `F` or `D` (digest element type) if no SIMD.
/// - `H`: Stateful hasher type.
/// - `C`: 2-to-1 compression function type.
/// - `WIDTH`: Permutation/state width.
/// - `DIGEST_ELEMS`: Number of elements in a digest.
///
/// # Example
///
/// ```ignore
/// use p3_miden_lmcs::LmcsMmcs;
///
/// let mmcs = LmcsMmcs::<P, P, Sponge, Compress, WIDTH, DIGEST>::new(sponge, compress);
///
/// // Use via Mmcs trait
/// let (commitment, prover_data) = mmcs.commit(matrices);
/// ```
#[derive(Copy, Clone, Debug)]
pub struct LmcsMmcs<PF, PD, H, C, const WIDTH: usize, const DIGEST_ELEMS: usize>
where
    PF: PackedValue,
    PD: PackedValue,
{
    /// The underlying LMCS configuration (non-hiding, SALT = 0).
    pub config: LmcsConfig<PF::Value, PD::Value, H, C, WIDTH, DIGEST_ELEMS, 0>,
    _phantom: PhantomData<(PF, PD)>,
}

impl<PF, PD, H, C, const WIDTH: usize, const DIGEST_ELEMS: usize>
    LmcsMmcs<PF, PD, H, C, WIDTH, DIGEST_ELEMS>
where
    PF: PackedValue,
    PD: PackedValue,
{
    /// Create a new non-hiding MMCS wrapper from sponge and compression function.
    #[inline]
    pub const fn new(sponge: H, compress: C) -> Self {
        Self {
            config: LmcsConfig::new(sponge, compress),
            _phantom: PhantomData,
        }
    }
}

impl<PF, PD, H, C, const WIDTH: usize, const DIGEST_ELEMS: usize> Mmcs<PF::Value>
    for LmcsMmcs<PF, PD, H, C, WIDTH, DIGEST_ELEMS>
where
    PF: PackedValue + Default,
    PD: PackedValue + Default,
    PF::Value: PartialEq,
    H: StatefulHasher<PF, [PD; DIGEST_ELEMS], State = [PD; WIDTH]>
        + StatefulHasher<PF::Value, [PD::Value; DIGEST_ELEMS], State = [PD::Value; WIDTH]>
        + Sync,
    C: PseudoCompressionFunction<[PD::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[PD; DIGEST_ELEMS], 2>
        + Sync,
    [PD::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
{
    type ProverData<M> = LiftedHidingMerkleTree<PF::Value, PD::Value, M, DIGEST_ELEMS, 0>;
    type Commitment = Hash<PF::Value, PD::Value, DIGEST_ELEMS>;
    type Proof = Vec<[PD::Value; DIGEST_ELEMS]>;
    type Error = LmcsError;

    fn commit<M: Matrix<PF::Value>>(
        &self,
        inputs: Vec<M>,
    ) -> (Self::Commitment, Self::ProverData<M>) {
        let tree = self.config.build_tree::<PF, PD, _>(inputs);
        let root = tree.root();
        (root, tree)
    }

    fn open_batch<M: Matrix<PF::Value>>(
        &self,
        index: usize,
        tree: &Self::ProverData<M>,
    ) -> BatchOpening<PF::Value, Self> {
        let final_height = tree.height();
        assert!(
            index < final_height,
            "index {index} out of range {final_height}"
        );

        let opened_rows = tree.rows(index);
        let proof = tree.authentication_path(index);

        BatchOpening::new(opened_rows, proof)
    }

    fn get_matrices<'a, M: Matrix<PF::Value>>(&self, tree: &'a Self::ProverData<M>) -> Vec<&'a M> {
        tree.leaves.iter().collect()
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        batch_opening: BatchOpeningRef<'_, PF::Value, Self>,
    ) -> Result<(), Self::Error> {
        let (opened_values, opening_proof) = batch_opening.unpack();

        // MMCS verify_batch uses no salt
        let no_salt: &[PF::Value; 0] = &[];
        let expected_root = compute_root(
            &self.config,
            opened_values,
            index,
            dimensions,
            opening_proof,
            no_salt,
        )?;

        if &expected_root == commit {
            Ok(())
        } else {
            Err(LmcsError::RootMismatch)
        }
    }
}

// ============================================================================
// Hiding MMCS
// ============================================================================

/// Hiding MMCS wrapper for LMCS configuration.
///
/// This wrapper generates random salt during commit and includes it in proofs,
/// following plonky3's `MerkleTreeHidingMmcs` pattern. The RNG is stored in a
/// `RefCell` to allow salt generation without `&mut self` (required by `Mmcs::commit`).
///
/// # Type Parameters
///
/// - `PF`: Packed field type (for SIMD operations). Use `F` (scalar field) if no SIMD.
/// - `PD`: Packed digest type. Use `F` or `D` (digest element type) if no SIMD.
/// - `H`: Stateful hasher type.
/// - `C`: 2-to-1 compression function type.
/// - `R`: Random number generator type.
/// - `WIDTH`: Permutation/state width.
/// - `DIGEST_ELEMS`: Number of elements in a digest.
/// - `SALT_ELEMS`: Number of salt elements per leaf (must be > 0 for hiding).
///
/// # Example
///
/// ```ignore
/// use p3_miden_lmcs::HidingLmcsMmcs;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// let rng = StdRng::seed_from_u64(42);
/// let mmcs = HidingLmcsMmcs::<P, P, Sponge, Compress, StdRng, WIDTH, DIGEST, 4>::new(sponge, compress, rng);
///
/// // Use via Mmcs trait - salt is automatically generated
/// let (commitment, prover_data) = mmcs.commit(matrices);
/// ```
#[derive(Debug)]
pub struct HidingLmcsMmcs<
    PF,
    PD,
    H,
    C,
    R,
    const WIDTH: usize,
    const DIGEST_ELEMS: usize,
    const SALT_ELEMS: usize,
> where
    PF: PackedValue,
    PD: PackedValue,
{
    /// The underlying LMCS configuration.
    pub config: LmcsConfig<PF::Value, PD::Value, H, C, WIDTH, DIGEST_ELEMS, SALT_ELEMS>,
    /// RNG for salt generation. Uses `RefCell` for interior mutability since
    /// `Mmcs::commit` takes `&self` but we need to mutate the RNG.
    rng: RefCell<R>,
    _phantom: PhantomData<(PF, PD)>,
}

impl<PF, PD, H, C, R, const WIDTH: usize, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    HidingLmcsMmcs<PF, PD, H, C, R, WIDTH, DIGEST_ELEMS, SALT_ELEMS>
where
    PF: PackedValue,
    PD: PackedValue,
{
    /// Create a new hiding MMCS wrapper from sponge, compression function, and RNG.
    ///
    /// # Compile-time Error
    ///
    /// Fails to compile if `SALT_ELEMS == 0`. Use [`LmcsMmcs`] for non-hiding commitments.
    #[inline]
    pub fn new(sponge: H, compress: C, rng: R) -> Self {
        const {
            assert!(
                SALT_ELEMS > 0,
                "HidingLmcsMmcs requires SALT_ELEMS > 0; use LmcsMmcs for non-hiding"
            )
        }
        Self {
            config: LmcsConfig::new(sponge, compress),
            rng: RefCell::new(rng),
            _phantom: PhantomData,
        }
    }
}

impl<PF, PD, H, C, R, const WIDTH: usize, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize> Clone
    for HidingLmcsMmcs<PF, PD, H, C, R, WIDTH, DIGEST_ELEMS, SALT_ELEMS>
where
    PF: PackedValue,
    PD: PackedValue,
    H: Clone,
    C: Clone,
    R: Clone,
{
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            rng: self.rng.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<PF, PD, H, C, R, const WIDTH: usize, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    Mmcs<PF::Value> for HidingLmcsMmcs<PF, PD, H, C, R, WIDTH, DIGEST_ELEMS, SALT_ELEMS>
where
    PF: PackedValue + Default,
    PD: PackedValue + Default,
    PF::Value: PartialEq,
    R: Rng + Clone,
    StandardUniform: Distribution<PF::Value>,
    H: StatefulHasher<PF, [PD; DIGEST_ELEMS], State = [PD; WIDTH]>
        + StatefulHasher<PF::Value, [PD::Value; DIGEST_ELEMS], State = [PD::Value; WIDTH]>
        + Clone
        + Sync,
    C: PseudoCompressionFunction<[PD::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[PD; DIGEST_ELEMS], 2>
        + Clone
        + Sync,
    [PD::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    [PF::Value; SALT_ELEMS]: Serialize + for<'de> Deserialize<'de>,
{
    type ProverData<M> = LiftedHidingMerkleTree<PF::Value, PD::Value, M, DIGEST_ELEMS, SALT_ELEMS>;
    type Commitment = Hash<PF::Value, PD::Value, DIGEST_ELEMS>;
    /// Proof includes salt and siblings: `([F; SALT_ELEMS], Vec<[D; DIGEST_ELEMS]>)`
    type Proof = ([PF::Value; SALT_ELEMS], Vec<[PD::Value; DIGEST_ELEMS]>);
    type Error = LmcsError;

    fn commit<M: Matrix<PF::Value>>(
        &self,
        inputs: Vec<M>,
    ) -> (Self::Commitment, Self::ProverData<M>) {
        // Use interior mutability to borrow the RNG mutably
        let tree = self
            .config
            .build_tree_hiding::<PF, PD, _>(inputs, &mut *self.rng.borrow_mut());
        let root = tree.root();
        (root, tree)
    }

    fn open_batch<M: Matrix<PF::Value>>(
        &self,
        index: usize,
        tree: &Self::ProverData<M>,
    ) -> BatchOpening<PF::Value, Self> {
        let final_height = tree.height();
        assert!(
            index < final_height,
            "index {index} out of range {final_height}"
        );

        let opened_rows = tree.rows(index);
        let siblings = tree.authentication_path(index);
        let salt = tree.salt(index);

        BatchOpening::new(opened_rows, (salt, siblings))
    }

    fn get_matrices<'a, M: Matrix<PF::Value>>(&self, tree: &'a Self::ProverData<M>) -> Vec<&'a M> {
        tree.leaves.iter().collect()
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        batch_opening: BatchOpeningRef<'_, PF::Value, Self>,
    ) -> Result<(), Self::Error> {
        let (opened_values, (salt, siblings)) = batch_opening.unpack();

        let expected_root = compute_root(
            &self.config,
            opened_values,
            index,
            dimensions,
            siblings,
            salt,
        )?;

        if &expected_root == commit {
            Ok(())
        } else {
            Err(LmcsError::RootMismatch)
        }
    }
}

// ============================================================================
// Internal Helpers for MMCS verify_batch
// ============================================================================

/// Recompute the Merkle root from opened rows and an authentication path.
///
/// This is used by the MMCS implementation which uses simple authentication paths
/// (not CompactProof).
fn compute_root<
    F,
    D,
    H,
    C,
    const WIDTH: usize,
    const DIGEST_ELEMS: usize,
    const SALT_ELEMS: usize,
    const CFG_SALT: usize,
>(
    config: &LmcsConfig<F, D, H, C, WIDTH, DIGEST_ELEMS, CFG_SALT>,
    rows: &[Vec<F>],
    index: usize,
    dimensions: &[Dimensions],
    proof: &[[D; DIGEST_ELEMS]],
    salt: &[F; SALT_ELEMS],
) -> Result<Hash<F, D, DIGEST_ELEMS>, LmcsError>
where
    F: Default + Copy,
    D: Default + Copy,
    H: StatefulHasher<F, [D; DIGEST_ELEMS], State = [D; WIDTH]>,
    C: PseudoCompressionFunction<[D; DIGEST_ELEMS], 2>,
{
    let final_height = dimensions.last().unwrap().height;
    let expected_proof_len = log2_strict_usize(final_height);
    if proof.len() != expected_proof_len {
        return Err(LmcsError::WrongProofLen);
    }

    let mut digest = crate::compute_leaf_digest::<F, D, H, WIDTH, DIGEST_ELEMS>(
        &config.sponge,
        rows,
        dimensions.iter().map(|d| d.width),
        salt,
    )?;

    let mut current_index = index;
    for sibling in proof {
        let (left, right) = if current_index & 1 == 0 {
            (digest, *sibling)
        } else {
            (*sibling, digest)
        };
        digest = config.compress.compress([left, right]);
        current_index >>= 1;
    }

    Ok(digest.into())
}
