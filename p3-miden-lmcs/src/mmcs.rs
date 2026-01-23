//! MMCS trait implementation for LMCS configurations.
//!
//! This module provides [`Mmcs`] implementations directly on [`LmcsConfig`] and
//! [`HidingLmcsConfig`] for integration with FRI and other proof systems.

use alloc::vec::Vec;

use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_field::PackedValue;
use p3_matrix::{Dimensions, Matrix};
use p3_miden_stateful_hasher::StatefulHasher;
use p3_symmetric::{Hash, PseudoCompressionFunction};
use p3_util::log2_strict_usize;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Serialize};

use crate::lifted_tree::LiftedMerkleTree;
use crate::{HidingLmcsConfig, Lmcs, LmcsConfig, LmcsError};

// ============================================================================
// Mmcs implementation for LmcsConfig (non-hiding)
// ============================================================================

impl<PF, PD, H, C, const WIDTH: usize, const DIGEST_ELEMS: usize> Mmcs<PF::Value>
    for LmcsConfig<PF, PD, H, C, WIDTH, DIGEST_ELEMS>
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
    type ProverData<M> = LiftedMerkleTree<PF::Value, PD::Value, M, DIGEST_ELEMS, 0>;
    type Commitment = Hash<PF::Value, PD::Value, DIGEST_ELEMS>;
    type Proof = Vec<[PD::Value; DIGEST_ELEMS]>;
    type Error = LmcsError;

    fn commit<M: Matrix<PF::Value>>(
        &self,
        inputs: Vec<M>,
    ) -> (Self::Commitment, Self::ProverData<M>) {
        let tree = self.build_tree(inputs);
        (tree.root(), tree)
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

        // Convert dimensions to widths + log_max_height for internal use.
        let widths: Vec<usize> = dimensions.iter().map(|d| d.width).collect();
        let log_max_height = dimensions
            .last()
            .map(|d| log2_strict_usize(d.height))
            .unwrap_or(0);

        let opening = crate::Opening {
            rows: opened_values.to_vec(),
            salt: [],
        };
        let expected_root = compute_root(
            &self.sponge,
            &self.compress,
            &opening,
            index,
            &widths,
            log_max_height,
            opening_proof,
        )?;

        if &expected_root == commit {
            Ok(())
        } else {
            Err(LmcsError::RootMismatch)
        }
    }
}

// ============================================================================
// Mmcs implementation for HidingLmcsConfig
// ============================================================================

impl<PF, PD, H, C, R, const WIDTH: usize, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    Mmcs<PF::Value> for HidingLmcsConfig<PF, PD, H, C, R, WIDTH, DIGEST_ELEMS, SALT_ELEMS>
where
    PF: PackedValue + Default,
    PD: PackedValue + Default,
    PF::Value: PartialEq,
    R: Rng + Clone,
    StandardUniform: Distribution<PF::Value>,
    H: StatefulHasher<PF, [PD; DIGEST_ELEMS], State = [PD; WIDTH]>
        + StatefulHasher<PF::Value, [PD::Value; DIGEST_ELEMS], State = [PD::Value; WIDTH]>
        + Sync,
    C: PseudoCompressionFunction<[PD::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[PD; DIGEST_ELEMS], 2>
        + Sync,
    [PD::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    [PF::Value; SALT_ELEMS]: Serialize + for<'de> Deserialize<'de>,
{
    type ProverData<M> = LiftedMerkleTree<PF::Value, PD::Value, M, DIGEST_ELEMS, SALT_ELEMS>;
    type Commitment = Hash<PF::Value, PD::Value, DIGEST_ELEMS>;
    /// Proof includes salt and siblings: `([F; SALT_ELEMS], Vec<[D; DIGEST_ELEMS]>)`
    type Proof = ([PF::Value; SALT_ELEMS], Vec<[PD::Value; DIGEST_ELEMS]>);
    type Error = LmcsError;

    fn commit<M: Matrix<PF::Value>>(
        &self,
        inputs: Vec<M>,
    ) -> (Self::Commitment, Self::ProverData<M>) {
        let tree = self.build_tree(inputs);
        (tree.root(), tree)
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

        // Convert dimensions to widths + log_max_height for internal use.
        let widths: Vec<usize> = dimensions.iter().map(|d| d.width).collect();
        let log_max_height = dimensions
            .last()
            .map(|d| log2_strict_usize(d.height))
            .unwrap_or(0);

        let opening = crate::Opening {
            rows: opened_values.to_vec(),
            salt: *salt,
        };
        let expected_root = compute_root(
            &self.inner.sponge,
            &self.inner.compress,
            &opening,
            index,
            &widths,
            log_max_height,
            siblings,
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

/// Recompute the Merkle root from an opening and authentication path.
///
/// This is used by the MMCS implementation which uses simple authentication paths
/// (not the full `Proof` type which includes opening data).
fn compute_root<
    F,
    D,
    H,
    C,
    const WIDTH: usize,
    const DIGEST_ELEMS: usize,
    const SALT_ELEMS: usize,
>(
    sponge: &H,
    compress: &C,
    opening: &crate::Opening<F, SALT_ELEMS>,
    index: usize,
    widths: &[usize],
    log_max_height: usize,
    proof: &[[D; DIGEST_ELEMS]],
) -> Result<Hash<F, D, DIGEST_ELEMS>, LmcsError>
where
    F: Default + Copy,
    D: Default + Copy,
    H: StatefulHasher<F, [D; DIGEST_ELEMS], State = [D; WIDTH]>,
    C: PseudoCompressionFunction<[D; DIGEST_ELEMS], 2>,
{
    if proof.len() != log_max_height {
        return Err(LmcsError::WrongProofLen);
    }

    let mut digest = opening.digest::<D, H, WIDTH, DIGEST_ELEMS>(sponge, widths)?;

    let mut current_index = index;
    for sibling in proof {
        let (left, right) = if current_index & 1 == 0 {
            (digest, *sibling)
        } else {
            (*sibling, digest)
        };
        digest = compress.compress([left, right]);
        current_index >>= 1;
    }

    Ok(digest.into())
}
