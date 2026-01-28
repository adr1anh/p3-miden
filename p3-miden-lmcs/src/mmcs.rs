//! MMCS trait implementation for LMCS configurations.
//!
//! This module provides [`Mmcs`] implementations directly on [`LmcsConfig`] and
//! [`HidingLmcsConfig`] for integration with FRI and other proof systems.

use alloc::vec::Vec;
use core::iter::zip;

use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_field::PackedValue;
use p3_matrix::{Dimensions, Matrix};
use p3_miden_stateful_hasher::StatefulHasher;
use p3_symmetric::{Hash, PseudoCompressionFunction};
use p3_util::log2_ceil_usize;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Serialize};

use crate::LmcsTree;
use crate::lifted_tree::LiftedMerkleTree;
use crate::utils::digest_rows_and_salt;
use crate::{HidingLmcsConfig, Lmcs, LmcsConfig, LmcsError};

// ============================================================================
// Mmcs implementation for LmcsConfig (non-hiding)
// ============================================================================

impl<PF, PD, H, C, const WIDTH: usize, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    Mmcs<PF::Value> for LmcsConfig<PF, PD, H, C, WIDTH, DIGEST_ELEMS, SALT_ELEMS>
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
    [PF::Value; SALT_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    [PD::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
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

        let crate::Proof {
            rows,
            salt,
            siblings,
        } = tree.single_proof(index);

        BatchOpening::new(rows, (salt, siblings))
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
        let (rows, (salt, siblings)) = batch_opening.unpack();

        // Convert dimensions to widths + log_max_height for internal use.
        let widths: Vec<usize> = dimensions.iter().map(|d| d.width).collect();
        if batch_opening.opened_values.len() != widths.len() {
            return Err(LmcsError::InvalidProof);
        }
        for (row, &width) in zip(batch_opening.opened_values, &widths) {
            if row.len() != width {
                return Err(LmcsError::InvalidProof);
            }
        }

        let leaf_digest =
            digest_rows_and_salt(&self.sponge, rows.iter().map(|row| row.as_slice()), salt);

        let max_height = dimensions
            .iter()
            .map(|d| d.height)
            .max()
            .ok_or(LmcsError::InvalidProof)?;
        let log_max_height = log2_ceil_usize(max_height);
        if siblings.len() != log_max_height {
            return Err(LmcsError::InvalidProof);
        }

        let computed_root = {
            let mut current = leaf_digest;
            let mut pos = index;

            for sibling_digest in siblings {
                let is_left = pos & 1 == 0;
                current = if is_left {
                    self.compress.compress([current, *sibling_digest])
                } else {
                    self.compress.compress([*sibling_digest, current])
                };
                pos >>= 1;
            }

            current
        };

        if Hash::from(computed_root) != *commit {
            return Err(LmcsError::RootMismatch);
        }

        Ok(())
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
        let BatchOpening {
            opened_values,
            opening_proof,
        } = Mmcs::open_batch(&self.inner, index, tree);
        BatchOpening {
            opened_values,
            opening_proof,
        }
    }

    fn get_matrices<'a, M: Matrix<PF::Value>>(&self, tree: &'a Self::ProverData<M>) -> Vec<&'a M> {
        self.inner.get_matrices(tree)
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        batch_opening: BatchOpeningRef<'_, PF::Value, Self>,
    ) -> Result<(), Self::Error> {
        let batch_opening = BatchOpeningRef {
            opened_values: batch_opening.opened_values,
            opening_proof: batch_opening.opening_proof,
        };
        self.inner
            .verify_batch(commit, dimensions, index, batch_opening)
    }
}
