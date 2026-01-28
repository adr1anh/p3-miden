//! LMCS configuration types.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::PackedValue;
use p3_matrix::Matrix;
use p3_miden_stateful_hasher::StatefulHasher;
use p3_miden_transcript::VerifierChannel;
use p3_symmetric::{Hash, PseudoCompressionFunction};

use crate::utils::digest_rows_and_salt;
use crate::{LiftedMerkleTree, Lmcs, LmcsError, Proof};

/// LMCS configuration holding cryptographic primitives (sponge + compression).
///
/// For hiding commitments with salt, use [`crate::HidingLmcsConfig`] instead.
#[derive(Clone, Debug)]
pub struct LmcsConfig<
    PF,
    PD,
    H,
    C,
    const WIDTH: usize,
    const DIGEST: usize,
    const SALT_ELEMS: usize = 0,
> {
    /// Stateful sponge for hashing matrix rows into leaf digests.
    pub sponge: H,
    /// 2-to-1 compression function for building internal tree nodes.
    pub compress: C,
    pub(crate) _phantom: PhantomData<(PF, PD)>,
}

impl<PF, PD, H, C, const WIDTH: usize, const DIGEST: usize>
    LmcsConfig<PF, PD, H, C, WIDTH, DIGEST, 0>
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

impl<PF, PD, H, C, const WIDTH: usize, const DIGEST: usize, const SALT_ELEMS: usize> Lmcs
    for LmcsConfig<PF, PD, H, C, WIDTH, DIGEST, SALT_ELEMS>
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
    type SingleProof = Proof<PF::Value, PD::Value, DIGEST, SALT_ELEMS>;
    type Tree<M: Matrix<PF::Value>> = LiftedMerkleTree<PF::Value, PD::Value, M, DIGEST, SALT_ELEMS>;

    fn build_tree<M: Matrix<Self::F>>(&self, leaves: Vec<M>) -> Self::Tree<M> {
        const { assert!(SALT_ELEMS == 0) }
        LiftedMerkleTree::build::<PF, PD, H, C, WIDTH>(&self.sponge, &self.compress, leaves, None)
    }

    fn open_batch<Ch>(
        &self,
        commitment: &Self::Commitment,
        widths: &[usize],
        log_max_height: usize,
        indices: &[usize],
        channel: &mut Ch,
    ) -> Result<Vec<Vec<Vec<Self::F>>>, LmcsError>
    where
        Ch: VerifierChannel<F = Self::F, Commitment = Self::Commitment>,
    {
        let mut openings = Vec::with_capacity(indices.len());
        let mut leaf_nodes: BTreeMap<usize, [PD::Value; DIGEST]> = BTreeMap::new();

        for &index in indices {
            let rows = widths
                .iter()
                .map(|&width| channel.receive_hint_field_slice(width).map(Vec::from))
                .collect::<Option<Vec<_>>>()
                .ok_or(LmcsError::InvalidProof)?;

            let salt: [PF::Value; SALT_ELEMS] = channel
                .receive_hint_field_slice(SALT_ELEMS)
                .ok_or(LmcsError::InvalidProof)?
                .try_into()
                .unwrap();

            let digest = digest_rows_and_salt(
                &self.sponge,
                rows.iter().map(|row| row.as_slice()),
                salt.as_slice(),
            );

            openings.push(rows);

            if leaf_nodes
                .insert(index, digest)
                .is_some_and(|existing| existing != digest)
            {
                return Err(LmcsError::InvalidProof);
            }
        }

        // Recompute root from known leaves and streamed siblings.
        //
        // For a node at position p:
        // - sibling: p ^ 1
        // - parent: p >> 1
        // - left child: p & 1 == 0
        //
        // We walk level-by-level. If a sibling is not already known at a level, it must be
        // provided by the proof. After log_max_height steps, we expect a single root at position 0.
        //
        // Security notes:
        // - Completeness: missing siblings return InvalidProof.
        // - Canonical order: siblings are consumed left-to-right, bottom-to-top.
        // - Extra siblings are ignored and remain unread.
        let computed_root = {
            // We alternate between two vectors: one holds the current level's nodes (children),
            // the other accumulates the next level's nodes (parents). After each level, we swap them.
            let mut children: Vec<(usize, [PD::Value; DIGEST])> = leaf_nodes.into_iter().collect();
            let mut parents = Vec::new();

            // Process each level from leaves (level 0) up to root (level tree_depth).
            for _level in 0..log_max_height {
                let mut children_iter = children.iter().peekable();

                while let Some((child_position, child_hash)) = children_iter.next() {
                    // Get sibling hash: either from known nodes (if next in sorted list) or from proof.
                    // When both children are known, the proof omits that sibling since it's redundant.
                    let sibling_position = child_position ^ 1;
                    let sibling_hash =
                        match children_iter.next_if(|(pos, _)| *pos == sibling_position) {
                            Some((_, hash)) => *hash,
                            None => channel
                                .receive_hint_commitment()
                                .copied()
                                .map(Into::into)
                                .ok_or(LmcsError::InvalidProof)?,
                        };

                    // Determine left/right ordering: left child has even position (bit 0 = 0).
                    let child_is_left = child_position & 1 == 0;
                    let (left_hash, right_hash) = if child_is_left {
                        (*child_hash, sibling_hash)
                    } else {
                        (sibling_hash, *child_hash)
                    };

                    let parent_hash = self.compress.compress([left_hash, right_hash]);
                    let parent_position = child_position >> 1;
                    parents.push((parent_position, parent_hash));
                }

                core::mem::swap(&mut children, &mut parents);
                parents.clear();
            }

            // Invariant: after `tree_depth` iterations, all leaf positions converge to a single root at 0.
            // If any of the indices were out of bounds, the final index would not be 0.
            // If no indices were provided, children is empty.
            match children.as_slice() {
                [(0, root)] => *root,
                _ => return Err(LmcsError::InvalidProof),
            }
        };

        // Compare against commitment.
        if Hash::from(computed_root) != *commitment {
            return Err(LmcsError::RootMismatch);
        }

        // Return opened rows in query order.
        Ok(openings)
    }

    fn read_batch_from_channel<Ch>(
        &self,
        widths: &[usize],
        log_max_height: usize,
        indices: &[usize],
        channel: &mut Ch,
    ) -> Result<Vec<Self::SingleProof>, LmcsError>
    where
        Ch: VerifierChannel<F = Self::F, Commitment = Self::Commitment>,
    {
        Proof::read_batch_from_channel(
            &self.sponge,
            &self.compress,
            widths,
            log_max_height,
            indices,
            channel,
        )
        .ok_or(LmcsError::InvalidProof)
    }
}
// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use crate::{Lmcs, LmcsConfig, LmcsError, LmcsTree};
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
    use p3_miden_transcript::{
        ProverTranscript, TranscriptData, VerifierChannel, VerifierTranscript,
    };
    use p3_util::log2_strict_usize;

    type TestLmcs =
        LmcsConfig<bb::P, bb::P, bb::Sponge, bb::Compress, { bb::WIDTH }, { bb::DIGEST }>;

    fn small_matrix(height: usize, width: usize, seed: u64) -> RowMajorMatrix<bb::F> {
        let values = (0..height * width)
            .map(|i| bb::F::from_u64(seed + i as u64))
            .collect();
        RowMajorMatrix::new(values, width)
    }

    #[test]
    fn open_batch_cases() {
        let (_, sponge, compress) = bb::test_components();
        let lmcs: TestLmcs = LmcsConfig::new(sponge, compress);
        let matrices = vec![small_matrix(4, 2, 0), small_matrix(4, 3, 100)];
        let tree = lmcs.build_tree(matrices);
        let widths: Vec<_> = tree.leaves().iter().map(|m| m.width()).collect();
        let log_max_height = log2_strict_usize(tree.height());
        let commitment = tree.root();

        let make_transcript = |indices: &[usize]| {
            let mut prover_channel = ProverTranscript::new(bb::test_challenger());
            tree.prove_batch(indices, &mut prover_channel);
            prover_channel.into_data()
        };

        let assert_open = |indices: &[usize]| {
            let transcript = make_transcript(indices);
            let mut verifier_channel =
                VerifierTranscript::from_data(bb::test_challenger(), &transcript);
            let opened = lmcs
                .open_batch(
                    &commitment,
                    &widths,
                    log_max_height,
                    indices,
                    &mut verifier_channel,
                )
                .unwrap();
            assert_eq!(opened.len(), indices.len());
            for (pos, &idx) in indices.iter().enumerate() {
                assert_eq!(opened[pos], tree.rows(idx));
            }
            assert!(verifier_channel.is_empty())
        };

        assert_open(&[0]);
        assert_open(&[0, 1]);
        assert_open(&[0, 2]);
        assert_open(&[0, 1, 2, 3]);
        assert_open(&[2, 2]);

        let tiny_tree = lmcs.build_tree(vec![small_matrix(1, 1, 7)]);
        let widths_tiny = vec![1];
        let log_tiny = log2_strict_usize(tiny_tree.height());
        let indices = [0usize];
        let mut prover_channel = ProverTranscript::new(bb::test_challenger());
        tiny_tree.prove_batch(&indices, &mut prover_channel);
        let transcript = prover_channel.into_data();
        let mut verifier_channel =
            VerifierTranscript::from_data(bb::test_challenger(), &transcript);
        let opened = lmcs
            .open_batch(
                &tiny_tree.root(),
                &widths_tiny,
                log_tiny,
                &indices,
                &mut verifier_channel,
            )
            .unwrap();
        assert_eq!(opened[0], tiny_tree.rows(0));

        // oob index
        let transcript = ProverTranscript::new(bb::test_challenger()).into_data();
        let mut verifier_channel =
            VerifierTranscript::from_data(bb::test_challenger(), &transcript);
        assert_eq!(
            lmcs.open_batch(
                &commitment,
                &widths,
                log_max_height,
                &[tree.height()],
                &mut verifier_channel,
            ),
            Err(LmcsError::InvalidProof)
        );

        // wrong tree
        let indices = [0usize];
        let transcript = make_transcript(&indices);
        let mut verifier_channel =
            VerifierTranscript::from_data(bb::test_challenger(), &transcript);
        let wrong_tree = lmcs.build_tree(vec![small_matrix(4, 2, 999)]);
        assert_eq!(
            lmcs.open_batch(
                &wrong_tree.root(),
                &widths,
                log_max_height,
                &indices,
                &mut verifier_channel,
            ),
            Err(LmcsError::RootMismatch)
        );

        // missing item from transcript
        let indices = [0usize];
        let transcript = make_transcript(&indices);
        let (fields, mut commitments) = transcript.into_parts();
        commitments.pop();
        let truncated = TranscriptData::new(fields, commitments);
        let mut verifier_channel = VerifierTranscript::from_data(bb::test_challenger(), &truncated);
        assert_eq!(
            lmcs.open_batch(
                &commitment,
                &widths,
                log_max_height,
                &indices,
                &mut verifier_channel,
            ),
            Err(LmcsError::InvalidProof)
        );

        // empty indices
        let transcript = ProverTranscript::new(bb::test_challenger()).into_data();
        let mut verifier_channel =
            VerifierTranscript::from_data(bb::test_challenger(), &transcript);
        assert_eq!(
            lmcs.open_batch(
                &commitment,
                &widths,
                log_max_height,
                &[],
                &mut verifier_channel,
            ),
            Err(LmcsError::InvalidProof)
        );
    }
}
