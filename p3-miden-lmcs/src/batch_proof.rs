//! Batch multi-opening proof for Merkle trees.
//!
//! When opening multiple leaves, many siblings on authentication paths are shared.
//! [`BatchProof`] stores each sibling only once, in canonical order (left-to-right,
//! bottom-to-top).

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use p3_miden_stateful_hasher::StatefulHasher;
use p3_symmetric::{Hash, PseudoCompressionFunction};

use crate::LmcsError;
use crate::opening::Opening;
use crate::proof::Proof;

// ============================================================================
// Public Types
// ============================================================================

/// Batch multi-opening proof containing openings and compact Merkle siblings.
///
/// Indices are not stored—verifier supplies them during verification.
///
/// Siblings are in canonical order (left-to-right, bottom-to-top per level).
/// When both children of a pair are known, no sibling is stored.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, [D; DIGEST_ELEMS]: Serialize, [F; SALT_ELEMS]: Serialize",
    deserialize = "F: Deserialize<'de>, [D; DIGEST_ELEMS]: Deserialize<'de>, [F; SALT_ELEMS]: Deserialize<'de>"
))]
pub struct BatchProof<F, D, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize = 0> {
    /// Openings: `openings[query_idx]` contains rows and salt for that query.
    pub(crate) openings: Vec<Opening<F, SALT_ELEMS>>,
    /// Merkle siblings in canonical order (left-to-right, bottom-to-top).
    pub(crate) siblings: Vec<[D; DIGEST_ELEMS]>,
}

impl<F, D, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    BatchProof<F, D, DIGEST_ELEMS, SALT_ELEMS>
where
    F: Default + Copy + PartialEq,
    D: Default + Copy + PartialEq,
{
    /// Open a batch proof against a commitment.
    ///
    /// Returns references to opened rows on success.
    pub fn open<'a, H, C, const WIDTH: usize>(
        &'a self,
        sponge: &H,
        compress: &C,
        commitment: &Hash<F, D, DIGEST_ELEMS>,
        widths: &[usize],
        log_max_height: usize,
        indices: &[usize],
    ) -> Result<Vec<Vec<&'a [F]>>, LmcsError>
    where
        H: StatefulHasher<F, [D; DIGEST_ELEMS], State = [D; WIDTH]>,
        C: PseudoCompressionFunction<[D; DIGEST_ELEMS], 2>,
    {
        let nodes =
            self.compute_sorted_leaf_nodes::<H, WIDTH>(sponge, widths, log_max_height, indices)?;
        let depth = log_max_height;

        // Recompute root from known leaves and proof siblings
        let computed_root = self
            .recompute_root_in_place(nodes, compress, depth)
            .ok_or(LmcsError::InvalidProof)?;

        // Compare against commitment
        if Hash::from(computed_root) != *commitment {
            return Err(LmcsError::RootMismatch);
        }

        // Return references to opened rows
        Ok(self
            .openings
            .iter()
            .map(|opening| opening.rows.iter().map(|r| r.as_slice()).collect())
            .collect())
    }

    /// Extract individual single-opening proofs from this batch proof.
    ///
    /// Returns one [`Proof`] per index, in the same order as `indices`.
    pub fn extract_proofs<H, C, const WIDTH: usize>(
        &self,
        sponge: &H,
        compress: &C,
        widths: &[usize],
        log_max_height: usize,
        indices: &[usize],
    ) -> Result<Vec<Proof<F, D, DIGEST_ELEMS, SALT_ELEMS>>, LmcsError>
    where
        H: StatefulHasher<F, [D; DIGEST_ELEMS], State = [D; WIDTH]>,
        C: PseudoCompressionFunction<[D; DIGEST_ELEMS], 2>,
    {
        // Reuse existing helper for leaf computation
        let leaf_nodes =
            self.compute_sorted_leaf_nodes::<H, WIDTH>(sponge, widths, log_max_height, indices)?;
        let depth = log_max_height;

        // Phase 1: Build partial tree containing all nodes.
        // Insert leaves at level 0. Maps (depth, index) -> digest.
        let mut tree: BTreeMap<(usize, usize), [D; DIGEST_ELEMS]> = leaf_nodes
            .into_iter()
            .map(|(leaf_index, digest)| ((0, leaf_index), digest))
            .collect();

        // Build level by level from leaves to root
        let mut siblings = self.siblings.iter();
        for current_depth in 0..depth {
            let mut processed = BTreeSet::new();

            let indices_at_depth: Vec<_> = tree
                .range((current_depth, 0)..=(current_depth, usize::MAX))
                .map(|(&(_, idx), _)| idx)
                .collect();

            for leaf_index in indices_at_depth {
                let parent_index = leaf_index / 2;
                if !processed.insert(parent_index) {
                    continue;
                }

                let sibling_index = leaf_index ^ 1;
                let digest = tree[&(current_depth, leaf_index)];

                // Get sibling from tree or consume from proof
                let sibling_digest = *tree
                    .entry((current_depth, sibling_index))
                    .or_insert_with(|| *siblings.next().unwrap());

                // Compute parent: left child has even index
                let is_left_child = leaf_index & 1 == 0;
                let (left, right) = if is_left_child {
                    (digest, sibling_digest)
                } else {
                    (sibling_digest, digest)
                };
                tree.insert(
                    (current_depth + 1, parent_index),
                    compress.compress([left, right]),
                );
            }
        }

        // Phase 2: Extract proofs by walking up the tree for each leaf.
        let proofs = indices
            .iter()
            .zip(&self.openings)
            .map(|(&leaf_index, opening)| {
                let mut path_siblings = Vec::with_capacity(depth);
                let mut current_index = leaf_index;

                for current_depth in 0..depth {
                    let sibling_index = current_index ^ 1;
                    path_siblings.push(tree[&(current_depth, sibling_index)]);
                    current_index >>= 1; // Move to parent
                }

                Proof {
                    opening: opening.clone(),
                    siblings: path_siblings,
                }
            })
            .collect();

        Ok(proofs)
    }

    /// Compute leaf digests, validate indices, sort, and deduplicate.
    ///
    /// Returns sorted, deduplicated `(index, digest)` pairs.
    fn compute_sorted_leaf_nodes<H, const WIDTH: usize>(
        &self,
        sponge: &H,
        widths: &[usize],
        log_max_height: usize,
        indices: &[usize],
    ) -> Result<SortedLeafNodes<D, DIGEST_ELEMS>, LmcsError>
    where
        H: StatefulHasher<F, [D; DIGEST_ELEMS], State = [D; WIDTH]>,
    {
        // Validate proof structure
        if indices.is_empty() || indices.len() != self.openings.len() {
            return Err(LmcsError::WrongBatchSize);
        }

        if widths.is_empty() {
            return Err(LmcsError::WrongBatchSize);
        }

        let max_height = 1 << log_max_height;

        // Compute (index, digest) pairs with validation
        let mut nodes: SortedLeafNodes<D, DIGEST_ELEMS> = indices
            .iter()
            .zip(self.openings.iter())
            .map(|(&index, opening)| {
                if index >= max_height {
                    return Err(LmcsError::IndexOutOfBounds);
                }
                let digest = opening.digest::<D, H, WIDTH, DIGEST_ELEMS>(sponge, widths)?;
                Ok((index, digest))
            })
            .collect::<Result<_, _>>()?;

        // Sort by index
        nodes.sort_by_key(|(idx, _)| *idx);

        // Check for conflicting leaves (same index, different digest)
        for window in nodes.windows(2) {
            let [(idx_a, leaf_a), (idx_b, leaf_b)] = window else {
                unreachable!()
            };
            if idx_a == idx_b && leaf_a != leaf_b {
                return Err(LmcsError::ConflictingLeaf);
            }
        }

        // Deduplicate
        nodes.dedup_by_key(|(idx, _)| *idx);
        Ok(nodes)
    }

    /// Recompute the Merkle root from known leaf hashes and this proof's siblings.
    ///
    /// # Arguments
    ///
    /// - `leaf_nodes`: Vector of `(leaf_position, leaf_hash)` pairs, **sorted by position**.
    /// - `compress`: 2-to-1 compression function for combining sibling pairs.
    /// - `tree_depth`: Depth of the tree (number of levels from leaves to root).
    ///
    /// # Binary Tree Position Arithmetic
    ///
    /// For a node at position `p` in a binary tree level:
    /// - **Sibling position**: `p ^ 1` (XOR with 1 flips the least significant bit)
    ///   - If `p = 4` (binary `100`), sibling is `5` (binary `101`)
    ///   - If `p = 5` (binary `101`), sibling is `4` (binary `100`)
    /// - **Is left child**: `p & 1 == 0` (left children have even positions)
    /// - **Parent position**: `p >> 1` (right-shift divides by 2)
    ///   - Children at positions 4 and 5 both have parent at position 2
    ///
    /// # Algorithm
    ///
    /// We process the tree level-by-level, from leaves (level 0) up to the root:
    ///
    /// ```text
    /// Level 2 (root):        [0]              <- final output
    ///                       /   \
    /// Level 1:           [0]     [1]          <- after 2nd iteration
    ///                   /   \   /   \
    /// Level 0 (leaves): [0] [1] [2] [3]       <- input leaf positions
    /// ```
    ///
    /// At each level, we iterate through known nodes left-to-right. For each node:
    /// 1. Check if its sibling is also known (would be the next node if positions are adjacent)
    /// 2. If sibling is known, use it; otherwise consume next sibling from proof
    /// 3. Order the pair correctly (left child first) and compress to get parent hash
    /// 4. The parent's position is half the child's position (integer division)
    ///
    /// After processing all levels, we should have exactly one node at position 0 (the root).
    ///
    /// # Security Properties
    ///
    /// - **COMPLETENESS**: Returns `None` if any required sibling is missing from proof.
    /// - **SOUNDNESS**: Returns `None` if proof contains extra unused siblings.
    /// - **CANONICAL ORDER**: Siblings must be provided in exact left-to-right, bottom-to-top order.
    fn recompute_root_in_place<C: PseudoCompressionFunction<[D; DIGEST_ELEMS], 2>>(
        &self,
        leaf_nodes: SortedLeafNodes<D, DIGEST_ELEMS>,
        compress: &C,
        tree_depth: usize,
    ) -> Option<[D; DIGEST_ELEMS]> {
        // We alternate between two vectors: one holds the current level's nodes (children),
        // the other accumulates the next level's nodes (parents). After each level, we swap them.
        let mut children = leaf_nodes;
        let mut parents = Vec::new();
        let mut siblings = self.siblings.iter();

        // Process each level from leaves (level 0) up to root (level tree_depth)
        for _level in 0..tree_depth {
            let mut children_iter = children.iter().peekable();

            while let Some((child_position, child_hash)) = children_iter.next() {
                let sibling_position = child_position ^ 1;

                // Get sibling hash: either from known nodes (if next in sorted list) or from proof.
                // When both children are known, the proof omits that sibling since it's redundant.
                let sibling_hash = match children_iter.next_if(|(pos, _)| *pos == sibling_position)
                {
                    Some((_, hash)) => *hash,
                    None => *siblings.next()?,
                };

                // Determine left/right ordering: left child has even position (bit 0 = 0)
                let child_is_left = child_position & 1 == 0;
                let (left_hash, right_hash) = if child_is_left {
                    (*child_hash, sibling_hash)
                } else {
                    (sibling_hash, *child_hash)
                };

                let parent_hash = compress.compress([left_hash, right_hash]);
                let parent_position = child_position >> 1;
                parents.push((parent_position, parent_hash));
            }

            // Swap: parents become children for the next iteration
            core::mem::swap(&mut children, &mut parents);
            parents.clear();
        }

        // Security: all proof siblings must be consumed (no unused data in proof)
        if siblings.next().is_some() {
            return None;
        }

        // Invariant: after `depth` iterations of pos >> 1, all valid leaf positions
        // in [0, 2^depth) converge to exactly one node at position 0.
        let (_, root_hash) = children.into_iter().next().unwrap();
        Some(root_hash)
    }
}

// ============================================================================
// Type Aliases
// ============================================================================

/// Sorted `(index, leaf_digest)` pairs for Merkle tree operations.
type SortedLeafNodes<D, const DIGEST_ELEMS: usize> = Vec<(usize, [D; DIGEST_ELEMS])>;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use p3_symmetric::PseudoCompressionFunction;

    #[derive(Clone)]
    struct TestCompress;

    impl PseudoCompressionFunction<[u64; 1], 2> for TestCompress {
        fn compress(&self, input: [[u64; 1]; 2]) -> [u64; 1] {
            [input[0][0].wrapping_add(input[1][0]).wrapping_mul(31)]
        }
    }

    /// Build test tree (depth 2, 4 leaves):
    /// ```text
    ///        [root]         (level 2)
    ///        /    \
    ///     [h01]  [h23]      (level 1, positions 0, 1)
    ///     /  \    /  \
    ///   [0] [1] [2] [3]     (level 0, positions 0, 1, 2, 3)
    /// ```
    fn build_test_tree() -> ([[u64; 1]; 4], [u64; 1], [u64; 1], [u64; 1]) {
        let leaves = [[0u64; 1], [1], [2], [3]];
        let c = TestCompress;
        let h01 = c.compress([leaves[0], leaves[1]]);
        let h23 = c.compress([leaves[2], leaves[3]]);
        let root = c.compress([h01, h23]);
        (leaves, h01, h23, root)
    }

    /// Test helper: wraps the `BatchProof::recompute_root_in_place` method for standalone testing.
    fn recompute_root_in_place<D: Default + Copy + PartialEq, C, const DIGEST_ELEMS: usize>(
        leaf_nodes: SortedLeafNodes<D, DIGEST_ELEMS>,
        siblings: &[[D; DIGEST_ELEMS]],
        compress: &C,
        tree_depth: usize,
    ) -> Option<[D; DIGEST_ELEMS]>
    where
        C: PseudoCompressionFunction<[D; DIGEST_ELEMS], 2>,
    {
        let batch_proof: BatchProof<(), D, DIGEST_ELEMS, 0> = BatchProof {
            openings: vec![],
            siblings: siblings.to_vec(),
        };
        batch_proof.recompute_root_in_place(leaf_nodes, compress, tree_depth)
    }

    #[test]
    fn recompute_root_cases() {
        let (leaves, _h01, h23, root) = build_test_tree();
        let c = TestCompress;
        const DEPTH: usize = 2;

        // Success: single leaf
        let nodes = vec![(0, leaves[0])];
        let siblings = [leaves[1], h23];
        assert_eq!(
            recompute_root_in_place(nodes, &siblings, &c, DEPTH),
            Some(root)
        );

        // Success: adjacent leaves (same pair, no level-0 sibling needed)
        let nodes = vec![(0, leaves[0]), (1, leaves[1])];
        let siblings = [h23];
        assert_eq!(
            recompute_root_in_place(nodes, &siblings, &c, DEPTH),
            Some(root)
        );

        // Success: non-adjacent leaves (different pairs)
        let nodes = vec![(0, leaves[0]), (2, leaves[2])];
        let siblings = [leaves[1], leaves[3]];
        assert_eq!(
            recompute_root_in_place(nodes, &siblings, &c, DEPTH),
            Some(root)
        );

        // Success: all leaves (no siblings needed)
        let nodes = vec![
            (0, leaves[0]),
            (1, leaves[1]),
            (2, leaves[2]),
            (3, leaves[3]),
        ];
        let siblings: [_; 0] = [];
        assert_eq!(
            recompute_root_in_place(nodes, &siblings, &c, DEPTH),
            Some(root)
        );

        // Success: single-node tree (depth 0)
        let nodes = vec![(0, [42u64])];
        let siblings: [_; 0] = [];
        assert_eq!(
            recompute_root_in_place(nodes, &siblings, &c, 0),
            Some([42u64])
        );

        // Error: missing sibling
        let nodes = vec![(0, leaves[0])];
        let siblings = [leaves[1]]; // Missing h23
        assert!(recompute_root_in_place(nodes, &siblings, &c, DEPTH).is_none());

        // Error: extra sibling
        let nodes = vec![(0, leaves[0])];
        let siblings = [leaves[1], h23, [99]];
        assert!(recompute_root_in_place(nodes, &siblings, &c, DEPTH).is_none());
    }
}
