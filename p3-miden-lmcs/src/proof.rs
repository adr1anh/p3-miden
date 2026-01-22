//! Multi-opening proof structures for Merkle trees.
//!
//! When opening multiple leaves of a Merkle tree, many siblings on the
//! authentication paths are shared. This module provides compact proof
//! representations that store each sibling only once.
//!
//! # Proof Types
//!
//! - [`Opening`]: Per-query row data with optional salt.
//! - [`BatchProof`]: Multi-opening proof with openings and compact siblings.
//! - [`Proof`]: Single-opening proof with opening and authentication path.
//!
//! # Usage
//!
//! **Prover** (has Merkle tree, creates compact proof):
//! ```ignore
//! // Use LiftedMerkleTree::open_multi to create a batch proof
//! let proof = tree.open_multi(&indices);
//! ```
//!
//! **Verifier** (checks batch proof against commitment):
//! ```ignore
//! let rows = batch_proof.open(&sponge, &compress, &commitment, &dims, &indices)?;
//! ```
//!
//! **Single-opening** (from extracted paths):
//! ```ignore
//! let proofs = batch_proof.extract_proofs(&sponge, &compress, &dims, &indices)?;
//! let rows = proofs[0].open(&sponge, &compress, &commitment, &dims, 0)?;
//! ```

use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use p3_matrix::Dimensions;
use p3_miden_stateful_hasher::StatefulHasher;
use p3_symmetric::{Hash, PseudoCompressionFunction};
use p3_util::log2_strict_usize;

use alloc::collections::{BTreeMap, BTreeSet};

use crate::{LmcsError, compute_leaf_digest};

// ============================================================================
// Public Types
// ============================================================================

/// Per-query row data with optional salt.
///
/// Groups the opened rows and salt for a single query index.
///
/// # Type Parameters
///
/// - `F`: Field element type.
/// - `SALT_ELEMS`: Number of salt elements. Use `0` for non-hiding, `N` for hiding with N salt elements.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, [F; SALT_ELEMS]: Serialize",
    deserialize = "F: Deserialize<'de>, [F; SALT_ELEMS]: Deserialize<'de>"
))]
pub struct Opening<F, const SALT_ELEMS: usize = 0> {
    /// Opened rows: `rows[matrix_idx]` = row data for that matrix.
    pub(crate) rows: Vec<Vec<F>>,
    /// Salt for this leaf. Zero-sized when `SALT_ELEMS = 0`.
    pub(crate) salt: [F; SALT_ELEMS],
}

/// Batch multi-opening proof for LMCS with optional salt.
///
/// Contains openings (rows + salt per query) and compact Merkle siblings.
/// The indices are **not** stored in the proof—they must be supplied by the
/// verifier during verification. This design ensures the verifier controls
/// which positions are being opened.
///
/// # Type Parameters
///
/// - `F`: Field element type.
/// - `D`: Digest element type.
/// - `DIGEST_ELEMS`: Number of elements in each digest.
/// - `SALT_ELEMS`: Number of salt elements. Use `0` for non-hiding, `N` for hiding with N salt elements.
///
/// # Structure
///
/// - `openings[query_idx]` contains the rows and salt for that query.
/// - `siblings` contains sibling hashes in **canonical order** (see below).
///
/// # Canonical Sibling Order
///
/// Siblings are ordered for level-by-level verification from leaves to root.
/// At each level, pairs are processed left-to-right. For pair `(2i, 2i+1)`:
/// - If both nodes are known from opened leaves: no sibling needed.
/// - If exactly one is known: the next sibling in the vector is the missing one.
///
/// This deterministic ordering makes verification trivial to audit.
///
/// # Verification
///
/// The verifier supplies the indices and calls `open()`, which:
/// 1. Validates that the proof structure matches the expected dimensions
/// 2. Recomputes the Merkle root from opened rows, salt, and siblings
/// 3. Compares against the commitment
/// 4. Returns references to the opened rows on success
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
    /// Validates the proof structure, computes leaf digests from opened rows,
    /// recomputes the Merkle root, and compares against the commitment.
    ///
    /// # Arguments
    ///
    /// - `sponge`: Stateful hasher for computing leaf digests.
    /// - `compress`: 2-to-1 compression function for tree nodes.
    /// - `commitment`: The committed root hash to verify against.
    /// - `dimensions`: Dimensions of committed matrices (width and height).
    /// - `indices`: Leaf indices that were opened.
    ///
    /// # Returns
    ///
    /// On success, returns references to opened rows for each query index.
    /// `result[query_idx][matrix_idx]` is the row slice for that query/matrix.
    ///
    /// # Errors
    ///
    /// - `WrongBatchSize`: Number of indices doesn't match proof's query count,
    ///   or dimensions is empty.
    /// - `IndexOutOfBounds`: An index exceeds the tree height.
    /// - `WrongWidth`: A row's width doesn't match its matrix's dimension.
    /// - `RootMismatch`: Recomputed root doesn't match commitment.
    pub fn open<'a, H, C, const WIDTH: usize>(
        &'a self,
        sponge: &H,
        compress: &C,
        commitment: &Hash<F, D, DIGEST_ELEMS>,
        dimensions: &[Dimensions],
        indices: &[usize],
    ) -> Result<Vec<Vec<&'a [F]>>, LmcsError>
    where
        H: StatefulHasher<F, [D; DIGEST_ELEMS], State = [D; WIDTH]>,
        C: PseudoCompressionFunction<[D; DIGEST_ELEMS], 2>,
    {
        let (nodes, depth) =
            self.compute_sorted_leaf_nodes::<H, WIDTH>(sponge, dimensions, indices)?;

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
    /// Reverses the sibling deduplication to produce complete per-leaf proofs.
    /// When two opened leaves are siblings at some level, the extracted proofs
    /// include the other's computed digest.
    ///
    /// # Arguments
    ///
    /// - `sponge`: Stateful hasher for computing leaf digests.
    /// - `compress`: 2-to-1 compression function for tree nodes.
    /// - `dimensions`: Dimensions of committed matrices (width and height).
    /// - `indices`: Leaf indices that were opened.
    ///
    /// # Returns
    ///
    /// One [`Proof`] per index, in the same order as `indices`.
    ///
    /// # Errors
    ///
    /// - `WrongBatchSize`: Number of indices doesn't match proof's query count,
    ///   or dimensions is empty.
    /// - `IndexOutOfBounds`: An index exceeds the tree height.
    /// - `WrongWidth`: A row's width doesn't match its matrix's dimension.
    pub fn extract_proofs<H, C, const WIDTH: usize>(
        &self,
        sponge: &H,
        compress: &C,
        dimensions: &[Dimensions],
        indices: &[usize],
    ) -> Result<Vec<Proof<F, D, DIGEST_ELEMS, SALT_ELEMS>>, LmcsError>
    where
        H: StatefulHasher<F, [D; DIGEST_ELEMS], State = [D; WIDTH]>,
        C: PseudoCompressionFunction<[D; DIGEST_ELEMS], 2>,
    {
        // Reuse existing helper for leaf computation
        let (leaf_nodes, depth) =
            self.compute_sorted_leaf_nodes::<H, WIDTH>(sponge, dimensions, indices)?;

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
    /// Returns sorted, deduplicated `(index, digest)` pairs and the tree depth.
    fn compute_sorted_leaf_nodes<H, const WIDTH: usize>(
        &self,
        sponge: &H,
        dimensions: &[Dimensions],
        indices: &[usize],
    ) -> Result<(SortedLeafNodes<D, DIGEST_ELEMS>, usize), LmcsError>
    where
        H: StatefulHasher<F, [D; DIGEST_ELEMS], State = [D; WIDTH]>,
    {
        // Validate proof structure
        if indices.is_empty() || indices.len() != self.openings.len() {
            return Err(LmcsError::WrongBatchSize);
        }

        let final_height = dimensions.last().ok_or(LmcsError::WrongBatchSize)?.height;

        // Compute (index, digest) pairs with validation
        let mut nodes: SortedLeafNodes<D, DIGEST_ELEMS> = indices
            .iter()
            .zip(self.openings.iter())
            .map(|(&index, opening)| {
                if index >= final_height {
                    return Err(LmcsError::IndexOutOfBounds);
                }
                let digest = compute_leaf_digest::<F, D, H, WIDTH, DIGEST_ELEMS>(
                    sponge,
                    &opening.rows,
                    dimensions.iter().map(|d| d.width),
                    &opening.salt,
                )?;
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

        let depth = log2_strict_usize(final_height);
        Ok((nodes, depth))
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
                    Some((_, hash)) => hash.clone(),
                    None => siblings.next()?.clone(),
                };

                // Determine left/right ordering: left child has even position (bit 0 = 0)
                let child_is_left = child_position & 1 == 0;
                let (left_hash, right_hash) = if child_is_left {
                    (child_hash.clone(), sibling_hash)
                } else {
                    (sibling_hash, child_hash.clone())
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

/// Single-opening Merkle proof with opening data and authentication path.
///
/// Contains the opening (rows + salt) and siblings (bottom-to-top) for a single leaf.
/// Use `open()` to verify against a commitment and retrieve the opened rows.
///
/// # Type Parameters
///
/// - `F`: Field element type.
/// - `D`: Digest element type.
/// - `DIGEST_ELEMS`: Number of elements in each digest.
/// - `SALT_ELEMS`: Number of salt elements. Use `0` for non-hiding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, [D; DIGEST_ELEMS]: Serialize, [F; SALT_ELEMS]: Serialize",
    deserialize = "F: Deserialize<'de>, [D; DIGEST_ELEMS]: Deserialize<'de>, [F; SALT_ELEMS]: Deserialize<'de>"
))]
pub struct Proof<F, D, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize = 0> {
    /// The opened row data (rows + salt).
    pub opening: Opening<F, SALT_ELEMS>,
    /// Sibling digests from leaf level to root (bottom-to-top).
    pub siblings: Vec<[D; DIGEST_ELEMS]>,
}

impl<F, D, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize> Proof<F, D, DIGEST_ELEMS, SALT_ELEMS>
where
    F: Default + Copy + PartialEq,
    D: Default + Copy + PartialEq,
{
    /// Open this proof against a commitment.
    ///
    /// Computes the leaf digest from the opening data, verifies the
    /// authentication path, and returns the opened rows on success.
    ///
    /// # Arguments
    ///
    /// - `sponge`: Stateful hasher for computing leaf digests.
    /// - `compress`: 2-to-1 compression function.
    /// - `commitment`: The committed root hash to verify against.
    /// - `dimensions`: Dimensions of committed matrices (width and height).
    /// - `index`: Leaf index that was opened.
    ///
    /// # Returns
    ///
    /// On success, returns references to the opened rows.
    /// `result[matrix_idx]` is the row slice for that matrix.
    ///
    /// # Errors
    ///
    /// - `WrongWidth`: A row's width doesn't match its matrix's dimension.
    /// - `RootMismatch`: Recomputed root doesn't match commitment.
    pub fn open<'a, H, C, const WIDTH: usize>(
        &'a self,
        sponge: &H,
        compress: &C,
        commitment: &Hash<F, D, DIGEST_ELEMS>,
        dimensions: &[Dimensions],
        index: usize,
    ) -> Result<Vec<&'a [F]>, LmcsError>
    where
        H: StatefulHasher<F, [D; DIGEST_ELEMS], State = [D; WIDTH]>,
        C: PseudoCompressionFunction<[D; DIGEST_ELEMS], 2>,
    {
        let leaf_digest = compute_leaf_digest::<F, D, H, WIDTH, DIGEST_ELEMS>(
            sponge,
            &self.opening.rows,
            dimensions.iter().map(|d| d.width),
            &self.opening.salt,
        )?;

        let computed_root = self.compute_root(index, leaf_digest, compress);

        if Hash::from(computed_root) != *commitment {
            return Err(LmcsError::RootMismatch);
        }

        Ok(self.opening.rows.iter().map(|r| r.as_slice()).collect())
    }

    /// Compute Merkle root from leaf digest and this path.
    ///
    /// # Arguments
    ///
    /// - `index`: Leaf position in tree (determines left/right at each level).
    /// - `leaf_digest`: Hash of the leaf data.
    /// - `compress`: 2-to-1 compression function.
    ///
    /// # Returns
    ///
    /// The computed root digest.
    pub fn compute_root<C>(
        &self,
        index: usize,
        leaf_digest: [D; DIGEST_ELEMS],
        compress: &C,
    ) -> [D; DIGEST_ELEMS]
    where
        C: PseudoCompressionFunction<[D; DIGEST_ELEMS], 2>,
    {
        let mut current = leaf_digest;
        let mut pos = index;

        for sibling in &self.siblings {
            let is_left = pos & 1 == 0;
            current = if is_left {
                compress.compress([current, sibling.clone()])
            } else {
                compress.compress([sibling.clone(), current])
            };
            pos >>= 1;
        }

        current
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
        // Create a dummy batch proof with the given siblings
        let batch_proof: BatchProof<(), D, DIGEST_ELEMS, 0> = BatchProof {
            openings: vec![],
            siblings: siblings.to_vec(),
        };
        batch_proof.recompute_root_in_place(leaf_nodes, compress, tree_depth)
    }

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

    const DEPTH: usize = 2;

    #[test]
    fn recompute_root_single_leaf() {
        let (leaf_hashes, _h01, h23, root) = build_test_tree();
        let c = TestCompress;

        // Open leaf 0 only
        let nodes = vec![(0, leaf_hashes[0])];

        // Canonical sibling order (left-to-right, bottom-to-top):
        // Level 0: pair (0,1) - have 0, need 1
        // Level 1: pair (0,1) - have h01, need h23
        let siblings = [leaf_hashes[1], h23];

        let computed = recompute_root_in_place(nodes, &siblings, &c, DEPTH).unwrap();
        assert_eq!(computed, root);
    }

    #[test]
    fn recompute_root_all_leaves_no_siblings() {
        let (leaf_hashes, _h01, _h23, root) = build_test_tree();
        let c = TestCompress;

        // All leaves known - no siblings needed
        let nodes = vec![
            (0, leaf_hashes[0]),
            (1, leaf_hashes[1]),
            (2, leaf_hashes[2]),
            (3, leaf_hashes[3]),
        ];
        let siblings: [_; 0] = [];

        let computed = recompute_root_in_place(nodes, &siblings, &c, DEPTH).unwrap();
        assert_eq!(computed, root);
    }

    #[test]
    fn recompute_root_single_node_tree() {
        let c = TestCompress;

        // Depth 0 tree: single leaf at index 0 IS the root
        let nodes = vec![(0, [42u64])];
        let siblings: [_; 0] = [];

        let result = recompute_root_in_place(nodes, &siblings, &c, 0);
        assert_eq!(result, Some([42u64]));
    }

    #[test]
    fn recompute_root_adjacent_leaves() {
        let (leaf_hashes, _h01, h23, root) = build_test_tree();
        let c = TestCompress;

        // Open leaves 0 and 1 (same pair)
        let nodes = vec![(0, leaf_hashes[0]), (1, leaf_hashes[1])];

        // Level 0: pair (0,1) both known - no sibling needed
        // Level 1: pair (0,1) - have h01, need h23
        let siblings = [h23];

        let computed = recompute_root_in_place(nodes, &siblings, &c, DEPTH).unwrap();
        assert_eq!(computed, root);
    }

    #[test]
    fn recompute_root_non_adjacent_leaves() {
        let (leaf_hashes, _h01, _h23, root) = build_test_tree();
        let c = TestCompress;

        // Open leaves 0 and 2 (different pairs)
        let nodes = vec![(0, leaf_hashes[0]), (2, leaf_hashes[2])];

        // Level 0: pair (0,1) - have 0, need 1; pair (2,3) - have 2, need 3
        // Level 1: pair (0,1) - both computed (h01, h23), no sibling needed
        let siblings = [leaf_hashes[1], leaf_hashes[3]];

        let computed = recompute_root_in_place(nodes, &siblings, &c, DEPTH).unwrap();
        assert_eq!(computed, root);
    }

    #[test]
    fn recompute_root_missing_sibling_fails() {
        let (leaf_hashes, _h01, _h23, _root) = build_test_tree();
        let c = TestCompress;

        let nodes = vec![(0, leaf_hashes[0])];
        let siblings = [leaf_hashes[1]]; // Missing h23

        let result = recompute_root_in_place(nodes, &siblings, &c, DEPTH);
        assert!(result.is_none());
    }

    #[test]
    fn recompute_root_extra_sibling_fails() {
        let (leaf_hashes, _h01, h23, _root) = build_test_tree();
        let c = TestCompress;

        let nodes = vec![(0, leaf_hashes[0])];
        let siblings = [leaf_hashes[1], h23, [99]]; // Extra sibling

        let result = recompute_root_in_place(nodes, &siblings, &c, DEPTH);
        assert!(result.is_none());
    }

    // ========================================================================
    // Proof::compute_root tests
    // ========================================================================

    /// Helper to create a dummy opening with no rows or salt.
    fn dummy_opening() -> Opening<u64, 0> {
        Opening {
            rows: vec![],
            salt: [],
        }
    }

    #[test]
    fn proof_compute_root() {
        // Tree:       [root]
        //            /      \
        //        [h01]      [h23]
        //        /    \     /    \
        //      [0]   [1]  [2]   [3]
        let (leaves, _h01, h23, root) = build_test_tree();
        let compress = TestCompress;

        // Path for leaf 0: siblings are [leaves[1], h23]
        let proof: Proof<u64, u64, 1, 0> = Proof {
            opening: dummy_opening(),
            siblings: vec![leaves[1], h23],
        };
        assert_eq!(proof.compute_root(0, leaves[0], &compress), root);

        // Path for leaf 1: siblings are [leaves[0], h23]
        let proof: Proof<u64, u64, 1, 0> = Proof {
            opening: dummy_opening(),
            siblings: vec![leaves[0], h23],
        };
        assert_eq!(proof.compute_root(1, leaves[1], &compress), root);

        // Path for leaf 2: siblings are [leaves[3], h01]
        let (_, h01, _, _) = build_test_tree();
        let proof: Proof<u64, u64, 1, 0> = Proof {
            opening: dummy_opening(),
            siblings: vec![leaves[3], h01],
        };
        assert_eq!(proof.compute_root(2, leaves[2], &compress), root);

        // Path for leaf 3: siblings are [leaves[2], h01]
        let proof: Proof<u64, u64, 1, 0> = Proof {
            opening: dummy_opening(),
            siblings: vec![leaves[2], h01],
        };
        assert_eq!(proof.compute_root(3, leaves[3], &compress), root);
    }

    #[test]
    fn proof_compute_root_depth_3() {
        // Depth 3 tree (8 leaves):
        //                 [root]
        //               /        \
        //           [h0123]    [h4567]
        //           /    \      /    \
        //       [h01]  [h23] [h45] [h67]
        //       / \    / \    / \   / \
        //      0   1  2   3  4   5 6   7
        let compress = TestCompress;
        let leaves: [[u64; 1]; 8] = [[0], [1], [2], [3], [4], [5], [6], [7]];

        let h01 = compress.compress([leaves[0], leaves[1]]);
        let h23 = compress.compress([leaves[2], leaves[3]]);
        let h45 = compress.compress([leaves[4], leaves[5]]);
        let h67 = compress.compress([leaves[6], leaves[7]]);
        let h0123 = compress.compress([h01, h23]);
        let h4567 = compress.compress([h45, h67]);
        let root = compress.compress([h0123, h4567]);

        // Path for leaf 5: siblings are [leaves[4], h67, h0123]
        let proof: Proof<u64, u64, 1, 0> = Proof {
            opening: dummy_opening(),
            siblings: vec![leaves[4], h67, h0123],
        };
        assert_eq!(proof.compute_root(5, leaves[5], &compress), root);
    }
}
