//! Multi-opening proof structures for Merkle trees.
//!
//! When opening multiple leaves of a Merkle tree, many siblings on the
//! authentication paths are shared. This module provides compact proof
//! representations that store each sibling only once.
//!
//! # Proof Types
//!
//! - [`Opening`]: Per-query row data with optional salt.
//! - [`Proof`]: Unified multi-opening proof with openings and compact siblings.
//!
//! # Usage
//!
//! **Prover** (has Merkle tree, creates compact proof):
//! ```ignore
//! // Collect siblings during tree traversal (see LiftedMerkleTree::open_multi)
//! let siblings: Vec<Hash> = collect_required_siblings(&tree, &indices);
//! let proof = Proof::new(openings, siblings);
//! ```
//!
//! **Verifier** (checks proof against commitment):
//! ```ignore
//! let root = proof.recompute_root(depth, &[(0, leaf0), (2, leaf2)], &compress)?;
//! assert_eq!(root, committed_root);
//! ```

use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use p3_matrix::Dimensions;
use p3_miden_stateful_hasher::StatefulHasher;
use p3_symmetric::{Hash, PseudoCompressionFunction};
use p3_util::log2_strict_usize;

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

/// Unified multi-opening proof for LMCS with optional salt.
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
/// The verifier supplies the indices and calls `verify()`, which:
/// 1. Validates that the proof structure matches the expected dimensions
/// 2. Recomputes the Merkle root from opened rows, salt, and siblings
/// 3. Compares against the commitment
/// 4. Returns references to the opened rows on success
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, [D; DIGEST_ELEMS]: Serialize, [F; SALT_ELEMS]: Serialize",
    deserialize = "F: Deserialize<'de>, [D; DIGEST_ELEMS]: Deserialize<'de>, [F; SALT_ELEMS]: Deserialize<'de>"
))]
pub struct Proof<F, D, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize = 0> {
    /// Openings: `openings[query_idx]` contains rows and salt for that query.
    pub(crate) openings: Vec<Opening<F, SALT_ELEMS>>,
    /// Merkle siblings in canonical order (left-to-right, bottom-to-top).
    pub(crate) siblings: Vec<[D; DIGEST_ELEMS]>,
}

impl<F, D, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    Proof<F, D, DIGEST_ELEMS, SALT_ELEMS>
{
    /// Verify a multi-opening proof against a commitment.
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
    pub fn verify<'a, H, C, const WIDTH: usize>(
        &'a self,
        sponge: &H,
        compress: &C,
        commitment: &Hash<F, D, DIGEST_ELEMS>,
        dimensions: &[Dimensions],
        indices: &[usize],
    ) -> Result<Vec<Vec<&'a [F]>>, LmcsError>
    where
        F: Default + Copy + PartialEq,
        D: Default + Copy + PartialEq,
        H: StatefulHasher<F, [D; DIGEST_ELEMS], State = [D; WIDTH]>,
        C: PseudoCompressionFunction<[D; DIGEST_ELEMS], 2>,
    {
        // Validate proof structure
        if indices.is_empty() || indices.len() != self.openings.len() {
            return Err(LmcsError::WrongBatchSize);
        }

        let final_height = dimensions.last().ok_or(LmcsError::WrongBatchSize)?.height;
        let depth = log2_strict_usize(final_height);

        // Collect (index, digest) pairs, sorted and deduplicated
        let nodes: Vec<(usize, [D; DIGEST_ELEMS])> = {
            let mut nodes: Vec<(usize, [D; DIGEST_ELEMS])> = indices
                .iter()
                .zip(self.openings.iter())
                .map(|(&index, opening)| {
                    if index >= (1 << depth) {
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

            nodes.sort_by_key(|(idx, _)| *idx);

            for ((idx_a, hash_a), (idx_b, hash_b)) in nodes.iter().zip(nodes.iter().skip(1)) {
                if idx_a == idx_b && hash_a != hash_b {
                    return Err(LmcsError::ConflictingLeaf);
                }
            }

            nodes.dedup_by_key(|(idx, _)| *idx);
            nodes
        };

        // Recompute root from known leaves and proof siblings
        let computed_root = recompute_root_in_place(nodes, &self.siblings, compress, depth)
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
}

/// Recompute the Merkle root from known leaf hashes and a sibling proof.
///
/// # Arguments
///
/// - `leaf_nodes`: Vector of `(leaf_position, leaf_hash)` pairs, **sorted by position**.
/// - `proof_siblings`: Sibling hashes in **canonical order** (left-to-right, bottom-to-top).
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
fn recompute_root_in_place<H: Clone, C: PseudoCompressionFunction<H, 2>>(
    leaf_nodes: Vec<(usize, H)>,
    proof_siblings: &[H],
    compress: &C,
    tree_depth: usize,
) -> Option<H> {
    // We alternate between two vectors: one holds the current level's nodes (children),
    // the other accumulates the next level's nodes (parents). After each level, we swap them.
    let mut children = leaf_nodes;
    let mut parents = Vec::new();
    let mut siblings = proof_siblings.iter();

    // Process each level from leaves (level 0) up to root (level tree_depth)
    for _level in 0..tree_depth {
        let mut children_iter = children.iter().peekable();

        while let Some((child_position, child_hash)) = children_iter.next() {
            let sibling_position = child_position ^ 1;

            // Get sibling hash: either from known nodes (if next in sorted list) or from proof.
            // When both children are known, the proof omits that sibling since it's redundant.
            let sibling_hash = match children_iter.next_if(|(pos, _)| *pos == sibling_position) {
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

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
}
