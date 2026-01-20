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
//! - [`CompactProof`]: Low-level compact Merkle siblings (deduplicated).
//!
//! # Usage
//!
//! **Prover** (has Merkle tree, creates compact proof):
//! ```ignore
//! let paths = [(0, leaf0, path0), (2, leaf2, path2)];
//! let proof = CompactProof::from_paths(depth, paths);
//! ```
//!
//! **Verifier** (checks proof against commitment):
//! ```ignore
//! let root = proof.recompute_root(depth, [(0, leaf0), (2, leaf2)], &compress)?;
//! assert_eq!(root, committed_root);
//! ```

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use p3_symmetric::PseudoCompressionFunction;
use thiserror::Error;

// ============================================================================
// High-Level Proof Types
// ============================================================================

/// Per-query row data with optional salt.
///
/// Groups the opened rows and salt for a single query index.
///
/// # Type Parameters
///
/// - `F`: Field element type.
/// - `Salt`: Salt type. Use `()` for non-hiding, `[F; N]` for hiding with N salt elements.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, Salt: Serialize",
    deserialize = "F: Deserialize<'de>, Salt: Deserialize<'de>"
))]
pub struct Opening<F, Salt = ()> {
    /// Opened rows: `rows[matrix_idx]` = row data for that matrix.
    rows: Vec<Vec<F>>,
    /// Salt for this leaf. Zero-sized when `Salt = ()`.
    salt: Salt,
}

impl<F, Salt> Opening<F, Salt> {
    /// Create a new opening from rows and salt.
    #[inline]
    pub fn new(rows: Vec<Vec<F>>, salt: Salt) -> Self {
        Self { rows, salt }
    }

    /// Returns the opened rows (one per committed matrix).
    #[inline]
    pub fn rows(&self) -> &[Vec<F>] {
        &self.rows
    }

    /// Returns the salt for this opening.
    #[inline]
    pub fn salt(&self) -> &Salt {
        &self.salt
    }
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
/// - `Salt`: Salt type. Use `()` for non-hiding, `[F; N]` for hiding with N salt elements.
///
/// # Structure
///
/// - `openings[query_idx]` contains the rows and salt for that query.
/// - `siblings` is a [`CompactProof`] containing deduplicated Merkle siblings.
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
    serialize = "F: Serialize, [D; DIGEST_ELEMS]: Serialize, Salt: Serialize",
    deserialize = "F: Deserialize<'de>, [D; DIGEST_ELEMS]: Deserialize<'de>, Salt: Deserialize<'de>"
))]
pub struct Proof<F, D, const DIGEST_ELEMS: usize, Salt = ()> {
    /// Openings: `openings[query_idx]` contains rows and salt for that query.
    openings: Vec<Opening<F, Salt>>,
    /// Compact Merkle siblings (deduplicated).
    siblings: CompactProof<[D; DIGEST_ELEMS]>,
}

impl<F, D, const DIGEST_ELEMS: usize, Salt> Proof<F, D, DIGEST_ELEMS, Salt> {
    /// Create a new proof from openings and compact siblings.
    #[inline]
    pub fn new(openings: Vec<Opening<F, Salt>>, siblings: CompactProof<[D; DIGEST_ELEMS]>) -> Self {
        Self { openings, siblings }
    }

    /// Returns the openings (one per query).
    #[inline]
    pub fn openings(&self) -> &[Opening<F, Salt>] {
        &self.openings
    }

    /// Returns the number of opened indices.
    #[inline]
    pub fn num_queries(&self) -> usize {
        self.openings.len()
    }

    /// Recompute the Merkle root from opened leaves and sibling proof.
    ///
    /// Delegates to [`CompactProof::recompute_root`].
    #[inline]
    pub fn recompute_root<C>(
        &self,
        depth: usize,
        leaves: &[(usize, [D; DIGEST_ELEMS])],
        compress: &C,
    ) -> Result<[D; DIGEST_ELEMS], CompactProofError>
    where
        [D; DIGEST_ELEMS]: Clone,
        C: PseudoCompressionFunction<[D; DIGEST_ELEMS], 2>,
    {
        self.siblings.recompute_root(depth, leaves, compress)
    }
}

// ============================================================================
// Low-Level Proof Structures
// ============================================================================

/// 1-based heap index for binary tree nodes.
///
/// Uses the standard binary heap indexing where:
/// - Root is at index 1
/// - Left child of node i is at 2*i
/// - Right child of node i is at 2*i + 1
/// - Parent of node i is at i/2
/// - Sibling of node i is at i XOR 1
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeIndex(u64);

impl NodeIndex {
    /// The root node index.
    pub const ROOT: Self = Self(1);

    /// Create a NodeIndex for a leaf at the given position (0-based) in a tree of given depth.
    ///
    /// Leaves in a tree of depth `d` occupy indices `2^d` to `2^(d+1) - 1`.
    #[inline]
    pub const fn from_leaf(leaf_idx: usize, depth: usize) -> Self {
        Self((1u64 << depth) + leaf_idx as u64)
    }

    #[inline]
    pub const fn sibling(self) -> Self {
        Self(self.0 ^ 1)
    }

    #[inline]
    pub const fn parent(self) -> Self {
        Self(self.0 >> 1)
    }

    #[inline]
    pub const fn left_child(self) -> Self {
        Self(self.0 << 1)
    }

    #[inline]
    pub const fn right_child(self) -> Self {
        Self((self.0 << 1) | 1)
    }
}

/// A leaf with its authentication path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedPath<H> {
    /// The leaf index.
    pub index: usize,
    /// The leaf hash.
    pub leaf: H,
    /// Authentication path (sibling hashes from leaf to root).
    pub path: Vec<H>,
}

/// Compact multi-opening proof with labeled siblings.
///
/// Each sibling hash is keyed by its NodeIndex, guaranteeing uniqueness
/// and providing natural ordering for deterministic serialization.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "H: Serialize", deserialize = "H: Deserialize<'de>"))]
pub struct CompactProof<H>(pub(crate) BTreeMap<NodeIndex, H>);

impl<H: Clone> CompactProof<H> {
    /// Create an empty proof.
    #[inline]
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    /// Returns the number of sibling hashes.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the proof contains no sibling hashes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Create a compact proof from authentication paths.
    ///
    /// Paths can be provided in any order. Only the minimal set of required
    /// siblings is included (siblings that can be computed from opened leaves
    /// are excluded).
    ///
    /// # Arguments
    /// - `depth`: Tree depth (number of levels from leaves to root).
    /// - `paths`: Iterator of [`IndexedPath`] values.
    ///
    /// # Panics (debug builds only)
    /// - If any path length doesn't match `depth`.
    /// - If paths contain inconsistent sibling hashes.
    ///
    /// # Note
    /// This is a prover-side function with trusted input from the Merkle tree.
    /// Debug assertions verify consistency but are elided in release builds.
    pub fn from_paths<I>(depth: usize, paths: I) -> Self
    where
        I: IntoIterator<Item = IndexedPath<H>>,
        H: PartialEq,
    {
        use alloc::collections::BTreeSet;

        let paths_vec: Vec<_> = paths.into_iter().collect();

        // Compute which siblings are required (not derivable from opened leaves).
        // Walk up from leaves, tracking known nodes. Siblings of known nodes that
        // aren't themselves known are required.
        let mut known: BTreeSet<NodeIndex> = paths_vec
            .iter()
            .map(|p| NodeIndex::from_leaf(p.index, depth))
            .collect();
        let mut required: BTreeSet<NodeIndex> = BTreeSet::new();

        while !known.is_empty() && !known.contains(&NodeIndex::ROOT) {
            let mut parents = BTreeSet::new();
            for &node in &known {
                let sibling = node.sibling();
                if !known.contains(&sibling) {
                    required.insert(sibling);
                }
                parents.insert(node.parent());
            }
            known = parents;
        }

        // Collect siblings from paths that are in the required set.
        let mut siblings: BTreeMap<NodeIndex, H> = BTreeMap::new();

        for IndexedPath { index, path, .. } in &paths_vec {
            debug_assert_eq!(path.len(), depth, "path length must equal depth");

            let mut current = NodeIndex::from_leaf(*index, depth);
            for sibling_hash in path {
                let sibling = current.sibling();
                if required.contains(&sibling) {
                    debug_assert!(
                        !siblings.contains_key(&sibling) || &siblings[&sibling] == sibling_hash,
                        "inconsistent sibling hash"
                    );
                    siblings
                        .entry(sibling)
                        .or_insert_with(|| sibling_hash.clone());
                }
                current = current.parent();
            }
        }

        Self(siblings)
    }

    /// Recompute the Merkle root from opened leaves and sibling proof.
    ///
    /// # Security Model
    ///
    /// ⚠ **CRITICAL**: This function computes a root hash from the provided proof.
    /// A malicious proof can produce ANY hash. The caller MUST compare the returned
    /// root against a trusted commitment to detect tampering.
    ///
    /// # Arguments
    /// - `depth`: Tree depth (number of levels from leaves to root).
    /// - `leaves`: Slice of `(leaf_index, leaf_hash)` pairs (any order).
    /// - `compress`: 2-to-1 compression function.
    ///
    /// # Errors
    /// - `MissingSibling` - Proof is missing required sibling hashes
    /// - `UnusedSibling` - Proof contains sibling hashes that weren't needed
    /// - `DuplicateLeaf` - Same leaf index provided multiple times
    pub fn recompute_root<C>(
        &self,
        depth: usize,
        leaves: &[(usize, H)],
        compress: &C,
    ) -> Result<H, CompactProofError>
    where
        C: PseudoCompressionFunction<H, 2>,
    {
        let mut tree = PartialTree::new(depth, leaves, self)?;
        Ok(tree.get(NodeIndex::ROOT, compress))
    }

    /// Expand the proof to full authentication paths.
    ///
    /// Reconstructs the complete authentication path for each leaf.
    ///
    /// # Arguments
    /// - `depth`: Tree depth.
    /// - `leaves`: Slice of `(leaf_index, leaf_hash)` pairs.
    /// - `compress`: 2-to-1 compression function.
    ///
    /// # Returns
    /// Vector of `IndexedPath` in the same order as input `leaves`.
    ///
    /// # Errors
    /// - `MissingSibling` - Proof is missing required sibling hashes
    /// - `UnusedSibling` - Proof contains sibling hashes that weren't needed
    /// - `DuplicateLeaf` - Same leaf index provided multiple times
    pub fn to_paths<C>(
        &self,
        depth: usize,
        leaves: &[(usize, H)],
        compress: &C,
    ) -> Result<Vec<IndexedPath<H>>, CompactProofError>
    where
        C: PseudoCompressionFunction<H, 2>,
    {
        let mut tree = PartialTree::new(depth, leaves, self)?;
        Ok(leaves
            .iter()
            .map(|(idx, _)| tree.path(*idx, compress))
            .collect())
    }
}

/// A validated partial Merkle tree built from leaves and proof siblings.
///
/// # Security
///
/// Construction is the trust boundary: `new()` validates that the proof
/// contains exactly the required siblings (no missing, no extras).
/// After construction, `root()` and `path()` operate on trusted state.
struct PartialTree<H> {
    depth: usize,
    nodes: BTreeMap<NodeIndex, H>,
}

impl<H: Clone> PartialTree<H> {
    /// Validate proof and build partial tree.
    ///
    /// # Security
    ///
    /// This is the trust boundary. Validates that:
    /// - No duplicate leaf indices
    /// - Proof contains exactly the siblings needed to compute the root
    /// - No missing siblings, no unused siblings
    fn new(
        depth: usize,
        leaves: &[(usize, H)],
        proof: &CompactProof<H>,
    ) -> Result<Self, CompactProofError> {
        // Build leaf map, checking for duplicates.
        let mut nodes: BTreeMap<NodeIndex, H> = leaves
            .iter()
            .map(|(idx, hash)| (NodeIndex::from_leaf(*idx, depth), hash.clone()))
            .collect();

        if nodes.len() != leaves.len() {
            return Err(CompactProofError::DuplicateLeaf);
        }

        // Validate proof by traversing tree and counting sibling usage.
        let used_count = Self::validate_subtree(NodeIndex::ROOT, 0, depth, &nodes, proof)?;

        // All proof siblings must have been used.
        if used_count != proof.len() {
            return Err(CompactProofError::UnusedSibling);
        }

        // Merge validated proof siblings into nodes.
        nodes.extend(proof.0.iter().map(|(k, v)| (*k, v.clone())));

        Ok(Self { depth, nodes })
    }

    /// Validate that a subtree can be computed from leaves and proof.
    ///
    /// Returns the count of proof siblings used in this subtree.
    /// Returns `Err(MissingSibling)` if any required node is missing.
    fn validate_subtree(
        idx: NodeIndex,
        current_depth: usize,
        max_depth: usize,
        leaves: &BTreeMap<NodeIndex, H>,
        proof: &CompactProof<H>,
    ) -> Result<usize, CompactProofError> {
        if leaves.contains_key(&idx) {
            return Ok(0);
        }
        if proof.0.contains_key(&idx) {
            return Ok(1);
        }
        if current_depth >= max_depth {
            return Err(CompactProofError::MissingSibling);
        }

        let left = Self::validate_subtree(
            idx.left_child(),
            current_depth + 1,
            max_depth,
            leaves,
            proof,
        )?;
        let right = Self::validate_subtree(
            idx.right_child(),
            current_depth + 1,
            max_depth,
            leaves,
            proof,
        )?;
        Ok(left + right)
    }

    /// Get or compute a node's hash (memoized).
    ///
    /// Tree is already validated, so this cannot fail.
    fn get<C: PseudoCompressionFunction<H, 2>>(&mut self, idx: NodeIndex, compress: &C) -> H {
        if let Some(h) = self.nodes.get(&idx) {
            return h.clone();
        }

        // Compute from children.
        let left = self.get(idx.left_child(), compress);
        let right = self.get(idx.right_child(), compress);
        let h = compress.compress([left, right]);
        self.nodes.insert(idx, h.clone());
        h
    }

    /// Build the authentication path for a leaf.
    ///
    /// Tree is already validated, so this cannot fail.
    fn path<C: PseudoCompressionFunction<H, 2>>(
        &mut self,
        leaf_idx: usize,
        compress: &C,
    ) -> IndexedPath<H> {
        let leaf_node = NodeIndex::from_leaf(leaf_idx, self.depth);
        let leaf = self.nodes.get(&leaf_node).unwrap().clone();

        let mut path = Vec::with_capacity(self.depth);
        let mut current = leaf_node;

        while current != NodeIndex::ROOT {
            path.push(self.get(current.sibling(), compress));
            current = current.parent();
        }

        IndexedPath {
            index: leaf_idx,
            leaf,
            path,
        }
    }
}

/// Error type for multi-opening proof operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CompactProofError {
    /// Proof is missing required sibling hashes.
    #[error("proof is missing required sibling hashes")]
    MissingSibling,
    /// Proof contains sibling hashes that weren't needed.
    #[error("proof contains unused sibling hashes")]
    UnusedSibling,
    /// Duplicate leaf index provided.
    #[error("duplicate leaf index provided")]
    DuplicateLeaf,
}

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
    ///        [root]         (index 1)
    ///        /    \
    ///     [h01]  [h23]      (indices 2, 3)
    ///     /  \    /  \
    ///   [0] [1] [2] [3]     (indices 4, 5, 6, 7)
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

    /// Node index for leaf `i` in test tree.
    const fn leaf(i: usize) -> NodeIndex {
        NodeIndex::from_leaf(i, DEPTH)
    }

    #[test]
    fn roundtrip() {
        let (leaves, h01, h23, root) = build_test_tree();
        let c = TestCompress;

        let original_paths = [
            IndexedPath {
                index: 0,
                leaf: leaves[0],
                path: vec![leaves[1], h23],
            },
            IndexedPath {
                index: 2,
                leaf: leaves[2],
                path: vec![leaves[3], h01],
            },
        ];
        let proof = CompactProof::from_paths(DEPTH, original_paths.clone());

        let expanded = proof
            .to_paths(DEPTH, &[(0, leaves[0]), (2, leaves[2])], &c)
            .unwrap();
        assert_eq!(expanded[0].path, original_paths[0].path);
        assert_eq!(expanded[1].path, original_paths[1].path);

        let computed_root = proof
            .recompute_root(DEPTH, &[(0, leaves[0]), (2, leaves[2])], &c)
            .unwrap();
        assert_eq!(computed_root, root);
    }

    #[test]
    fn all_leaves_empty_proof() {
        let (leaves, _h01, _h23, root) = build_test_tree();
        let c = TestCompress;

        let proof = CompactProof::new();
        let computed = proof
            .recompute_root(
                DEPTH,
                &[
                    (0, leaves[0]),
                    (1, leaves[1]),
                    (2, leaves[2]),
                    (3, leaves[3]),
                ],
                &c,
            )
            .unwrap();

        assert_eq!(computed, root);
    }

    #[test]
    fn single_leaf_tree() {
        let c = TestCompress;
        let proof = CompactProof::<[u64; 1]>::new();
        let result = proof.recompute_root(0, &[(0, [42u64])], &c);
        assert_eq!(result, Ok([42u64]));
    }

    #[test]
    fn rejects_missing_sibling() {
        let (leaves, _h01, _h23, _root) = build_test_tree();
        let c = TestCompress;

        // Leaf 0 needs siblings at nodes 5 (leaf 1) and 3 (h23), but we only provide leaf 1.
        let proof = CompactProof([(leaf(1), leaves[1])].into_iter().collect());
        let result = proof.recompute_root(DEPTH, &[(0, leaves[0])], &c);

        assert_eq!(result, Err(CompactProofError::MissingSibling));
    }

    #[test]
    fn rejects_unused_sibling() {
        let (leaves, _h01, h23, _root) = build_test_tree();
        let c = TestCompress;

        // Leaf 0 needs siblings at leaf(1) and leaf(2).parent(), but we also include leaf(3).
        let proof = CompactProof(
            [
                (leaf(1), leaves[1]),
                (leaf(2).parent(), h23),
                (leaf(3), leaves[3]),
            ]
            .into_iter()
            .collect(),
        );
        let result = proof.recompute_root(DEPTH, &[(0, leaves[0])], &c);

        assert_eq!(result, Err(CompactProofError::UnusedSibling));
    }

    #[test]
    fn rejects_duplicate_leaf() {
        let (leaves, _h01, h23, _root) = build_test_tree();
        let c = TestCompress;

        let proof = CompactProof(
            [(leaf(1), leaves[1]), (leaf(2).parent(), h23)]
                .into_iter()
                .collect(),
        );
        let result = proof.recompute_root(DEPTH, &[(0, leaves[0]), (0, [99])], &c);

        assert_eq!(result, Err(CompactProofError::DuplicateLeaf));
    }
}
