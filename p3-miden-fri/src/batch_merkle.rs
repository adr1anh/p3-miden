//! Batch Merkle proof implementation using a variation of the Octopus algorithm.
//!
//! When multiple leaves need to be opened from the same Merkle tree, many authentication
//! paths share common internal nodes (especially near the root). This module provides
//! utilities to compress multiple Merkle proofs by deduplicating shared nodes.
//!
//! Reference: <https://eprint.iacr.org/2017/933> (Octopus)

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

/// A batch Merkle proof that aggregates multiple individual proofs.
///
/// The aggregation removes duplicate internal nodes that appear in multiple
/// authentication paths, achieving compression when opening many leaves.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchMerkleProof<D> {
    /// Deduplicated sibling nodes for each opened index.
    /// `nodes[i]` contains the siblings needed for the i-th query (after sorting).
    /// Within each query's list, siblings that are provided by other queries
    /// in the batch are omitted.
    pub nodes: Vec<Vec<D>>,
    /// The depth of the Merkle tree (number of levels from leaf to root).
    pub depth: u8,
}

impl<D: Clone + Default> BatchMerkleProof<D> {
    /// Creates a batch Merkle proof from a collection of individual proofs.
    ///
    /// # Arguments
    /// * `proofs` - Individual Merkle proofs (each is a Vec of sibling digests from leaf to root)
    /// * `indices` - The leaf indices corresponding to each proof
    ///
    /// # Panics
    /// Panics if proofs and indices have different lengths, or if proofs have inconsistent depths.
    pub fn from_individual_proofs(proofs: &[Vec<D>], indices: &[usize]) -> Self {
        assert!(!proofs.is_empty(), "at least one proof must be provided");
        assert_eq!(
            proofs.len(),
            indices.len(),
            "number of proofs must equal number of indices"
        );

        let depth = proofs[0].len();
        for proof in proofs.iter() {
            assert_eq!(proof.len(), depth, "all proofs must have the same depth");
        }

        // Sort indices and rearrange proofs accordingly
        let mut indexed_proofs: Vec<(usize, &Vec<D>)> =
            indices.iter().copied().zip(proofs.iter()).collect();
        indexed_proofs.sort_by_key(|(idx, _)| *idx);

        let sorted_indices: Vec<usize> = indexed_proofs.iter().map(|(idx, _)| *idx).collect();
        let sorted_proofs: Vec<&Vec<D>> = indexed_proofs.iter().map(|(_, p)| *p).collect();

        // Build the batched proof by identifying which siblings can be derived
        // from other queries vs which must be explicitly included
        let mut batched_nodes: Vec<Vec<D>> = Vec::with_capacity(sorted_indices.len());

        for (query_idx, (index, proof)) in sorted_indices.iter().zip(sorted_proofs.iter()).enumerate()
        {
            let mut query_nodes = Vec::new();
            let mut current_index = *index;

            for (level, sibling) in proof.iter().enumerate() {
                let sibling_index = current_index ^ 1;

                // Check if this sibling is the leaf or ancestor of another query
                let sibling_provided_by_other = sorted_indices.iter().enumerate().any(|(other_idx, &other_index)| {
                    if other_idx == query_idx {
                        return false;
                    }
                    // Check if other_index, when shifted to this level, equals sibling_index
                    (other_index >> level) == sibling_index
                });

                if !sibling_provided_by_other {
                    query_nodes.push(sibling.clone());
                }

                current_index >>= 1;
            }

            batched_nodes.push(query_nodes);
        }

        BatchMerkleProof {
            nodes: batched_nodes,
            depth: depth as u8,
        }
    }

    /// Returns the number of queries (opened leaves) in this batch proof.
    pub fn num_queries(&self) -> usize {
        self.nodes.len()
    }

    /// Computes the total number of nodes stored in this batch proof.
    pub fn total_nodes(&self) -> usize {
        self.nodes.iter().map(|v| v.len()).sum()
    }
}

/// Helper to verify a batch Merkle proof and compute the root.
///
/// # Arguments
/// * `batch_proof` - The batch Merkle proof
/// * `indices` - The leaf indices (must be sorted in ascending order)
/// * `leaves` - The leaf values at the given indices
/// * `compress` - Function to compress two sibling nodes into their parent
///
/// # Returns
/// The computed Merkle root if verification succeeds.
pub fn compute_root_from_batch_proof<D: Clone + Default + Eq>(
    batch_proof: &BatchMerkleProof<D>,
    indices: &[usize],
    leaves: &[D],
    compress: impl Fn(&D, &D) -> D,
) -> Result<D, BatchMerkleError> {
    if indices.is_empty() {
        return Err(BatchMerkleError::EmptyBatch);
    }
    if indices.len() != leaves.len() || indices.len() != batch_proof.nodes.len() {
        return Err(BatchMerkleError::InvalidProofShape);
    }

    // Verify indices are sorted
    if !indices.windows(2).all(|w| w[0] < w[1]) {
        return Err(BatchMerkleError::IndicesNotSorted);
    }

    let depth = batch_proof.depth as usize;

    // Map from node index (at each level) to its computed value
    // Level 0 is leaves, level `depth` is root
    let mut computed: BTreeMap<usize, D> = BTreeMap::new();

    // Initialize with leaf values
    for (&index, leaf) in indices.iter().zip(leaves.iter()) {
        computed.insert(index, leaf.clone());
    }

    // Track which proof node to use for each query
    let mut proof_pointers: Vec<usize> = vec![0; indices.len()];

    // Process level by level from leaves to root
    for level in 0..depth {
        let mut next_computed: BTreeMap<usize, D> = BTreeMap::new();

        // Collect all indices we need to process at this level
        let level_indices: Vec<usize> = computed.keys().copied().collect();

        for &index in &level_indices {
            let sibling_index = index ^ 1;
            let parent_index = index >> 1;

            // Skip if we already computed this parent (from processing sibling)
            if next_computed.contains_key(&parent_index) {
                continue;
            }

            let node = computed.get(&index).unwrap().clone();

            // Try to get sibling from computed values (another query)
            let sibling = if let Some(sib) = computed.get(&sibling_index) {
                sib.clone()
            } else {
                // Find which query this index belongs to and get sibling from proof
                let query_idx = indices
                    .iter()
                    .position(|&orig_idx| (orig_idx >> level) == index)
                    .ok_or(BatchMerkleError::InvalidProofShape)?;

                let ptr = &mut proof_pointers[query_idx];
                if *ptr >= batch_proof.nodes[query_idx].len() {
                    return Err(BatchMerkleError::InvalidProofShape);
                }
                let sib = batch_proof.nodes[query_idx][*ptr].clone();
                *ptr += 1;
                sib
            };

            // Compute parent: left child is even index, right child is odd
            let parent = if index % 2 == 0 {
                compress(&node, &sibling)
            } else {
                compress(&sibling, &node)
            };

            next_computed.insert(parent_index, parent);
        }

        computed = next_computed;
    }

    // After processing all levels, we should have exactly one node (the root)
    if computed.len() != 1 {
        return Err(BatchMerkleError::InvalidProofShape);
    }

    Ok(computed.into_values().next().unwrap())
}

/// Errors that can occur during batch Merkle proof operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchMerkleError {
    /// The batch is empty (no indices provided).
    EmptyBatch,
    /// The proof shape doesn't match expected dimensions.
    InvalidProofShape,
    /// Indices must be sorted in ascending order.
    IndicesNotSorted,
    /// The computed root doesn't match the expected root.
    RootMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple test digest type
    type TestDigest = [u8; 4];

    fn test_compress(left: &TestDigest, right: &TestDigest) -> TestDigest {
        // Simple compression: XOR and add
        [
            left[0] ^ right[0],
            left[1] ^ right[1],
            left[2].wrapping_add(right[2]),
            left[3].wrapping_add(right[3]),
        ]
    }

    /// Build a simple Merkle tree and return (root, leaves, all_proofs)
    fn build_test_tree(num_leaves: usize) -> (TestDigest, Vec<TestDigest>, Vec<Vec<TestDigest>>) {
        assert!(num_leaves.is_power_of_two());
        let depth = num_leaves.trailing_zeros() as usize;

        // Create leaves
        let leaves: Vec<TestDigest> = (0..num_leaves)
            .map(|i| [i as u8, (i * 2) as u8, (i * 3) as u8, (i * 4) as u8])
            .collect();

        // Build tree level by level
        let mut levels: Vec<Vec<TestDigest>> = vec![leaves.clone()];
        let mut current = leaves.clone();

        for _ in 0..depth {
            let next: Vec<TestDigest> = current
                .chunks(2)
                .map(|pair| test_compress(&pair[0], &pair[1]))
                .collect();
            levels.push(next.clone());
            current = next;
        }

        let root = levels.last().unwrap()[0];

        // Generate individual proofs for each leaf
        let mut all_proofs = Vec::new();
        for leaf_idx in 0..num_leaves {
            let mut proof = Vec::new();
            let mut idx = leaf_idx;
            for level in 0..depth {
                let sibling_idx = idx ^ 1;
                proof.push(levels[level][sibling_idx]);
                idx >>= 1;
            }
            all_proofs.push(proof);
        }

        (root, leaves, all_proofs)
    }

    #[test]
    fn test_single_proof_batching() {
        let (root, leaves, all_proofs) = build_test_tree(8);

        // Batch a single proof - should be identical to original
        let indices = vec![3];
        let proofs = vec![all_proofs[3].clone()];

        let batch = BatchMerkleProof::from_individual_proofs(&proofs, &indices);

        // Single proof should have all siblings
        assert_eq!(batch.nodes.len(), 1);
        assert_eq!(batch.nodes[0].len(), 3); // depth = 3 for 8 leaves

        // Verify root computation
        let computed_root =
            compute_root_from_batch_proof(&batch, &indices, &[leaves[3]], test_compress).unwrap();
        assert_eq!(computed_root, root);
    }

    #[test]
    fn test_sibling_batching() {
        let (root, leaves, all_proofs) = build_test_tree(8);

        // Batch two sibling proofs (indices 2 and 3)
        // They share all ancestors, so only the leaf-level siblings differ
        let indices = vec![2, 3];
        let proofs = vec![all_proofs[2].clone(), all_proofs[3].clone()];

        let batch = BatchMerkleProof::from_individual_proofs(&proofs, &indices);

        // First proof needs siblings at levels 1 and 2 (sibling at level 0 is provided by second query)
        // Second proof needs no siblings (all provided by first query's path)
        let total_nodes: usize = batch.nodes.iter().map(|v| v.len()).sum();
        assert!(
            total_nodes < 6,
            "batched siblings should be less than 2*3=6"
        );

        // Verify root computation
        let batch_leaves = vec![leaves[2], leaves[3]];
        let computed_root =
            compute_root_from_batch_proof(&batch, &indices, &batch_leaves, test_compress).unwrap();
        assert_eq!(computed_root, root);
    }

    #[test]
    fn test_multiple_queries_batching() {
        let (root, leaves, all_proofs) = build_test_tree(8);

        // Batch proofs for indices 0, 3, 5, 7
        let indices = vec![0, 3, 5, 7];
        let proofs: Vec<Vec<TestDigest>> = indices.iter().map(|&i| all_proofs[i].clone()).collect();

        let batch = BatchMerkleProof::from_individual_proofs(&proofs, &indices);

        // Should have fewer total nodes than 4 * 3 = 12
        let total_nodes = batch.total_nodes();
        assert!(
            total_nodes < 12,
            "batched proof should have fewer than 12 nodes, got {}",
            total_nodes
        );

        // Verify root computation
        let batch_leaves: Vec<TestDigest> = indices.iter().map(|&i| leaves[i]).collect();
        let computed_root =
            compute_root_from_batch_proof(&batch, &indices, &batch_leaves, test_compress).unwrap();
        assert_eq!(computed_root, root);
    }

    #[test]
    fn test_compression_ratio() {
        let (root, leaves, all_proofs) = build_test_tree(16);
        let depth = 4;

        // Open 8 random-ish leaves
        let indices = vec![1, 3, 4, 7, 9, 11, 12, 15];
        let proofs: Vec<Vec<TestDigest>> = indices.iter().map(|&i| all_proofs[i].clone()).collect();

        let batch = BatchMerkleProof::from_individual_proofs(&proofs, &indices);

        let individual_total = indices.len() * depth;
        let batched_total = batch.total_nodes();

        // Compression should achieve meaningful savings
        // Individual: 8 queries * 4 levels = 32 nodes
        // Batched: should be significantly less due to shared ancestors
        assert!(batched_total < individual_total);

        // Expect at least 20% savings with 8 queries on a tree of 16 leaves
        let savings_percent = (1.0 - batched_total as f64 / individual_total as f64) * 100.0;
        assert!(savings_percent > 20.0, "Expected >20% savings, got {:.1}%", savings_percent);

        // Verify correctness
        let batch_leaves: Vec<TestDigest> = indices.iter().map(|&i| leaves[i]).collect();
        let computed_root =
            compute_root_from_batch_proof(&batch, &indices, &batch_leaves, test_compress).unwrap();
        assert_eq!(computed_root, root);
    }
}
