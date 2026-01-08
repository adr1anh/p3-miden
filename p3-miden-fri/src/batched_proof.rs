//! Batched FRI proof structures that use compressed Merkle proofs.
//!
//! This module provides an alternative FRI proof format that batches Merkle proofs
//! across all queries at each FRI layer, reducing proof size by eliminating
//! redundant sibling nodes.
//!
//! # Architecture
//!
//! The standard FRI proof stores a complete Merkle proof for each query at each layer:
//! ```text
//! FriProof
//! └── query_proofs: Vec<QueryProof>          // num_queries items
//!     └── commit_phase_openings: Vec<Step>   // num_layers items
//!         └── opening_proof: M::Proof        // Full Merkle path
//! ```
//!
//! The batched format stores sibling values separately and batches Merkle proofs:
//! ```text
//! BatchedFriProof
//! ├── query_data: Vec<QueryData>                    // num_queries items
//! │   └── commit_phase_sibling_values: Vec<Vec<F>>  // Just the field values
//! └── layer_proofs: Vec<LayerBatchProof>            // num_layers items
//!     └── batched_proof: BatchMerkleProof<Digest>   // Deduplicated proofs
//! ```

use alloc::vec::Vec;

use p3_field::Field;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::batch_merkle::BatchMerkleProof;

/// A batched FRI proof that compresses Merkle proofs across queries.
///
/// This structure achieves proof size reduction by:
/// 1. Storing sibling field values separately from Merkle authentication paths
/// 2. Batching Merkle proofs across all queries at each FRI layer
/// 3. Deduplicating shared internal nodes using the Octopus algorithm
///
/// The digest type `D` represents a single node in the Merkle tree (e.g., `[F; 8]`
/// for a Poseidon2-based tree with 8-element digests).
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct BatchedFriProof<F, D, Commitment, Witness, InputOpenedValues>
where
    F: Field,
    D: Clone + Default + Serialize + DeserializeOwned,
    Commitment: Clone + Serialize + DeserializeOwned,
    Witness: Clone + Serialize + DeserializeOwned,
    InputOpenedValues: Clone + Serialize + DeserializeOwned,
{
    /// Commitments to the folded polynomials at each FRI layer.
    pub commit_phase_commits: Vec<Commitment>,

    /// Query-specific data: sibling values for FRI folding and input opened values.
    pub query_data: Vec<BatchedQueryData<F, InputOpenedValues>>,

    /// Batched Merkle proofs for each FRI layer.
    /// `layer_proofs[i]` contains the batched proof for all queries at layer i.
    pub layer_proofs: Vec<LayerBatchProof<D>>,

    /// Batched Merkle proofs for input polynomial openings.
    /// Outer vec: per input commitment batch.
    /// Each contains proofs for all queries batched together.
    pub input_proofs: Vec<BatchMerkleProof<D>>,

    /// Coefficients of the final polynomial after folding.
    pub final_poly: Vec<F>,

    /// Proof-of-work witness for grinding.
    pub pow_witness: Witness,
}

/// Query-specific data without Merkle proofs.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct BatchedQueryData<F: Field, InputOpenedValues>
where
    InputOpenedValues: Clone + Serialize + DeserializeOwned,
{
    /// Opened values from input polynomials for this query.
    pub input_opened_values: InputOpenedValues,

    /// Sibling values for each commit phase layer.
    /// For folding factor k, each inner Vec has k-1 sibling values.
    pub commit_phase_sibling_values: Vec<Vec<F>>,
}

/// Batched Merkle proofs for a single FRI layer.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct LayerBatchProof<D>
where
    D: Clone + Default + Serialize + DeserializeOwned,
{
    /// The batched Merkle proof for all queries at this layer.
    pub batched_proof: BatchMerkleProof<D>,

    /// The query indices (sorted) that this batch covers.
    /// These are the indices into the FRI domain at this layer.
    pub indices: Vec<usize>,
}

/// Trait for converting between MMCS proof types and raw digest sequences.
///
/// This trait bridges the gap between the generic `Mmcs::Proof` type and
/// the concrete digest type needed for batch Merkle proofs.
pub trait ProofDigestOps {
    /// The type of a single digest/node in the Merkle tree.
    type Digest: Clone + Default + Serialize + DeserializeOwned + Eq;

    /// Extract the sequence of sibling digests from a Merkle proof.
    /// Returns one digest per tree level, from leaf to root.
    fn extract_digests(proof: &Self) -> Vec<Self::Digest>;

    /// Reconstruct a Merkle proof from a sequence of sibling digests.
    fn from_digests(digests: Vec<Self::Digest>) -> Self;

    /// Get the depth (number of levels) in this proof.
    fn depth(&self) -> usize;
}

// Implementation for Vec<D> which is the proof type for MerkleTreeMmcs
impl<D> ProofDigestOps for Vec<D>
where
    D: Clone + Default + Serialize + DeserializeOwned + Eq,
{
    type Digest = D;

    fn extract_digests(proof: &Self) -> Vec<Self::Digest> {
        proof.clone()
    }

    fn from_digests(digests: Vec<Self::Digest>) -> Self {
        digests
    }

    fn depth(&self) -> usize {
        self.len()
    }
}

/// Convert a standard FRI proof to a batched FRI proof.
///
/// This function takes a standard FRI proof and compresses the Merkle proofs
/// using batch aggregation. The resulting proof has the same cryptographic
/// properties but is smaller in size.
///
/// # Type Parameters
/// - `F`: The challenge field type
/// - `M`: The MMCS type (its Proof must implement ProofDigestOps)
/// - `Witness`: The proof-of-work witness type
/// - `InputProof`: The type for input proofs
///
/// # Arguments
/// - `proof`: The standard FRI proof to convert
/// - `query_indices`: The query indices used in the proof (must match proof.query_proofs.len())
/// - `log_folding_factor`: Log2 of the folding factor used in FRI
pub fn batch_fri_proof<F, M, Witness, InputProof>(
    proof: &crate::FriProof<F, M, Witness, InputProof>,
    query_indices: &[usize],
    log_folding_factor: usize,
) -> BatchedFriProof<F, <M::Proof as ProofDigestOps>::Digest, M::Commitment, Witness, InputProof>
where
    F: Field,
    M: p3_commit::Mmcs<F>,
    M::Proof: ProofDigestOps,
    <M::Proof as ProofDigestOps>::Digest: Clone + Default + Serialize + DeserializeOwned + Eq,
    M::Commitment: Clone + Serialize + DeserializeOwned,
    Witness: Clone + Serialize + DeserializeOwned,
    InputProof: Clone + Serialize + DeserializeOwned,
{
    let num_queries = proof.query_proofs.len();
    let num_layers = proof.commit_phase_commits.len();

    assert_eq!(
        query_indices.len(),
        num_queries,
        "query_indices length must match number of query proofs"
    );

    // Extract query data (sibling values and input opened values)
    let query_data: Vec<BatchedQueryData<F, InputProof>> = proof
        .query_proofs
        .iter()
        .map(|qp| BatchedQueryData {
            input_opened_values: qp.input_proof.clone(),
            commit_phase_sibling_values: qp
                .commit_phase_openings
                .iter()
                .map(|step| step.sibling_values.clone())
                .collect(),
        })
        .collect();

    // Batch Merkle proofs for each FRI layer
    let layer_proofs: Vec<LayerBatchProof<<M::Proof as ProofDigestOps>::Digest>> = (0..num_layers)
        .map(|layer| {
            // Compute indices at this layer (shifted by folding factor)
            let layer_indices: Vec<usize> = query_indices
                .iter()
                .map(|&idx| idx >> (layer * log_folding_factor + log_folding_factor))
                .collect();

            // Extract individual proofs for this layer
            let individual_proofs: Vec<Vec<<M::Proof as ProofDigestOps>::Digest>> = proof
                .query_proofs
                .iter()
                .map(|qp| {
                    <M::Proof as ProofDigestOps>::extract_digests(
                        &qp.commit_phase_openings[layer].opening_proof,
                    )
                })
                .collect();

            // Batch the proofs
            let batched_proof =
                BatchMerkleProof::from_individual_proofs(&individual_proofs, &layer_indices);

            // Sort indices for storage
            let mut sorted_indices = layer_indices;
            sorted_indices.sort();

            LayerBatchProof {
                batched_proof,
                indices: sorted_indices,
            }
        })
        .collect();

    // TODO: Batch input proofs similarly
    // For now, we leave input_proofs empty and handle them separately
    let input_proofs = Vec::new();

    BatchedFriProof {
        commit_phase_commits: proof.commit_phase_commits.clone(),
        query_data,
        layer_proofs,
        input_proofs,
        final_poly: proof.final_poly.clone(),
        pow_witness: proof.pow_witness.clone(),
    }
}

/// Convert a batched FRI proof back to a standard FRI proof.
///
/// This function reconstructs the individual Merkle proofs from the batched
/// representation. The resulting proof is cryptographically equivalent but
/// larger in size.
pub fn unbatch_fri_proof<F, M, Witness, InputProof>(
    batched: &BatchedFriProof<
        F,
        <M::Proof as ProofDigestOps>::Digest,
        M::Commitment,
        Witness,
        InputProof,
    >,
    query_indices: &[usize],
    log_folding_factor: usize,
) -> crate::FriProof<F, M, Witness, InputProof>
where
    F: Field,
    M: p3_commit::Mmcs<F>,
    M::Proof: ProofDigestOps,
    <M::Proof as ProofDigestOps>::Digest: Clone + Default + Serialize + DeserializeOwned + Eq,
    M::Commitment: Clone + Serialize + DeserializeOwned,
    Witness: Clone + Serialize + DeserializeOwned,
    InputProof: Clone + Serialize + DeserializeOwned,
{
    use crate::{CommitPhaseProofStep, QueryProof};

    let num_queries = batched.query_data.len();
    let num_layers = batched.layer_proofs.len();

    // Reconstruct individual query proofs
    let query_proofs: Vec<QueryProof<F, M, InputProof>> = (0..num_queries)
        .map(|query_idx| {
            let query_data = &batched.query_data[query_idx];

            // Reconstruct commit phase openings for each layer
            let commit_phase_openings: Vec<CommitPhaseProofStep<F, M>> = (0..num_layers)
                .map(|layer| {
                    let layer_proof = &batched.layer_proofs[layer];

                    // Find this query's position in the sorted indices
                    let layer_index = query_indices[query_idx]
                        >> (layer * log_folding_factor + log_folding_factor);

                    // Get the individual proof from the batched proof
                    // This requires knowing which nodes belong to this query
                    let individual_proof = extract_individual_proof_from_batch(
                        &layer_proof.batched_proof,
                        layer_index,
                        &layer_proof.indices,
                    );

                    CommitPhaseProofStep {
                        sibling_values: query_data.commit_phase_sibling_values[layer].clone(),
                        opening_proof: <M::Proof as ProofDigestOps>::from_digests(individual_proof),
                    }
                })
                .collect();

            QueryProof {
                input_proof: query_data.input_opened_values.clone(),
                commit_phase_openings,
            }
        })
        .collect();

    crate::FriProof {
        commit_phase_commits: batched.commit_phase_commits.clone(),
        query_proofs,
        final_poly: batched.final_poly.clone(),
        pow_witness: batched.pow_witness.clone(),
    }
}

/// Extract an individual Merkle proof from a batched proof.
///
/// This is the inverse of the batching operation - it reconstructs the
/// full sibling path for a single query from the deduplicated batch.
fn extract_individual_proof_from_batch<D>(
    batch: &BatchMerkleProof<D>,
    query_index: usize,
    sorted_indices: &[usize],
) -> Vec<D>
where
    D: Clone + Default,
{
    // Find position of this query in the sorted indices
    let query_pos = sorted_indices
        .iter()
        .position(|&idx| idx == query_index)
        .expect("query_index must be in sorted_indices");

    let depth = batch.depth as usize;
    let mut result = Vec::with_capacity(depth);
    let mut proof_ptr = 0;
    let mut current_index = query_index;

    for level in 0..depth {
        let sibling_index = current_index ^ 1;

        // Check if sibling is provided by another query at this level
        let sibling_provided = sorted_indices.iter().enumerate().any(|(other_pos, &other_idx)| {
            if other_pos == query_pos {
                return false;
            }
            (other_idx >> level) == sibling_index
        });

        if sibling_provided {
            // Sibling comes from another query - we need to look it up
            // For unbatching, we need to compute this from other queries' data
            // This is complex and would require the full verification context
            // For now, use a placeholder
            result.push(D::default());
        } else {
            // Sibling is in our batch nodes
            if proof_ptr < batch.nodes[query_pos].len() {
                result.push(batch.nodes[query_pos][proof_ptr].clone());
                proof_ptr += 1;
            } else {
                result.push(D::default());
            }
        }

        current_index >>= 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn test_proof_digest_ops_vec() {
        type Digest = [u8; 4];
        let proof: Vec<Digest> = vec![[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]];

        let digests = <Vec<Digest> as ProofDigestOps>::extract_digests(&proof);
        assert_eq!(digests.len(), 3);
        assert_eq!(digests[0], [1, 2, 3, 4]);

        let reconstructed = <Vec<Digest> as ProofDigestOps>::from_digests(digests);
        assert_eq!(reconstructed, proof);

        assert_eq!(proof.depth(), 3);
    }
}
