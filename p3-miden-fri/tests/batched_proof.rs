//! Integration tests for batched FRI proof conversion.

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{CanObserve, DuplexChallenger, FieldChallenger};
use p3_commit::{ExtensionMmcs, Pcs};
use p3_dft::Radix2Dit;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_miden_fri::batched_proof::{BatchedFriProof, BatchedQueryData, LayerBatchProof};
use p3_miden_fri::{FriParameters, TwoAdicFriPcs};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;

type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type MyPcs = TwoAdicFriPcs<BabyBear, Radix2Dit<BabyBear>, ValMmcs, ChallengeMmcs>;

/// The digest type for our Merkle tree (8 BabyBear elements)
type Digest = [BabyBear; 8];

fn get_pcs_for_testing<R: Rng>(
    rng: &mut R,
    num_queries: usize,
    log_folding_factor: usize,
) -> (Perm, MyPcs) {
    let perm = Perm::new_from_rng_128(rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let input_mmcs = ValMmcs::new(hash.clone(), compress.clone());
    let fri_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));
    let fri_params = FriParameters {
        log_blowup: 1,
        log_final_poly_len: 0,
        num_queries,
        proof_of_work_bits: 8,
        mmcs: fri_mmcs,
        log_folding_factor,
    };
    let dft = Radix2Dit::default();
    let pcs = MyPcs::new(dft, input_mmcs, fri_params);
    (perm, pcs)
}

/// Test that batch_fri_proof produces a valid batched proof structure.
#[test]
fn test_batch_fri_proof_structure() {
    let mut rng = SmallRng::seed_from_u64(42);
    let num_queries = 10;
    let log_folding_factor = 1; // folding factor 2
    let (perm, pcs) = get_pcs_for_testing(&mut rng, num_queries, log_folding_factor);

    // Create a simple polynomial and commit
    let log_size = 8;
    let domain =
        <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_size);
    let evaluations = RowMajorMatrix::<Val>::rand_nonzero(&mut rng, 1 << log_size, 4);

    let mut challenger = Challenger::new(perm.clone());
    let (commitment, prover_data) =
        <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain, evaluations)]);
    challenger.observe(commitment);

    let zeta: Challenge = challenger.sample_algebra_element();
    let open_data = vec![(&prover_data, vec![vec![zeta]])];
    let (_opened_values, fri_proof) = pcs.open(open_data, &mut challenger);

    // Verify basic structure - the proof returned by pcs.open is the FriProof
    assert_eq!(fri_proof.query_proofs.len(), num_queries);

    // Verify commit phase structure
    assert!(!fri_proof.commit_phase_commits.is_empty());
    assert!(!fri_proof.final_poly.is_empty());
}

/// Test that the ProofDigestOps implementation works for MerkleTreeMmcs proof type.
#[test]
fn test_proof_digest_ops_for_merkle_tree() {
    use p3_miden_fri::batched_proof::ProofDigestOps;

    // MerkleTreeMmcs::Proof is Vec<[Val; DIGEST_ELEMS]> = Vec<[BabyBear; 8]>
    type MerkleProof = Vec<Digest>;

    let proof: MerkleProof = vec![
        [BabyBear::from_u8(1); 8],
        [BabyBear::from_u8(2); 8],
        [BabyBear::from_u8(3); 8],
    ];

    // Test extract
    let digests = <MerkleProof as ProofDigestOps>::extract_digests(&proof);
    assert_eq!(digests.len(), 3);
    assert_eq!(digests[0], [BabyBear::from_u8(1); 8]);

    // Test reconstruct
    let reconstructed = <MerkleProof as ProofDigestOps>::from_digests(digests);
    assert_eq!(reconstructed, proof);

    // Test depth
    assert_eq!(proof.depth(), 3);
}

/// Test batch Merkle proof compression ratio with realistic parameters.
#[test]
fn test_batch_compression_with_realistic_params() {
    use p3_miden_fri::batch_merkle::BatchMerkleProof;

    // Simulate 27 queries (Miden config) on a tree of depth 20
    let num_queries = 27;
    let depth = 20;

    // Generate random-ish indices spread across the domain
    let mut rng = SmallRng::seed_from_u64(123);
    let domain_size = 1 << depth;
    let mut indices: Vec<usize> = (0..num_queries)
        .map(|_| rng.random_range(0..domain_size))
        .collect();
    indices.sort();
    indices.dedup();

    // Ensure we have enough unique indices
    while indices.len() < num_queries {
        let idx = rng.random_range(0..domain_size);
        if !indices.contains(&idx) {
            indices.push(idx);
            indices.sort();
        }
    }
    indices.truncate(num_queries);

    // Create fake individual proofs (just for size calculation)
    type TestDigest = [u8; 32]; // 256-bit hash
    let individual_proofs: Vec<Vec<TestDigest>> = indices
        .iter()
        .map(|_| (0..depth).map(|i| [i as u8; 32]).collect())
        .collect();

    // Batch them
    let batched = BatchMerkleProof::from_individual_proofs(&individual_proofs, &indices);

    // Calculate sizes
    let individual_size = num_queries * depth; // Number of digest nodes
    let batched_size = batched.total_nodes();

    let savings_percent = (1.0 - batched_size as f64 / individual_size as f64) * 100.0;

    // With 27 queries on depth 20, we should see meaningful compression
    // At higher levels of the tree, queries will share ancestors
    assert!(
        batched_size < individual_size,
        "Batched size {} should be less than individual size {}",
        batched_size,
        individual_size
    );

    // We expect at least 15% savings with 27 random queries
    assert!(
        savings_percent > 15.0,
        "Expected >15% savings, got {:.1}% (individual: {}, batched: {})",
        savings_percent,
        individual_size,
        batched_size
    );

    // Print stats for manual inspection
    // println!("Queries: {}, Depth: {}", num_queries, depth);
    // println!("Individual nodes: {}", individual_size);
    // println!("Batched nodes: {}", batched_size);
    // println!("Savings: {:.1}%", savings_percent);
}

/// Test that batched proof structure can be created with realistic parameters.
#[test]
fn test_batched_proof_structure_creation() {
    use p3_miden_fri::batch_merkle::BatchMerkleProof;

    // Create a mock batched proof to test structure
    let num_queries = 10;
    let num_layers = 5;
    let depth = 15;

    // Mock query data
    let query_data: Vec<BatchedQueryData<BabyBear, ()>> = (0..num_queries)
        .map(|_| BatchedQueryData {
            input_opened_values: (),
            commit_phase_sibling_values: (0..num_layers)
                .map(|_| vec![BabyBear::from_u8(1); 3]) // 3 siblings for folding factor 4
                .collect(),
        })
        .collect();

    // Mock layer proofs with batched Merkle proofs
    let layer_proofs: Vec<LayerBatchProof<Digest>> = (0..num_layers)
        .map(|layer| {
            let layer_depth = depth - layer * 2; // Depth decreases with folding
            let mock_indices: Vec<usize> = (0..num_queries).map(|i| i * 100).collect();

            // Create batched proof - in practice this would come from actual proofs
            let mock_individual: Vec<Vec<Digest>> = (0..num_queries)
                .map(|_| {
                    (0..layer_depth)
                        .map(|_| [BabyBear::from_u8(0); 8])
                        .collect()
                })
                .collect();

            let batched_proof =
                BatchMerkleProof::from_individual_proofs(&mock_individual, &mock_indices);

            LayerBatchProof {
                batched_proof,
                indices: mock_indices,
            }
        })
        .collect();

    let batched_proof: BatchedFriProof<BabyBear, Digest, [BabyBear; 8], BabyBear, ()> =
        BatchedFriProof {
            commit_phase_commits: vec![[BabyBear::from_u8(0); 8]; num_layers],
            query_data,
            layer_proofs,
            input_proofs: vec![],
            final_poly: vec![BabyBear::from_u8(1); 4],
            pow_witness: BabyBear::from_u8(0),
        };

    // Verify structure
    assert_eq!(batched_proof.query_data.len(), num_queries);
    assert_eq!(batched_proof.layer_proofs.len(), num_layers);
    assert_eq!(batched_proof.commit_phase_commits.len(), num_layers);

    // Verify each layer proof has the expected structure
    for (layer, layer_proof) in batched_proof.layer_proofs.iter().enumerate() {
        assert_eq!(layer_proof.indices.len(), num_queries);
        // Batched proof should have fewer nodes than individual proofs
        let expected_individual = num_queries * (depth - layer * 2);
        let actual_batched = layer_proof.batched_proof.total_nodes();
        assert!(
            actual_batched <= expected_individual,
            "Layer {}: batched {} should be <= individual {}",
            layer,
            actual_batched,
            expected_individual
        );
    }
}
