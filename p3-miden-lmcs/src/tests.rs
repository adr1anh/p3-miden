//! Integration tests for LMCS.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
use p3_miden_stateful_hasher::{Alignable, StatefulHasher};
use p3_miden_transcript::{ProverTranscript, TranscriptData, VerifierChannel, VerifierTranscript};
use p3_symmetric::Hash;
use p3_util::log2_strict_usize;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::utils::aligned_len;
use crate::{
    BatchProof, HidingLmcsConfig, LiftedMerkleTree, Lmcs, LmcsConfig, LmcsError, LmcsTree, Proof,
};

// ============================================================================
// Test Helpers and Re-exports
// ============================================================================

pub use bb::{Compress, DIGEST, F, P, RATE, Sponge, WIDTH};
pub use p3_miden_dev_utils::matrix::concatenate_matrices;

/// Type alias for local LMCS config.
pub type BaseLmcs = LmcsConfig<P, P, Sponge, Compress, WIDTH, DIGEST>;
type Commitment = <BaseLmcs as Lmcs>::Commitment;
type TestTranscriptData = TranscriptData<F, Commitment>;
type OpenedRows = BTreeMap<usize, Vec<Vec<F>>>;

/// Create a local LMCS config.
pub fn lmcs() -> BaseLmcs {
    let (_, sponge, compress) = bb::test_components();
    LmcsConfig::new(sponge, compress)
}

/// Build leaf hashes for a single matrix (used for equivalence testing).
pub fn build_leaves_single(matrix: &RowMajorMatrix<F>, sponge: &Sponge) -> Vec<[F; DIGEST]> {
    matrix
        .rows()
        .map(|row| {
            let mut state = [F::ZERO; WIDTH];
            sponge.absorb_into(&mut state, row);
            sponge.squeeze(&state)
        })
        .collect()
}

fn verify_open_batch<C>(
    lmcs: &C,
    commitment: &Commitment,
    widths: &[usize],
    log_max_height: usize,
    indices: &[usize],
    transcript: &TestTranscriptData,
) -> Result<OpenedRows, LmcsError>
where
    C: Lmcs<F = F, Commitment = Commitment>,
{
    let mut verifier_channel = VerifierTranscript::from_data(bb::test_challenger(), transcript);
    let result = lmcs.open_batch(
        commitment,
        widths,
        log_max_height,
        indices.iter().copied(),
        &mut verifier_channel,
    );
    if result.is_ok() {
        assert!(
            verifier_channel.is_empty(),
            "transcript should be fully consumed"
        );
    }
    result
}

fn roundtrip_open_batch<C, M>(
    lmcs: &C,
    tree: &C::Tree<M>,
    indices: &[usize],
) -> Result<(TestTranscriptData, OpenedRows), LmcsError>
where
    C: Lmcs<F = F, Commitment = Commitment>,
    M: Matrix<F>,
{
    let widths = tree.widths();
    let log_max_height = log2_strict_usize(tree.height());

    let transcript = {
        let mut prover_channel = ProverTranscript::new(bb::test_challenger());
        tree.prove_batch(indices.iter().copied(), &mut prover_channel);
        prover_channel.into_data()
    };
    let opened_rows = verify_open_batch(
        lmcs,
        &tree.root(),
        &widths,
        log_max_height,
        indices,
        &transcript,
    )?;
    Ok((transcript, opened_rows))
}

// ============================================================================
// Hiding LMCS Types and Helpers
// ============================================================================

const SALT: usize = 4;
type HidingTree<M> = LiftedMerkleTree<F, F, M, DIGEST, SALT>;
type HidingConfig = HidingLmcsConfig<P, P, Sponge, Compress, SmallRng, WIDTH, DIGEST, SALT>;

fn hiding_lmcs(rng: SmallRng) -> HidingConfig {
    let (_, sponge, compress) = bb::test_components();
    HidingLmcsConfig::new(sponge, compress, rng)
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn lmcs_roundtrip() {
    let test = |seed: u64, matrices: &[(usize, usize)], num_queries: usize| {
        let mut rng = SmallRng::seed_from_u64(seed);
        let lmcs = lmcs();
        let matrices: Vec<_> = matrices
            .iter()
            .map(|&(h, w)| RowMajorMatrix::rand(&mut rng, h, w))
            .collect();

        let tree = lmcs.build_tree(matrices, None);
        let widths = tree.widths();
        let max_height = tree.height();
        let indices: Vec<usize> = (0..num_queries)
            .map(|_| rng.random_range(0..max_height))
            .collect();
        let (_transcript, opened_rows) =
            roundtrip_open_batch(&lmcs, &tree, &indices).expect("batch opening should verify");

        for (&leaf_idx, rows_for_query) in &opened_rows {
            assert_eq!(rows_for_query.len(), widths.len());
            let expected_rows = tree.rows(leaf_idx);
            for (matrix_idx, expected_row) in expected_rows.iter().enumerate() {
                assert_eq!(
                    rows_for_query[matrix_idx].as_slice(),
                    expected_row.as_slice(),
                    "mismatch at leaf {leaf_idx}, matrix {matrix_idx}"
                );
            }
        }
    };

    test(1, &[(8, 4)], 1); // single matrix
    test(42, &[(4, 3), (8, 5), (16, 7)], 4); // multi-height
    test(99, &[(32, 2)], 8); // tall matrix
}

#[test]
fn lmcs_duplicate_indices_roundtrip() {
    let mut rng = SmallRng::seed_from_u64(123);
    let lmcs = lmcs();
    let matrices = vec![
        RowMajorMatrix::rand(&mut rng, 4, 5),
        RowMajorMatrix::rand(&mut rng, 8, 3),
    ];

    let tree = lmcs.build_tree(matrices, None);
    let widths = tree.widths();
    let log_max_height = log2_strict_usize(tree.height());
    let indices = [3usize, 1, 3, 0, 1];

    let (transcript, opened_rows) =
        roundtrip_open_batch(&lmcs, &tree, &indices).expect("batch opening should verify");

    // BTreeMap coalesces duplicates: 5 indices → 3 unique keys
    assert_eq!(opened_rows.len(), 3);

    for (&index, rows) in &opened_rows {
        let expected_rows = tree.rows(index);
        assert_eq!(*rows, expected_rows, "row mismatch for index {index}");
    }

    let mut verifier_channel = VerifierTranscript::from_data(bb::test_challenger(), &transcript);
    let batch = BatchProof::<F, Hash<F, F, DIGEST>>::read_from_channel(
        &widths,
        log_max_height,
        &indices,
        &mut verifier_channel,
    )
    .expect("batch proof should parse from transcript");

    assert_eq!(batch.openings.len(), 3);
    for &index in &[0usize, 1, 3] {
        let opening = batch.openings.get(&index).expect("opening for index");
        let expected_rows = tree.rows(index);
        assert_eq!(
            opening.rows, expected_rows,
            "batch opening rows mismatch for index {index}"
        );
    }

    let proofs = batch
        .single_proofs(&lmcs, &widths, log_max_height)
        .expect("batch proof should reconstruct proofs");
    assert_eq!(proofs.len(), 3);
}

#[test]
fn hiding_roundtrip() {
    let test = |seed: u64, matrices: &[(usize, usize)], indices: &[usize]| {
        let mut rng = SmallRng::seed_from_u64(seed);
        let matrices: Vec<_> = matrices
            .iter()
            .map(|&(h, w)| RowMajorMatrix::rand(&mut rng, h, w))
            .collect();

        let config = hiding_lmcs(rng);
        let tree: HidingTree<_> = config.build_tree(matrices, None);
        let (_transcript, opened_rows) =
            roundtrip_open_batch(&config, &tree, indices).expect("batch opening should verify");

        for (&leaf_idx, rows) in &opened_rows {
            let expected_rows = tree.rows(leaf_idx);
            for (matrix_idx, expected_row) in expected_rows.iter().enumerate() {
                assert_eq!(
                    rows[matrix_idx].as_slice(),
                    expected_row.as_slice(),
                    "mismatch at leaf {leaf_idx}, matrix {matrix_idx}"
                );
            }
        }
    };

    test(99, &[(4, 3), (8, 5)], &[1, 3, 5]);

    // Different salts should produce different commitments
    let matrices1 = vec![RowMajorMatrix::rand(
        &mut SmallRng::seed_from_u64(100),
        4,
        3,
    )];
    let matrices2 = matrices1.clone();

    let config1 = hiding_lmcs(SmallRng::seed_from_u64(1));
    let config2 = hiding_lmcs(SmallRng::seed_from_u64(2));

    let tree1: HidingTree<_> = config1.build_tree(matrices1, None);
    let tree2: HidingTree<_> = config2.build_tree(matrices2, None);

    assert_ne!(tree1.root(), tree2.root());
}

#[test]
fn open_batch_handles_empty_or_oob() {
    let mut rng = SmallRng::seed_from_u64(7);
    let lmcs = lmcs();
    let matrices = vec![RowMajorMatrix::rand(&mut rng, 4, 3)];
    let tree = lmcs.build_tree(matrices, None);
    let widths = tree.widths();
    let log_max_height = log2_strict_usize(tree.height());
    let commitment = tree.root();

    let transcript = ProverTranscript::new(bb::test_challenger()).into_data();

    assert_eq!(
        verify_open_batch(
            &lmcs,
            &commitment,
            &widths,
            log_max_height,
            &[],
            &transcript,
        ),
        Err(LmcsError::InvalidProof)
    );

    assert_eq!(
        verify_open_batch(
            &lmcs,
            &commitment,
            &widths,
            log_max_height,
            &[tree.height()],
            &transcript,
        ),
        Err(LmcsError::InvalidProof)
    );
}

#[test]
fn build_tree_alignment_modes() {
    let mut rng = SmallRng::seed_from_u64(123);
    let lmcs = lmcs();
    let matrices = vec![
        RowMajorMatrix::rand(&mut rng, 4, 3),
        RowMajorMatrix::rand(&mut rng, 8, 5),
    ];

    let tree_unaligned = lmcs.build_tree(matrices.clone(), None);
    let tree_aligned = lmcs.build_aligned_tree(matrices, None);
    let alignment = tree_aligned.alignment();
    let expected_alignment = <Sponge as Alignable<F, F>>::ALIGNMENT;

    assert_eq!(tree_unaligned.alignment(), 1);
    assert_eq!(alignment, expected_alignment);
    assert_eq!(tree_unaligned.root(), tree_aligned.root());

    let widths_aligned = tree_aligned.widths();
    assert_eq!(widths_aligned[0], aligned_len(3, expected_alignment));
    assert_eq!(widths_aligned[1], aligned_len(5, expected_alignment));

    let widths_unaligned = tree_unaligned.widths();
    assert_eq!(widths_unaligned, vec![3, 5]);
    if expected_alignment > 1 {
        assert_ne!(widths_unaligned, widths_aligned);
    }

    let rows_aligned = tree_aligned.rows(0);
    assert_eq!(rows_aligned[0].len(), widths_aligned[0]);
    assert_eq!(rows_aligned[1].len(), widths_aligned[1]);

    let rows_unaligned = tree_unaligned.rows(0);
    assert_eq!(rows_unaligned[0].len(), widths_unaligned[0]);
    assert_eq!(rows_unaligned[1].len(), widths_unaligned[1]);

    let indices = [0usize, 1usize];
    let (_transcript, opened_rows) = roundtrip_open_batch(&lmcs, &tree_aligned, &indices)
        .expect("aligned opening should verify");
    for (&idx, rows) in &opened_rows {
        assert_eq!(*rows, tree_aligned.rows(idx));
    }
}

#[test]
fn batch_proof_handles_empty_or_oob() {
    let mut rng = SmallRng::seed_from_u64(9);
    let lmcs = lmcs();
    let matrices = vec![RowMajorMatrix::rand(&mut rng, 4, 3)];
    let tree = lmcs.build_tree(matrices, None);
    let widths = tree.widths();
    let log_max_height = log2_strict_usize(tree.height());

    let mut prover_channel = ProverTranscript::new(bb::test_challenger());
    tree.prove_batch([0], &mut prover_channel);
    let transcript = prover_channel.into_data();

    let mut verifier_channel = VerifierTranscript::from_data(bb::test_challenger(), &transcript);
    let batch = BatchProof::<F, Hash<F, F, DIGEST>>::read_from_channel(
        &widths,
        log_max_height,
        &[],
        &mut verifier_channel,
    )
    .unwrap();
    assert!(batch.openings.is_empty());
    assert!(batch.siblings.is_empty());
    let proofs = batch.single_proofs(&lmcs, &widths, log_max_height).unwrap();
    assert!(proofs.is_empty());

    let mut verifier_channel = VerifierTranscript::from_data(bb::test_challenger(), &transcript);
    let batch = BatchProof::<F, Hash<F, F, DIGEST>>::read_from_channel(
        &[],
        log_max_height,
        &[0],
        &mut verifier_channel,
    )
    .unwrap();
    let proofs = batch.single_proofs(&lmcs, &[], log_max_height).unwrap();
    assert_eq!(proofs.len(), 1);
    let proof = proofs.get(&0).expect("proof for index 0");
    let Proof {
        rows,
        salt,
        siblings,
    } = proof;
    assert!(rows.is_empty());
    assert!(salt.is_empty());
    assert_eq!(siblings.len(), 2);

    // Out-of-range indices are not rejected at parse time; they produce proofs that
    // fail verification. Here we just confirm parsing succeeds (verification tested elsewhere).
    let mut verifier_channel = VerifierTranscript::from_data(bb::test_challenger(), &transcript);
    let _ = BatchProof::<F, Hash<F, F, DIGEST>>::read_from_channel(
        &widths,
        log_max_height,
        &[tree.height()],
        &mut verifier_channel,
    )
    .unwrap();
}

// ============================================================================
// Target Height Tests
// ============================================================================

/// Non-hiding LMCS with target height larger than the tallest matrix.
/// Digests are upsampled (nearest-neighbor) to match target height.
/// Pairs of identical sibling digests produce self-compressions.
#[test]
fn target_height_non_hiding_roundtrip() {
    let test = |seed: u64, matrices: &[(usize, usize)], log_target: usize, num_queries: usize| {
        let mut rng = SmallRng::seed_from_u64(seed);
        let lmcs = lmcs();
        let matrices: Vec<_> = matrices
            .iter()
            .map(|&(h, w)| RowMajorMatrix::rand(&mut rng, h, w))
            .collect();

        let tree = lmcs.build_tree(matrices, Some(log_target));
        let target_height = 1usize << log_target;
        assert_eq!(tree.height(), target_height);

        let widths = tree.widths();
        let indices: Vec<usize> = (0..num_queries)
            .map(|_| rng.random_range(0..target_height))
            .collect();
        let (_transcript, opened_rows) =
            roundtrip_open_batch(&lmcs, &tree, &indices).expect("batch opening should verify");

        for (&leaf_idx, rows_for_query) in &opened_rows {
            assert_eq!(rows_for_query.len(), widths.len());
            let expected_rows = tree.rows(leaf_idx);
            for (matrix_idx, expected_row) in expected_rows.iter().enumerate() {
                assert_eq!(
                    rows_for_query[matrix_idx].as_slice(),
                    expected_row.as_slice(),
                    "mismatch at leaf {leaf_idx}, matrix {matrix_idx}"
                );
            }
        }
    };

    // single matrix, target height = 2x
    test(1, &[(4, 3)], 3, 4);
    // single matrix, target height = 4x
    test(2, &[(4, 3)], 4, 8);
    // multi-height matrices, target height above tallest
    test(3, &[(4, 3), (8, 5)], 5, 8);
    // target height equals natural height (no upsampling)
    test(4, &[(8, 4)], 3, 4);
}

/// Non-hiding: commitment with target_height == natural height should match
/// the commitment without explicit target_height.
#[test]
fn target_height_none_equals_natural() {
    let mut rng = SmallRng::seed_from_u64(50);
    let lmcs = lmcs();
    let matrices = vec![
        RowMajorMatrix::rand(&mut rng, 4, 3),
        RowMajorMatrix::rand(&mut rng, 8, 5),
    ];

    let tree_none = lmcs.build_tree(matrices.clone(), None);
    let tree_explicit = lmcs.build_tree(matrices, Some(3)); // log2(8) = 3
    assert_eq!(tree_none.root(), tree_explicit.root());
    assert_eq!(tree_none.height(), tree_explicit.height());
}

/// Non-hiding: upsampled leaf rows should repeat via nearest-neighbor.
#[test]
fn target_height_rows_are_upsampled() {
    let mut rng = SmallRng::seed_from_u64(60);
    let lmcs = lmcs();
    let matrices = vec![RowMajorMatrix::rand(&mut rng, 4, 3)];

    let tree = lmcs.build_tree(matrices, Some(4)); // target = 16, natural = 4
    assert_eq!(tree.height(), 16);

    // Each original row should repeat 4 times (16/4 = 4)
    for i in 0..16 {
        let rows_i = tree.rows(i);
        let original_idx = i / 4;
        let rows_orig = tree.rows(original_idx * 4);
        assert_eq!(rows_i, rows_orig, "row {i} should match row {}", original_idx * 4);
    }
}

/// Hiding LMCS with target height: states are upsampled then salted.
/// Each upsampled leaf gets independent salt → unique hashes.
#[test]
fn target_height_hiding_roundtrip() {
    let test = |seed: u64, matrices: &[(usize, usize)], log_target: usize, indices: &[usize]| {
        let mut rng = SmallRng::seed_from_u64(seed);
        let matrices: Vec<_> = matrices
            .iter()
            .map(|&(h, w)| RowMajorMatrix::rand(&mut rng, h, w))
            .collect();

        let config = hiding_lmcs(rng);
        let tree: HidingTree<_> = config.build_tree(matrices, Some(log_target));
        let target_height = 1usize << log_target;
        assert_eq!(tree.height(), target_height);

        let (_transcript, opened_rows) =
            roundtrip_open_batch(&config, &tree, indices).expect("batch opening should verify");

        for (&leaf_idx, rows) in &opened_rows {
            let expected_rows = tree.rows(leaf_idx);
            for (matrix_idx, expected_row) in expected_rows.iter().enumerate() {
                assert_eq!(
                    rows[matrix_idx].as_slice(),
                    expected_row.as_slice(),
                    "mismatch at leaf {leaf_idx}, matrix {matrix_idx}"
                );
            }
        }
    };

    // 2x upsampling with salt
    test(100, &[(4, 3)], 3, &[0, 1, 5, 7]);
    // multi-height with salt, target above tallest
    test(101, &[(4, 3), (8, 5)], 5, &[1, 10, 20, 31]);
}

/// Hiding: upsampled sibling leaves should have different hashes because
/// of independent salt.
#[test]
fn target_height_hiding_unique_leaves() {
    let rng = SmallRng::seed_from_u64(200);
    let matrices = vec![RowMajorMatrix::rand(
        &mut SmallRng::seed_from_u64(200),
        4,
        3,
    )];

    let config = hiding_lmcs(rng);
    let tree: HidingTree<_> = config.build_tree(matrices, Some(4)); // target=16, natural=4
    assert_eq!(tree.height(), 16);

    // Leaf 0 and leaf 1 come from the same original row but have different salt.
    // Their salts should differ (with overwhelming probability).
    let salt_0 = tree.salt(0);
    let salt_1 = tree.salt(1);
    assert_ne!(salt_0, salt_1, "upsampled siblings should have independent salt");
}
