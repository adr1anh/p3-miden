//! Integration tests for LMCS.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
use p3_miden_stateful_hasher::StatefulHasher;
use p3_miden_transcript::{ProverTranscript, VerifierTranscript};
use p3_util::log2_strict_usize;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::{HidingLmcsConfig, LiftedMerkleTree, Lmcs, LmcsConfig, LmcsError, LmcsTree, Proof};

// ============================================================================
// Test Helpers and Re-exports
// ============================================================================

pub use bb::{Compress, DIGEST, F, P, RATE, Sponge, WIDTH};
pub use p3_miden_dev_utils::matrix::concatenate_matrices;

/// Type alias for local LMCS config.
pub type BaseLmcs = LmcsConfig<P, P, Sponge, Compress, WIDTH, DIGEST>;

/// Create a local LMCS config.
pub fn lmcs() -> BaseLmcs {
    let (_, sponge, compress) = bb::test_components();
    LmcsConfig::new(sponge, compress)
}

/// Create sponge and compressor for Merkle tree tests.
pub fn components() -> (Sponge, Compress) {
    let (_, sponge, compress) = bb::test_components();
    (sponge, compress)
}

/// Common matrix group scenarios for testing lifting with varying heights.
pub fn matrix_scenarios() -> Vec<Vec<(usize, usize)>> {
    p3_miden_dev_utils::fixtures::matrix_scenarios::<P>(RATE)
}

/// Build leaf digests for a single matrix (used for equivalence testing).
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

        let tree = lmcs.build_tree(matrices);
        let widths: Vec<_> = tree.leaves().iter().map(|m| m.width()).collect();
        let max_height = tree.leaves().last().map(|m| m.height()).unwrap_or(0);
        let log_max_height = log2_strict_usize(max_height);

        let indices: Vec<usize> = (0..num_queries)
            .map(|_| rng.random_range(0..max_height))
            .collect();

        let mut prover_channel = ProverTranscript::new(bb::test_challenger());
        tree.prove_batch(&indices, &mut prover_channel);
        let transcript = prover_channel.into_data();

        let mut verifier_channel =
            VerifierTranscript::from_data(bb::test_challenger(), &transcript);
        let opened_rows = lmcs
            .open_batch(
                &tree.root(),
                &widths,
                log_max_height,
                &indices,
                &mut verifier_channel,
            )
            .expect("batch opening should verify");

        assert_eq!(opened_rows.len(), indices.len());
        for rows_for_query in &opened_rows {
            assert_eq!(rows_for_query.len(), widths.len());
        }

        for (query_idx, &leaf_idx) in indices.iter().enumerate() {
            let expected_rows = tree.rows(leaf_idx);
            for (matrix_idx, expected_row) in expected_rows.iter().enumerate() {
                assert_eq!(
                    opened_rows[query_idx][matrix_idx].as_slice(),
                    expected_row.as_slice(),
                    "mismatch at query {query_idx} (leaf {leaf_idx}), matrix {matrix_idx}"
                );
            }
        }
    };

    test(1, &[(8, 4)], 1); // single matrix
    test(42, &[(4, 3), (8, 5), (16, 7)], 4); // multi-height
    test(99, &[(32, 2)], 8); // tall matrix
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
        let tree: HidingTree<_> = config.build_tree(matrices);
        let widths: Vec<_> = tree.leaves().iter().map(|m| m.width()).collect();
        let max_height = tree.leaves().last().map(|m| m.height()).unwrap_or(0);
        let log_max_height = log2_strict_usize(max_height);
        let mut prover_channel = ProverTranscript::new(bb::test_challenger());
        tree.prove_batch(indices, &mut prover_channel);
        let transcript = prover_channel.into_data();

        let mut verifier_channel =
            VerifierTranscript::from_data(bb::test_challenger(), &transcript);
        let opened_rows = config
            .open_batch(
                &tree.root(),
                &widths,
                log_max_height,
                indices,
                &mut verifier_channel,
            )
            .expect("batch opening should verify");

        assert_eq!(opened_rows.len(), indices.len());

        for (query_idx, &leaf_idx) in indices.iter().enumerate() {
            let expected_rows = tree.rows(leaf_idx);
            for (matrix_idx, expected_row) in expected_rows.iter().enumerate() {
                assert_eq!(
                    opened_rows[query_idx][matrix_idx].as_slice(),
                    expected_row.as_slice(),
                    "mismatch at query {query_idx} (leaf {leaf_idx}), matrix {matrix_idx}"
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

    let tree1: HidingTree<_> = config1.build_tree(matrices1);
    let tree2: HidingTree<_> = config2.build_tree(matrices2);

    assert_ne!(tree1.root(), tree2.root());
}

#[test]
fn open_batch_handles_empty_or_oob() {
    let mut rng = SmallRng::seed_from_u64(7);
    let lmcs = lmcs();
    let matrices = vec![RowMajorMatrix::rand(&mut rng, 4, 3)];
    let tree = lmcs.build_tree(matrices);
    let widths: Vec<_> = tree.leaves().iter().map(|m| m.width()).collect();
    let log_max_height = log2_strict_usize(tree.height());
    let commitment = tree.root();

    let transcript = ProverTranscript::new(bb::test_challenger()).into_data();

    let mut verifier_channel = VerifierTranscript::from_data(bb::test_challenger(), &transcript);
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

    let mut verifier_channel = VerifierTranscript::from_data(bb::test_challenger(), &transcript);
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
}

#[test]
fn read_batch_handles_empty_or_oob() {
    let mut rng = SmallRng::seed_from_u64(9);
    let lmcs = lmcs();
    let matrices = vec![RowMajorMatrix::rand(&mut rng, 4, 3)];
    let tree = lmcs.build_tree(matrices);
    let widths: Vec<_> = tree.leaves().iter().map(|m| m.width()).collect();
    let log_max_height = log2_strict_usize(tree.height());

    let mut prover_channel = ProverTranscript::new(bb::test_challenger());
    tree.prove_batch(&[0], &mut prover_channel);
    let transcript = prover_channel.into_data();

    let mut verifier_channel = VerifierTranscript::from_data(bb::test_challenger(), &transcript);
    assert_eq!(
        lmcs.read_batch_from_channel(&widths, log_max_height, &[], &mut verifier_channel),
        Ok(vec![])
    );

    let mut verifier_channel = VerifierTranscript::from_data(bb::test_challenger(), &transcript);
    let proofs = lmcs
        .read_batch_from_channel(&[], log_max_height, &[0], &mut verifier_channel)
        .unwrap();
    assert_eq!(proofs.len(), 1);
    let Proof {
        rows,
        salt,
        siblings,
    } = &proofs[0];
    assert!(rows.is_empty());
    assert!(salt.is_empty());
    assert_eq!(siblings.len(), 2);

    let mut verifier_channel = VerifierTranscript::from_data(bb::test_challenger(), &transcript);
    assert_eq!(
        lmcs.read_batch_from_channel(
            &widths,
            log_max_height,
            &[tree.height()],
            &mut verifier_channel,
        ),
        Err(LmcsError::InvalidProof)
    );
}
