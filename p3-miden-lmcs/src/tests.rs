//! Test fixtures and integration tests for the LMCS crate.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
use p3_miden_stateful_hasher::StatefulHasher;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::{HidingLmcsConfig, LiftedMerkleTree, Lmcs, LmcsConfig};

// ============================================================================
// Re-exports from dev-utils (for external use)
// ============================================================================

pub use bb::{Compress, DIGEST, F, P, RATE, Sponge, WIDTH};
pub use p3_miden_dev_utils::matrix::concatenate_matrices;

/// Type alias for local LMCS config (uses crate's own types to avoid cross-crate issues).
pub type BaseLmcs = LmcsConfig<P, P, Sponge, Compress, WIDTH, DIGEST>;

/// Create a local LMCS config using crate's own types.
///
/// This avoids cross-crate type mismatches when running tests.
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
// Integration Tests
// ============================================================================

#[test]
fn commit_open_verify_roundtrip() {
    let lmcs = lmcs();

    let mut rng = SmallRng::seed_from_u64(9);
    let matrices = vec![
        RowMajorMatrix::rand(&mut rng, 2, 3),
        RowMajorMatrix::rand(&mut rng, 4, 5),
        RowMajorMatrix::rand(&mut rng, 8, 7),
    ];
    let dims: Vec<_> = matrices
        .iter()
        .map(|m: &RowMajorMatrix<F>| m.dimensions())
        .collect();

    let tree = lmcs.build_tree(matrices);
    let commitment = tree.root();
    let final_height = dims.last().unwrap().height;

    // Open and verify using multi-opening API
    let indices = [final_height - 1];
    let proof = tree.open_multi(&indices);
    let opened_rows = lmcs.verify(&commitment, &dims, &indices, &proof).unwrap();

    assert_eq!(opened_rows.len(), 1);
    assert_eq!(opened_rows[0].len(), dims.len());
}

#[test]
fn multi_opening_roundtrip() {
    let lmcs = lmcs();

    let mut rng = SmallRng::seed_from_u64(42);
    let matrices = vec![
        RowMajorMatrix::rand(&mut rng, 4, 3),
        RowMajorMatrix::rand(&mut rng, 8, 5),
        RowMajorMatrix::rand(&mut rng, 16, 7),
    ];
    let dims: Vec<_> = matrices.iter().map(|m| m.dimensions()).collect();

    let tree = lmcs.build_tree(matrices);
    let commitment = tree.root();
    let final_height = dims.last().unwrap().height;

    // Open multiple indices
    let indices = [0, 5, 10, final_height - 1];
    let proof = tree.open_multi(&indices);

    // Verify and get back opened rows
    let opened_rows = lmcs.verify(&commitment, &dims, &indices, &proof).unwrap();

    // Check that we get the expected number of queries and matrices
    assert_eq!(opened_rows.len(), indices.len());
    for rows_for_query in &opened_rows {
        assert_eq!(rows_for_query.len(), dims.len());
    }

    // Verify row contents match the original tree
    for (query_idx, &leaf_idx) in indices.iter().enumerate() {
        let expected_rows = tree.rows(leaf_idx);
        for (matrix_idx, expected_row) in expected_rows.iter().enumerate() {
            assert_eq!(
                opened_rows[query_idx][matrix_idx],
                expected_row.as_slice(),
                "mismatch at query {query_idx} (leaf {leaf_idx}), matrix {matrix_idx}"
            );
        }
    }
}

#[test]
fn multi_opening_single_index() {
    let lmcs = lmcs();

    let mut rng = SmallRng::seed_from_u64(123);
    let matrices = vec![RowMajorMatrix::rand(&mut rng, 8, 4)];
    let dims: Vec<_> = matrices.iter().map(|m| m.dimensions()).collect();

    let tree = lmcs.build_tree(matrices);
    let commitment = tree.root();

    // Open a single index using multi-opening API
    let indices = [3];
    let proof = tree.open_multi(&indices);

    let opened_rows = lmcs.verify(&commitment, &dims, &indices, &proof).unwrap();
    assert_eq!(opened_rows.len(), 1);
    assert_eq!(opened_rows[0].len(), 1);
}

/// Hiding tree type alias with SALT_ELEMS = 4.
type HidingTree<M> = LiftedMerkleTree<F, F, M, DIGEST, 4>;

/// Hiding config type alias with SALT_ELEMS = 4.
type HidingConfig = HidingLmcsConfig<P, P, Sponge, Compress, SmallRng, WIDTH, DIGEST, 4>;

/// Create a hiding LMCS config.
fn hiding_lmcs(rng: SmallRng) -> HidingConfig {
    let (_, sponge, compress) = bb::test_components();
    HidingLmcsConfig::new(sponge, compress, rng)
}

#[test]
fn hiding_commit_open_verify_roundtrip() {
    let mut rng = SmallRng::seed_from_u64(99);
    let matrices = vec![
        RowMajorMatrix::rand(&mut rng, 4, 3),
        RowMajorMatrix::rand(&mut rng, 8, 5),
    ];
    let dims: Vec<_> = matrices.iter().map(|m| m.dimensions()).collect();

    // Create config with RNG for salt generation
    let config = hiding_lmcs(rng);

    // Use config builder for hiding commitment (RNG is internal to config)
    let tree: HidingTree<_> = config.build_tree(matrices);
    let commitment = tree.root();

    // Open multiple indices (open_multi includes salt for SALT > 0)
    let indices = [1, 3, 5];
    let proof = tree.open_multi(&indices);

    // Verify hiding proof (SALT = 4 inferred from config)
    let opened_rows = config.verify(&commitment, &dims, &indices, &proof).unwrap();

    // Check structure
    assert_eq!(opened_rows.len(), indices.len());
    assert_eq!(proof.openings.len(), indices.len());

    // Verify row contents
    for (query_idx, &leaf_idx) in indices.iter().enumerate() {
        let expected_rows = tree.rows(leaf_idx);
        for (matrix_idx, expected_row) in expected_rows.iter().enumerate() {
            assert_eq!(
                opened_rows[query_idx][matrix_idx],
                expected_row.as_slice(),
                "mismatch at query {query_idx} (leaf {leaf_idx}), matrix {matrix_idx}"
            );
        }
    }
}

#[test]
fn hiding_different_salts_different_roots() {
    // Commit the same data with different RNGs should produce different roots
    let matrices1 = vec![RowMajorMatrix::rand(
        &mut SmallRng::seed_from_u64(100),
        4,
        3,
    )];
    let matrices2 = matrices1.clone();

    // Create two configs with different RNGs
    let config1 = hiding_lmcs(SmallRng::seed_from_u64(1));
    let config2 = hiding_lmcs(SmallRng::seed_from_u64(2));

    let tree1: HidingTree<_> = config1.build_tree(matrices1);
    let tree2: HidingTree<_> = config2.build_tree(matrices2);

    let commitment1 = tree1.root();
    let commitment2 = tree2.root();

    // Different salt should produce different commitments
    assert_ne!(commitment1, commitment2);
}
