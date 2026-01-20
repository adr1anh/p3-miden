//! Test fixtures and integration tests for the LMCS crate.

use alloc::vec;
use alloc::vec::Vec;

use p3_commit::{BatchOpeningRef, Mmcs};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
use p3_miden_stateful_hasher::StatefulHasher;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::{HidingLmcsMmcs, LiftedHidingMerkleTree, LmcsMmcs};

// ============================================================================
// Re-exports from dev-utils (for external use)
// ============================================================================

pub use bb::{Compress, DIGEST, F, P, RATE, Sponge, WIDTH};
pub use p3_miden_dev_utils::matrix::concatenate_matrices;

/// Type alias for local LMCS MMCS (uses crate's own types to avoid cross-crate issues).
pub type BaseLmcs = LmcsMmcs<P, P, Sponge, Compress, WIDTH, DIGEST>;

/// Create a local LMCS MMCS using crate's own types.
///
/// This avoids cross-crate type mismatches when running tests.
pub fn lmcs() -> BaseLmcs {
    let (_, sponge, compress) = bb::test_components();
    LmcsMmcs::new(sponge, compress)
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

    let (commitment, tree) = lmcs.commit(matrices);
    let final_height = dims.last().unwrap().height;
    let index = final_height - 1; // valid index within range

    let opening = lmcs.open_batch(index, &tree);
    let opening_ref: BatchOpeningRef<'_, F, _> = (&opening).into();
    assert!(
        lmcs.verify_batch(&commitment, &dims, index, opening_ref)
            .is_ok()
    );
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

    let (commitment, tree) = lmcs.commit(matrices);
    let final_height = dims.last().unwrap().height;

    // Open multiple indices (opening methods are now on the tree)
    let indices = [0, 5, 10, final_height - 1];
    let proof = tree.open_multi(&indices);

    // Verify and get back opened rows
    let opened_rows = lmcs
        .config
        .verify(&commitment, &dims, &indices, &proof)
        .unwrap();

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

    let (commitment, tree) = lmcs.commit(matrices);

    // Open a single index using multi-opening API
    let indices = [3];
    let proof = tree.open_multi(&indices);

    let opened_rows = lmcs
        .config
        .verify(&commitment, &dims, &indices, &proof)
        .unwrap();
    assert_eq!(opened_rows.len(), 1);
    assert_eq!(opened_rows[0].len(), 1);
}

/// Hiding tree type alias with SALT_ELEMS = 4.
type HidingTree<M> = LiftedHidingMerkleTree<F, F, M, DIGEST, 4>;

#[test]
fn hiding_commit_open_verify_roundtrip() {
    let (_, sponge, compress) = bb::test_components();
    // Config with SALT = 4 for hiding commitment
    let config = crate::LmcsConfig::<F, F, _, _, WIDTH, DIGEST, 4>::new(sponge, compress);

    let mut rng = SmallRng::seed_from_u64(99);
    let matrices = vec![
        RowMajorMatrix::rand(&mut rng, 4, 3),
        RowMajorMatrix::rand(&mut rng, 8, 5),
    ];
    let dims: Vec<_> = matrices.iter().map(|m| m.dimensions()).collect();

    // Use config builder for hiding commitment
    let tree: HidingTree<_> = config.build_tree_hiding::<P, P, _>(matrices, &mut rng);
    let commitment = tree.root();

    // Open multiple indices (open_multi includes salt for SALT > 0)
    let indices = [1, 3, 5];
    let proof = tree.open_multi(&indices);

    // Verify hiding proof (SALT = 4 inferred from config)
    let opened_rows = config.verify(&commitment, &dims, &indices, &proof).unwrap();

    // Check structure
    assert_eq!(opened_rows.len(), indices.len());
    assert_eq!(proof.openings().len(), indices.len());

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

/// Hiding MMCS type alias with SALT_ELEMS = 4.
type HidingMmcs = HidingLmcsMmcs<P, P, Sponge, Compress, SmallRng, WIDTH, DIGEST, 4>;

/// Create a hiding LMCS MMCS with the given RNG.
fn hiding_lmcs(rng: SmallRng) -> HidingMmcs {
    let (_, sponge, compress) = bb::test_components();
    HidingLmcsMmcs::new(sponge, compress, rng)
}

#[test]
fn hiding_mmcs_roundtrip() {
    let rng = SmallRng::seed_from_u64(42);
    let mmcs = hiding_lmcs(rng);

    let mut rng = SmallRng::seed_from_u64(123);
    let matrices = vec![
        RowMajorMatrix::rand(&mut rng, 2, 3),
        RowMajorMatrix::rand(&mut rng, 4, 5),
        RowMajorMatrix::rand(&mut rng, 8, 7),
    ];
    let dims: Vec<_> = matrices
        .iter()
        .map(|m: &RowMajorMatrix<F>| m.dimensions())
        .collect();

    let (commitment, tree) = mmcs.commit(matrices);
    let final_height = dims.last().unwrap().height;

    // Test multiple indices
    for index in [0, 3, final_height - 1] {
        let opening = mmcs.open_batch(index, &tree);
        let opening_ref: BatchOpeningRef<'_, F, _> = (&opening).into();
        assert!(
            mmcs.verify_batch(&commitment, &dims, index, opening_ref)
                .is_ok(),
            "verification failed for index {index}"
        );
    }
}

#[test]
fn hiding_mmcs_different_salts_different_roots() {
    // Commit the same data with different RNGs should produce different roots
    let matrices1 = vec![RowMajorMatrix::rand(
        &mut SmallRng::seed_from_u64(100),
        4,
        3,
    )];
    let matrices2 = matrices1.clone();

    let mmcs1 = hiding_lmcs(SmallRng::seed_from_u64(1));
    let mmcs2 = hiding_lmcs(SmallRng::seed_from_u64(2));

    let (commitment1, _) = mmcs1.commit(matrices1);
    let (commitment2, _) = mmcs2.commit(matrices2);

    // Different salt should produce different commitments
    assert_ne!(commitment1, commitment2);
}
