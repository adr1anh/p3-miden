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

// ============================================================================
// Re-exports from dev-utils
// ============================================================================

pub use bb::{Compress, DIGEST, F, P, RATE, Sponge, WIDTH, base_lmcs};

/// Create sponge and compressor for Merkle tree tests.
pub fn components() -> (Sponge, Compress) {
    let (_, sponge, compress) = bb::test_components();
    (sponge, compress)
}
pub use p3_miden_dev_utils::matrix::{concatenate_matrices, lift_matrix};

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
    let lmcs = base_lmcs();

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
