//! Common test fixtures and end-to-end tests for the lifted FRI PCS.
//!
//! Re-exports test fixtures from `p3_miden_dev_utils` for use in tests.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::CanObserve;
use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_lmcs::{Lmcs, LmcsTree};
use p3_util::log2_strict_usize;
use rand::distr::StandardUniform;
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};

use crate::deep::DeepParams;
use crate::fri::{FriFold, FriParams};
use crate::{PcsParams, prover::open, verifier::verify};

pub use p3_miden_dev_utils::configs::baby_bear_poseidon2::{base_lmcs as test_lmcs, *};
pub use p3_miden_dev_utils::matrix::random_lde_matrix;

// ============================================================================
// End-to-end test
// ============================================================================

#[test]
fn test_pcs_open_verify_roundtrip() {
    let rng = &mut SmallRng::seed_from_u64(42);
    let lmcs = test_lmcs();

    let params = PcsParams {
        fri: FriParams {
            log_blowup: 2,
            fold: FriFold::ARITY_2,
            log_final_degree: 2,
            proof_of_work_bits: 1, // Low for fast tests (per-round)
        },
        deep: DeepParams {
            alignment: RATE,
            proof_of_work_bits: 1, // Low for fast tests
        },
        num_queries: 5,
        query_proof_of_work_bits: 1, // Low for fast tests
    };

    // Create a matrix of LDE evaluations.
    // Each column is a random polynomial of degree < 2^log_poly_degree,
    // evaluated on the coset gK (g = F::GENERATOR, K = subgroup of order 2^log_n).
    //
    // The DEEP quotient Q(X) computed from these polynomials will have degree
    // at most 2^log_poly_degree - 1, satisfying FRI's low-degree requirement.
    let log_poly_degree = 6; // polynomial degree = 64
    let num_columns = 3;
    let matrix = random_lde_matrix(
        rng,
        log_poly_degree,
        params.fri.log_blowup,
        num_columns,
        F::GENERATOR,
    );
    let matrices: Vec<RowMajorMatrix<F>> = vec![matrix];

    // Commit matrices via LMCS
    let tree = lmcs.build_tree(matrices);
    let commitment = tree.root();
    let widths: Vec<usize> = tree.leaves().iter().map(|m| m.width()).collect();
    let max_height = tree.leaves().last().map(|m| m.height()).unwrap_or(0);
    let log_max_height = log2_strict_usize(max_height);

    // Evaluation points
    let z1: EF = rng.sample(StandardUniform);
    let z2: EF = rng.sample(StandardUniform);
    let eval_points = [z1, z2];

    // Create slice of tree references for multi-tree API (single tree in this case)
    let trace_trees: &[&_] = &[&tree];

    // Prover
    let mut prover_challenger = test_challenger();
    prover_challenger.observe(commitment);

    let proof = open::<F, EF, _, _, _, 2>(
        &params,
        &lmcs,
        eval_points,
        trace_trees,
        &mut prover_challenger,
    );

    // Create commitments slice for multi-commitment API (single commitment in this case)
    let commitments: &[_] = &[(commitment, widths)];

    // Verifier
    let mut verifier_challenger = test_challenger();
    verifier_challenger.observe(commitment);

    let result = verify::<F, EF, _, _, 2>(
        &params,
        &lmcs,
        commitments,
        log_max_height,
        eval_points,
        &proof,
        &mut verifier_challenger,
    );

    assert!(result.is_ok(), "Verification should succeed: {:?}", result);
    let verified_evals = result.unwrap();
    assert_eq!(verified_evals.len(), 2, "Should have 2 evaluation points");
}
