//! End-to-end tests for DEEP quotient prover/verifier agreement.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::CanObserve;
use p3_field::FieldArray;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_lmcs::{Lmcs, LmcsTree};
use rand::distr::StandardUniform;
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};

use super::DeepParams;
use super::interpolate::PointQuotients;
use super::prover::DeepPoly;
use super::verifier::DeepOracle;
use crate::tests::{EF, F, RATE, test_challenger, test_lmcs};
use crate::utils::{MatrixGroupEvals, bit_reversed_coset_points};

// ============================================================================
// End-to-end test
// ============================================================================

/// End-to-end: prover's `DeepPoly.open()` must match verifier's `DeepOracle.query()`.
#[test]
fn deep_quotient_end_to_end() {
    let rng = &mut SmallRng::seed_from_u64(42);
    let lmcs = test_lmcs();

    // Parameters
    let log_blowup: usize = 2;
    let log_n = 10;
    let n = 1 << log_n;

    let params = DeepParams {
        alignment: RATE,       // Use sponge rate for coefficient alignment
        proof_of_work_bits: 1, // Low for fast tests
    };

    // Two random opening points
    let z1: EF = rng.sample(StandardUniform);
    let z2: EF = rng.sample(StandardUniform);

    // Coset points in bit-reversed order
    let coset_points = bit_reversed_coset_points::<F>(log_n);

    // Create matrices of varying heights (ascending order required)
    // specs: (log_scaling, width) where height = n >> log_scaling
    let specs: Vec<(usize, usize)> = vec![(2, 2), (1, 3), (0, 4)]; // heights: n/4, n/2, n
    let matrices: Vec<RowMajorMatrix<F>> = specs
        .iter()
        .map(|&(log_scaling, width)| {
            let height = n >> log_scaling;
            RowMajorMatrix::rand(rng, height, width)
        })
        .collect();

    // Step 1: Commit matrices via LMCS
    let tree = lmcs.build_tree(matrices);
    let commitment = tree.root();
    let dims: Vec<_> = tree.leaves().iter().map(|m| m.dimensions()).collect();

    // Step 2: Compute batched evaluations at both opening points
    let quotient = PointQuotients::<F, EF, 2>::new(FieldArray([z1, z2]), &coset_points);

    let matrices_ref: Vec<&RowMajorMatrix<F>> = tree.leaves().iter().collect();
    let matrices_groups = vec![matrices_ref];
    let batched_evals = quotient.batch_eval_lifted(&matrices_groups, &coset_points, log_blowup);

    // Transpose batched evals to per-point format: [point][group][matrix][col]
    let evals: Vec<Vec<MatrixGroupEvals<EF>>> = (0..2)
        .map(|point_idx| {
            batched_evals
                .iter()
                .map(|g| g.map(|arr| arr[point_idx]))
                .collect()
        })
        .collect();

    // Step 3: Prover constructs DeepPoly (handles observe, grind, sample internally)
    let mut prover_challenger = test_challenger();
    prover_challenger.observe(commitment);
    let (deep_poly, deep_proof) = DeepPoly::new(
        &params,
        &matrices_groups,
        &evals,
        &batched_evals,
        &quotient,
        &mut prover_challenger,
    );

    // Create commitments slice for multi-commitment API (single commitment in this case)
    let commitments = vec![(commitment, dims)];

    // Step 4: Verifier constructs DeepOracle with same transcript state
    let mut verifier_challenger = test_challenger();
    verifier_challenger.observe(commitment);
    let deep_oracle = DeepOracle::new(
        &params,
        [z1, z2],
        &evals,
        commitments,
        &mut verifier_challenger,
        &deep_proof,
    )
    .expect("DeepOracle construction should succeed");

    // Step 5: Verify at multiple query indices
    let sample_indices = vec![0, 1, n / 4, n / 2, n - 1];

    // Prover opens at all indices at once (one proof per tree)
    let trace_query_proofs = vec![tree.open_multi(&sample_indices)];

    // Verifier evaluates at all indices (also verifies Merkle proofs)
    let verifier_evals = deep_oracle
        .query(&lmcs, &sample_indices, &trace_query_proofs)
        .expect("Merkle verification should pass");

    for (i, &index) in sample_indices.iter().enumerate() {
        let prover_eval = deep_poly.deep_evals[index];
        let verifier_eval = verifier_evals[i];
        assert_eq!(
            prover_eval, verifier_eval,
            "Prover and verifier disagree at index {index}"
        );
    }
}
