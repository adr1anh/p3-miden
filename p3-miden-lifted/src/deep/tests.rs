//! End-to-end tests for DEEP quotient prover/verifier agreement.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::CanObserve;
use p3_commit::Mmcs;
use p3_field::FieldArray;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use rand::distr::StandardUniform;
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};

use super::interpolate::PointQuotients;
use super::prover::DeepPoly;
use super::verifier::DeepOracle;
use super::{DeepParams, MatrixGroupEvals};
use crate::tests::{EF, F, RATE, base_lmcs, challenger};
use crate::utils::bit_reversed_coset_points;

// ============================================================================
// End-to-end test
// ============================================================================

/// End-to-end: prover's `DeepPoly.open()` must match verifier's `DeepOracle.query()`.
#[test]
fn deep_quotient_end_to_end() {
    let rng = &mut SmallRng::seed_from_u64(42);
    let lmcs = base_lmcs();

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
    let (commitment, prover_data) = lmcs.commit(matrices.clone());
    let dims: Vec<_> = matrices.iter().map(|m| m.dimensions()).collect();

    // Step 2: Compute batched evaluations at both opening points
    let quotient = PointQuotients::<F, EF, 2>::new(FieldArray([z1, z2]), &coset_points);

    let matrices_ref: Vec<&RowMajorMatrix<F>> = matrices.iter().collect();
    let matrices_groups = [matrices_ref];
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
    let mut prover_challenger = challenger();
    prover_challenger.observe(commitment);
    let (deep_poly, deep_proof) = DeepPoly::new(
        &params,
        &lmcs,
        vec![&prover_data],
        &evals,
        &batched_evals,
        &quotient,
        &mut prover_challenger,
    );

    // Step 4: Verifier constructs DeepOracle with same transcript state
    let mut verifier_challenger = challenger();
    verifier_challenger.observe(commitment);
    let deep_oracle = DeepOracle::new(
        &params,
        [z1, z2],
        &evals,
        vec![(commitment, dims)],
        &mut verifier_challenger,
        &deep_proof,
    )
    .expect("DeepOracle construction should succeed");

    // Step 5: Verify at random query indices
    let sample_indices = [0, 1, n / 4, n / 2, n - 1];
    for &index in &sample_indices {
        // Prover opens at index
        let deep_query = deep_poly.open(&lmcs, index);

        // Verifier evaluates at index (also verifies Merkle proofs)
        let verifier_eval = deep_oracle
            .query(&lmcs, index, &deep_query)
            .expect("Merkle verification should pass");

        let prover_eval = deep_poly.deep_evals[index];
        assert_eq!(
            prover_eval, verifier_eval,
            "Prover and verifier disagree at index {index}"
        );
    }
}
