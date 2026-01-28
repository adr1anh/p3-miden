//! End-to-end tests for DEEP quotient prover/verifier agreement.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::FieldArray;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_lmcs::{Lmcs, LmcsTree};
use rand::distr::StandardUniform;
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};

use super::DeepParams;
use super::interpolate::PointQuotients;
use super::prover::DeepPoly;
use super::verifier::DeepOracle;
use crate::tests::{
    EF, F, prover_channel_with_commitment, test_lmcs, verifier_channel_with_commitment,
};
use crate::utils::bit_reversed_coset_points;

/// End-to-end: prover's `DeepPoly.open()` must match verifier's channel-based openings.
#[test]
fn deep_quotient_end_to_end() {
    let rng = &mut SmallRng::seed_from_u64(42);
    let lmcs = test_lmcs();

    // Parameters
    let log_blowup: usize = 2;
    let log_max_height = 10;
    let max_height = 1 << log_max_height;

    let params = DeepParams {
        proof_of_work_bits: 1, // Low for fast tests
    };
    let alignment = lmcs.alignment();

    // Two random opening points
    let z1: EF = rng.sample(StandardUniform);
    let z2: EF = rng.sample(StandardUniform);

    // Coset points in bit-reversed order
    let coset_points = bit_reversed_coset_points::<F>(log_max_height);

    // Create matrices of varying heights (ascending order required)
    // specs: (log_scaling, width) where height = n >> log_scaling
    let specs: Vec<(usize, usize)> = vec![(2, 2), (1, 3), (0, 4)]; // heights: n/4, n/2, n
    let matrices: Vec<RowMajorMatrix<F>> = specs
        .iter()
        .map(|&(log_scaling, width)| {
            let height = max_height >> log_scaling;
            RowMajorMatrix::rand(rng, height, width)
        })
        .collect();

    // Step 1: Commit matrices via LMCS
    let tree = lmcs.build_tree(matrices);
    let commitment = tree.root();
    let widths = tree.widths();

    // Step 2: Compute batched evaluations at both opening points
    let quotient = PointQuotients::<F, EF, 2>::new(FieldArray([z1, z2]), &coset_points);

    let matrices_ref: Vec<&RowMajorMatrix<F>> = tree.leaves().iter().collect();
    let matrices_groups = vec![matrices_ref];
    let batched_evals = quotient.batch_eval_lifted(&matrices_groups, &coset_points, log_blowup);

    // Step 3: Prover constructs DeepPoly (handles observe, grind, sample internally)
    let mut prover_channel = prover_channel_with_commitment(&commitment);
    let deep_poly = DeepPoly::new(
        &params,
        &matrices_groups,
        batched_evals,
        &quotient,
        alignment,
        &mut prover_channel,
    );
    let sample_indices = vec![0, 1, max_height / 4, max_height / 2, max_height - 1];
    tree.prove_batch(&sample_indices, &mut prover_channel);
    let transcript = prover_channel.into_data();

    // Create commitments slice for multi-commitment API (single commitment in this case)
    let commitments = vec![(commitment, widths)];

    // Step 4: Verifier constructs DeepOracle with same transcript state
    let mut verifier_channel = verifier_channel_with_commitment(&transcript, &commitment);
    let (deep_oracle, _evals) = DeepOracle::new(
        &params,
        &[z1, z2],
        commitments,
        log_max_height,
        &mut verifier_channel,
    )
    .expect("DeepOracle construction should succeed");

    // Step 5: Verify at multiple query indices (proofs are read from transcript)
    let verifier_evals = deep_oracle
        .open_batch(&lmcs, &sample_indices, &mut verifier_channel)
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
