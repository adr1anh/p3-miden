//! End-to-end tests for DEEP quotient prover/verifier agreement.

use alloc::vec;
use alloc::vec::Vec;

use p3_matrix::dense::RowMajorMatrix;
use p3_miden_lmcs::{Lmcs, LmcsTree};
use rand::distr::StandardUniform;
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};

use super::DeepParams;
use super::prover::DeepPoly;
use super::verifier::DeepOracle;
use crate::tests::{
    EF, F, prover_channel_with_commitment, test_lmcs, verifier_channel_with_commitment,
};

/// End-to-end: prover's `DeepPoly.open()` must match verifier's channel-based openings.
#[test]
fn deep_quotient_end_to_end() {
    let rng = &mut SmallRng::seed_from_u64(42);
    let lmcs = test_lmcs();

    // Parameters
    let log_blowup: usize = 2;
    let log_lde_height = 10;
    let lde_height = 1 << log_lde_height;

    let params = DeepParams {
        proof_of_work_bits: 1, // Low for fast tests
    };
    // Two random opening points
    let z1: EF = rng.sample(StandardUniform);
    let z2: EF = rng.sample(StandardUniform);

    // Create matrices of varying heights (ascending order required)
    // specs: (log_scaling, width) where height = n >> log_scaling
    let specs: Vec<(usize, usize)> = vec![(2, 2), (1, 3), (0, 4)]; // heights: n/4, n/2, n
    let matrices: Vec<RowMajorMatrix<F>> = specs
        .iter()
        .map(|&(log_scaling, width)| {
            let height = lde_height >> log_scaling;
            RowMajorMatrix::rand(rng, height, width)
        })
        .collect();

    // Step 1: Commit matrices via LMCS (aligned for trace commitments)
    let tree = lmcs.build_aligned_tree(matrices);
    let commitment = tree.root();
    let widths = tree.widths();

    // Step 3: Prover constructs DeepPoly (handles observe, grind, sample internally)
    let mut prover_channel = prover_channel_with_commitment(&commitment);
    let trace_trees: &[&_] = &[&tree];
    let deep_poly = DeepPoly::from_trees::<crate::tests::BaseLmcs, _, 2, _>(
        &params,
        trace_trees,
        [z1, z2],
        log_blowup,
        &mut prover_channel,
    );
    let sample_indices = vec![0, 1, lde_height / 4, lde_height / 2, lde_height - 1];
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
        log_lde_height,
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
