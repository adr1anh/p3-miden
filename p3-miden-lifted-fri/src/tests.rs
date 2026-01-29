//! Common test fixtures and end-to-end tests for the lifted FRI PCS.
//!
//! Re-exports test fixtures from `p3_miden_dev_utils` for use in tests.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::CanObserve;
use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_lmcs::{Lmcs, LmcsConfig, LmcsTree};
use p3_miden_transcript::{ProverTranscript, TranscriptData, VerifierChannel, VerifierTranscript};
use p3_util::log2_strict_usize;
use rand::distr::StandardUniform;
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};

use crate::deep::DeepParams;
use crate::fri::{FriFold, FriParams};
use crate::{PcsParams, prover::open_with_channel, verifier::verify_with_channel};

pub use p3_miden_dev_utils::configs::baby_bear_poseidon2::*;
pub use p3_miden_dev_utils::matrix::random_lde_matrix;

pub type BaseLmcs = LmcsConfig<P, P, Sponge, Compress, WIDTH, DIGEST>;
pub type TestCommitment = <BaseLmcs as Lmcs>::Commitment;
pub type TestTranscriptData = TranscriptData<F, TestCommitment>;
pub type TestProverChannel = ProverTranscript<F, TestCommitment, Challenger>;
pub type TestVerifierChannel<'a> = VerifierTranscript<'a, F, TestCommitment, Challenger>;

pub fn test_lmcs() -> BaseLmcs {
    let (_, sponge, compress) = test_components();
    LmcsConfig::new_aligned(sponge, compress)
}

pub fn prover_channel() -> TestProverChannel {
    ProverTranscript::new(test_challenger())
}

pub fn prover_channel_with_commitment(commitment: &TestCommitment) -> TestProverChannel {
    let mut challenger = test_challenger();
    challenger.observe(*commitment);
    ProverTranscript::new(challenger)
}

pub fn verifier_channel(data: &'_ TestTranscriptData) -> TestVerifierChannel<'_> {
    VerifierTranscript::from_data(test_challenger(), data)
}

pub fn verifier_channel_with_commitment<'a>(
    data: &'a TestTranscriptData,
    commitment: &TestCommitment,
) -> TestVerifierChannel<'a> {
    let mut challenger = test_challenger();
    challenger.observe(*commitment);
    VerifierTranscript::from_data(challenger, data)
}

pub fn sample_indices<R: Rng>(rng: &mut R, upper: usize, count: usize) -> Vec<usize> {
    let mut indices = Vec::with_capacity(count);
    for _ in 0..count {
        indices.push(rng.random_range(0..upper));
    }
    indices
}

pub fn evals_at<T: Copy>(values: &[T], indices: &[usize]) -> Vec<T> {
    indices.iter().map(|&idx| values[idx]).collect()
}

// ============================================================================
// End-to-end test
// ============================================================================

#[test]
fn test_pcs_open_verify_roundtrip() {
    let rng = &mut SmallRng::seed_from_u64(42);
    let lmcs = test_lmcs();

    let log_blowup = 2;
    let log_final_degree = 2;
    let log_poly_degree = 6; // polynomial degree = 64

    let fri = FriParams {
        log_blowup,
        fold: FriFold::ARITY_2,
        log_final_degree,
        proof_of_work_bits: 1, // Low for fast tests (per-round)
    };
    let deep = DeepParams {
        proof_of_work_bits: 1, // Low for fast tests
    };
    let params = PcsParams {
        deep,
        fri,
        num_queries: 5,
        query_proof_of_work_bits: 1,
    };

    // Create a matrix of LDE evaluations.
    // Each column is a random polynomial of degree < 2^log_poly_degree,
    // evaluated on the coset gK (g = F::GENERATOR, K = subgroup of order 2^log_n).
    //
    // The DEEP quotient Q(X) computed from these polynomials will have degree
    // at most 2^log_poly_degree - 1, satisfying FRI's low-degree requirement.
    let num_columns = 3;
    let matrix = random_lde_matrix(rng, log_poly_degree, log_blowup, num_columns, F::GENERATOR);
    let matrices: Vec<RowMajorMatrix<F>> = vec![matrix];

    // Commit matrices via LMCS
    let tree = lmcs.build_tree(matrices);
    let commitment = tree.root();
    let widths = tree.widths();
    let lde_height = tree.leaves().last().map(|m| m.height()).unwrap_or(0);
    let log_lde_height = log2_strict_usize(lde_height);

    // Evaluation points
    let z1: EF = rng.sample(StandardUniform);
    let z2: EF = rng.sample(StandardUniform);
    let eval_points = [z1, z2];

    // Create slice of tree references for multi-tree API (single tree in this case)
    let trace_trees: &[&_] = &[&tree];

    // Prover
    let mut prover_channel = prover_channel_with_commitment(&commitment);

    open_with_channel::<F, EF, _, _, _, 2>(
        &params,
        &lmcs,
        log_lde_height,
        eval_points,
        trace_trees,
        &mut prover_channel,
    );
    let transcript = prover_channel.into_data();

    // Create commitments slice for multi-commitment API (single commitment in this case)
    let commitments: &[_] = &[(commitment, widths)];

    // Verifier
    let mut verifier_channel = verifier_channel_with_commitment(&transcript, &commitment);

    let result = verify_with_channel::<F, EF, _, _, 2>(
        &params,
        &lmcs,
        commitments,
        log_lde_height,
        eval_points,
        &mut verifier_channel,
    );

    assert!(result.is_ok(), "Verification should succeed: {:?}", result);
    let verified_evals = result.unwrap();
    assert_eq!(
        verified_evals.num_points(),
        2,
        "Should have 2 evaluation points"
    );
    assert!(
        verifier_channel.is_empty(),
        "transcript should be fully consumed"
    );
}
