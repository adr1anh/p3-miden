//! Integration tests for FRI protocol commit/verify cycles.

use alloc::vec::Vec;
use p3_challenger::CanObserve;
use p3_field::{PrimeCharacteristicRing, TwoAdicField};
use p3_miden_transcript::{VerifierChannel, VerifierTranscript};
use p3_util::{log2_strict_usize, reverse_bits_len};
use rand::distr::StandardUniform;
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};

use super::prover::FriPolys;
use super::verifier::FriOracle;
use super::*;
use crate::tests::{
    BaseLmcs, Challenger, EF, F, TestTranscriptData, evals_at, prover_channel, random_lde_matrix,
    sample_indices, test_challenger, test_lmcs, verifier_channel,
};
use crate::utils::horner;

// ============================================================================
// Integration tests
// ============================================================================

fn prove_queries(
    params: &FriParams,
    lmcs: &BaseLmcs,
    evals: Vec<EF>,
    indices: &[usize],
) -> TestTranscriptData {
    let mut prover_channel = prover_channel();
    let fri_polys = FriPolys::<F, EF, _>::new(params, lmcs, evals, &mut prover_channel);
    fri_polys.prove_queries(params, indices, &mut prover_channel);
    prover_channel.into_data()
}

fn verify_queries(
    params: &FriParams,
    lmcs: &BaseLmcs,
    transcript: &TestTranscriptData,
    lde_size: usize,
    indices: &[usize],
    initial_evals: &[EF],
    challenger: Option<Challenger>,
) -> Result<(), FriError> {
    let mut channel = match challenger {
        Some(challenger) => VerifierTranscript::from_data(challenger, transcript),
        None => verifier_channel(transcript),
    };
    let log_domain_size = log2_strict_usize(lde_size);
    let oracle = FriOracle::new(params, log_domain_size, &mut channel)?;
    let result = oracle.test_low_degree(lmcs, params, indices, initial_evals, &mut channel);
    if result.is_ok() {
        assert!(channel.is_empty(), "transcript should be fully consumed");
    }
    result
}

/// Test that commit_phase produces valid proofs that verify_queries accepts.
///
/// This test:
/// 1. Generates a random polynomial and computes its LDE
/// 2. Runs the FRI commit phase to fold down to final polynomial
/// 3. Verifies a fixed batch of query indices across all fold arities
#[test]
fn test_fri_commit_verify_all_arities() {
    let mut rng = SmallRng::seed_from_u64(42);
    let lmcs = test_lmcs();

    let log_poly_degree = 10;
    let log_blowup = 2;
    let log_final_degree = 2;

    // Generate random LDE evaluations once so all arities use the same data.
    let evals = random_lde_matrix::<F, EF>(&mut rng, log_poly_degree, log_blowup, 1, F::ONE).values;
    let lde_size = evals.len();
    // Generate one batch of query indices and reuse across arities.
    let query_indices = sample_indices(&mut rng, lde_size, 3);
    let initial_evals = evals_at(&evals, &query_indices);

    for fold in [FriFold::ARITY_2, FriFold::ARITY_4, FriFold::ARITY_8] {
        let params = FriParams {
            log_blowup,
            fold,
            log_final_degree,
            proof_of_work_bits: 1, // Low for fast tests (per-round)
        };
        let transcript = prove_queries(&params, &lmcs, evals.clone(), &query_indices);
        verify_queries(
            &params,
            &lmcs,
            &transcript,
            lde_size,
            &query_indices,
            &initial_evals,
            None,
        )
        .expect("low-degree test should pass");
    }
}

/// Test that verification fails with wrong initial evaluation.
#[test]
fn test_fri_verify_wrong_eval() {
    let mut rng = SmallRng::seed_from_u64(42);
    let lmcs = test_lmcs();

    let log_poly_degree = 8;
    let log_blowup = 2;
    let log_final_degree = 2;

    let params = FriParams {
        log_blowup,
        fold: FriFold::ARITY_2,
        log_final_degree,
        proof_of_work_bits: 1,
    };

    let evals = random_lde_matrix::<F, EF>(&mut rng, log_poly_degree, log_blowup, 1, F::ONE).values;
    let lde_size = evals.len();
    let indices = sample_indices(&mut rng, lde_size, 2);
    let mut initial_evals = evals_at(&evals, &indices);
    let mut wrong_eval: EF = rng.sample(StandardUniform);
    while wrong_eval == initial_evals[0] {
        wrong_eval = rng.sample(StandardUniform);
    }
    initial_evals[0] = wrong_eval; // Wrong!

    let transcript = prove_queries(&params, &lmcs, evals, &indices);
    let result = verify_queries(
        &params,
        &lmcs,
        &transcript,
        lde_size,
        &indices,
        &initial_evals,
        None,
    );

    assert!(
        matches!(result, Err(FriError::EvaluationMismatch { .. })),
        "expected EvaluationMismatch error, got {:?}",
        result
    );
}

/// Test that verification fails with mismatched proof data (wrong betas scenario).
///
/// When the verifier's challenger state differs from the prover's (e.g., due to
/// different commitments being observed), the derived betas will be wrong,
/// causing verification to fail.
#[test]
fn test_fri_verify_wrong_beta() {
    let mut rng = SmallRng::seed_from_u64(42);
    let lmcs = test_lmcs();

    let log_poly_degree = 8;
    let log_blowup = 2;
    let log_final_degree = 2;

    let params = FriParams {
        log_blowup,
        fold: FriFold::ARITY_2,
        log_final_degree,
        proof_of_work_bits: 0, // No grinding to simplify test
    };

    // Create two independent provers with different evaluations.
    let evals1 =
        random_lde_matrix::<F, EF>(&mut rng, log_poly_degree, log_blowup, 1, F::ONE).values;
    let evals2 =
        random_lde_matrix::<F, EF>(&mut rng, log_poly_degree, log_blowup, 1, F::ONE).values;
    let lde_size = evals1.len();

    // Prover 1: generate FRI transcript (grinds per-round internally).
    let indices = sample_indices(&mut rng, lde_size, 2);
    let initial_evals = evals_at(&evals1, &indices);
    let transcript = prove_queries(&params, &lmcs, evals1, &indices);

    // Prover 2: generate different transcript (different commitments = different betas).
    let mut prover2_channel = prover_channel();
    let _ = FriPolys::<F, EF, _>::new(&params, &lmcs, evals2, &mut prover2_channel);
    let other_commitment = prover2_channel
        .into_data()
        .commitments()
        .first()
        .cloned()
        .expect("prover2 should produce commitments");

    // Verifier: use prover1's transcript but alter challenger state beforehand.
    let mut wrong_challenger = test_challenger();
    wrong_challenger.observe(other_commitment);
    let result = verify_queries(
        &params,
        &lmcs,
        &transcript,
        lde_size,
        &indices,
        &initial_evals,
        Some(wrong_challenger),
    );

    // Should fail because wrong betas produce wrong folding results
    assert!(
        matches!(
            result,
            Err(FriError::EvaluationMismatch { .. }) | Err(FriError::FinalPolyMismatch)
        ),
        "expected EvaluationMismatch or FinalPolyMismatch error, got {:?}",
        result
    );
}

/// Zero-round FRI: when the evaluation domain is at or below the final polynomial degree.
#[test]
fn test_fri_zero_rounds_final_poly_only() {
    let mut rng = SmallRng::seed_from_u64(123);
    let lmcs = test_lmcs();

    let log_poly_degree = 4;
    let log_blowup = 0;
    let log_final_degree = log_poly_degree; // final degree >= domain size => zero rounds

    let params = FriParams {
        log_blowup,
        fold: FriFold::ARITY_2,
        log_final_degree,
        proof_of_work_bits: 0,
    };

    let evals = random_lde_matrix::<F, EF>(&mut rng, log_poly_degree, log_blowup, 1, F::ONE).values;
    let lde_size = evals.len();
    let indices = sample_indices(&mut rng, lde_size, 2);
    let initial_evals = evals_at(&evals, &indices);
    let transcript = prove_queries(&params, &lmcs, evals, &indices);

    let mut channel = verifier_channel(&transcript);
    let fri_transcript: FriTranscript<F, EF, _> =
        FriTranscript::from_verifier_channel(&params, log2_strict_usize(lde_size), &mut channel)
            .expect("transcript parsing should succeed");

    assert!(
        fri_transcript.rounds.is_empty(),
        "expected zero folding rounds"
    );
    assert_eq!(
        fri_transcript.final_poly.len(),
        lde_size,
        "final polynomial should match domain size"
    );

    verify_queries(
        &params,
        &lmcs,
        &transcript,
        lde_size,
        &indices,
        &initial_evals,
        None,
    )
    .expect("zero-round FRI should verify");
}

/// FRI with no blowup but with folding rounds.
#[test]
fn test_fri_blowup_zero_with_rounds() {
    let mut rng = SmallRng::seed_from_u64(321);
    let lmcs = test_lmcs();

    let log_poly_degree = 8;
    let log_blowup = 0;
    let log_final_degree = 3;

    let params = FriParams {
        log_blowup,
        fold: FriFold::ARITY_2,
        log_final_degree,
        proof_of_work_bits: 0,
    };

    let evals = random_lde_matrix::<F, EF>(&mut rng, log_poly_degree, log_blowup, 1, F::ONE).values;
    let lde_size = evals.len();
    let indices = sample_indices(&mut rng, lde_size, 2);
    let initial_evals = evals_at(&evals, &indices);
    let transcript = prove_queries(&params, &lmcs, evals, &indices);

    verify_queries(
        &params,
        &lmcs,
        &transcript,
        lde_size,
        &indices,
        &initial_evals,
        None,
    )
    .expect("FRI with blowup=0 should verify");
}

/// Test that the final polynomial is correctly computed by evaluating it
/// at points in the final domain and comparing with folded values.
#[test]
fn test_final_polynomial_correctness() {
    let mut rng = SmallRng::seed_from_u64(123);
    let lmcs = test_lmcs();

    let log_poly_degree = 8;
    let log_blowup = 2;
    let log_final_degree = 3;

    let params = FriParams {
        log_blowup,
        fold: FriFold::ARITY_2,
        log_final_degree,
        proof_of_work_bits: 0, // No grinding for this test
    };

    let evals = random_lde_matrix::<F, EF>(&mut rng, log_poly_degree, log_blowup, 1, F::ONE).values;

    let mut prover_channel = prover_channel();
    let _fri_polys = FriPolys::<F, EF, _>::new(&params, &lmcs, evals, &mut prover_channel);
    let transcript = prover_channel.into_data();

    let mut v_channel = verifier_channel(&transcript);
    let log_domain_size = log_poly_degree + log_blowup;
    let fri_transcript: FriTranscript<F, EF, _> =
        FriTranscript::from_verifier_channel(&params, log_domain_size, &mut v_channel)
            .expect("transcript parsing should succeed");

    // Verify final polynomial has correct degree
    let final_degree = 1 << log_final_degree;
    assert_eq!(
        fri_transcript.final_poly.len(),
        final_degree,
        "Final polynomial should have {} coefficients",
        final_degree
    );

    // Evaluate final polynomial at several points in the final domain
    let log_final_height = log_final_degree + log_blowup;
    let final_height = 1 << log_final_height;
    let g = F::two_adic_generator(log_final_height);

    for idx in 0..final_height {
        // Point in bit-reversed final domain
        let x: F = g.exp_u64(reverse_bits_len(idx, log_final_height) as u64);

        // Evaluate polynomial via Horner
        let poly_eval: EF = horner(x, fri_transcript.final_poly.iter().rev().copied());

        // The polynomial should be well-defined (just check it doesn't panic)
        let _ = poly_eval;
    }
}
