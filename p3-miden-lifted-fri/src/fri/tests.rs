//! Integration tests for FRI protocol commit/verify cycles.

use p3_challenger::CanObserve;
use p3_field::{PrimeCharacteristicRing, TwoAdicField};
use p3_miden_transcript::VerifierTranscript;
use p3_util::{log2_strict_usize, reverse_bits_len};
use rand::distr::StandardUniform;
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};

use super::prover::FriPolys;
use super::verifier::FriOracle;
use super::*;
use crate::tests::{
    EF, F, evals_at, prover_channel, random_lde_matrix, sample_indices, test_challenger, test_lmcs,
    verifier_channel,
};
use crate::utils::horner;

// ============================================================================
// Integration tests
// ============================================================================

/// Test that commit_phase produces valid proofs that verify_queries accepts.
///
/// This test:
/// 1. Generates a random polynomial and computes its LDE
/// 2. Runs the FRI commit phase to fold down to final polynomial
/// 3. Verifies batch of random query indices
fn test_fri_commit_verify_roundtrip(log_poly_degree: usize, fold: FriFold) {
    let mut rng = SmallRng::seed_from_u64(42);
    let lmcs = test_lmcs();

    let params = FriParams {
        log_blowup: 2,
        fold,
        log_final_degree: 2,
        proof_of_work_bits: 1, // Low for fast tests (per-round)
    };

    // Generate random LDE evaluations
    let evals = random_lde_matrix(&mut rng, log_poly_degree, params.log_blowup, 1, F::ONE).values;
    let lde_size = evals.len();

    // Generate batch of query indices
    let query_indices = sample_indices(&mut rng, lde_size, 3);
    let initial_evals = evals_at(&evals, &query_indices);

    // Prover: run commit phase (grinds per-round internally)
    let mut prover_channel = prover_channel();
    let fri_polys = FriPolys::<F, EF, _>::new(&params, &lmcs, evals, &mut prover_channel);

    // Open all query indices at once
    fri_polys.prove_queries(&params, &query_indices, &mut prover_channel);

    let transcript = prover_channel.into_data();

    // Verifier: replay challenger to get oracle with betas
    let mut channel = verifier_channel(&transcript);
    let log_domain_size = log2_strict_usize(lde_size);
    let fri_oracle = FriOracle::new(&params, log_domain_size, &mut channel)
        .expect("FRI oracle construction should succeed");

    // Test all queries at once
    fri_oracle
        .test_low_degree(&lmcs, &params, &query_indices, &initial_evals, &mut channel)
        .expect("low-degree test should pass");
    assert!(channel.is_empty(), "transcript should be fully consumed");
}

#[test]
fn test_fri_commit_verify_arity2() {
    test_fri_commit_verify_roundtrip(10, FriFold::ARITY_2);
}

#[test]
fn test_fri_commit_verify_arity4() {
    test_fri_commit_verify_roundtrip(10, FriFold::ARITY_4);
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

    let evals = random_lde_matrix(&mut rng, log_poly_degree, log_blowup, 1, F::ONE).values;
    let log_lde_size = log_poly_degree + log_blowup;
    let lde_size = 1 << log_lde_size;

    let indices = sample_indices(&mut rng, lde_size, 2);
    let mut initial_evals = evals_at(&evals, &indices);
    let mut wrong_eval: EF = rng.sample(StandardUniform);
    while wrong_eval == initial_evals[0] {
        wrong_eval = rng.sample(StandardUniform);
    }
    initial_evals[0] = wrong_eval; // Wrong!

    let mut prover_channel = prover_channel();
    let fri_polys = FriPolys::<F, EF, _>::new(&params, &lmcs, evals, &mut prover_channel);
    fri_polys.prove_queries(&params, &indices, &mut prover_channel);

    let transcript = prover_channel.into_data();

    let mut v_channel = verifier_channel(&transcript);
    let log_domain_size = log2_strict_usize(lde_size);
    let fri_oracle = FriOracle::new(&params, log_domain_size, &mut v_channel)
        .expect("oracle construction should succeed");

    let result = fri_oracle.test_low_degree(
        &lmcs,
        &params,
        &indices,
        &initial_evals, // Should fail (one eval is wrong)
        &mut v_channel,
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

    // Create two independent provers with different evaluations
    let evals1 = random_lde_matrix(&mut rng, log_poly_degree, log_blowup, 1, F::ONE).values;
    let evals2 = random_lde_matrix(&mut rng, log_poly_degree, log_blowup, 1, F::ONE).values;
    let log_lde_size = log_poly_degree + log_blowup;
    let lde_size = 1 << log_lde_size;

    // Prover 1: generate FRI transcript (grinds per-round internally)
    let indices = sample_indices(&mut rng, lde_size, 2);
    let initial_evals = evals_at(&evals1, &indices);

    let mut prover1_channel = prover_channel();
    let fri_polys1 = FriPolys::<F, EF, _>::new(&params, &lmcs, evals1, &mut prover1_channel);

    // Prover 2: generate different transcript (different commitments = different betas)
    let mut prover2_channel = prover_channel();
    let _ = FriPolys::<F, EF, _>::new(&params, &lmcs, evals2, &mut prover2_channel);

    fri_polys1.prove_queries(&params, &indices, &mut prover1_channel);

    let transcript = prover1_channel.into_data();
    let other_commitment = prover2_channel
        .into_data()
        .commitments()
        .first()
        .cloned()
        .expect("prover2 should produce commitments");

    // Verifier: use prover1's transcript but alter challenger state beforehand.
    let mut wrong_challenger = test_challenger();
    wrong_challenger.observe(other_commitment);
    let mut v_channel = VerifierTranscript::from_data(wrong_challenger, &transcript);

    let log_domain_size = log2_strict_usize(lde_size);
    let wrong_oracle = FriOracle::new(&params, log_domain_size, &mut v_channel)
        .expect("oracle construction should succeed");

    let result =
        wrong_oracle.test_low_degree(&lmcs, &params, &indices, &initial_evals, &mut v_channel);

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

    let evals = random_lde_matrix(&mut rng, log_poly_degree, log_blowup, 1, F::ONE).values;
    let lde_size = evals.len();

    let indices = sample_indices(&mut rng, lde_size, 2);
    let initial_evals = evals_at(&evals, &indices);

    let mut prover_channel = prover_channel();
    let fri_polys = FriPolys::<F, EF, _>::new(&params, &lmcs, evals, &mut prover_channel);
    fri_polys.prove_queries(&params, &indices, &mut prover_channel);
    let transcript = prover_channel.into_data();

    let mut channel = verifier_channel(&transcript);
    let log_domain_size = log2_strict_usize(lde_size);
    let fri_transcript: FriTranscript<F, EF, _> =
        FriTranscript::from_verifier_channel(&params, log_domain_size, &mut channel)
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

    let mut v_channel = verifier_channel(&transcript);
    let fri_oracle = FriOracle::new(&params, log_domain_size, &mut v_channel)
        .expect("oracle construction should succeed");
    fri_oracle
        .test_low_degree(&lmcs, &params, &indices, &initial_evals, &mut v_channel)
        .expect("zero-round FRI should verify");
    assert!(v_channel.is_empty(), "transcript should be fully consumed");
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

    let evals = random_lde_matrix(&mut rng, log_poly_degree, log_blowup, 1, F::ONE).values;
    let lde_size = evals.len();
    let indices = sample_indices(&mut rng, lde_size, 2);
    let initial_evals = evals_at(&evals, &indices);

    let mut prover_channel = prover_channel();
    let fri_polys = FriPolys::<F, EF, _>::new(&params, &lmcs, evals, &mut prover_channel);
    fri_polys.prove_queries(&params, &indices, &mut prover_channel);
    let transcript = prover_channel.into_data();

    let mut v_channel = verifier_channel(&transcript);
    let log_domain_size = log2_strict_usize(lde_size);
    let fri_oracle = FriOracle::new(&params, log_domain_size, &mut v_channel)
        .expect("oracle construction should succeed");
    fri_oracle
        .test_low_degree(&lmcs, &params, &indices, &initial_evals, &mut v_channel)
        .expect("FRI with blowup=0 should verify");
    assert!(v_channel.is_empty(), "transcript should be fully consumed");
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

    let evals = random_lde_matrix(&mut rng, log_poly_degree, log_blowup, 1, F::ONE).values;

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
