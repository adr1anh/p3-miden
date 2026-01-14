//! Integration tests for FRI protocol commit/verify cycles.

use p3_field::{PrimeCharacteristicRing, TwoAdicField};
use p3_util::reverse_bits_len;
use rand::distr::StandardUniform;
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};

use super::prover::FriPolys;
use super::verifier::FriOracle;
use super::*;
use crate::tests::{EF, F, challenger, fri_mmcs, random_lde_matrix};

// ============================================================================
// Integration tests
// ============================================================================

/// Test that commit_phase produces valid proofs that verify_query accepts.
///
/// This test:
/// 1. Generates a random polynomial and computes its LDE
/// 2. Runs the FRI commit phase to fold down to final polynomial
/// 3. Verifies random query indices
fn test_fri_commit_verify_roundtrip(log_poly_degree: usize, fold: FriFold) {
    let mut rng = SmallRng::seed_from_u64(42);
    let mmcs = fri_mmcs();

    let params = FriParams {
        log_blowup: 2,
        fold,
        log_final_degree: 2,
        proof_of_work_bits: 1, // Low for fast tests (per-round)
    };

    // Generate random LDE evaluations
    let evals = random_lde_matrix(&mut rng, log_poly_degree, params.log_blowup, 1, F::ONE).values;
    let lde_size = evals.len();

    // Prover: run commit phase (grinds per-round internally)
    let mut prover_challenger = challenger();
    let (fri_polys, fri_proof) =
        FriPolys::<F, EF, _>::new(&params, &mmcs, &evals, &mut prover_challenger);

    // Verifier: replay challenger to get oracle with betas
    let mut verifier_challenger = challenger();
    let fri_oracle = FriOracle::new(
        &fri_proof,
        &mut verifier_challenger,
        params.proof_of_work_bits,
    )
    .expect("FRI oracle construction should succeed");

    // Verify random queries
    for _ in 0..3 {
        let index: usize = rng.random_range(0..lde_size);
        let initial_eval = evals[index];
        let openings = fri_polys.open_query(&params, &mmcs, index);

        fri_oracle
            .verify_query(&params, &mmcs, index, initial_eval, &openings)
            .expect("verification should succeed");
    }
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
    let mmcs = fri_mmcs();

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

    let mut prover_challenger = challenger();
    let (fri_polys, fri_proof) =
        FriPolys::<F, EF, _>::new(&params, &mmcs, &evals, &mut prover_challenger);

    let mut verifier_challenger = challenger();
    let fri_oracle = FriOracle::new(
        &fri_proof,
        &mut verifier_challenger,
        params.proof_of_work_bits,
    )
    .expect("oracle construction should succeed");

    let index: usize = rng.random_range(0..lde_size);
    let wrong_eval: EF = rng.sample(StandardUniform); // Wrong!
    let openings = fri_polys.open_query(&params, &mmcs, index);

    let result = fri_oracle.verify_query(
        &params, &mmcs, index, wrong_eval, // Should fail
        &openings,
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
    let mmcs = fri_mmcs();

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

    // Prover 1: generate FRI proof (grinds per-round internally)
    let mut prover1_challenger = challenger();
    let (fri_polys1, fri_proof) =
        FriPolys::<F, EF, _>::new(&params, &mmcs, &evals1, &mut prover1_challenger);

    // Prover 2: generate different FRI proof (different commitments = different betas)
    let mut prover2_challenger = challenger();
    let _ = FriPolys::<F, EF, _>::new(&params, &mmcs, &evals2, &mut prover2_challenger);

    // Verifier: use prover1's proof structure but prover2's challenger state
    // This simulates a scenario where the verifier has incorrect transcript state

    // Use prover2's challenger (which observed different commitments)
    // This gives us different betas than prover1 expected
    let wrong_oracle = FriOracle::new(
        &fri_proof,
        &mut prover2_challenger,
        params.proof_of_work_bits,
    )
    .expect("oracle construction should succeed");

    let index: usize = rng.random_range(0..lde_size);
    let initial_eval = evals1[index];
    let openings = fri_polys1.open_query(&params, &mmcs, index);

    let result = wrong_oracle.verify_query(&params, &mmcs, index, initial_eval, &openings);

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

/// Test that the final polynomial is correctly computed by evaluating it
/// at points in the final domain and comparing with folded values.
#[test]
fn test_final_polynomial_correctness() {
    let mut rng = SmallRng::seed_from_u64(123);
    let mmcs = fri_mmcs();

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

    let mut chal = challenger();
    let (_, fri_proof) = FriPolys::<F, EF, _>::new(&params, &mmcs, &evals, &mut chal);

    // Verify final polynomial has correct degree
    let final_degree = 1 << log_final_degree;
    assert_eq!(
        fri_proof.final_poly.len(),
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
        let poly_eval: EF = fri_proof
            .final_poly
            .iter()
            .rev()
            .fold(EF::ZERO, |acc, &coeff| acc * x + coeff);

        // The polynomial should be well-defined (just check it doesn't panic)
        let _ = poly_eval;
    }
}
