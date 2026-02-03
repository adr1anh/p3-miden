use p3_dft::Radix2DitParallel;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_air::{MidenAir, MidenAirBuilder};
use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
use p3_miden_lifted_fri::PcsParams;
use p3_miden_lifted_fri::deep::DeepParams;
use p3_miden_lifted_fri::fri::{FriFold, FriParams};
use p3_miden_lifted_prover::prove_single;
use p3_miden_lifted_verifier::{StarkConfig, VerifierError, verify_single};
use p3_miden_lmcs::LmcsConfig;
use p3_miden_transcript::{ProverTranscript, TranscriptData, VerifierTranscript};
use p3_util::log2_strict_usize;

#[derive(Clone, Copy, Debug)]
struct TinyAir;

impl MidenAir<bb::F, bb::EF> for TinyAir {
    fn width(&self) -> usize {
        1
    }

    fn aux_width(&self) -> usize {
        1
    }

    fn build_aux_trace(
        &self,
        main: &RowMajorMatrix<bb::F>,
        _challenges: &[bb::EF],
    ) -> Option<RowMajorMatrix<bb::F>> {
        let height = main.height();
        let width = <bb::EF as BasedVectorSpace<bb::F>>::DIMENSION;
        Some(RowMajorMatrix::new(
            vec![bb::F::ZERO; height * width],
            width,
        ))
    }

    fn eval<AB: MidenAirBuilder<F = bb::F>>(&self, _builder: &mut AB) {}
}

type TestLmcs = LmcsConfig<bb::P, bb::P, bb::Sponge, bb::Compress, { bb::WIDTH }, { bb::DIGEST }>;
type TestDft = Radix2DitParallel<bb::F>;

fn test_config() -> StarkConfig<TestLmcs, TestDft> {
    let pcs = PcsParams {
        fri: FriParams {
            log_blowup: 1,
            fold: FriFold::ARITY_2,
            log_final_degree: 2,
            proof_of_work_bits: 0,
        },
        deep: DeepParams {
            proof_of_work_bits: 0,
        },
        num_queries: 2,
        query_proof_of_work_bits: 0,
    };

    let (_, sponge, compress) = bb::test_components();
    let lmcs: TestLmcs = LmcsConfig::new(sponge, compress);
    let dft = TestDft::default();

    StarkConfig { pcs, lmcs, dft }
}

#[test]
fn malformed_transcript_is_rejected() {
    let config = test_config();
    let air = TinyAir;

    let trace = RowMajorMatrix::new(vec![bb::F::ZERO, bb::F::ONE, bb::F::ONE, bb::F::ZERO], 1);
    let log_trace_height = log2_strict_usize(trace.height());
    let public_values = vec![];

    // Prove
    let mut prover_channel = ProverTranscript::new(bb::test_challenger());
    prove_single::<bb::F, bb::EF, _, _, _, _>(
        &config,
        &air,
        &trace,
        &public_values,
        &mut prover_channel,
    )
    .expect("proving should succeed");
    let transcript = prover_channel.into_data();

    // Verify baseline proof
    let mut verifier_channel = VerifierTranscript::from_data(bb::test_challenger(), &transcript);
    let baseline = verify_single::<bb::F, bb::EF, _, _, _, _>(
        &config,
        &air,
        log_trace_height,
        &public_values,
        &mut verifier_channel,
    );
    assert!(
        baseline.is_ok(),
        "baseline proof should verify: {baseline:?}"
    );

    // Create bad transcript with extra field element
    let (mut fields, commitments) = transcript.clone().into_parts();
    fields.push(bb::F::ONE);
    let bad_transcript = TranscriptData::new(fields, commitments);

    let mut bad_verifier_channel =
        VerifierTranscript::from_data(bb::test_challenger(), &bad_transcript);
    let err = verify_single::<bb::F, bb::EF, _, _, _, _>(
        &config,
        &air,
        log_trace_height,
        &public_values,
        &mut bad_verifier_channel,
    )
    .expect_err("extra transcript data should fail verification");
    assert!(matches!(err, VerifierError::TranscriptNotConsumed));
}
