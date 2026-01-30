use core::marker::PhantomData;

use p3_dft::Radix2DitParallel;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_air::{MidenAir, MidenAirBuilder};
use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
use p3_miden_lifted_fri::PcsParams;
use p3_miden_lifted_fri::deep::DeepParams;
use p3_miden_lifted_fri::fri::{FriFold, FriParams};
use p3_miden_lifted_prover::prove;
use p3_miden_lifted_verifier::{LiftedStarkConfig, Proof, VerifierError, verify};
use p3_miden_lmcs::LmcsConfig;
use p3_miden_transcript::TranscriptData;

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

fn test_config() -> LiftedStarkConfig<bb::F, TestLmcs, TestDft> {
    let params = PcsParams {
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
    let alignment = bb::RATE;
    let dft = TestDft::default();

    LiftedStarkConfig {
        params,
        lmcs,
        dft,
        alignment,
        _phantom: PhantomData,
    }
}

type TestLmcs = LmcsConfig<bb::P, bb::P, bb::Sponge, bb::Compress, { bb::WIDTH }, { bb::DIGEST }>;
type TestDft = Radix2DitParallel<bb::F>;

#[test]
fn malformed_transcript_is_rejected() {
    let config = test_config();
    let air = TinyAir;

    let trace = RowMajorMatrix::new(vec![bb::F::ZERO, bb::F::ONE, bb::F::ONE, bb::F::ZERO], 1);
    let traces = vec![trace];
    let public_values = vec![vec![]];

    let proof = prove::<bb::F, bb::EF, _, _, _, _>(
        &config,
        &[air],
        &traces,
        &public_values,
        bb::test_challenger(),
    );

    let baseline = verify::<bb::F, bb::EF, _, _, _, _>(
        &config,
        &[air],
        &proof,
        &public_values,
        bb::test_challenger(),
    );
    assert!(
        baseline.is_ok(),
        "baseline proof should verify: {baseline:?}"
    );

    let (mut fields, commitments) = proof.transcript.clone().into_parts();
    fields.push(bb::F::ONE);
    let bad_proof = Proof {
        transcript: TranscriptData::new(fields, commitments),
    };

    let err = verify::<bb::F, bb::EF, _, _, _, _>(
        &config,
        &[air],
        &bad_proof,
        &public_values,
        bb::test_challenger(),
    )
    .expect_err("extra transcript data should fail verification");
    assert!(matches!(err, VerifierError::TranscriptNotConsumed));
}
