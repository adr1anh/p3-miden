use p3_dft::Radix2DitParallel;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_air::{MidenAir, MidenAirBuilder};
use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
use p3_miden_lifted_fri::PcsParams;
use p3_miden_lifted_fri::deep::DeepParams;
use p3_miden_lifted_fri::fri::{FriFold, FriParams};
use p3_miden_lifted_prover::{StarkConfig, prove_single};
use p3_miden_lmcs::LmcsConfig;
use p3_miden_transcript::ProverTranscript;

#[derive(Clone, Copy, Debug)]
struct BadAuxWidthAir;

impl MidenAir<bb::F, bb::EF> for BadAuxWidthAir {
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
        let width = <bb::EF as BasedVectorSpace<bb::F>>::DIMENSION + 1;
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
#[should_panic(expected = "aux trace width mismatch")]
fn aux_width_mismatch_panics() {
    let config = test_config();
    let air = BadAuxWidthAir;

    let trace = RowMajorMatrix::new(vec![bb::F::ZERO, bb::F::ONE, bb::F::ONE, bb::F::ZERO], 1);
    let public_values = vec![];

    let mut channel = ProverTranscript::new(bb::test_challenger());

    let _result = prove_single::<bb::F, bb::EF, _, _, _, _>(
        &config,
        &air,
        &trace,
        &public_values,
        &mut channel,
    );
}
