mod common;

use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_air::{MidenAir, MidenAirBuilder};
use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
use p3_miden_lifted_prover::prove_single;
use p3_miden_transcript::ProverTranscript;

use common::test_config;

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
