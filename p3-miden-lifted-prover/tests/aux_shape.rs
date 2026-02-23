mod common;

use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
use p3_miden_lifted_air::{
    AirWithPeriodicColumns, BaseAir, BaseAirWithPublicValues, LiftedAir, LiftedAirBuilder,
};
use p3_miden_lifted_prover::prove_single;
use p3_miden_transcript::ProverTranscript;

use common::test_config;

#[derive(Clone, Copy, Debug)]
struct BadAuxWidthAir;

impl BaseAir<bb::F> for BadAuxWidthAir {
    fn width(&self) -> usize {
        1
    }
}

impl BaseAirWithPublicValues<bb::F> for BadAuxWidthAir {}

impl AirWithPeriodicColumns<bb::F> for BadAuxWidthAir {
    fn periodic_columns(&self) -> &[Vec<bb::F>] {
        &[]
    }
}

impl LiftedAir<bb::F, bb::EF> for BadAuxWidthAir {
    fn aux_width(&self) -> usize {
        1
    }

    fn build_aux_trace(
        &self,
        main: &RowMajorMatrix<bb::F>,
        _challenges: &[bb::EF],
    ) -> Option<RowMajorMatrix<bb::EF>> {
        let height = main.height();
        // Return 2 EF columns when aux_width() declares 1
        Some(RowMajorMatrix::new(vec![bb::EF::ZERO; height * 2], 2))
    }

    fn eval<AB: LiftedAirBuilder<F = bb::F>>(&self, _builder: &mut AB) {}
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
