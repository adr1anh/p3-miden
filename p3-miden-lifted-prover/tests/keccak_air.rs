mod common;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{BasedVectorSpace, ExtensionField, Field};
use p3_keccak_air::{KeccakAir, generate_trace_rows};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_air::{MidenAir, MidenAirBuilder};
use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;

use common::prove_and_verify;

// ---------------------------------------------------------------------------
// Adapter: MidenAirBuilder → AirBuilder
// ---------------------------------------------------------------------------

/// Adapter that makes a `MidenAirBuilder` usable as a standard p3-air `AirBuilder`.
///
/// Truncates the main trace matrix to the correct width to compensate for
/// LMCS alignment padding that standard p3 AIRs (like KeccakAir) don't expect.
struct MidenToAirBuilder<'a, AB: MidenAirBuilder> {
    inner: &'a mut AB,
    width: usize,
}

impl<'a, AB: MidenAirBuilder> MidenToAirBuilder<'a, AB>
where
    AB::Var: Clone,
{
    /// Build a truncated RowMajorMatrix from the inner builder's main trace.
    fn truncated_main(&self) -> RowMajorMatrix<AB::Var> {
        let m = self.inner.main();
        let full_width = m.width();
        let height = m.height();

        if full_width == self.width {
            // No truncation needed; collect directly
            let values: Vec<AB::Var> = (0..height)
                .flat_map(|r| m.row(r).unwrap().into_iter())
                .collect();
            return RowMajorMatrix::new(values, full_width);
        }

        // Truncate each row to `self.width`
        let mut values = Vec::with_capacity(height * self.width);
        for r in 0..height {
            values.extend(m.row(r).unwrap().into_iter().take(self.width));
        }
        RowMajorMatrix::new(values, self.width)
    }
}

impl<AB: MidenAirBuilder> AirBuilder for MidenToAirBuilder<'_, AB>
where
    AB::Var: Clone,
{
    type F = AB::F;
    type Expr = AB::Expr;
    type Var = AB::Var;
    type M = RowMajorMatrix<AB::Var>;

    fn main(&self) -> Self::M {
        self.truncated_main()
    }

    fn is_first_row(&self) -> Self::Expr {
        self.inner.is_first_row()
    }

    fn is_last_row(&self) -> Self::Expr {
        self.inner.is_last_row()
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        self.inner.is_transition_window(size)
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.inner.assert_zero(x)
    }
}

// ---------------------------------------------------------------------------
// MidenKeccakAir: wraps KeccakAir to implement MidenAir
// ---------------------------------------------------------------------------

/// Wrapper around `p3_keccak_air::KeccakAir` that implements `MidenAir`.
///
/// The lifted STARK prover always requires an aux trace, so this wrapper
/// provides a dummy single-column zero aux trace.
struct MidenKeccakAir;

impl<F: Field, EF: ExtensionField<F>> MidenAir<F, EF> for MidenKeccakAir {
    fn width(&self) -> usize {
        BaseAir::<F>::width(&KeccakAir {})
    }

    fn num_randomness(&self) -> usize {
        1
    }

    fn aux_width(&self) -> usize {
        1
    }

    fn build_aux_trace(
        &self,
        main: &RowMajorMatrix<F>,
        _challenges: &[EF],
    ) -> Option<RowMajorMatrix<F>> {
        let dim = <EF as BasedVectorSpace<F>>::DIMENSION;
        Some(RowMajorMatrix::new(
            vec![F::ZERO; main.height() * dim],
            dim,
        ))
    }

    fn eval<AB: MidenAirBuilder<F = F>>(&self, builder: &mut AB) {
        let width = BaseAir::<F>::width(&KeccakAir {});
        let mut adapter = MidenToAirBuilder {
            inner: builder,
            width,
        };
        Air::eval(&KeccakAir {}, &mut adapter);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn keccak_single_input() {
    let air = MidenKeccakAir;
    let inputs = vec![[0u64; 25]];
    let trace = generate_trace_rows::<bb::F>(inputs, 0);
    let public_values = vec![];
    prove_and_verify(&air, &[(trace, public_values)]);
}

#[test]
fn keccak_multiple_inputs() {
    let air = MidenKeccakAir;
    let inputs: Vec<[u64; 25]> = (0..4)
        .map(|i| {
            let mut state = [0u64; 25];
            state[0] = i as u64;
            state
        })
        .collect();
    let trace = generate_trace_rows::<bb::F>(inputs, 0);
    let public_values = vec![];
    prove_and_verify(&air, &[(trace, public_values)]);
}
