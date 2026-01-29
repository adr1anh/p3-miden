//! Constraint folder (EF-only) for evaluating MidenAir constraints.
//!
//! This is intentionally minimal: no preprocessed trace, and aux is always
//! treated as a permutation trace in EF form.

use core::marker::PhantomData;

use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_air::MidenAirBuilder;

/// Minimal constraint folder used by both prover and verifier.
#[derive(Clone, Debug)]
pub struct ConstraintFolder<'a, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    pub main: RowMajorMatrix<EF>,
    pub aux: RowMajorMatrix<EF>,
    pub randomness: &'a [EF],
    pub public_values: &'a [EF],
    pub periodic_values: &'a [EF],
    pub is_first_row: EF,
    pub is_last_row: EF,
    pub is_transition: EF,
    pub alpha: EF,
    pub accumulator: EF,
    pub _phantom: PhantomData<F>,
}

impl<'a, F, EF> MidenAirBuilder for ConstraintFolder<'a, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type F = F;
    type Expr = EF;
    type Var = EF;
    type M = RowMajorMatrix<EF>;
    type PublicVar = EF;
    type PeriodicVal = EF;
    type EF = EF;
    type ExprEF = EF;
    type VarEF = EF;
    type MP = RowMajorMatrix<EF>;
    type RandomVar = EF;

    fn main(&self) -> Self::M {
        self.main.clone()
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("only window size 2 supported in this prototype")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.accumulator = self.accumulator * self.alpha + x.into();
    }

    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }

    fn periodic_evals(&self) -> &[Self::PeriodicVal] {
        self.periodic_values
    }

    fn preprocessed(&self) -> Self::M {
        panic!("preprocessed trace not supported in this prototype")
    }

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.accumulator = self.accumulator * self.alpha + x.into();
    }

    fn permutation(&self) -> Self::MP {
        self.aux.clone()
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.randomness
    }

    fn aux_bus_boundary_values(&self) -> &[Self::VarEF] {
        &[]
    }
}
