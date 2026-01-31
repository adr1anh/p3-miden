use alloc::vec::Vec;
use p3_field::{BasedVectorSpace, PackedField, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::ViewPair;
use p3_miden_air::MidenAirBuilder;
use p3_miden_uni_stark::PackedChallengeLinearCombination;

use crate::{PackedChallenge, PackedVal, StarkGenericConfig, Val};

// Batch size for constraint linear-combination chunks.
// Kept small to reduce overhead without inflating stack/regs.
pub(crate) const CONSTRAINT_BATCH: usize = 8;

/// Handles constraint accumulation for the prover in a STARK system.
///
/// This struct evaluates constraints for a given row in the trace matrix.
/// It accumulates them into a single value using a randomized challenge:
/// `C_0 + alpha C_1 + alpha^2 C_2 + ...`
#[derive(Debug)]
pub struct ProverConstraintFolder<'a, SC: StarkGenericConfig> {
    /// The matrix containing rows on which the constraint polynomial is to be evaluated
    pub main: RowMajorMatrixView<'a, PackedVal<SC>>,
    /// The matrix containing rows on which the aux constraint polynomial is to be evaluated
    /// (may have zero width)
    pub aux: RowMajorMatrixView<'a, PackedChallenge<SC>>,
    /// The randomness used to compute the aux trace; can be zero width.
    /// Cached EF randomness packed from base randomness to avoid temporary leaks
    pub packed_randomness: &'a [PackedChallenge<SC>],
    /// Aux trace bus boundary values packed from base field to extension field
    pub aux_bus_boundary_values: &'a [PackedChallenge<SC>],
    /// The preprocessed columns (if any)
    pub preprocessed: Option<RowMajorMatrixView<'a, PackedVal<SC>>>,
    /// Public inputs to the AIR
    pub public_values: &'a [Val<SC>],
    /// Periodic column values (precomputed for the current row)
    pub periodic_values: &'a [PackedVal<SC>],
    /// Evaluations of the Selector polynomial for the first row of the trace
    pub is_first_row: PackedVal<SC>,
    /// Evaluations of the Selector polynomial for the last row of the trace
    pub is_last_row: PackedVal<SC>,
    /// Evaluations of the Selector polynomial for rows where transition constraints
    /// should be applied
    pub is_transition: PackedVal<SC>,
    /// Base-field alpha powers ordered to match base constraint emission.
    pub base_alpha_powers: &'a [Vec<Val<SC>>],
    /// Extension-field alpha powers ordered to match extension constraint emission.
    pub ext_alpha_powers: &'a [SC::Challenge],
    /// Running accumulator for all constraints multiplied by challenge powers
    /// `C_0 + alpha C_1 + alpha^2 C_2 + ...`
    pub accumulator: PackedChallenge<SC>,
    /// Current constraint index being processed
    pub constraint_index: usize,
    /// Total constraint count expected for this AIR.
    pub constraint_count: usize,
    /// Collected base-field constraints for this row.
    pub base_constraints: Vec<PackedVal<SC>>,
    /// Collected extension-field constraints for this row.
    pub ext_constraints: Vec<PackedChallenge<SC>>,
}

impl<'a, SC: StarkGenericConfig> ProverConstraintFolder<'a, SC> {
    #[inline(always)]
    fn packed_linear_combination_ext<const N: usize>(
        coeffs: &[SC::Challenge],
        exprs: &[PackedChallenge<SC>],
    ) -> PackedChallenge<SC> {
        let combine =
            <PackedChallenge<SC> as PackedChallengeLinearCombination<SC::Challenge>>
                ::packed_linear_combination::<N>;
        combine(coeffs, exprs)
    }

    #[inline]
    pub fn finalize_constraints(&mut self) {
        debug_assert_eq!(self.constraint_index, self.constraint_count);
        debug_assert_eq!(
            self.base_constraints.len(),
            self.base_alpha_powers.first().map_or(0, Vec::len)
        );
        debug_assert_eq!(self.ext_constraints.len(), self.ext_alpha_powers.len());

        let base_constraints = &self.base_constraints;
        let base_alpha_powers = self.base_alpha_powers;
        let base_len = base_constraints.len();

        self.accumulator = PackedChallenge::<SC>::from_basis_coefficients_fn(|i| {
            let coeffs = &base_alpha_powers[i];
            let mut acc = PackedVal::<SC>::ZERO;
            let mut start = 0;
            while start + CONSTRAINT_BATCH <= base_len {
                acc += <PackedVal<SC> as PackedField>::packed_linear_combination::<CONSTRAINT_BATCH>(
                    &coeffs[start..start + CONSTRAINT_BATCH],
                    &base_constraints[start..start + CONSTRAINT_BATCH],
                );
                start += CONSTRAINT_BATCH;
            }
            for (coeff, expr) in coeffs[start..base_len]
                .iter()
                .zip(base_constraints[start..base_len].iter())
            {
                acc += *expr * *coeff;
            }
            acc
        });

        let ext_constraints = &self.ext_constraints;
        let ext_alpha_powers = self.ext_alpha_powers;
        let ext_len = ext_constraints.len();
        let mut start = 0;
        while start + CONSTRAINT_BATCH <= ext_len {
            self.accumulator += Self::packed_linear_combination_ext::<CONSTRAINT_BATCH>(
                &ext_alpha_powers[start..start + CONSTRAINT_BATCH],
                &ext_constraints[start..start + CONSTRAINT_BATCH],
            );
            start += CONSTRAINT_BATCH;
        }
        for (coeff, expr) in ext_alpha_powers[start..ext_len]
            .iter()
            .zip(ext_constraints[start..ext_len].iter())
        {
            self.accumulator += *expr * *coeff;
        }
    }
}

impl<'a, SC: StarkGenericConfig> MidenAirBuilder for ProverConstraintFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = PackedVal<SC>;
    type Var = PackedVal<SC>;
    type M = RowMajorMatrixView<'a, PackedVal<SC>>;
    type PublicVar = Val<SC>;
    type EF = SC::Challenge;
    type ExprEF = PackedChallenge<SC>;
    type VarEF = PackedChallenge<SC>;
    type MP = RowMajorMatrixView<'a, PackedChallenge<SC>>;
    type RandomVar = PackedChallenge<SC>;
    type PeriodicVal = PackedVal<SC>;

    #[inline]
    fn main(&self) -> Self::M {
        self.main
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    #[inline]
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("miden-prover only supports a window size of 2")
        }
    }

    #[inline]
    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }

    #[inline]
    fn preprocessed(&self) -> Self::M {
        self.preprocessed
            .expect("Air does not provide preprocessed columns, hence can not be consumed")
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x = x.into();
        self.base_constraints.push(x);
        self.constraint_index += 1;
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        let expr_array = array.map(Into::into);
        self.base_constraints.extend(expr_array);
        self.constraint_index += N;
    }

    #[inline]
    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let x = x.into();
        self.ext_constraints.push(x);
        self.constraint_index += 1;
    }

    #[inline]
    fn permutation(&self) -> Self::MP {
        self.aux
    }

    #[inline]
    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.packed_randomness
    }

    fn aux_bus_boundary_values(&self) -> &[Self::VarEF] {
        self.aux_bus_boundary_values
    }

    fn periodic_evals(&self) -> &[Self::PeriodicVal] {
        self.periodic_values
    }
}

/// Handles constraint verification for the verifier in a STARK system.
///
/// Similar to ProverConstraintFolder but operates on committed values rather than the full trace,
/// using a more efficient accumulation method for verification.
#[derive(Debug)]
pub struct VerifierConstraintFolder<'a, SC: StarkGenericConfig> {
    /// Pair of consecutive rows from the committed polynomial evaluations
    pub main: ViewPair<'a, SC::Challenge>,
    /// Pair of consecutive rows from the committed polynomial evaluations (may have
    /// zero width)
    pub aux: ViewPair<'a, SC::Challenge>,
    /// The randomness used to compute the aux tract; can be zero width.
    pub randomness: &'a [SC::Challenge],
    /// Aux trace bus boundary values; can be zero width.
    pub aux_bus_boundary_values: &'a [SC::Challenge],
    /// The preprocessed columns (if any)
    pub preprocessed: Option<ViewPair<'a, SC::Challenge>>,
    /// Public values that are inputs to the computation
    pub public_values: &'a [Val<SC>],
    /// Periodic column values (precomputed for the current row)
    pub periodic_values: &'a [SC::Challenge],
    /// Evaluations of the Selector polynomial for the first row of the trace
    pub is_first_row: SC::Challenge,
    /// Evaluations of the Selector polynomial for the last row of the trace
    pub is_last_row: SC::Challenge,
    /// Evaluations of the Selector polynomial for rows where transition constraints
    /// should be applied
    pub is_transition: SC::Challenge,
    /// Single challenge value used for constraint combination
    pub alpha: SC::Challenge,
    /// Running accumulator for all constraints
    pub accumulator: SC::Challenge,
}

impl<'a, SC: StarkGenericConfig> MidenAirBuilder for VerifierConstraintFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = SC::Challenge;
    type Var = SC::Challenge;
    type M = ViewPair<'a, SC::Challenge>;
    type PublicVar = Val<SC>;
    type EF = SC::Challenge;
    type ExprEF = SC::Challenge;
    type VarEF = SC::Challenge;
    type MP = ViewPair<'a, SC::Challenge>;
    type RandomVar = SC::Challenge;
    type PeriodicVal = SC::Challenge;

    #[inline]
    fn main(&self) -> Self::M {
        self.main
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    #[inline]
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("miden-prover only supports a window size of 2")
        }
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.accumulator = self.accumulator * self.alpha + x.into();
    }

    #[inline]
    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }

    #[inline]
    fn preprocessed(&self) -> Self::M {
        self.preprocessed
            .expect("Air does not provide preprocessed columns, hence can not be consumed")
    }

    #[inline]
    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.accumulator = self.accumulator * self.alpha + x.into();
    }

    #[inline]
    fn permutation(&self) -> Self::MP {
        self.aux
    }

    #[inline]
    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.randomness
    }

    fn aux_bus_boundary_values(&self) -> &[Self::VarEF] {
        self.aux_bus_boundary_values
    }

    fn periodic_evals(&self) -> &[Self::PeriodicVal] {
        self.periodic_values
    }
}
