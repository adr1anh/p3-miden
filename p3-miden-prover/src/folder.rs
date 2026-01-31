use alloc::vec::Vec;
use p3_field::{BasedVectorSpace, PackedField, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::ViewPair;
use p3_miden_air::MidenAirBuilder;

use crate::{PackedChallenge, PackedVal, StarkGenericConfig, Val};

// Batch size for buffered base-field constraints.
// Kept small to reduce per-constraint overhead without inflating stack/regs.
const PENDING_BASE_BATCH: usize = 16;

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
    pub packed_randomness: Vec<PackedChallenge<SC>>,
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
    /// Challenge powers used for randomized constraint combination
    pub alpha_powers: &'a [SC::Challenge],
    /// Challenge powers decomposed into their base field component.
    pub decomposed_alpha_powers: &'a [Vec<Val<SC>>],
    /// Running accumulator for all constraints multiplied by challenge powers
    /// `C_0 + alpha C_1 + alpha^2 C_2 + ...`
    pub accumulator: PackedChallenge<SC>,
    /// Current constraint index being processed
    pub constraint_index: usize,

    // Buffer base-field `assert_zero` calls in small batches.
    // Flush before any extension-field constraint to keep ordering intact.
    pub(crate) pending_base_start: usize,
    pub(crate) pending_base_len: usize,
    pub(crate) pending_base: [PackedVal<SC>; PENDING_BASE_BATCH],
}

impl<'a, SC: StarkGenericConfig> ProverConstraintFolder<'a, SC> {
    #[inline]
    pub fn flush_pending_base(&mut self) {
        let n = self.pending_base_len;
        if n == 0 {
            return;
        }

        let start = self.pending_base_start;
        // Copy refs so we can update `accumulator` without borrow checker conflicts.
        let decomposed_alpha_powers = self.decomposed_alpha_powers;
        let pending_base = &self.pending_base;

        if n == PENDING_BASE_BATCH {
            // Fast path: flush a full batch via the const-generic kernel.
            self.accumulator += PackedChallenge::<SC>::from_basis_coefficients_fn(|i| {
                let alpha_powers = &decomposed_alpha_powers[i][start..(start + PENDING_BASE_BATCH)];
                PackedVal::<SC>::packed_linear_combination::<PENDING_BASE_BATCH>(
                    alpha_powers,
                    pending_base,
                )
            });
        } else {
            // Tail flush before `assert_zero_ext`/end-of-eval; use a small loop for n.
            let pending_base = &pending_base[..n];
            self.accumulator += PackedChallenge::<SC>::from_basis_coefficients_fn(|i| {
                let alpha_powers = &decomposed_alpha_powers[i][start..(start + n)];
                let mut acc = PackedVal::<SC>::ZERO;
                for (coeff, expr) in alpha_powers.iter().zip(pending_base.iter()) {
                    acc += *expr * *coeff;
                }
                acc
            });
        }

        self.pending_base_len = 0;
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
        if self.pending_base_len == 0 {
            self.pending_base_start = self.constraint_index;
        }

        self.pending_base[self.pending_base_len] = x;
        self.pending_base_len += 1;
        self.constraint_index += 1;

        if self.pending_base_len == PENDING_BASE_BATCH {
            self.flush_pending_base();
        }
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        // `assert_zero` is buffered; flush to avoid interleaving and breaking batching order.
        self.flush_pending_base();

        let expr_array = array.map(Into::into);
        self.accumulator += PackedChallenge::<SC>::from_basis_coefficients_fn(|i| {
            let alpha_powers = &self.decomposed_alpha_powers[i]
                [self.constraint_index..(self.constraint_index + N)];
            PackedVal::<SC>::packed_linear_combination::<N>(alpha_powers, &expr_array)
        });
        self.constraint_index += N;
    }

    #[inline]
    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        // Flush base-field pending constraints to keep the stream contiguous.
        self.flush_pending_base();

        let alpha_power = self.alpha_powers[self.constraint_index];
        self.accumulator += Into::<PackedChallenge<SC>>::into(alpha_power) * x.into();
        self.constraint_index += 1;
    }

    #[inline]
    fn permutation(&self) -> Self::MP {
        self.aux
    }

    #[inline]
    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.packed_randomness.as_slice()
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
