//! Constraint evaluation and quotient reconstruction for the verifier.
//!
//! This module provides:
//! - [`ConstraintFolder`]: Minimal EF-only folder for verifier constraint evaluation
//! - [`reconstruct_quotient`]: Reconstructs Q(ζ) from quotient chunk evaluations
//! - [`row_to_packed_ext`]: Reconstitutes EF elements from opened base field evaluations

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_air::MidenAirBuilder;

use crate::VerifierError;

// ============================================================================
// ConstraintFolder
// ============================================================================

/// Minimal constraint folder for verifier OOD evaluation.
///
/// Implements [`MidenAirBuilder`] for evaluating AIR constraints at out-of-domain
/// points. Uses extension field throughout since the verifier only evaluates at
/// a single EF point (ζ).
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

// ============================================================================
// Constants
// ============================================================================

/// Constraint degree for quotient decomposition (must match prover).
pub const CONSTRAINT_DEGREE: usize = 4;
pub const LOG_CONSTRAINT_DEGREE: usize = 2;

// ============================================================================
// Quotient Reconstruction
// ============================================================================

/// Reconstruct Q(ζ) from D quotient chunk evaluations using barycentric interpolation.
///
/// The quotient Q is decomposed into D chunks q_0, ..., q_{D-1} where each q_t
/// interpolates Q on the coset g·ω_J^t·H.
///
/// Let ω_S = ω_J^N (the D-th root of unity) and u = ζ^N / g^N.
/// For t = 0..D−1 define:
///   a_t = u − ω_S^t
///   w_t = ω_S^t / a_t
///
/// The reconstruction formula is:
///   Q(ζ) = (Σ_t w_t · q_t(ζ)) / (Σ_t w_t)
pub fn reconstruct_quotient<F, EF>(zeta: EF, trace_height: usize, chunks: &[EF]) -> EF
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    let d = CONSTRAINT_DEGREE;
    debug_assert_eq!(chunks.len(), d);

    let g = EF::from(F::GENERATOR);
    // ω_S = ω_J^N where J has size N*D, so ω_S has order D
    let omega_s = EF::from(F::two_adic_generator(LOG_CONSTRAINT_DEGREE));

    // u = ζ^N / g^N
    let zeta_n = zeta.exp_u64(trace_height as u64);
    let g_n = g.exp_u64(trace_height as u64);
    let u = zeta_n * g_n.inverse();

    // Compute weighted sum: Σ_t w_t · q_t(ζ) and Σ_t w_t
    let mut numerator = EF::ZERO;
    let mut denominator = EF::ZERO;
    let mut omega_s_t = EF::ONE; // ω_S^t

    for &q_t in chunks.iter() {
        // a_t = u - ω_S^t
        let a_t = u - omega_s_t;
        // w_t = ω_S^t / a_t
        let w_t = omega_s_t * a_t.inverse();

        numerator += w_t * q_t;
        denominator += w_t;

        omega_s_t *= omega_s;
    }

    numerator * denominator.inverse()
}

/// Extract quotient chunk evaluations from opened values.
///
/// The quotient has D chunks, each as an EF element. When committed, each EF element
/// becomes `EF::DIMENSION` base field polynomials. Opening at EF point ζ gives
/// `D * EF::DIMENSION` EF values which we recombine into D quotient evaluations.
pub fn extract_quotient_chunks<F, EF>(quotient_row: &[EF]) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    (0..CONSTRAINT_DEGREE)
        .map(|t| {
            let start = t * EF::DIMENSION;
            (0..EF::DIMENSION)
                .map(|j| EF::ith_basis_element(j).unwrap() * quotient_row[start + j])
                .sum()
        })
        .collect()
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Reconstitute EF elements from opened base field polynomial evaluations.
///
/// When an EF polynomial is committed, it becomes DIM base field polynomials.
/// Opening at EF point z gives DIM EF values (F-polys evaluated at EF point).
/// Reconstruct each EF element: ef_i = Σ_j basis_j * row[i*DIM + j]
pub fn row_to_packed_ext<F, EF, I>(row: I) -> Result<Vec<EF>, VerifierError>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    I: IntoIterator<Item = EF>,
{
    let evals: Vec<EF> = row.into_iter().collect();
    if !evals.len().is_multiple_of(EF::DIMENSION) {
        return Err(VerifierError::InvalidAuxShape {
            expected_divisor: EF::DIMENSION,
            actual_len: evals.len(),
        });
    }
    let num_elements = evals.len() / EF::DIMENSION;
    Ok((0..num_elements)
        .map(|i| {
            let start = i * EF::DIMENSION;
            (0..EF::DIMENSION)
                .map(|j| EF::ith_basis_element(j).unwrap() * evals[start + j])
                .sum()
        })
        .collect())
}

/// Align width to the given alignment.
#[inline]
pub fn align_width(width: usize, alignment: usize) -> usize {
    width.div_ceil(alignment) * alignment
}
