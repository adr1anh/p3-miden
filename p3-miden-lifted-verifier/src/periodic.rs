//! Verifier-side periodic column handling.
//!
//! Periodic columns are stored as polynomial coefficients for efficient evaluation
//! at the OOD point using Horner's method.

extern crate alloc;

use alloc::vec::Vec;

use p3_dft::{NaiveDft, TwoAdicSubgroupDft};
use p3_field::{ExtensionField, TwoAdicField};

/// Verifier-side periodic polynomials for OOD evaluation.
///
/// Stores polynomial coefficients computed from the AIR's periodic columns.
/// Used to evaluate periodic values at the OOD point during verification.
#[derive(Clone, Debug)]
pub struct PeriodicPolys<F> {
    /// Polynomial coefficients for each column.
    polys: Vec<Vec<F>>,
}

impl<F: TwoAdicField> PeriodicPolys<F> {
    /// Construct from subgroup evaluations (canonical order).
    ///
    /// Converts subgroup evaluations to polynomial coefficients via inverse DFT.
    ///
    /// # Returns
    /// `None` if any column length is zero or not a power of two.
    pub fn new(column_evals: &[Vec<F>]) -> Option<Self> {
        let dft = NaiveDft;
        let mut polys = Vec::with_capacity(column_evals.len());

        for column in column_evals {
            let p = column.len();
            if p == 0 || !p.is_power_of_two() {
                return None;
            }
            let coeffs = dft.idft(column.clone());
            polys.push(coeffs);
        }

        Some(Self { polys })
    }

    /// Evaluate all periodic polynomials at the OOD point.
    ///
    /// For a column with period `p`, evaluates at `zeta^(trace_height / p)`.
    /// Uses Horner's method for efficient polynomial evaluation.
    ///
    /// # Arguments
    /// - `trace_height`: Height of the trace
    /// - `zeta`: The OOD evaluation point
    pub fn eval_at<EF>(&self, trace_height: usize, zeta: EF) -> Vec<EF>
    where
        EF: ExtensionField<F>,
    {
        let mut result = Vec::with_capacity(self.polys.len());

        for coeffs in &self.polys {
            let period = coeffs.len();
            let y = zeta.exp_u64((trace_height / period) as u64);
            result.push(horner_eval(coeffs, y));
        }

        result
    }
}

/// Evaluate a polynomial at a point using Horner's method.
fn horner_eval<F, EF>(coeffs: &[F], x: EF) -> EF
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    let mut acc = EF::ZERO;
    for coeff in coeffs.iter().rev() {
        acc = acc * x + *coeff;
    }
    acc
}
