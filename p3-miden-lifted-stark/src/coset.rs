//! Lifted coset domain abstraction with selector and vanishing computation.
//!
//! This module provides [`LiftedCoset`], the central abstraction for domain operations
//! in lifted STARKs where traces of different heights share a common evaluation domain.

use alloc::vec::Vec;

use p3_field::{ExtensionField, TwoAdicField, batch_multiplicative_inverse};

use crate::selectors::Selectors;

// ============================================================================
// LiftedCoset
// ============================================================================

/// Lifted coset for polynomial evaluation.
///
/// Represents a coset (gK)^r where:
/// - K is the evaluation domain of size 2^log_lde_height
/// - r = 2^log_lift_ratio is the lift factor (row repetition)
/// - The shift is g^r where g = F::GENERATOR
///
/// Key relationships:
/// - log_blowup = log_lde_height - log_trace_height
/// - log_lift_ratio = log_max_lde_height - log_lde_height
/// - lde_shift = g^r = F::GENERATOR.exp_power_of_2(log_lift_ratio)
///
/// # Invariants
///
/// - `log_lde_height = log_trace_height + log_blowup`
/// - `log_lde_height <= log_max_lde_height`
/// - All heights are powers of two (stored as log values)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiftedCoset {
    /// Log₂ of the original trace height.
    pub log_trace_height: usize,
    /// Log₂ of this matrix's LDE height.
    pub log_lde_height: usize,
    /// Log₂ of the maximum LDE height in the commitment.
    pub log_max_lde_height: usize,
}

impl LiftedCoset {
    /// Create a new `LiftedCoset` from the given parameters.
    ///
    /// # Panics
    ///
    /// Panics if `log_lde_height > log_max_lde_height` or if
    /// `log_trace_height > log_lde_height`.
    #[inline]
    pub fn new(log_trace_height: usize, log_lde_height: usize, log_max_lde_height: usize) -> Self {
        debug_assert!(
            log_trace_height <= log_lde_height,
            "trace height cannot exceed LDE height"
        );
        debug_assert!(
            log_lde_height <= log_max_lde_height,
            "LDE height cannot exceed max LDE height"
        );
        Self {
            log_trace_height,
            log_lde_height,
            log_max_lde_height,
        }
    }

    // ============ Existing methods ============

    /// Log₂ of the blowup factor for this matrix.
    ///
    /// Returns `log_lde_height - log_trace_height`.
    #[inline]
    pub fn log_blowup(&self) -> usize {
        self.log_lde_height - self.log_trace_height
    }

    /// Log₂ of the lift ratio for this matrix.
    ///
    /// The lift ratio is how many times this matrix's rows are virtually repeated
    /// to match the max LDE height: `max_lde_height / lde_height`.
    ///
    /// Returns `log_max_lde_height - log_lde_height`.
    #[inline]
    pub fn log_lift_ratio(&self) -> usize {
        self.log_max_lde_height - self.log_lde_height
    }

    /// Whether this matrix is lifted (its LDE height is less than the max).
    #[inline]
    pub fn is_lifted(&self) -> bool {
        self.log_lde_height < self.log_max_lde_height
    }

    /// Compute the coset shift for this matrix's LDE domain.
    ///
    /// For a matrix with lift ratio `r = 2^log_lift_ratio`, the coset shift is
    /// `g^r` where `g` is the field generator. This places the LDE on the
    /// nested coset `(gK)^r` within the max LDE domain.
    #[inline]
    pub fn lde_shift<F: TwoAdicField>(&self) -> F {
        F::GENERATOR.exp_power_of_2(self.log_lift_ratio())
    }

    /// The trace height (number of constraint rows).
    #[inline]
    pub fn trace_height(&self) -> usize {
        1 << self.log_trace_height
    }

    /// The LDE height for this matrix.
    #[inline]
    pub fn lde_height(&self) -> usize {
        1 << self.log_lde_height
    }

    /// The maximum LDE height across all matrices.
    #[inline]
    pub fn max_lde_height(&self) -> usize {
        1 << self.log_max_lde_height
    }

    /// The blowup factor for this matrix.
    #[inline]
    pub fn blowup(&self) -> usize {
        1 << self.log_blowup()
    }

    // ============ Domain derivation ============

    /// Derive the quotient domain coset from this LDE coset.
    ///
    /// For constraint evaluation, we need a coset of size trace_height × constraint_degree.
    /// This transforms (gK)^r into (gJ)^r while preserving the lift ratio.
    ///
    /// The transformation scales all heights by 2^log_diff where
    /// log_diff = log_blowup - log_constraint_degree, preserving:
    /// - The blowup ratio (constraint_degree after transformation)
    /// - The lift ratio (same row repetition factor)
    ///
    /// # Panics
    /// Panics if log_constraint_degree > log_blowup.
    pub fn quotient_domain(&self, log_constraint_degree: usize) -> Self {
        let log_blowup = self.log_blowup();
        assert!(
            log_constraint_degree <= log_blowup,
            "constraint degree cannot exceed blowup"
        );
        let log_diff = log_blowup - log_constraint_degree;
        Self {
            log_trace_height: self.log_trace_height + log_diff,
            log_lde_height: self.log_lde_height + log_diff,
            log_max_lde_height: self.log_max_lde_height + log_diff,
        }
    }

    // ============ Selector computation ============

    /// Compute selectors for evaluation over this coset in natural order (prover).
    ///
    /// Returns is_first_row, is_last_row, is_transition for each point
    /// in the coset (gK)^r. The trace domain H has size 2^log_trace_height.
    pub fn selectors<F: TwoAdicField>(&self) -> Selectors<Vec<F>> {
        let shift: F = self.lde_shift();
        let coset_size = self.lde_height();
        let rate_bits = self.log_blowup();

        // Z_H(x) = x^n - 1 evaluated at coset points.
        // Due to periodicity, only need 2^rate_bits distinct values.
        let s_pow_n = shift.exp_power_of_2(self.log_trace_height);
        let z_h_evals: Vec<F> = F::two_adic_generator(rate_bits)
            .powers()
            .take(1 << rate_bits)
            .map(|x| s_pow_n * x - F::ONE)
            .collect();

        // Coset points in natural order: shift · ω_J^i
        let omega_j = F::two_adic_generator(self.log_lde_height);
        let xs: Vec<F> = omega_j
            .powers()
            .take(coset_size)
            .map(|w| shift * w)
            .collect();

        let omega_h_inv = F::two_adic_generator(self.log_trace_height).inverse();

        // Helper: compute selector for a single Lagrange basis point
        let single_point_selector = |basis_point: F| -> Vec<F> {
            let denoms: Vec<F> = xs.iter().map(|&x| x - basis_point).collect();
            let invs = batch_multiplicative_inverse(&denoms);
            z_h_evals
                .iter()
                .cycle()
                .zip(invs)
                .map(|(&z_h, inv)| z_h * inv)
                .collect()
        };

        Selectors {
            is_first_row: single_point_selector(F::ONE),
            is_last_row: single_point_selector(omega_h_inv),
            is_transition: xs.iter().map(|&x| x - omega_h_inv).collect(),
        }
    }

    /// Compute selectors at an out-of-domain point (verifier).
    ///
    /// # Formulas (unnormalized)
    /// - `is_first_row = Z_H(z) / (z - 1)`
    /// - `is_last_row = Z_H(z) / (z - ω^{-1})`
    /// - `is_transition = z - ω^{-1}`
    ///
    /// where `Z_H(z) = z^n - 1` is the vanishing polynomial of the trace domain.
    pub fn selectors_at<F, EF>(&self, zeta: EF) -> Selectors<EF>
    where
        F: TwoAdicField,
        EF: ExtensionField<F>,
    {
        let z_n = zeta.exp_power_of_2(self.log_trace_height);
        let vanishing = z_n - EF::ONE;
        let omega_inv = EF::from(F::two_adic_generator(self.log_trace_height).inverse());

        Selectors {
            is_first_row: vanishing / (zeta - EF::ONE),
            is_last_row: vanishing / (zeta - omega_inv),
            is_transition: zeta - omega_inv,
        }
    }

    // ============ Vanishing polynomial ============

    /// Compute inverse vanishing polynomial over this coset (prover).
    ///
    /// The vanishing polynomial for the trace domain H at point x is:
    ///   Z_H(x) = x^n - 1
    /// where n = 2^log_trace_height is the trace height.
    ///
    /// Returns a `Vec<F>` of inverse vanishing values for each point in the coset.
    pub fn inv_vanishing<F: TwoAdicField>(&self) -> Vec<F> {
        let shift: F = self.lde_shift();
        let coset_size = self.lde_height();
        let rate_bits = self.log_blowup();

        // Z_H(x) = x^n - 1 evaluated at coset points.
        // Due to periodicity, only need 2^rate_bits distinct values.
        let s_pow_n = shift.exp_power_of_2(self.log_trace_height);
        let z_h_evals: Vec<F> = F::two_adic_generator(rate_bits)
            .powers()
            .take(1 << rate_bits)
            .map(|x| s_pow_n * x - F::ONE)
            .collect();

        batch_multiplicative_inverse(&z_h_evals)
            .into_iter()
            .cycle()
            .take(coset_size)
            .collect()
    }

    /// Compute inverse vanishing at an out-of-domain point (verifier).
    ///
    /// Returns 1/Z_H(zeta) where Z_H(z) = z^n - 1.
    pub fn inv_vanishing_at<F, EF>(&self, zeta: EF) -> EF
    where
        F: TwoAdicField,
        EF: ExtensionField<F>,
    {
        let z_n = zeta.exp_power_of_2(self.log_trace_height);
        (z_n - EF::ONE).inverse()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;
    use p3_field::{Field, PrimeCharacteristicRing};

    type F = BabyBear;

    #[test]
    fn domain_info_basic() {
        // Trace height 2^10, blowup 2^3, max LDE 2^15
        let info = LiftedCoset::new(10, 13, 15);

        assert_eq!(info.log_trace_height, 10);
        assert_eq!(info.log_lde_height, 13);
        assert_eq!(info.log_max_lde_height, 15);

        assert_eq!(info.log_blowup(), 3);
        assert_eq!(info.log_lift_ratio(), 2);
        assert!(info.is_lifted());

        assert_eq!(info.trace_height(), 1024);
        assert_eq!(info.lde_height(), 8192);
        assert_eq!(info.max_lde_height(), 32768);
        assert_eq!(info.blowup(), 8);
    }

    #[test]
    fn domain_info_no_lift() {
        // Matrix at max height (no lifting needed)
        let info = LiftedCoset::new(10, 13, 13);

        assert_eq!(info.log_lift_ratio(), 0);
        assert!(!info.is_lifted());
    }

    #[test]
    fn domain_info_lde_shift() {
        // Test that lde_shift computes correctly
        let info = LiftedCoset::new(10, 13, 15);
        let shift: F = info.lde_shift();

        // shift = g^(2^2) = g^4
        let expected = F::GENERATOR.exp_power_of_2(2);
        assert_eq!(shift, expected);
    }

    #[test]
    fn domain_info_no_lift_shift() {
        // When not lifted, shift should be g^1 = g
        let info = LiftedCoset::new(10, 13, 13);
        let shift: F = info.lde_shift();

        // shift = g^(2^0) = g
        assert_eq!(shift, F::GENERATOR);
    }
}
