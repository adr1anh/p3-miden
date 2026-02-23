//! Quotient polynomial helpers: accumulation, vanishing division, decomposition.
//!
//! The prover orchestrates the quotient pipeline (loop over instances, accumulate,
//! divide, commit). This module provides the building blocks:
//!
//! - `cyclic_extend_and_scale`: Horner-style beta scaling + cyclic extension
//! - `divide_by_vanishing_in_place`: Divide by Z_H on the quotient domain
//! - [`commit_quotient`]: Decompose Q(gJ) into chunks and commit on gK

use alloc::vec;
use alloc::vec::Vec;

use p3_dft::TwoAdicSubgroupDft;
use p3_field::{
    BasedVectorSpace, ExtensionField, Field, TwoAdicField, batch_multiplicative_inverse,
};
use p3_matrix::Matrix;
use p3_matrix::bitrev::BitReversibleMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_miden_lifted_stark::LiftedCoset;
use p3_miden_lmcs::Lmcs;
use p3_util::log2_strict_usize;

use crate::StarkConfig;
use crate::commit::Committed;

// ============================================================================
// Accumulation
// ============================================================================

/// Cyclically extend the accumulator to `target_len` and scale every element by `beta`.
///
/// On the first call (empty accumulator) this simply zero-fills to `target_len`.
/// On subsequent calls it scales the existing buffer by `beta` (Horner folding)
/// then doubles via `extend_from_within` until it reaches `target_len`.
///
/// Both `accumulator.len()` and `target_len` must be powers of two, and
/// `target_len >= accumulator.len()`.
pub(crate) fn cyclic_extend_and_scale<EF: Field>(
    accumulator: &mut Vec<EF>,
    target_len: usize,
    beta: EF,
) {
    if accumulator.is_empty() {
        accumulator.resize(target_len, EF::ZERO);
    } else {
        // Horner: scale the smaller buffer by beta before upsampling
        accumulator.par_iter_mut().for_each(|v| *v *= beta);
        // Cyclic extension by repeated doubling (all sizes are powers of 2)
        while accumulator.len() < target_len {
            accumulator.extend_from_within(..);
        }
    }
}

// ============================================================================
// Vanishing division
// ============================================================================

/// Divide quotient numerator by vanishing polynomial in-place (natural order).
///
/// Replaces each `numerator[i]` with `numerator[i] / Z_H(x_i)` where
/// Z_H(x) = x^N - 1 and N is the trace height.
///
/// Exploits periodicity: Z_H(x) has only 2^rate_bits distinct values on gJ,
/// so we compute only the distinct inverse values and use modular indexing.
pub(crate) fn divide_by_vanishing_in_place<F, EF>(numerator: &mut [EF], coset: &LiftedCoset)
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    let rate_bits = coset.log_blowup();
    let num_distinct = 1 << rate_bits;

    // Compute only the distinct inverse vanishing values (2^rate_bits of them)
    // Z_H(x) = x^n - 1 has periodicity on the coset
    let shift: F = coset.lde_shift();
    let s_pow_n = shift.exp_power_of_2(coset.log_trace_height);
    let z_h_evals: Vec<F> = F::two_adic_generator(rate_bits)
        .powers()
        .take(num_distinct)
        .map(|x| s_pow_n * x - F::ONE)
        .collect();

    let inv_van = batch_multiplicative_inverse(&z_h_evals);

    // Parallel division using modular indexing for periodicity.
    // Z_H has only num_distinct unique values on gJ; power-of-2 size
    // lets us use bitmask: i & (num_distinct - 1) == i % num_distinct.
    numerator.par_iter_mut().enumerate().for_each(|(i, n)| {
        *n *= inv_van[i & (num_distinct - 1)];
    });
}

// ============================================================================
// Quotient decomposition + commitment
// ============================================================================

/// Commit quotient polynomial using the fused scaling pipeline.
///
/// Takes Q(gJ) evaluations in natural order, decomposes into D quotient
/// components, and commits their LDE evaluations on gK.
///
/// `q_evals` is consumed, flattened to base field, and zero-padded to
/// `N * B * D * EF::DIMENSION` base elements for the LDE. Callers that
/// pre-allocate `q_evals` with capacity `N * B` (in EF elements) allow the
/// flatten + resize to reuse the same allocation.
///
/// # Pipeline
///
/// 1. Reshape Q(gJ) as N×D matrix (column t = coset g·ω_J^t·H)
/// 2. Batch iDFT over H
/// 3. Fused scaling: multiply by (ω_J^t)^{-k} to bake g^k into coefficients
/// 4. Flatten to base field (width D → D·`EF::DIMENSION`)
/// 5. Zero-pad to N·B rows
/// 6. Batch plain DFT on base field → evaluations on gK
/// 7. Bit-reverse rows and commit via LMCS
///
/// # Arguments
///
/// - `config`: STARK configuration (provides DFT, LMCS, blowup)
/// - `q_evals`: Q(gJ) evaluations in natural order, length N·D
/// - `coset`: The [`LiftedCoset`] for the trace (provides trace height and blowup)
///
/// # Returns
///
/// A `Committed` wrapper around the quotient tree with base field matrix.
///
/// # Panics
///
/// - If `q_evals.len()` is not divisible by the trace height
/// - If blowup B < constraint degree D
pub fn commit_quotient<F, EF, L, Dft>(
    config: &StarkConfig<L, Dft>,
    q_evals: Vec<EF>,
    coset: &LiftedCoset,
) -> Committed<F, RowMajorMatrix<F>, L>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    L: Lmcs<F = F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    let n = coset.trace_height();
    let d = q_evals.len() / n;
    let log_d = log2_strict_usize(d);
    let log_blowup = config.pcs.fri.log_blowup;
    let b = 1usize << log_blowup;

    debug_assert!(
        q_evals.len().is_multiple_of(n),
        "q_evals length must be divisible by N"
    );
    debug_assert!(b >= d, "blowup B must be >= constraint degree D");

    // ═══════════════════════════════════════════════════════════════════════
    // Step 0: Reshape to N × D matrix
    // ═══════════════════════════════════════════════════════════════════════
    // Column t = evaluations on coset g·ω_J^t·H
    // q_evals[r*D + t] = Q(g·ω_J^t·ω_H^r)
    let m = RowMajorMatrix::new(q_evals, d);

    // ═══════════════════════════════════════════════════════════════════════
    // Step 1: Batched iDFT over H (using algebra methods for EF)
    // ═══════════════════════════════════════════════════════════════════════
    // Treats each column as evaluations on H (not the actual coset g·ω_J^t·H)
    // Result: C0[k, t] = a_{t,k} · (g·ω_J^t)^k
    let mut coeffs = config.dft.idft_algebra_batch(m);

    // ═══════════════════════════════════════════════════════════════════════
    // Step 2: Fused coefficient scaling
    // ═══════════════════════════════════════════════════════════════════════
    // Multiply by (ω_J^t)^{-k} to get a_{t,k} · g^k
    // This bakes the coset shift g^k into the coefficients
    let omega_j_inv = F::two_adic_generator(coset.log_trace_height + log_d).inverse();

    // Precompute ω_J^{-k} for k = 0..n with sequential multiplications
    // (N base-field muls vs N exponentiations)
    let row_bases: Vec<F> = omega_j_inv.powers().take(n).collect();

    // Parallel row-first scaling: row k has d entries, column t gets scale (ω_J^{-k})^t
    coeffs
        .par_rows_mut()
        .zip(row_bases.par_iter())
        .for_each(|(row, &row_base)| {
            for (val, scale) in row.iter_mut().zip(row_base.powers()) {
                *val *= scale;
            }
        });

    // ═══════════════════════════════════════════════════════════════════════
    // Step 3: Flatten to base field, zero-pad, and DFT
    // ═══════════════════════════════════════════════════════════════════════
    // Flatten EF → F before the DFT rather than after: dft_algebra_batch
    // internally does flatten → dft_batch → reconstitute, but we need base
    // field for commitment anyway, so flattening first skips the reconstitute.
    let base_width = d * EF::DIMENSION;
    let mut base_coeffs = <EF as BasedVectorSpace<F>>::flatten_to_base(coeffs.values);
    base_coeffs.resize(n * b * base_width, F::ZERO);
    let coeffs_padded = RowMajorMatrix::new(base_coeffs, base_width);

    // ═══════════════════════════════════════════════════════════════════════
    // Step 4: Batched forward DFT (PLAIN, not coset) on base field
    // ═══════════════════════════════════════════════════════════════════════
    // Because g^k is baked into coefficients, plain DFT gives evaluations on gK
    // Result: E[i, t] = q_t(g·ω_K^i)
    let lde = config.dft.dft_batch(coeffs_padded);

    // ═══════════════════════════════════════════════════════════════════════
    // Step 5: Bit-reverse rows for commitment
    // ═══════════════════════════════════════════════════════════════════════
    let quotient_matrix = lde.bit_reverse_rows().to_row_major_matrix();

    let tree = config.lmcs.build_aligned_tree(vec![quotient_matrix]);

    Committed::new(tree, log_blowup)
}
