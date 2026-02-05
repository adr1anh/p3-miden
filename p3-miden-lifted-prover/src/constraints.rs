//! Constraint evaluation and quotient commitment for the prover.
//!
//! This module provides:
//! - [`commit_quotient`]: Commits evaluated quotient polynomial using fused scaling pipeline
//! - [`ProverConstraintFolder`]: SIMD-optimized folder for constraint evaluation

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_dft::TwoAdicSubgroupDft;
use p3_field::{Algebra, BasedVectorSpace, ExtensionField, Field, PackedField, TwoAdicField};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_miden_air::MidenAirBuilder;
use p3_miden_lmcs::Lmcs;
use p3_util::log2_strict_usize;

use crate::{Committed, StarkConfig};

// ============================================================================
// Quotient Commitment
// ============================================================================

/// Commit quotient polynomial using the fused scaling pipeline.
///
/// Takes Q(gJ) evaluations in natural order, decomposes into D quotient
/// components, and commits their LDE evaluations on gK.
///
/// # Pipeline
///
/// 1. Reshape Q(gJ) as N×D matrix (column t = coset g·ω_J^t·H)
/// 2. Batch iDFT over H
/// 3. Fused scaling: multiply by (ω_J^t)^{-k} to bake g^k into coefficients
/// 4. Zero-pad to N·B rows
/// 5. Batch plain DFT → evaluations on gK
/// 6. Bit-reverse, flatten to base field, and commit via LMCS
///
/// # Arguments
///
/// - `config`: STARK configuration (provides DFT, LMCS, blowup)
/// - `q_evals`: Q(gJ) evaluations in natural order, length N·D
/// - `log_trace_height`: Log₂ of trace height N
///
/// # Returns
///
/// A [`Committed`] wrapper around the quotient tree with base field matrix.
///
/// # Panics
///
/// - If `q_evals.len()` is not divisible by `(1 << log_trace_height)`
/// - If blowup B < constraint degree D
pub fn commit_quotient<F, EF, L, Dft>(
    config: &StarkConfig<L, Dft>,
    q_evals: Vec<EF>,
    log_trace_height: usize,
) -> Committed<F, RowMajorMatrix<F>, L>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    L: Lmcs<F = F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    let n = 1usize << log_trace_height;
    let d = q_evals.len() / n;
    let log_d = log2_strict_usize(d);
    let log_blowup = config.pcs.fri.log_blowup;
    let b = 1usize << log_blowup;

    debug_assert_eq!(
        q_evals.len() % n,
        0,
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
    let omega_j_inv = EF::from(F::two_adic_generator(log_trace_height + log_d).inverse());

    for t in 0..d {
        let base = omega_j_inv.exp_u64(t as u64); // ω_J^{-t}
        let mut scale = EF::ONE;
        for k in 0..n {
            coeffs.values[k * d + t] *= scale;
            scale *= base;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Step 3: Zero-pad to height N·B
    // ═══════════════════════════════════════════════════════════════════════
    coeffs.values.resize(n * b * d, EF::ZERO);
    let coeffs_padded = RowMajorMatrix::new(coeffs.values, d);

    // ═══════════════════════════════════════════════════════════════════════
    // Step 4: Batched forward DFT (PLAIN, not coset)
    // ═══════════════════════════════════════════════════════════════════════
    // Because g^k is baked into coefficients, plain DFT gives evaluations on gK
    // Result: E[i, t] = q_t(g·ω_K^i)
    let lde = config.dft.dft_algebra_batch(coeffs_padded);

    // ═══════════════════════════════════════════════════════════════════════
    // Step 5: Bit-reverse for commitment
    // ═══════════════════════════════════════════════════════════════════════
    let mut lde_br = lde;
    p3_util::reverse_slice_index_bits(&mut lde_br.values);

    // ═══════════════════════════════════════════════════════════════════════
    // Step 6: Flatten to base field and commit via LMCS
    // ═══════════════════════════════════════════════════════════════════════
    // Flatten EF values to base field for compatibility with trace commitments
    let base_values = <EF as BasedVectorSpace<F>>::flatten_to_base(lde_br.values);
    // Width = D * EF::DIMENSION (D quotient chunks, each as EF elements flattened)
    let quotient_matrix = RowMajorMatrix::new(base_values, d * EF::DIMENSION);

    let tree = config.lmcs.build_aligned_tree(vec![quotient_matrix]);

    Committed::new(tree, log_blowup)
}

// ============================================================================
// Prover Constraint Folder (SIMD-optimized)
// ============================================================================

/// Packed constraint folder for SIMD-optimized prover evaluation.
///
/// Uses packed types to evaluate constraints on multiple domain points simultaneously:
/// - `P`: Packed base field (e.g., `PackedBabyBear`)
/// - `PE`: Packed extension field - must be `Algebra<EF> + Algebra<P> + BasedVectorSpace<P>`
///
/// Unlike [`ConstraintFolder`](p3_miden_lifted_verifier::ConstraintFolder), this folder
/// uses pre-computed alpha powers with constraint indexing for efficient accumulation.
///
/// # Type Parameters
/// - `F`: Base field scalar
/// - `EF`: Extension field scalar
/// - `P`: Packed base field (with `P::Scalar = F`)
/// - `PE`: Packed extension field (must implement appropriate algebra traits)
#[derive(Debug)]
pub struct ProverConstraintFolder<'a, F, EF, P, PE>
where
    F: Field,
    EF: ExtensionField<F>,
    P: PackedField<Scalar = F>,
    PE: Algebra<EF> + Algebra<P> + BasedVectorSpace<P> + Copy + Send + Sync,
{
    /// Main trace matrix view (packed base field)
    pub main: RowMajorMatrixView<'a, P>,
    /// Aux/permutation trace matrix view (packed extension field)
    pub aux: RowMajorMatrixView<'a, PE>,
    /// Randomness for aux trace (packed extension field)
    pub packed_randomness: Vec<PE>,
    /// Public values (base field scalars)
    pub public_values: &'a [F],
    /// Periodic column values (packed base field - F polynomials evaluated at F coset points)
    pub periodic_values: &'a [P],
    /// Selector for first row (packed base field)
    pub is_first_row: P,
    /// Selector for last row (packed base field)
    pub is_last_row: P,
    /// Selector for transition rows (packed base field)
    pub is_transition: P,
    /// Pre-computed alpha powers in reverse order: [α^{n-1}, ..., α^1, α^0]
    pub alpha_powers: &'a [EF],
    /// Alpha powers decomposed into base field coefficients (transposed for SIMD)
    pub decomposed_alpha_powers: &'a [Vec<F>],
    /// Running accumulator (packed extension field)
    pub accumulator: PE,
    /// Current constraint index being processed
    pub constraint_index: usize,
    /// Phantom marker for EF
    pub _phantom: PhantomData<EF>,
}

impl<'a, F, EF, P, PE> MidenAirBuilder for ProverConstraintFolder<'a, F, EF, P, PE>
where
    F: Field,
    EF: ExtensionField<F>,
    P: PackedField<Scalar = F>,
    PE: Algebra<EF> + Algebra<P> + BasedVectorSpace<P> + Copy + Send + Sync,
{
    type F = F;
    type Expr = P;
    type Var = P;
    type M = RowMajorMatrixView<'a, P>;
    type PublicVar = F;
    type PeriodicVal = P;
    type EF = EF;
    type ExprEF = PE;
    type VarEF = PE;
    type MP = RowMajorMatrixView<'a, PE>;
    type RandomVar = PE;

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
            panic!("only window size 2 supported in this prototype")
        }
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let alpha_power = self.alpha_powers[self.constraint_index];
        self.accumulator += PE::from(alpha_power) * x.into();
        self.constraint_index += 1;
    }

    #[inline]
    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }

    #[inline]
    fn periodic_evals(&self) -> &[Self::PeriodicVal] {
        self.periodic_values
    }

    fn preprocessed(&self) -> Self::M {
        panic!("preprocessed trace not supported in this prototype")
    }

    #[inline]
    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let alpha_power = self.alpha_powers[self.constraint_index];
        self.accumulator += PE::from(alpha_power) * x.into();
        self.constraint_index += 1;
    }

    #[inline]
    fn permutation(&self) -> Self::MP {
        self.aux
    }

    #[inline]
    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        &self.packed_randomness
    }

    fn aux_bus_boundary_values(&self) -> &[Self::VarEF] {
        &[]
    }
}

impl<'a, F, EF, P, PE> ProverConstraintFolder<'a, F, EF, P, PE>
where
    F: Field,
    EF: ExtensionField<F>,
    P: PackedField<Scalar = F>,
    PE: Algebra<EF> + Algebra<P> + BasedVectorSpace<P> + Copy + Send + Sync,
{
    /// Create pre-reversed alpha powers for efficient constraint accumulation.
    ///
    /// The alpha powers are stored in reverse order so that `alpha_powers[i]`
    /// gives the correct power for constraint `i` (which gets α^{n-1-i}).
    pub fn prepare_alpha_powers(alpha: EF, constraint_count: usize) -> Vec<EF> {
        let mut powers: Vec<EF> = alpha.powers().take(constraint_count).collect();
        powers.reverse();
        powers
    }

    /// Decompose alpha powers into base field coefficients for SIMD operations.
    ///
    /// Returns a vector of length `EF::DIMENSION`, where each element is a
    /// vector of base field coefficients for that component.
    pub fn decompose_alpha_powers(alpha_powers: &[EF]) -> Vec<Vec<F>> {
        (0..EF::DIMENSION)
            .map(|i| {
                alpha_powers
                    .iter()
                    .map(|x| x.as_basis_coefficients_slice()[i])
                    .collect()
            })
            .collect()
    }
}
