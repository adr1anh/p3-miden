//! Constraint evaluation and quotient commitment for the prover.
//!
//! This module provides:
//! - [`commit_quotient`]: Commits evaluated quotient polynomial using fused scaling pipeline
//! - [`ProverConstraintFolder`]: SIMD-optimized folder for constraint evaluation

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_dft::TwoAdicSubgroupDft;
use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, PackedField, PackedValue,
    PrimeCharacteristicRing, TwoAdicField,
};
use p3_matrix::Matrix;
use p3_matrix::bitrev::BitReversibleMatrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_miden_lifted_air::{
    AirBuilder, AirBuilderWithPublicValues, ConstraintLayout, ExtensionBuilder, LiftedAir,
    LiftedAirBuilder, PairBuilder, PeriodicAirBuilder, PermutationAirBuilder,
};
use p3_miden_lifted_stark::{LiftedCoset, Selectors};
use p3_miden_lmcs::Lmcs;
use p3_util::log2_strict_usize;

use crate::StarkConfig;
use crate::commit::Committed;
use crate::periodic::PeriodicLde;

// ============================================================================
// Quotient Commitment
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
/// A [`Committed`] wrapper around the quotient tree with base field matrix.
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

// ============================================================================
// Constraint Evaluation
// ============================================================================

/// Type alias for packed base field from F.
type PackedVal<F> = <F as Field>::Packing;

/// Type alias for packed extension field from EF.
type PackedExt<F, EF> = <EF as ExtensionField<F>>::ExtensionPacking;

/// Batch size for constraint linear-combination chunks in [`finalize_constraints`].
const CONSTRAINT_BATCH: usize = 8;

/// Batched linear combination of packed extension field values with EF coefficients.
///
/// Extension-field analogue of [`PackedField::packed_linear_combination`]. Processes
/// `coeffs` and `values` in chunks of [`CONSTRAINT_BATCH`], then handles the remainder.
#[inline]
fn batched_ext_linear_combination<PE, EF>(coeffs: &[EF], values: &[PE]) -> PE
where
    EF: Field,
    PE: PrimeCharacteristicRing + Algebra<EF> + Copy,
{
    debug_assert_eq!(coeffs.len(), values.len());
    let len = coeffs.len();
    let mut acc = PE::ZERO;
    let mut start = 0;
    while start + CONSTRAINT_BATCH <= len {
        let batch: [PE; CONSTRAINT_BATCH] =
            core::array::from_fn(|i| values[start + i] * coeffs[start + i]);
        acc += PE::sum_array::<CONSTRAINT_BATCH>(&batch);
        start += CONSTRAINT_BATCH;
    }
    for (&coeff, &val) in coeffs[start..].iter().zip(&values[start..]) {
        acc += val * coeff;
    }
    acc
}

/// Batched linear combination of packed base field values with F coefficients.
///
/// Wraps [`PackedField::packed_linear_combination`] with batched chunking
/// and remainder handling, mirroring [`batched_ext_linear_combination`].
#[inline]
fn batched_base_linear_combination<P: PackedField>(coeffs: &[P::Scalar], values: &[P]) -> P {
    debug_assert_eq!(coeffs.len(), values.len());
    let len = coeffs.len();
    let mut acc = P::ZERO;
    let mut start = 0;
    while start + CONSTRAINT_BATCH <= len {
        acc += P::packed_linear_combination::<CONSTRAINT_BATCH>(
            &coeffs[start..start + CONSTRAINT_BATCH],
            &values[start..start + CONSTRAINT_BATCH],
        );
        start += CONSTRAINT_BATCH;
    }
    for (&coeff, &val) in coeffs[start..].iter().zip(&values[start..]) {
        acc += val * coeff;
    }
    acc
}

/// Evaluate constraints on the quotient domain (natural order).
///
/// Evaluates constraints at each point of gJ, collects them into base/ext streams,
/// and combines using decomposed alpha powers with `packed_linear_combination`.
/// Input matrices must be in natural order on gJ.
///
/// Uses SIMD-packed parallel iteration via rayon for optimal performance:
/// - Processes `WIDTH` points simultaneously using packed field types
/// - Main trace stays in base field, only aux trace uses extension field
/// - Selectors are packed base field values
/// - Constraints are collected then finalized in batches for efficient accumulation
///
/// # Arguments
/// - `air`: The AIR definition
/// - `main_on_gj`: Main trace matrix on quotient domain (natural order)
/// - `aux_on_gj`: Aux trace matrix on quotient domain (natural order)
/// - `coset`: Quotient domain coset information
/// - `alpha`: Challenge for constraint folding
/// - `randomness`: Randomness for aux trace
/// - `public_values`: Public values for constraint evaluation
/// - `periodic_lde`: Periodic column LDE values
/// - `layout`: Pre-computed constraint layout (base/ext index mapping)
#[allow(clippy::too_many_arguments)]
pub fn evaluate_constraints<F, EF, A, M>(
    air: &A,
    main_on_gj: &M,
    aux_on_gj: &M,
    coset: &LiftedCoset,
    alpha: EF,
    randomness: &[EF],
    public_values: &[F],
    periodic_lde: &PeriodicLde<F>,
    layout: &ConstraintLayout,
) -> Vec<EF>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F>,
    PackedExt<F, EF>: Algebra<EF> + Algebra<PackedVal<F>> + BasedVectorSpace<PackedVal<F>>,
    A: LiftedAir<F, EF>,
    M: Matrix<F> + Sync,
{
    type P<F> = PackedVal<F>;
    type PE<F, EF> = PackedExt<F, EF>;

    let gj_height = coset.lde_height();
    let constraint_degree = coset.blowup();
    let width = P::<F>::WIDTH;

    // Precompute selectors via coset method
    let mut sels = coset.selectors::<F>();

    // Pad selectors to WIDTH alignment for safe packed access
    for _ in gj_height..gj_height.next_multiple_of(width) {
        sels.is_first_row.push(F::ZERO);
        sels.is_last_row.push(F::ZERO);
        sels.is_transition.push(F::ZERO);
    }

    // ─── Decompose alpha powers by constraint layout ───
    let aux_ef_width = aux_on_gj.width() / EF::DIMENSION;
    let constraint_count = layout.total_constraints();
    let base_count = layout.base_indices.len();
    let ext_count = layout.ext_indices.len();
    let (base_alpha_powers, ext_alpha_powers) = layout.decompose_alpha(alpha);

    // Main trace width
    let main_width = main_on_gj.width();

    // Pack randomness for aux trace
    let packed_randomness: Vec<PE<F, EF>> = randomness.iter().copied().map(Into::into).collect();

    // Parallel iteration over quotient domain points, step by WIDTH
    (0..gj_height)
        .into_par_iter()
        .step_by(width)
        .flat_map_iter(|i_start| {
            // Extract packed selectors from precomputed vectors
            let selectors = sels.packed_at::<P<F>>(i_start);

            // Get main trace as packed row pair (stays in base field)
            let main_packed: Vec<P<F>> =
                main_on_gj.vertically_packed_row_pair(i_start, constraint_degree);
            let main = RowMajorMatrix::new(main_packed, main_width);

            // Get aux trace as packed row pair and convert to packed extension field
            let aux_base_packed: Vec<P<F>> =
                aux_on_gj.vertically_packed_row_pair(i_start, constraint_degree);

            // Convert from packed base field to packed extension field
            // Each EF element is formed from DIMENSION consecutive base field elements
            let aux_packed: Vec<PE<F, EF>> = (0..aux_ef_width * 2)
                .map(|i| {
                    PE::<F, EF>::from_basis_coefficients_fn(|j| {
                        aux_base_packed[i * EF::DIMENSION + j]
                    })
                })
                .collect();
            let aux = RowMajorMatrix::new(aux_packed, aux_ef_width);

            // Get packed periodic values
            let periodic_values: Vec<P<F>> = periodic_lde.packed_values_at(i_start).collect();

            // Build packed folder and evaluate constraints
            let mut folder: ProverConstraintFolder<'_, F, EF, P<F>, PE<F, EF>> =
                ProverConstraintFolder {
                    main: main.as_view(),
                    aux: aux.as_view(),
                    packed_randomness: &packed_randomness,
                    public_values,
                    periodic_values: &periodic_values,
                    selectors,
                    base_alpha_powers: &base_alpha_powers,
                    ext_alpha_powers: &ext_alpha_powers,
                    constraint_index: 0,
                    constraint_count,
                    base_constraints: Vec::with_capacity(base_count),
                    ext_constraints: Vec::with_capacity(ext_count),
                    _phantom: PhantomData,
                };

            air.eval(&mut folder);
            let accumulator = folder.finalize_constraints();

            // Unpack WIDTH results into scalar extension field values
            // PE stores DIMENSION components, each as a packed base field
            let num_results = core::cmp::min(gj_height.saturating_sub(i_start), width);
            (0..num_results).map(move |idx| {
                EF::from_basis_coefficients_fn(|coeff_idx| {
                    accumulator.as_basis_coefficients_slice()[coeff_idx].as_slice()[idx]
                })
            })
        })
        .collect()
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
/// Collects constraints during `air.eval()` into separate base/ext vectors, then
/// combines them in [`finalize_constraints`] using decomposed alpha powers and
/// `packed_linear_combination` for efficient SIMD accumulation.
///
/// # Type Parameters
/// - `F`: Base field scalar
/// - `EF`: Extension field scalar
/// - `P`: Packed base field (with `P::Scalar = F`)
/// - `PE`: Packed extension field (must implement appropriate algebra traits)
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
    pub packed_randomness: &'a [PE],
    /// Public values (base field scalars)
    pub public_values: &'a [F],
    /// Periodic column values (packed base field)
    pub periodic_values: &'a [P],
    /// Constraint selectors (packed base field)
    pub selectors: Selectors<P>,
    /// Base-field alpha powers, reordered to match base constraint emission order.
    /// `base_alpha_powers[d][j]` = d-th basis coefficient of alpha power for j-th base constraint.
    pub base_alpha_powers: &'a [Vec<F>],
    /// Extension-field alpha powers, reordered to match ext constraint emission order.
    pub ext_alpha_powers: &'a [EF],
    /// Current constraint index (debug-only bookkeeping)
    pub constraint_index: usize,
    /// Total expected constraint count (debug-only bookkeeping)
    pub constraint_count: usize,
    /// Collected base-field constraints for this row
    pub base_constraints: Vec<P>,
    /// Collected extension-field constraints for this row
    pub ext_constraints: Vec<PE>,
    pub _phantom: PhantomData<EF>,
}

impl<'a, F, EF, P, PE> ProverConstraintFolder<'a, F, EF, P, PE>
where
    F: Field,
    EF: ExtensionField<F>,
    P: PackedField<Scalar = F>,
    PE: Algebra<EF> + Algebra<P> + BasedVectorSpace<P> + Copy + Send + Sync,
{
    /// Combine all collected constraints with their pre-computed alpha powers.
    ///
    /// Base constraints use [`batched_base_linear_combination`] per basis dimension,
    /// decomposing the extension-field multiply into D base-field SIMD dot products.
    /// Extension constraints use [`batched_ext_linear_combination`] with scalar EF
    /// coefficients. Both process in chunks of [`CONSTRAINT_BATCH`].
    #[inline]
    pub fn finalize_constraints(self) -> PE {
        debug_assert_eq!(self.constraint_index, self.constraint_count);
        debug_assert_eq!(
            self.base_constraints.len(),
            self.base_alpha_powers.first().map_or(0, Vec::len)
        );
        debug_assert_eq!(self.ext_constraints.len(), self.ext_alpha_powers.len());

        // Base constraints: D independent base-field dot products
        let base = &self.base_constraints;
        let base_powers = self.base_alpha_powers;
        let acc = PE::from_basis_coefficients_fn(|d| {
            batched_base_linear_combination(&base_powers[d], base)
        });

        // Extension constraints: EF-coefficient dot product
        acc + batched_ext_linear_combination(self.ext_alpha_powers, &self.ext_constraints)
    }
}

impl<'a, F, EF, P, PE> AirBuilder for ProverConstraintFolder<'a, F, EF, P, PE>
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

    #[inline]
    fn main(&self) -> Self::M {
        self.main
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        self.selectors.is_first_row
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        self.selectors.is_last_row
    }

    #[inline]
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.selectors.is_transition
        } else {
            panic!("only window size 2 supported")
        }
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.base_constraints.push(x.into());
        self.constraint_index += 1;
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        let expr_array = array.map(Into::into);
        self.base_constraints.extend(expr_array);
        self.constraint_index += N;
    }
}

impl<'a, F, EF, P, PE> AirBuilderWithPublicValues for ProverConstraintFolder<'a, F, EF, P, PE>
where
    F: Field,
    EF: ExtensionField<F>,
    P: PackedField<Scalar = F>,
    PE: Algebra<EF> + Algebra<P> + BasedVectorSpace<P> + Copy + Send + Sync,
{
    type PublicVar = F;

    #[inline]
    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
}

impl<'a, F, EF, P, PE> PairBuilder for ProverConstraintFolder<'a, F, EF, P, PE>
where
    F: Field,
    EF: ExtensionField<F>,
    P: PackedField<Scalar = F>,
    PE: Algebra<EF> + Algebra<P> + BasedVectorSpace<P> + Copy + Send + Sync,
{
    fn preprocessed(&self) -> Self::M {
        panic!("preprocessed trace not supported in this prototype")
    }
}

impl<'a, F, EF, P, PE> ExtensionBuilder for ProverConstraintFolder<'a, F, EF, P, PE>
where
    F: Field,
    EF: ExtensionField<F>,
    P: PackedField<Scalar = F>,
    PE: Algebra<EF> + Algebra<P> + BasedVectorSpace<P> + Copy + Send + Sync,
{
    type EF = EF;
    type ExprEF = PE;
    type VarEF = PE;

    #[inline]
    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.ext_constraints.push(x.into());
        self.constraint_index += 1;
    }
}

impl<'a, F, EF, P, PE> PermutationAirBuilder for ProverConstraintFolder<'a, F, EF, P, PE>
where
    F: Field,
    EF: ExtensionField<F>,
    P: PackedField<Scalar = F>,
    PE: Algebra<EF> + Algebra<P> + BasedVectorSpace<P> + Copy + Send + Sync,
{
    type MP = RowMajorMatrixView<'a, PE>;
    type RandomVar = PE;

    #[inline]
    fn permutation(&self) -> Self::MP {
        self.aux
    }

    #[inline]
    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.packed_randomness
    }
}

impl<'a, F, EF, P, PE> PeriodicAirBuilder for ProverConstraintFolder<'a, F, EF, P, PE>
where
    F: Field,
    EF: ExtensionField<F>,
    P: PackedField<Scalar = F>,
    PE: Algebra<EF> + Algebra<P> + BasedVectorSpace<P> + Copy + Send + Sync,
{
    type PeriodicVar = P;

    #[inline]
    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        self.periodic_values
    }
}

impl<'a, F, EF, P, PE> LiftedAirBuilder for ProverConstraintFolder<'a, F, EF, P, PE>
where
    F: Field,
    EF: ExtensionField<F>,
    P: PackedField<Scalar = F>,
    PE: Algebra<EF> + Algebra<P> + BasedVectorSpace<P> + Copy + Send + Sync,
{
}
