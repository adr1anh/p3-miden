//! Lifted STARK prover.
//!
//! This module provides:
//! - [`prove_single`]: Prove a single AIR instance
//! - [`prove_multi`]: Prove multiple AIRs with traces of different heights
//!
//! Uses the lifted STARK protocol with LMCS commitments. For multi-trace proving,
//! traces must be provided in ascending height order. Numerators are accumulated
//! using cyclic extension before a single vanishing division.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::{CanSample, CanSampleBits};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, PrimeCharacteristicRing, PrimeField64, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_miden_air::MidenAir;
use p3_miden_lifted_fri::prover::open_with_channel;
use p3_miden_lmcs::Lmcs;
use p3_miden_transcript::ProverChannel;
use p3_util::log2_strict_usize;
use thiserror::Error;

use p3_miden_lifted_stark::{
    LiftedCoset, StarkConfig, row_as_ext, row_pair_matrix, row_to_ext, sample_ext, sample_ood_zeta,
};
use p3_miden_lifted_verifier::ConstraintFolder;

use crate::{commit_quotient, commit_traces};

use crate::PeriodicLde;

/// Log₂ of constraint degree for quotient decomposition.
///
/// The quotient polynomial Q is evaluated on the coset gJ of size N×D
/// where D = 2^LOG_CONSTRAINT_DEGREE = 4, then decomposed into D chunks
/// and extended to gK for commitment.
const LOG_CONSTRAINT_DEGREE: usize = 2;

/// Errors that can occur during proving.
#[derive(Debug, Error)]
pub enum ProverError {
    #[error("aux trace required but not provided")]
    AuxTraceRequired,
    #[error("trace height mismatch: expected {expected}, got {actual}")]
    TraceHeightMismatch { expected: usize, actual: usize },
    #[error("trace width mismatch: expected {expected}, got {actual}")]
    TraceWidthMismatch { expected: usize, actual: usize },
    #[error("invalid periodic table")]
    InvalidPeriodicTable,
    #[error(
        "traces must be in ascending height order: trace {index} has height {height}, but previous max was {prev_max}"
    )]
    TracesNotAscending {
        index: usize,
        height: usize,
        prev_max: usize,
    },
    #[error("no traces provided")]
    NoTraces,
}

/// Prove a single AIR.
///
/// Assumes the channel has been initialized with domain separator and public values.
///
/// # Arguments
/// - `config`: STARK configuration (PCS params, LMCS, DFT)
/// - `air`: The AIR definition
/// - `trace`: Main trace matrix
/// - `public_values`: Public values for this AIR
/// - `channel`: Prover channel for transcript
///
/// # Returns
/// `Ok(())` on success, or a `ProverError` if validation fails.
pub fn prove_single<F, EF, A, L, Dft, Ch>(
    config: &StarkConfig<L, Dft>,
    air: &A,
    trace: &RowMajorMatrix<F>,
    public_values: &[F],
    channel: &mut Ch,
) -> Result<(), ProverError>
where
    F: TwoAdicField + PrimeField64,
    EF: ExtensionField<F>,
    A: MidenAir<F, EF>,
    L: Lmcs<F = F>,
    L::Commitment: Copy,
    Dft: TwoAdicSubgroupDft<F>,
    Ch: ProverChannel<F = F, Commitment = L::Commitment> + CanSample<F> + CanSampleBits<usize>,
{
    // Validate inputs
    if trace.width() != air.width() {
        return Err(ProverError::TraceWidthMismatch {
            expected: air.width(),
            actual: trace.width(),
        });
    }
    assert!(
        trace.height().is_power_of_two(),
        "trace height must be power of two"
    );

    let trace_height = trace.height();
    let log_trace_height = log2_strict_usize(trace_height);
    let log_blowup = config.pcs.fri.log_blowup;
    let log_lde_height = log_trace_height + log_blowup;

    // Create LDE coset (single AIR at max height, no lifting)
    let lde_coset = LiftedCoset::new(log_trace_height, log_lde_height, log_lde_height);

    // Derive quotient domain coset (size N×D where D = CONSTRAINT_DEGREE)
    let quotient_coset = lde_coset.quotient_domain(LOG_CONSTRAINT_DEGREE);

    // 1. Commit main trace
    let main_committed = commit_traces(config, vec![trace.clone()]);
    channel.send_commitment(main_committed.root());

    // 2. Sample randomness and build aux trace
    let num_randomness = air.num_randomness();
    let randomness: Vec<EF> = (0..num_randomness)
        .map(|_| sample_ext::<F, EF, _>(channel))
        .collect();

    let aux_trace = air
        .build_aux_trace(trace, &randomness)
        .ok_or(ProverError::AuxTraceRequired)?;

    if aux_trace.height() != trace_height {
        return Err(ProverError::TraceHeightMismatch {
            expected: trace_height,
            actual: aux_trace.height(),
        });
    }

    let expected_aux_width = air.aux_width() * EF::DIMENSION;
    assert_eq!(
        aux_trace.width(),
        expected_aux_width,
        "aux trace width mismatch: expected {expected_aux_width}, got {}",
        aux_trace.width()
    );

    // 3. Commit aux trace
    let aux_committed = commit_traces(config, vec![aux_trace.clone()]);
    channel.send_commitment(aux_committed.root());

    // 4. Sample constraint folding challenge
    let alpha: EF = sample_ext::<F, EF, _>(channel);

    // 5. Build periodic LDEs via quotient coset method
    let periodic_lde: PeriodicLde<F> = PeriodicLde::build(&quotient_coset, air.periodic_table())
        .ok_or(ProverError::InvalidPeriodicTable)?;

    // 6. Compute quotient Q(gJ) on the quotient domain (natural order)
    // Get views into committed LDEs: truncate to gJ, then bit-reverse for natural order
    // gJ is the first N×D rows of gK in bit-reversed order
    let constraint_degree = 1 << LOG_CONSTRAINT_DEGREE;
    let main_on_gj = main_committed
        .matrix_view(0)
        .quotient_domain_natural(constraint_degree);
    let aux_on_gj = aux_committed
        .matrix_view(0)
        .quotient_domain_natural(constraint_degree);

    let quotient_numerator = compute_quotient_numerator::<F, EF, A, _>(
        air,
        &main_on_gj,
        &aux_on_gj,
        &quotient_coset,
        alpha,
        &randomness,
        public_values,
        &periodic_lde,
    );

    // 7. Divide by vanishing polynomial (natural order on gJ)
    let q_evals = divide_by_vanishing_natural::<F, EF>(quotient_numerator, &quotient_coset);

    // 8. Commit quotient using fused scaling pipeline
    // commit_quotient decomposes into D chunks and extends to gK
    let quotient_committed = commit_quotient(config, q_evals, log_trace_height);
    channel.send_commitment(quotient_committed.root());

    // 9. Sample OOD point
    let zeta: EF = sample_ood_zeta::<F, EF, _>(channel, log_trace_height, log_lde_height);
    let h = F::two_adic_generator(log_trace_height);
    let zeta_next = zeta * EF::from(h);

    // 10. Open via PCS
    open_with_channel::<F, EF, L, RowMajorMatrix<F>, _, 2>(
        &config.pcs,
        &config.lmcs,
        log_lde_height,
        [zeta, zeta_next],
        &[
            main_committed.tree(),
            aux_committed.tree(),
            quotient_committed.tree(),
        ],
        channel,
    );

    Ok(())
}

/// An AIR with its associated trace for multi-trace proving.
pub struct AirWithTrace<'a, F, A> {
    /// The AIR definition
    pub air: &'a A,
    /// Main trace matrix
    pub trace: &'a RowMajorMatrix<F>,
    /// Public values for this AIR
    pub public_values: &'a [F],
}

impl<'a, F, A> AirWithTrace<'a, F, A> {
    /// Create a new AIR-trace pair.
    pub fn new(air: &'a A, trace: &'a RowMajorMatrix<F>, public_values: &'a [F]) -> Self {
        Self {
            air,
            trace,
            public_values,
        }
    }
}

/// Prove multiple AIRs with traces of different heights.
///
/// Traces must be provided in ascending height order (smallest first). Each trace
/// may have a different height that is a power of 2. The quotient numerators are
/// accumulated using cyclic extension:
///
/// 1. Compute numerator N_0 on the smallest quotient domain
/// 2. For each subsequent trace j:
///    - Extend accumulator to trace j's quotient domain size
///    - Fold: `acc = acc * beta + N_j`
/// 3. Divide by Z_H once on the largest quotient domain
///
/// # Arguments
/// - `config`: STARK configuration (PCS params, LMCS, DFT)
/// - `airs_and_traces`: AIR definitions with their traces, sorted by height (ascending)
/// - `channel`: Prover channel for transcript
///
/// # Returns
/// `Ok(())` on success, or a `ProverError` if validation fails.
#[allow(clippy::too_many_arguments)]
pub fn prove_multi<F, EF, A, L, Dft, Ch>(
    config: &StarkConfig<L, Dft>,
    airs_and_traces: &[AirWithTrace<'_, F, A>],
    channel: &mut Ch,
) -> Result<(), ProverError>
where
    F: TwoAdicField + PrimeField64,
    EF: ExtensionField<F>,
    A: MidenAir<F, EF>,
    L: Lmcs<F = F>,
    L::Commitment: Copy,
    Dft: TwoAdicSubgroupDft<F>,
    Ch: ProverChannel<F = F, Commitment = L::Commitment> + CanSample<F> + CanSampleBits<usize>,
{
    if airs_and_traces.is_empty() {
        return Err(ProverError::NoTraces);
    }

    // Validate traces are in ascending height order
    let mut prev_height = 0;
    for (i, awt) in airs_and_traces.iter().enumerate() {
        let height = awt.trace.height();
        if height < prev_height {
            return Err(ProverError::TracesNotAscending {
                index: i,
                height,
                prev_max: prev_height,
            });
        }
        assert!(
            height.is_power_of_two(),
            "trace height must be power of two"
        );
        if awt.trace.width() != awt.air.width() {
            return Err(ProverError::TraceWidthMismatch {
                expected: awt.air.width(),
                actual: awt.trace.width(),
            });
        }
        prev_height = height;
    }

    let log_blowup = config.pcs.fri.log_blowup;

    // Max trace height determines the LDE domain
    let max_trace_height = airs_and_traces.last().unwrap().trace.height();
    let log_max_trace_height = log2_strict_usize(max_trace_height);
    let log_lde_height = log_max_trace_height + log_blowup;

    // Max LDE coset (for the largest trace, no lifting)
    let max_lde_coset = LiftedCoset::new(log_max_trace_height, log_lde_height, log_lde_height);
    let max_quotient_coset = max_lde_coset.quotient_domain(LOG_CONSTRAINT_DEGREE);
    let max_quotient_height = max_quotient_coset.lde_height();

    // 1. Commit all main traces
    let main_traces: Vec<_> = airs_and_traces
        .iter()
        .map(|awt| awt.trace.clone())
        .collect();
    let main_committed = commit_traces(config, main_traces);
    channel.send_commitment(main_committed.root());

    // 2. Sample randomness and build aux traces for all AIRs
    // First, collect max randomness needed across all AIRs
    let max_num_randomness = airs_and_traces
        .iter()
        .map(|awt| awt.air.num_randomness())
        .max()
        .unwrap_or(0);

    let randomness: Vec<EF> = (0..max_num_randomness)
        .map(|_| sample_ext::<F, EF, _>(channel))
        .collect();

    // Build aux traces for all AIRs
    let aux_traces: Vec<RowMajorMatrix<F>> = airs_and_traces
        .iter()
        .map(|awt| {
            let num_rand = awt.air.num_randomness();
            awt.air
                .build_aux_trace(awt.trace, &randomness[..num_rand])
                .expect("aux trace required")
        })
        .collect();

    // 3. Commit all aux traces
    let aux_committed = commit_traces(config, aux_traces.clone());
    channel.send_commitment(aux_committed.root());

    // 4. Sample per-trace alpha and accumulation beta
    let alphas: Vec<EF> = airs_and_traces
        .iter()
        .map(|_| sample_ext::<F, EF, _>(channel))
        .collect();
    let beta: EF = sample_ext::<F, EF, _>(channel);

    // 5. Compute numerators for each trace and accumulate
    let constraint_degree = 1 << LOG_CONSTRAINT_DEGREE;
    let mut numerators: Vec<Vec<EF>> = Vec::with_capacity(airs_and_traces.len());

    for (i, awt) in airs_and_traces.iter().enumerate() {
        let trace_height = awt.trace.height();
        let log_trace_height = log2_strict_usize(trace_height);

        // Compute LDE height for this trace
        let log_this_lde_height = log_trace_height + log_blowup;

        // Create LiftedCoset for this trace (may be lifted relative to max)
        let this_lde_coset =
            LiftedCoset::new(log_trace_height, log_this_lde_height, log_lde_height);
        let this_quotient_coset = this_lde_coset.quotient_domain(LOG_CONSTRAINT_DEGREE);

        // Get views into committed LDEs for this trace
        let main_on_gj = main_committed
            .matrix_view(i)
            .quotient_domain_natural(constraint_degree);
        let aux_on_gj = aux_committed
            .matrix_view(i)
            .quotient_domain_natural(constraint_degree);

        // Build periodic LDE for this trace via coset method
        let periodic_lde = PeriodicLde::build(&this_quotient_coset, awt.air.periodic_table())
            .ok_or(ProverError::InvalidPeriodicTable)?;

        // Compute numerator (NO vanishing division yet)
        let numerator = compute_quotient_numerator::<F, EF, A, _>(
            awt.air,
            &main_on_gj,
            &aux_on_gj,
            &this_quotient_coset,
            alphas[i],
            &randomness[..awt.air.num_randomness()],
            awt.public_values,
            &periodic_lde,
        );

        numerators.push(numerator);
    }

    // 6. Accumulate numerators with beta folding
    let accumulated = accumulate_numerators(numerators, beta);

    // Verify we have the expected size (max quotient domain)
    assert_eq!(
        accumulated.len(),
        max_quotient_height,
        "accumulated numerator should be on max quotient domain"
    );

    // 7. Divide by vanishing polynomial once on full gJ
    // Use the max quotient coset since all numerators are now on the max domain
    let q_evals = divide_by_vanishing_natural::<F, EF>(accumulated, &max_quotient_coset);

    // 8. Commit quotient
    let quotient_committed = commit_quotient(config, q_evals, log_max_trace_height);
    channel.send_commitment(quotient_committed.root());

    // 9. Sample OOD point
    let zeta: EF = sample_ood_zeta::<F, EF, _>(channel, log_max_trace_height, log_lde_height);
    let h = F::two_adic_generator(log_max_trace_height);
    let zeta_next = zeta * EF::from(h);

    // 10. Open via PCS
    // Collect all trees for opening
    let mut trees: Vec<_> = vec![main_committed.tree(), aux_committed.tree()];
    trees.push(quotient_committed.tree());

    open_with_channel::<F, EF, L, RowMajorMatrix<F>, _, 2>(
        &config.pcs,
        &config.lmcs,
        log_lde_height,
        [zeta, zeta_next],
        &trees,
        channel,
    );

    Ok(())
}

/// Compute folded constraint numerators on gJ (natural order).
///
/// Evaluates constraints at each point of gJ and folds them with alpha.
/// Input matrices must be in natural order on gJ.
///
/// Uses parallel iteration via rayon when the `parallel` feature is enabled.
/// Precomputes selectors via the LiftedCoset for efficiency.
#[allow(clippy::too_many_arguments)]
fn compute_quotient_numerator<F, EF, A, M>(
    air: &A,
    main_on_gj: &M,
    aux_on_gj: &M,
    coset: &LiftedCoset,
    alpha: EF,
    randomness: &[EF],
    public_values: &[F],
    periodic_lde: &PeriodicLde<F>,
) -> Vec<EF>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F>,
    A: MidenAir<F, EF>,
    M: Matrix<F> + Sync,
{
    let gj_height = coset.lde_height();
    let constraint_degree = coset.blowup();

    // Precompute selectors via coset method
    let sels = coset.selectors::<F>();

    // Convert public values to EF once
    let public_values_ef: Vec<EF> = public_values.iter().copied().map(EF::from).collect();

    // Parallel iteration over quotient domain points
    (0..gj_height)
        .into_par_iter()
        .map(|i| {
            let next_i = (i + constraint_degree) % gj_height;

            // Get main trace rows (base field → EF)
            // Note: row_as_ext promotes F to EF efficiently
            let main_local_row: Vec<F> = main_on_gj.row(i).unwrap().into_iter().collect();
            let main_next_row: Vec<F> = main_on_gj.row(next_i).unwrap().into_iter().collect();
            let main_local = row_as_ext::<F, EF>(&main_local_row);
            let main_next = row_as_ext::<F, EF>(&main_next_row);

            // Get aux trace rows (base field → EF, packed representation)
            let aux_local_row: Vec<F> = aux_on_gj.row(i).unwrap().into_iter().collect();
            let aux_next_row: Vec<F> = aux_on_gj.row(next_i).unwrap().into_iter().collect();
            let aux_local = row_to_ext::<F, EF>(&aux_local_row);
            let aux_next = row_to_ext::<F, EF>(&aux_next_row);

            // Get precomputed selectors at this point and promote to EF
            // Arithmetic ordering: EF::from(F) is efficient
            let is_first_row = EF::from(sels.is_first_row[i]);
            let is_last_row = EF::from(sels.is_last_row[i]);
            let is_transition = EF::from(sels.is_transition[i]);

            // Get periodic values at this row (F values, promoted to EF)
            let periodic_values: Vec<EF> = periodic_lde.values_at(i).map(EF::from).collect();

            // Build folder and evaluate constraints
            let main_pair = row_pair_matrix(&main_local, &main_next);
            let aux_pair = row_pair_matrix(&aux_local, &aux_next);

            let mut folder = ConstraintFolder {
                main: main_pair,
                aux: aux_pair,
                randomness,
                public_values: &public_values_ef,
                periodic_values: &periodic_values,
                is_first_row,
                is_last_row,
                is_transition,
                alpha,
                accumulator: EF::ZERO,
                _phantom: PhantomData,
            };

            air.eval(&mut folder);
            folder.accumulator
        })
        .collect()
}

/// Divide quotient numerator by vanishing polynomial (natural order).
///
/// Takes constraint numerator on gJ in natural order and divides by the
/// vanishing polynomial Z_H(x) = x^N - 1 where N is trace height.
///
/// Uses precomputed batch inverse for efficiency - due to periodicity,
/// Z_H(x) has only 2^rate_bits distinct values on gJ.
///
/// # Arguments
///
/// - `numerator`: Constraint numerator values on gJ in natural order
/// - `coset`: The quotient domain coset providing domain information
///
/// # Returns
///
/// Q(gJ) = numerator / Z_H in natural order, ready for `commit_quotient`.
fn divide_by_vanishing_natural<F, EF>(numerator: Vec<EF>, coset: &LiftedCoset) -> Vec<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    // Precompute inverse vanishing values via coset method
    let inv_van = coset.inv_vanishing::<F>();

    // Parallel division: num * inv_vanishing
    // Arithmetic ordering: EF * F (EF on left for efficiency)
    numerator
        .into_par_iter()
        .enumerate()
        .map(|(i, num)| {
            // EF * F is more efficient than F * EF
            num * EF::from(inv_van[i])
        })
        .collect()
}

// ============================================================================
// Multi-Trace Support
// ============================================================================

/// Cyclically extend values to a larger domain.
///
/// For lifted STARK, when accumulating numerators from traces of different heights,
/// we need to extend smaller numerators to larger domains. This is done by cycling
/// the values: `values[i % original_len]`.
///
/// This preserves the algebraic property that if `f(x)` is divisible by `Z_{H_small}(x)`,
/// then the cyclic extension is divisible by `Z_{H_large}(x)` because:
/// `Z_{H_large}(x) = Z_{H_small}(x) · Φ_r(x)` where `Φ_r` is a cyclotomic factor.
///
/// # Arguments
/// - `values`: Original values on a smaller domain
/// - `new_len`: Target length (must be a multiple of `values.len()`)
///
/// # Panics
/// - If `new_len` is not a multiple of `values.len()`
pub fn cyclic_extend<T: Copy>(values: &[T], new_len: usize) -> Vec<T> {
    let original_len = values.len();
    assert!(
        new_len.is_multiple_of(original_len),
        "new_len ({new_len}) must be a multiple of original_len ({original_len})"
    );

    if new_len == original_len {
        return values.to_vec();
    }

    values.iter().cycle().take(new_len).copied().collect()
}

/// Accumulate numerators from multiple traces using beta folding.
///
/// For lifted STARK with traces of heights h_0 < h_1 < ... < h_{k-1}, we:
/// 1. Start with the smallest trace's numerator
/// 2. Cyclically extend to the next larger size
/// 3. Fold: `acc = acc * beta + numerator_j`
/// 4. Repeat until all traces are accumulated
///
/// The final result is on the largest trace's quotient domain.
///
/// # Arguments
/// - `numerators`: Numerators in ascending height order (smallest domain first)
/// - `beta`: Folding challenge
///
/// # Returns
/// Accumulated numerator on the largest domain, ready for vanishing division.
pub fn accumulate_numerators<EF: ExtensionField<EF>>(
    numerators: Vec<Vec<EF>>,
    beta: EF,
) -> Vec<EF> {
    let mut iter = numerators.into_iter();

    // Start with the first (smallest) numerator
    let mut acc = match iter.next() {
        Some(first) => first,
        None => return vec![],
    };

    // Accumulate remaining numerators in ascending size order
    for num_j in iter {
        let new_len = num_j.len();

        // Extend accumulator to match the new numerator's size
        let acc_extended = cyclic_extend(&acc, new_len);

        // Fold: acc = acc * beta + num_j
        acc = acc_extended
            .into_iter()
            .zip(num_j)
            .map(|(a, n)| a * beta + n)
            .collect();
    }

    acc
}
