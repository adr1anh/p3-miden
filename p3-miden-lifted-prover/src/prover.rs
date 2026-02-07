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

use p3_challenger::{CanSample, CanSampleBits};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, PrimeField64, TwoAdicField, batch_multiplicative_inverse};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_miden_air::MidenAir;
use p3_miden_lifted_fri::prover::open_with_channel;
use p3_miden_lmcs::Lmcs;
use p3_miden_transcript::ProverChannel;
use p3_util::log2_strict_usize;
use thiserror::Error;

use p3_miden_lifted_stark::{LiftedCoset, StarkConfig};

use crate::commit::commit_traces;
use crate::constraints::{commit_quotient, evaluate_constraints};

use crate::periodic::PeriodicLde;

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
/// This is a convenience wrapper around [`prove_multi`] for the single-AIR case.
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
    prove_multi(
        config,
        &[AirWithTrace::new(air, trace, public_values)],
        channel,
    )
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
    let max_lde_coset = LiftedCoset::unlifted(log_max_trace_height, log_blowup);
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
        .map(|_| channel.sample_algebra_element::<EF>())
        .collect();

    // Build and validate aux traces for all AIRs
    let aux_traces: Vec<RowMajorMatrix<F>> = airs_and_traces
        .iter()
        .map(|awt| {
            let num_rand = awt.air.num_randomness();
            let aux = awt
                .air
                .build_aux_trace(awt.trace, &randomness[..num_rand])
                .expect("aux trace required");

            let expected_width = awt.air.aux_width() * EF::DIMENSION;
            assert_eq!(
                aux.width(),
                expected_width,
                "aux trace width mismatch: expected {expected_width}, got {}",
                aux.width()
            );
            assert_eq!(
                aux.height(),
                awt.trace.height(),
                "aux trace height mismatch: expected {}, got {}",
                awt.trace.height(),
                aux.height()
            );

            aux
        })
        .collect();

    // 3. Commit all aux traces
    let aux_committed = commit_traces(config, aux_traces.clone());
    channel.send_commitment(aux_committed.root());

    // 4. Sample constraint folding alpha and accumulation beta
    let alpha: EF = channel.sample_algebra_element::<EF>();
    let beta: EF = channel.sample_algebra_element::<EF>();

    // 5. Compute numerators for each trace and accumulate
    let constraint_degree = 1 << LOG_CONSTRAINT_DEGREE;
    let mut numerators: Vec<Vec<EF>> = Vec::with_capacity(airs_and_traces.len());

    for (i, awt) in airs_and_traces.iter().enumerate() {
        let trace_height = awt.trace.height();
        let log_trace_height = log2_strict_usize(trace_height);

        // Create LiftedCoset for this trace (may be lifted relative to max)
        let this_lde_coset = LiftedCoset::new(log_trace_height, log_blowup, log_max_trace_height);
        let this_quotient_coset = this_lde_coset.quotient_domain(LOG_CONSTRAINT_DEGREE);

        // Get views into committed LDEs for this trace
        let main_on_gj = main_committed.quotient_domain_natural(i, constraint_degree);
        let aux_on_gj = aux_committed.quotient_domain_natural(i, constraint_degree);

        // Build periodic LDE for this trace via coset method
        let periodic_lde = PeriodicLde::build(&this_quotient_coset, &awt.air.periodic_table())
            .ok_or(ProverError::InvalidPeriodicTable)?;

        // Compute numerator (NO vanishing division yet)
        let numerator = evaluate_constraints::<F, EF, A, _>(
            awt.air,
            &main_on_gj,
            &aux_on_gj,
            &this_quotient_coset,
            alpha,
            &randomness[..awt.air.num_randomness()],
            awt.public_values,
            &periodic_lde,
        );

        numerators.push(numerator);
    }

    // 6. Accumulate numerators with beta folding: acc = cyclic_extend(acc) * beta + num
    let accumulated = numerators
        .into_iter()
        .reduce(|acc, mut num| {
            let acc_mask = acc.len() - 1;
            num.par_iter_mut().enumerate().for_each(|(i, n)| {
                *n = acc[i & acc_mask] * beta + *n;
            });
            num
        })
        .unwrap_or_default();

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
    let quotient_committed = commit_quotient(config, q_evals, &max_lde_coset);
    channel.send_commitment(quotient_committed.root());

    // 9. Sample OOD point (outside H and gK)
    let zeta: EF = loop {
        let z: EF = channel.sample_algebra_element::<EF>();
        if !max_lde_coset.is_in_trace_domain::<F, _>(z) && !max_lde_coset.is_in_lde_coset::<F, _>(z)
        {
            break z;
        }
    };
    let h = F::two_adic_generator(log_max_trace_height);
    let zeta_next = zeta * h;

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

/// Divide quotient numerator by vanishing polynomial (natural order).
///
/// Takes constraint numerator on gJ in natural order and divides by the
/// vanishing polynomial Z_H(x) = x^N - 1 where N is trace height.
///
/// Exploits periodicity: Z_H(x) has only 2^rate_bits distinct values on gJ,
/// so we compute only the distinct inverse values and use modular indexing.
/// This saves memory compared to expanding to full coset size.
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

    // Parallel division using modular indexing for periodicity
    // Arithmetic ordering: EF * F (EF on left for efficiency)
    numerator
        .into_par_iter()
        .enumerate()
        .map(|(i, num)| {
            // Use periodicity: only num_distinct unique values
            // For power-of-2, i % num_distinct == i & (num_distinct - 1)
            num * inv_van[i & (num_distinct - 1)]
        })
        .collect()
}
