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
use p3_field::{
    BasedVectorSpace, ExtensionField, Field, PrimeField64, TwoAdicField,
    batch_multiplicative_inverse,
};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_miden_lifted_air::LiftedAir;
use p3_miden_lifted_fri::prover::open_with_channel;
use p3_miden_lmcs::Lmcs;
use p3_miden_transcript::ProverChannel;
use p3_util::log2_strict_usize;
use thiserror::Error;

use p3_miden_lifted_stark::{AirWitness, LiftedCoset, StarkConfig, ValidationError};

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
    #[error("invalid instances: {0}")]
    Validation(#[from] ValidationError),
    #[error("aux trace required but not provided")]
    AuxTraceRequired,
    #[error("invalid periodic table")]
    InvalidPeriodicTable,
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
    A: LiftedAir<F, EF>,
    L: Lmcs<F = F>,
    L::Commitment: Copy,
    Dft: TwoAdicSubgroupDft<F>,
    Ch: ProverChannel<F = F, Commitment = L::Commitment> + CanSample<F> + CanSampleBits<usize>,
{
    let witness = AirWitness::new(trace, public_values);
    prove_multi(config, &[(air, witness)], channel)
}

/// Prove multiple AIRs with traces of different heights.
///
/// Instances must be provided in ascending height order (smallest first). Each trace
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
/// - `instances`: Pairs of (AIR, witness) sorted by trace height (ascending)
/// - `channel`: Prover channel for transcript
///
/// # Returns
/// `Ok(())` on success, or a `ProverError` if validation fails.
#[allow(clippy::too_many_arguments)]
pub fn prove_multi<F, EF, A, L, Dft, Ch>(
    config: &StarkConfig<L, Dft>,
    instances: &[(&A, AirWitness<'_, F>)],
    channel: &mut Ch,
) -> Result<(), ProverError>
where
    F: TwoAdicField + PrimeField64,
    EF: ExtensionField<F>,
    A: LiftedAir<F, EF>,
    L: Lmcs<F = F>,
    L::Commitment: Copy,
    Dft: TwoAdicSubgroupDft<F>,
    Ch: ProverChannel<F = F, Commitment = L::Commitment> + CanSample<F> + CanSampleBits<usize>,
{
    validate_inputs(instances)?;

    let log_blowup = config.pcs.fri.log_blowup;

    // Max trace height determines the LDE domain
    let max_trace_height = instances.last().unwrap().1.trace.height();
    let log_max_trace_height = log2_strict_usize(max_trace_height);
    let log_lde_height = log_max_trace_height + log_blowup;

    // Max LDE coset (for the largest trace, no lifting)
    let max_lde_coset = LiftedCoset::unlifted(log_max_trace_height, log_blowup);
    let max_quotient_coset = max_lde_coset.quotient_domain(LOG_CONSTRAINT_DEGREE);
    let max_quotient_height = max_quotient_coset.lde_height();

    // 1. Commit all main traces
    let main_traces: Vec<_> = instances.iter().map(|(_, w)| w.trace.clone()).collect();
    let main_committed = commit_traces(config, main_traces);
    channel.send_commitment(main_committed.root());

    // 2. Sample randomness and build aux traces for all AIRs
    let max_num_randomness = instances
        .iter()
        .map(|(air, _)| air.num_randomness())
        .max()
        .unwrap_or(0);

    let randomness: Vec<EF> = (0..max_num_randomness)
        .map(|_| channel.sample_algebra_element::<EF>())
        .collect();

    // Build and validate aux traces for all AIRs, then flatten to base field
    let aux_traces: Vec<RowMajorMatrix<F>> = instances
        .iter()
        .map(|(air, w)| {
            let num_rand = air.num_randomness();
            let aux = air
                .build_aux_trace(w.trace, &randomness[..num_rand])
                .expect("aux trace required");

            assert_eq!(
                aux.width(),
                air.aux_width(),
                "aux trace width mismatch: expected {}, got {}",
                air.aux_width(),
                aux.width()
            );
            assert_eq!(
                aux.height(),
                w.trace.height(),
                "aux trace height mismatch: expected {}, got {}",
                w.trace.height(),
                aux.height()
            );

            // Flatten EF -> F for commitment
            let base_width = aux.width() * EF::DIMENSION;
            let base_values = <EF as BasedVectorSpace<F>>::flatten_to_base(aux.values);
            RowMajorMatrix::new(base_values, base_width)
        })
        .collect();

    // 3. Commit all aux traces
    let aux_committed = commit_traces(config, aux_traces);
    channel.send_commitment(aux_committed.root());

    // 4. Sample constraint folding alpha and accumulation beta
    let alpha: EF = channel.sample_algebra_element::<EF>();
    let beta: EF = channel.sample_algebra_element::<EF>();

    // 5. Compute numerators for each trace and accumulate
    let constraint_degree = 1 << LOG_CONSTRAINT_DEGREE;
    let mut numerators: Vec<Vec<EF>> = Vec::with_capacity(instances.len());

    for (i, (air, w)) in instances.iter().enumerate() {
        let trace_height = w.trace.height();
        let log_trace_height = log2_strict_usize(trace_height);

        // Create LiftedCoset for this trace (may be lifted relative to max)
        let this_lde_coset = LiftedCoset::new(log_trace_height, log_blowup, log_max_trace_height);
        let this_quotient_coset = this_lde_coset.quotient_domain(LOG_CONSTRAINT_DEGREE);

        // Get views into committed LDEs for this trace
        let main_on_gj = main_committed.quotient_domain_natural(i, constraint_degree);
        let aux_on_gj = aux_committed.quotient_domain_natural(i, constraint_degree);

        // Build periodic LDE for this trace via coset method
        let periodic_lde = PeriodicLde::build(&this_quotient_coset, air.periodic_columns())
            .ok_or(ProverError::InvalidPeriodicTable)?;

        // Compute numerator (NO vanishing division yet)
        let numerator = evaluate_constraints::<F, EF, A, _>(
            *air,
            &main_on_gj,
            &aux_on_gj,
            &this_quotient_coset,
            alpha,
            &randomness[..air.num_randomness()],
            w.public_values,
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

/// Validate prover inputs: width match, non-empty, ascending height.
///
/// Power-of-two height is enforced by [`AirWitness::new`].
fn validate_inputs<F, EF, A>(instances: &[(&A, AirWitness<'_, F>)]) -> Result<(), ProverError>
where
    F: Field,
    EF: ExtensionField<F>,
    A: LiftedAir<F, EF>,
{
    for (i, (air, w)) in instances.iter().enumerate() {
        if w.trace.width() != air.width() {
            return Err(ValidationError::WidthMismatch {
                index: i,
                expected: air.width(),
                actual: w.trace.width(),
            }
            .into());
        }
    }
    let verifier_instances: Vec<_> = instances.iter().map(|(_, w)| w.to_instance()).collect();
    p3_miden_lifted_stark::validate_instances(&verifier_instances)?;
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
