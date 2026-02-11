//! Lifted STARK verifier.
//!
//! This module provides:
//! - [`verify_single`]: Verify a single AIR instance
//! - [`verify_multi`]: Verify multiple AIRs with traces of different heights
//!
//! Uses the lifted STARK protocol with LMCS commitments. For multi-trace verification,
//! instances must be provided in ascending height order. Constraint evaluations are
//! accumulated using Horner folding with beta before a single quotient identity check.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::{CanSample, CanSampleBits};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, PrimeCharacteristicRing, PrimeField64, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_lifted_air::{LiftedAir, SymbolicExpression, get_log_quotient_degree};
use p3_miden_lifted_fri::verifier::{PcsError, verify_with_channel as verify_pcs_with_channel};
use p3_miden_lmcs::Lmcs;
use p3_miden_transcript::{TranscriptError, VerifierChannel};
use thiserror::Error;

use p3_miden_lifted_stark::{AirInstance, LiftedCoset, StarkConfig, ValidationError};

use crate::constraints::{
    ConstraintFolder, align_width, reconstruct_quotient, row_to_packed_ext,
};
use crate::periodic::PeriodicPolys;

/// Errors that can occur during verification.
#[derive(Debug, Error)]
pub enum VerifierError {
    #[error("invalid instances: {0}")]
    Validation(#[from] ValidationError),
    #[error("PCS verification failed: {0}")]
    Pcs(#[from] PcsError),
    #[error("transcript error: {0}")]
    Transcript(#[from] TranscriptError),
    #[error("invalid aux shape: expected length divisible by {expected_divisor}, got {actual_len}")]
    InvalidAuxShape {
        expected_divisor: usize,
        actual_len: usize,
    },
    #[error("constraint mismatch: quotient * vanishing != folded constraints")]
    ConstraintMismatch,
    #[error("transcript not fully consumed")]
    TranscriptNotConsumed,
    #[error("invalid periodic table")]
    InvalidPeriodicTable,
    #[error("invalid PCS opening groups: expected 3")]
    InvalidOpeningGroups,
}

/// Verify a single AIR.
///
/// Assumes the channel has been initialized with domain separator and public values.
/// This is a convenience wrapper around [`verify_multi`] for the single-AIR case.
///
/// # Arguments
/// - `config`: STARK configuration (PCS params, LMCS, DFT)
/// - `air`: The AIR definition
/// - `log_trace_height`: Log₂ of the trace height
/// - `public_values`: Public values for this AIR
/// - `channel`: Verifier channel for transcript
///
/// # Returns
/// `Ok(())` on success, or a `VerifierError` if verification fails.
pub fn verify_single<F, EF, A, L, Dft, Ch>(
    config: &StarkConfig<L, Dft>,
    air: &A,
    log_trace_height: usize,
    public_values: &[F],
    channel: &mut Ch,
) -> Result<(), VerifierError>
where
    F: TwoAdicField + PrimeField64 + PrimeCharacteristicRing,
    EF: ExtensionField<F>,
    A: LiftedAir<F, EF>,
    L: Lmcs<F = F>,
    L::Commitment: Copy,
    Dft: TwoAdicSubgroupDft<F>,
    Ch: VerifierChannel<F = F, Commitment = L::Commitment> + CanSample<F> + CanSampleBits<usize>,
    SymbolicExpression<EF>: From<SymbolicExpression<F>>,
{
    let instance = AirInstance::new(log_trace_height, public_values);
    verify_multi(config, &[(air, instance)], channel)
}

/// Verify multiple AIRs with traces of different heights.
///
/// Instances must be provided in ascending height order (smallest first). Each trace
/// may have a different height that is a power of 2. The verifier mirrors the prover's
/// protocol:
///
/// 1. Receive commitments and sample challenges in the same order as the prover
/// 2. For each AIR, evaluate constraints at the lifted OOD point `y_j = ζ^{r_j}`
/// 3. Accumulate folded constraints with beta: `acc = acc * β + folded_j`
/// 4. Check quotient identity: `acc == Q(ζ) * Z_{H_max}(ζ)`
///
/// Lifting ensures correctness: for a trace of height `n_j` lifted by factor `r_j`,
/// the committed polynomial is `p(X^{r_j})`, so the PCS opening at `[ζ, ζ·h_max]`
/// gives `[p(ζ^{r_j}), p(h_{n_j}·ζ^{r_j})]` — exactly the local/next pair in the
/// original domain.
///
/// # Arguments
/// - `config`: STARK configuration (PCS params, LMCS, DFT)
/// - `instances`: Pairs of (AIR, instance) sorted by trace height (ascending)
/// - `channel`: Verifier channel for transcript
///
/// # Returns
/// `Ok(())` on success, or a `VerifierError` if verification fails.
pub fn verify_multi<F, EF, A, L, Dft, Ch>(
    config: &StarkConfig<L, Dft>,
    instances: &[(&A, AirInstance<'_, F>)],
    channel: &mut Ch,
) -> Result<(), VerifierError>
where
    F: TwoAdicField + PrimeField64 + PrimeCharacteristicRing,
    EF: ExtensionField<F>,
    A: LiftedAir<F, EF>,
    L: Lmcs<F = F>,
    L::Commitment: Copy,
    Dft: TwoAdicSubgroupDft<F>,
    Ch: VerifierChannel<F = F, Commitment = L::Commitment> + CanSample<F> + CanSampleBits<usize>,
    SymbolicExpression<EF>: From<SymbolicExpression<F>>,
{
    let air_instances: Vec<_> = instances.iter().map(|(_, inst)| *inst).collect();
    p3_miden_lifted_stark::validate_instances(&air_instances)?;

    let log_blowup = config.pcs.fri.log_blowup;
    let alignment = config.lmcs.alignment();

    // Derive constraint degree from the AIR definitions (max across all instances)
    let log_constraint_degree = instances
        .iter()
        .map(|(air, inst)| get_log_quotient_degree::<F, EF, A>(*air, inst.public_values.len()))
        .max()
        .unwrap_or(0);
    let constraint_degree = 1usize << log_constraint_degree;

    // Max trace height determines the LDE domain
    let log_max_trace_height = instances.last().unwrap().1.log_trace_height;
    let max_trace_height = 1usize << log_max_trace_height;
    let log_lde_height = log_max_trace_height + log_blowup;

    // Max LDE coset (for the largest trace, no lifting)
    let max_lde_coset = LiftedCoset::unlifted(log_max_trace_height, log_blowup);

    // 1. Receive main trace commitment
    let main_commit = *channel.receive_commitment()?;

    // 2. Sample randomness for aux traces
    let max_num_randomness = instances
        .iter()
        .map(|(air, _)| air.num_randomness())
        .max()
        .unwrap_or(0);

    let randomness: Vec<EF> = (0..max_num_randomness)
        .map(|_| channel.sample_algebra_element::<EF>())
        .collect();

    // 3. Receive aux trace commitment
    let aux_commit = *channel.receive_commitment()?;

    // 4. Sample constraint folding alpha and accumulation beta
    let alpha: EF = channel.sample_algebra_element::<EF>();
    let beta: EF = channel.sample_algebra_element::<EF>();

    // 5. Receive quotient commitment
    let quotient_commit = *channel.receive_commitment()?;

    // 6. Sample OOD point (outside max trace domain H and max LDE coset gK)
    let zeta: EF = loop {
        let z: EF = channel.sample_algebra_element::<EF>();
        if !max_lde_coset.is_in_trace_domain::<F, _>(z) && !max_lde_coset.is_in_lde_coset::<F, _>(z)
        {
            break z;
        }
    };
    let h = F::two_adic_generator(log_max_trace_height);
    let zeta_next = zeta * h;

    // 7. Build commitment widths for PCS verification
    // Each commitment group has one matrix per AIR instance (except quotient: single matrix)
    let main_widths: Vec<usize> = instances
        .iter()
        .map(|(air, _)| align_width(air.width(), alignment))
        .collect();
    let aux_widths: Vec<usize> = instances
        .iter()
        .map(|(air, _)| align_width(air.aux_width() * EF::DIMENSION, alignment))
        .collect();
    let quotient_width = align_width(constraint_degree * EF::DIMENSION, alignment);

    let commitments = vec![
        (main_commit, main_widths),
        (aux_commit, aux_widths),
        (quotient_commit, vec![quotient_width]),
    ];

    // 8. Verify PCS openings
    let evals = verify_pcs_with_channel::<F, EF, L, _, 2>(
        &config.pcs,
        &config.lmcs,
        &commitments,
        log_lde_height,
        [zeta, zeta_next],
        channel,
    )?;

    // 9. Extract opened values and check structure
    let [main, aux, quotient] = evals.groups() else {
        return Err(VerifierError::InvalidOpeningGroups);
    };

    // Quotient: single matrix, reconstruct Q(ζ) using max coset
    let quotient_matrix = &quotient[0];
    let quotient_chunks = row_to_packed_ext::<F, EF, _>(quotient_matrix.row(0).unwrap())?;
    let quotient_zeta = reconstruct_quotient::<F, EF>(zeta, &max_lde_coset, &quotient_chunks);

    // 10. Per-AIR constraint evaluation and beta accumulation
    let mut accumulated = EF::ZERO;

    for (j, (air, inst)) in instances.iter().enumerate() {
        let log_n_j = inst.log_trace_height;
        let n_j = 1usize << log_n_j;
        let log_lift_ratio = log_max_trace_height - log_n_j;

        // Virtual evaluation point for lifted trace: y_j = ζ^{r_j}
        // For unlifted traces (r_j = 1), y_j = ζ
        let y_j = zeta.exp_power_of_2(log_lift_ratio);

        // Extract main trace opened values
        let main_matrix = &main[j];
        let main_local: Vec<EF> = main_matrix.row(0).unwrap().into_iter().collect();
        let main_next: Vec<EF> = main_matrix.row(1).unwrap().into_iter().collect();

        // Extract aux trace opened values (reconstitute EF from base field components)
        let aux_matrix = &aux[j];
        let aux_local = row_to_packed_ext::<F, EF, _>(aux_matrix.row(0).unwrap())?;
        let aux_next = row_to_packed_ext::<F, EF, _>(aux_matrix.row(1).unwrap())?;

        // Selectors at virtual point y_j (relative to this trace's domain)
        let coset_j = LiftedCoset::new(log_n_j, log_blowup, log_max_trace_height);
        let selectors = coset_j.selectors_at::<F, _>(y_j);

        // Periodic values at virtual point y_j
        let periodic_polys = PeriodicPolys::new(air.periodic_columns())
            .ok_or(VerifierError::InvalidPeriodicTable)?;
        let periodic_values = periodic_polys.eval_at::<EF>(n_j, y_j);

        // Public values as EF
        let public_values_ef: Vec<EF> = inst.public_values.iter().copied().map(EF::from).collect();

        // Build 2-row matrices for the folder (row 0 = local, row 1 = next)
        let trace_width = align_width(air.width(), alignment);
        let aux_ef_width = align_width(air.aux_width() * EF::DIMENSION, alignment) / EF::DIMENSION;

        let main_pair = RowMajorMatrix::new([main_local, main_next].concat(), trace_width);
        let aux_pair = RowMajorMatrix::new([aux_local, aux_next].concat(), aux_ef_width);

        let num_rand = air.num_randomness();
        let mut folder = ConstraintFolder {
            main: main_pair,
            aux: aux_pair,
            randomness: &randomness[..num_rand],
            public_values: &public_values_ef,
            periodic_values: &periodic_values,
            selectors,
            alpha,
            accumulator: EF::ZERO,
            _phantom: PhantomData,
        };

        air.eval(&mut folder);

        // Accumulate: acc = acc * β + folded_j
        accumulated = accumulated * beta + folder.accumulator;
    }

    // 11. Check quotient identity: accumulated == Q(ζ) * Z_{H_max}(ζ)
    let vanishing = zeta.exp_u64(max_trace_height as u64) - EF::ONE;
    if accumulated != quotient_zeta * vanishing {
        return Err(VerifierError::ConstraintMismatch);
    }

    // 12. Ensure transcript is fully consumed
    if !channel.is_empty() {
        return Err(VerifierError::TranscriptNotConsumed);
    }

    Ok(())
}
