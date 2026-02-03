//! Lifted STARK prover (single AIR).
//!
//! This module provides `prove_single` for proving a single AIR instance using
//! the lifted STARK protocol with LMCS commitments.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::{CanSample, CanSampleBits};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    BasedVectorSpace, ExtensionField, PrimeCharacteristicRing, PrimeField64, TwoAdicField,
};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_air::MidenAir;
use p3_miden_lifted_fri::prover::open_with_channel;
use p3_miden_lmcs::{Lmcs, LmcsTree};
use p3_miden_transcript::ProverChannel;
use p3_util::log2_strict_usize;
use thiserror::Error;

use p3_miden_lifted_stark::{
    ConstraintFolder, StarkConfig, commit_traces, row_as_ext, row_pair_matrix, row_to_ext,
    sample_ext, sample_ood_zeta, selectors_at,
};

use crate::PeriodicLde;

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
    let shift = F::GENERATOR; // Single AIR at max height

    // 1. Commit main trace
    let main_tree = commit_traces(config, vec![trace.clone()]);
    channel.send_commitment(main_tree.root());

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
    let aux_tree = commit_traces(config, vec![aux_trace.clone()]);
    channel.send_commitment(aux_tree.root());

    // 4. Sample constraint folding challenge
    let alpha: EF = sample_ext::<F, EF, _>(channel);

    // 5. Build periodic LDEs
    let target_lde_coset =
        TwoAdicMultiplicativeCoset::new(shift, log_lde_height).expect("valid LDE coset");
    let periodic_lde: PeriodicLde<F> =
        PeriodicLde::new(config, air.periodic_table(), target_lde_coset)
            .ok_or(ProverError::InvalidPeriodicTable)?;

    // 6. Compute quotient numerator on natural-order LDE domain
    // We need natural-order LDEs for constraint evaluation (not bit-reversed)
    let main_lde_nat = config
        .dft
        .coset_lde_batch(trace.clone(), log_blowup, shift)
        .to_row_major_matrix();
    let aux_lde_nat = config
        .dft
        .coset_lde_batch(aux_trace.clone(), log_blowup, shift)
        .to_row_major_matrix();

    let mut quotient_numerator = compute_quotient_numerator::<F, EF, A>(
        air,
        &main_lde_nat,
        &aux_lde_nat,
        trace_height,
        alpha,
        &randomness,
        public_values,
        shift,
        &periodic_lde,
    );

    // 7. Convert to bit-reversed order and divide by vanishing polynomial
    p3_util::reverse_slice_index_bits(&mut quotient_numerator);
    let quotient =
        divide_by_vanishing::<F, EF>(&quotient_numerator, log_trace_height, log_lde_height);

    // 8. Commit quotient
    let quotient_matrix = RowMajorMatrix::new(
        <EF as BasedVectorSpace<F>>::flatten_to_base(quotient),
        EF::DIMENSION,
    );
    let quotient_tree = config.lmcs.build_aligned_tree(vec![quotient_matrix]);
    channel.send_commitment(quotient_tree.root());

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
        &[&main_tree, &aux_tree, &quotient_tree],
        channel,
    );

    Ok(())
}

/// Compute folded constraint numerators on the LDE domain (natural order).
#[allow(clippy::too_many_arguments)]
fn compute_quotient_numerator<F, EF, A>(
    air: &A,
    main_lde: &RowMajorMatrix<F>,
    aux_lde: &RowMajorMatrix<F>,
    trace_height: usize,
    alpha: EF,
    randomness: &[EF],
    public_values: &[F],
    shift: F,
    periodic_lde: &PeriodicLde<F>,
) -> Vec<EF>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F>,
    A: MidenAir<F, EF>,
{
    let lde_height = main_lde.height();
    let next_step = lde_height / trace_height;

    let log_lde_height = log2_strict_usize(lde_height);
    let generator = F::two_adic_generator(log_lde_height);
    let mut x = shift;

    let public_values_ef: Vec<EF> = public_values.iter().copied().map(EF::from).collect();
    let mut result = Vec::with_capacity(lde_height);

    for i in 0..lde_height {
        let next_i = (i + next_step) % lde_height;

        // Get main trace rows (base field → EF, one element per entry)
        let main_local_row: Vec<F> = main_lde.row(i).unwrap().into_iter().collect();
        let main_next_row: Vec<F> = main_lde.row(next_i).unwrap().into_iter().collect();
        let main_local = row_as_ext::<F, EF>(&main_local_row);
        let main_next = row_as_ext::<F, EF>(&main_next_row);

        // Get aux trace rows (base field → EF, packed representation)
        let aux_local_row: Vec<F> = aux_lde.row(i).unwrap().into_iter().collect();
        let aux_next_row: Vec<F> = aux_lde.row(next_i).unwrap().into_iter().collect();
        let aux_local = row_to_ext::<F, EF>(&aux_local_row);
        let aux_next = row_to_ext::<F, EF>(&aux_next_row);

        // Compute selectors at this point
        let selectors = selectors_at::<F, EF>(EF::from(x), trace_height);

        // Get periodic values at this row (convert to extension field)
        let periodic_values: Vec<EF> = periodic_lde.values_at_ext(i);

        // Build folder and evaluate constraints
        let main_pair = row_pair_matrix(&main_local, &main_next);
        let aux_pair = row_pair_matrix(&aux_local, &aux_next);

        let mut folder = ConstraintFolder {
            main: main_pair,
            aux: aux_pair,
            randomness,
            public_values: &public_values_ef,
            periodic_values: &periodic_values,
            is_first_row: selectors.is_first_row,
            is_last_row: selectors.is_last_row,
            is_transition: selectors.is_transition,
            alpha,
            accumulator: EF::ZERO,
            _phantom: PhantomData,
        };

        air.eval(&mut folder);
        result.push(folder.accumulator);

        x *= generator;
    }

    result
}

/// Divide quotient numerator by vanishing polynomial (in bit-reversed order).
fn divide_by_vanishing<F, EF>(
    quotient: &[EF],
    log_trace_height: usize,
    log_lde_height: usize,
) -> Vec<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    let trace_size = 1usize << log_trace_height;
    let g = F::GENERATOR;

    let omega_lde = F::two_adic_generator(log_lde_height);

    quotient
        .iter()
        .enumerate()
        .map(|(i, &num)| {
            // Compute x = g * omega_lde^{bitrev(i)}
            let bitrev_i = p3_util::reverse_bits_len(i, log_lde_height);
            let x = g * omega_lde.exp_u64(bitrev_i as u64);

            // vanishing(x) = x^trace_size - 1
            let vanishing = EF::from(x).exp_u64(trace_size as u64) - EF::ONE;

            num * vanishing.inverse()
        })
        .collect()
}
