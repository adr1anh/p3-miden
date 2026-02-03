//! Lifted STARK verifier (single AIR).
//!
//! This module provides `verify_single` for verifying a single AIR instance using
//! the lifted STARK protocol with LMCS commitments.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::{CanSample, CanSampleBits};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, PrimeCharacteristicRing, PrimeField64, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_air::MidenAir;
use p3_miden_lifted_fri::verifier::{PcsError, verify_with_channel as verify_pcs_with_channel};
use p3_miden_lmcs::Lmcs;
use p3_miden_transcript::{TranscriptError, VerifierChannel};
use thiserror::Error;

use p3_miden_lifted_stark::{
    ConstraintFolder, StarkConfig, sample_ext, sample_ood_zeta, selectors_at,
};

use crate::PeriodicPolys;

/// Errors that can occur during verification.
#[derive(Debug, Error)]
pub enum VerifierError {
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
    #[error("invalid PCS opening groups: expected 3, got {0}")]
    InvalidOpeningGroups(usize),
}

/// Verify a single AIR.
///
/// Assumes the channel has been initialized with domain separator and public values.
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
    A: MidenAir<F, EF>,
    L: Lmcs<F = F>,
    L::Commitment: Copy,
    Dft: TwoAdicSubgroupDft<F>,
    Ch: VerifierChannel<F = F, Commitment = L::Commitment> + CanSample<F> + CanSampleBits<usize>,
{
    let trace_height = 1usize << log_trace_height;
    let log_blowup = config.pcs.fri.log_blowup;
    let log_lde_height = log_trace_height + log_blowup;
    let alignment = config.lmcs.alignment();

    // 1. Receive main trace commitment
    let main_commit = *channel.receive_commitment()?;

    // 2. Sample randomness for aux trace
    let num_randomness = air.num_randomness();
    let randomness: Vec<EF> = (0..num_randomness)
        .map(|_| sample_ext::<F, EF, _>(channel))
        .collect();

    // 3. Receive aux trace commitment
    let aux_commit = *channel.receive_commitment()?;

    // 4. Sample constraint folding challenge
    let alpha: EF = sample_ext::<F, EF, _>(channel);

    // 5. Receive quotient commitment
    let quotient_commit = *channel.receive_commitment()?;

    // 6. Sample OOD point
    let zeta: EF = sample_ood_zeta::<F, EF, _>(channel, log_trace_height, log_lde_height);
    let h = F::two_adic_generator(log_trace_height);
    let zeta_next = zeta * EF::from(h);

    // 7. Verify PCS openings
    // Widths include alignment padding (as stored in the committed trees).
    // The folder ignores padding columns since the AIR only accesses columns it knows about.
    let trace_width = align_width(air.width(), alignment);
    let aux_width = align_width(air.aux_width() * EF::DIMENSION, alignment);
    let quotient_width = align_width(EF::DIMENSION, alignment);

    let commitments = vec![
        (main_commit, vec![trace_width]),
        (aux_commit, vec![aux_width]),
        (quotient_commit, vec![quotient_width]),
    ];

    let evals = verify_pcs_with_channel::<F, EF, L, _, 2>(
        &config.pcs,
        &config.lmcs,
        &commitments,
        log_lde_height,
        [zeta, zeta_next],
        channel,
    )?;

    // 8. Extract opened values
    let groups = evals.groups();
    if groups.len() != 3 {
        return Err(VerifierError::InvalidOpeningGroups(groups.len()));
    }

    let main_matrix = &groups[0][0]; // Single main trace matrix
    let aux_matrix = &groups[1][0]; // Single aux trace matrix
    let quotient_matrix = &groups[2][0]; // Single quotient matrix

    // Main trace: collect rows directly (padding columns are ignored by the AIR)
    let main_local: Vec<EF> = main_matrix.row(0).unwrap().into_iter().collect();
    let main_next: Vec<EF> = main_matrix.row(1).unwrap().into_iter().collect();

    // Aux trace: extract base coefficients and reconstitute as EF
    // Padding columns are included but ignored by the AIR
    let aux_local = row_to_packed_ext::<F, EF, _>(aux_matrix.row(0).unwrap())?;
    let aux_next = row_to_packed_ext::<F, EF, _>(aux_matrix.row(1).unwrap())?;

    // Quotient: extract first EF::DIMENSION base coefficients and reconstitute
    let quotient_row: Vec<EF> = quotient_matrix.row(0).unwrap().into_iter().collect();
    let quotient_coeffs: Vec<F> = quotient_row
        .iter()
        .take(EF::DIMENSION)
        .map(|ef| ef.as_basis_coefficients_slice()[0])
        .collect();
    let quotient_zeta = EF::from_basis_coefficients_slice(&quotient_coeffs).ok_or(
        VerifierError::InvalidAuxShape {
            expected_divisor: EF::DIMENSION,
            actual_len: quotient_coeffs.len(),
        },
    )?;

    // 9. Evaluate periodic values at zeta
    let periodic_polys =
        PeriodicPolys::new(air.periodic_table()).ok_or(VerifierError::InvalidPeriodicTable)?;
    let periodic_values = periodic_polys.eval_at::<EF>(trace_height, zeta);

    // 10. Evaluate constraints at zeta
    let selectors = selectors_at::<F, EF>(zeta, trace_height);
    let public_values_ef: Vec<EF> = public_values.iter().copied().map(EF::from).collect();

    // Build 2-row matrices for the folder (row 0 = local, row 1 = next)
    let main_pair = RowMajorMatrix::new([main_local, main_next].concat(), trace_width);
    let aux_pair = RowMajorMatrix::new([aux_local, aux_next].concat(), aux_width / EF::DIMENSION);

    let mut folder = ConstraintFolder {
        main: main_pair,
        aux: aux_pair,
        randomness: &randomness,
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
    let folded = folder.accumulator;

    // 11. Check quotient identity: folded * inv_vanishing == quotient_zeta
    // Equivalently: folded == quotient_zeta * vanishing
    let vanishing = zeta.exp_u64(trace_height as u64) - EF::ONE;
    if folded != quotient_zeta * vanishing {
        return Err(VerifierError::ConstraintMismatch);
    }

    // 12. Ensure transcript is fully consumed
    if !channel.is_empty() {
        return Err(VerifierError::TranscriptNotConsumed);
    }

    Ok(())
}

/// Align width to the given alignment.
fn align_width(width: usize, alignment: usize) -> usize {
    width.div_ceil(alignment) * alignment
}

/// Extract base field coefficients from EF row and reconstitute as packed EF elements.
fn row_to_packed_ext<F, EF, I>(row: I) -> Result<Vec<EF>, VerifierError>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    I: IntoIterator<Item = EF>,
{
    let base_coeffs: Vec<F> = row
        .into_iter()
        .map(|ef| ef.as_basis_coefficients_slice()[0])
        .collect();
    if !base_coeffs.len().is_multiple_of(EF::DIMENSION) {
        return Err(VerifierError::InvalidAuxShape {
            expected_divisor: EF::DIMENSION,
            actual_len: base_coeffs.len(),
        });
    }
    Ok(EF::reconstitute_from_base(base_coeffs))
}
