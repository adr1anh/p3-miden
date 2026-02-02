#![allow(dead_code, unused_imports)]
//! Exploratory lifted STARK verifier (LMCS-based).
//!
//! Suggested long-term placement (mirrors original prototype comments):
//! - `config.rs`: public params (protocol-level) + LMCS/Dft config.
//! - `layout.rs`: instance layout (per-AIR degrees/widths/permutation).
//! - `prover.rs`: end-to-end proving flow.
//! - `verifier.rs`: end-to-end verification flow (this file).
//! - `folder.rs`: MidenAir builders (prover/verifier).
//! - `quotient.rs`: lifting + quotient combine/verify helpers.
//! - `periodic.rs`: periodic table encoding + eval helpers.

// -----------------------------------------------------------------------------
// Lifted STARK verifier (exploratory, not wired into the workspace yet)
// -----------------------------------------------------------------------------

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::{CanObserve, CanSample, CanSampleBits, GrindingChallenger};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{
    BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing, PrimeField64, TwoAdicField,
};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_air::MidenAir;
use p3_miden_lifted_fri::verifier::verify_with_channel as verify_pcs_with_channel;
use p3_miden_lifted_fri::{PcsError, PcsTranscript};
use p3_miden_lmcs::Lmcs;
use p3_miden_transcript::{
    InitTranscript, TranscriptData, TranscriptError, VerifierChannel, VerifierTranscript,
};
use p3_util::log2_strict_usize;

use p3_miden_lifted_stark::{
    ConstraintFolder, LayoutSnapshot, LiftedStarkConfig, ParamsSnapshot, Proof, align_width,
    eval_periodic_values, observe_init_domain_sep, observe_init_public_values,
    read_periodic_tables, row_pair_matrix, sample_ext, sample_ood_zeta, selectors_at, trim_row,
};

// -----------------------------------------------------------------------------
// Error + replayable transcript view
// -----------------------------------------------------------------------------

#[derive(Debug)]
pub enum VerifierError {
    Pcs(PcsError),
    Transcript(TranscriptError),
    InvalidTranscript,
    InvalidAuxShape,
    ConstraintMismatch,
    TranscriptNotConsumed,
    ParamsMismatch,
    PublicValuesMismatch,
}

impl From<PcsError> for VerifierError {
    fn from(err: PcsError) -> Self {
        Self::Pcs(err)
    }
}

impl From<TranscriptError> for VerifierError {
    fn from(err: TranscriptError) -> Self {
        Self::Transcript(err)
    }
}

/// Replayable transcript view for debugging or audit tooling.
pub struct StarkTranscript<EF, L>
where
    L: Lmcs,
    L::F: Field,
    L::Commitment: Copy,
    EF: ExtensionField<L::F>,
{
    pub params: ParamsSnapshot,
    pub layout: LayoutSnapshot,
    pub periodic_tables: Vec<Vec<Vec<L::F>>>,
    pub commitments: Commitments<L::Commitment>,
    pub randomness: Vec<Vec<EF>>,
    pub alphas: Vec<EF>,
    pub beta: EF,
    pub zeta: EF,
    pub pcs: PcsTranscript<EF, L>,
}

#[derive(Clone, Debug)]
pub struct Commitments<C> {
    pub main: C,
    pub aux: C,
    pub quotient: C,
}

impl<EF, L> StarkTranscript<EF, L>
where
    L: Lmcs,
    L::F: PrimeField64 + TwoAdicField,
    L::Commitment: Copy,
    EF: ExtensionField<L::F>,
{
    /// Parse a full transcript by replaying a verifier channel.
    pub fn from_verifier_channel<Ch>(lmcs: &L, channel: &mut Ch) -> Result<Self, VerifierError>
    where
        Ch: VerifierChannel<F = L::F, Commitment = L::Commitment>
            + CanSample<L::F>
            + CanSampleBits<usize>,
    {
        let params = ParamsSnapshot::read_from_channel::<L::F, _>(channel)?;
        let layout = LayoutSnapshot::read_from_channel::<L::F, _>(channel)?;
        let periodic_tables = read_periodic_tables::<L::F, _>(channel, layout.num_airs)?;

        let main = *channel.receive_commitment()?;

        let mut randomness = Vec::with_capacity(layout.num_airs);
        for &num_r in &layout.num_randomness {
            let mut rs = Vec::with_capacity(num_r);
            for _ in 0..num_r {
                rs.push(sample_ext::<L::F, EF, _>(channel));
            }
            randomness.push(rs);
        }

        let aux = *channel.receive_commitment()?;

        let mut alphas = Vec::with_capacity(layout.num_airs);
        for _ in 0..layout.num_airs {
            alphas.push(sample_ext::<L::F, EF, _>(channel));
        }

        let beta: EF = sample_ext::<L::F, EF, _>(channel);
        let quotient = *channel.receive_commitment()?;

        let zeta: EF =
            sample_ood_zeta::<L::F, EF, _>(channel, layout.log_max_degree, layout.log_max_height);
        let h_max = L::F::two_adic_generator(layout.log_max_degree);
        let zeta_next = zeta * EF::from(h_max);

        let commitments = Commitments {
            main,
            aux,
            quotient,
        };

        let (trace_widths, aux_widths): (Vec<usize>, Vec<usize>) = layout
            .air_widths
            .iter()
            .map(|widths| (widths.trace, widths.aux))
            .unzip();
        let commitments_vec = vec![
            (commitments.main, trace_widths),
            (commitments.aux, aux_widths),
            (commitments.quotient, layout.quotient_widths.clone()),
        ];

        let pcs_params = params
            .to_pcs_params()
            .ok_or(VerifierError::InvalidTranscript)?;
        let pcs = PcsTranscript::from_verifier_channel::<Ch, 2>(
            &pcs_params,
            lmcs,
            &commitments_vec,
            layout.log_max_height,
            [zeta, zeta_next],
            channel,
        )?;

        Ok(Self {
            params,
            layout,
            periodic_tables,
            commitments,
            randomness,
            alphas,
            beta,
            zeta,
            pcs,
        })
    }
}

// -----------------------------------------------------------------------------
// Verifier entrypoint
// -----------------------------------------------------------------------------
//
// Intern guide (repeat from notes.md for quick context):
// - The verifier must replay the transcript in the exact order written by the
//   prover. Any order changes require synchronized updates.
// - Commitments are observed in this order: main, aux, quotient, interleaved with sampling.
// - Randomness is sampled per AIR (AIR order) after main commitment.
// - Alphas are sampled after aux commitment, then beta, then quotient commitment, then zeta.
// - Zeta is rejection-sampled outside H and gK (loop ~1x); zeta_next is derived from zeta * h_max.
// - PCS openings return groups:
//   - groups[0]: main trace openings in permutation order
//   - groups[1]: aux trace openings in permutation order
//   - groups[2]: quotient openings (single matrix for now)
// - Each group matrix has row 0 = eval at zeta^r, row 1 = eval at zeta_next^r.
// - Folded constraints are recomputed at zeta^r and compared to quotient(zeta).

pub fn verify_with_channel<F, EF, A, L, Dft, Ch>(
    config: &LiftedStarkConfig<F, L, Dft>,
    airs: &[A],
    public_values: &[Vec<F>],
    channel: &mut Ch,
) -> Result<(), VerifierError>
where
    F: TwoAdicField + PrimeCharacteristicRing + PrimeField64,
    EF: ExtensionField<F>,
    A: MidenAir<F, EF>,
    L: Lmcs<F = F>,
    L::Commitment: Copy,
    Dft: TwoAdicSubgroupDft<F>,
    Ch: VerifierChannel<F = F, Commitment = L::Commitment> + CanSample<F> + CanSampleBits<usize>,
{
    assert_eq!(airs.len(), public_values.len());
    for (air, values) in airs.iter().zip(public_values) {
        if values.len() != air.num_public_values() {
            return Err(VerifierError::PublicValuesMismatch);
        }
    }

    // === Public parameters ===
    let params_snapshot = ParamsSnapshot::read_from_channel::<F, _>(channel)?;
    if !params_snapshot.matches_config(config) {
        return Err(VerifierError::ParamsMismatch);
    }

    // === Instance layout ===
    let layout = LayoutSnapshot::read_from_channel::<F, _>(channel)?;
    if layout.num_airs != airs.len() {
        return Err(VerifierError::InvalidTranscript);
    }
    let expected_log_max_height = layout.log_max_degree + params_snapshot.log_blowup;
    if layout.log_max_height != expected_log_max_height {
        return Err(VerifierError::InvalidTranscript);
    }
    let max_log_degree = layout.log_degrees.iter().copied().max().unwrap_or(0);
    if max_log_degree != layout.log_max_degree {
        return Err(VerifierError::InvalidTranscript);
    }
    let expected_quotient_width = align_width(EF::DIMENSION, params_snapshot.alignment);
    if layout.quotient_widths.first() != Some(&expected_quotient_width) {
        return Err(VerifierError::InvalidTranscript);
    }
    for (pos, &idx) in layout.permutation.iter().enumerate() {
        let expected_trace_width = align_width(airs[idx].width(), params_snapshot.alignment);
        if layout.air_widths.get(pos).map(|w| w.trace) != Some(expected_trace_width) {
            return Err(VerifierError::InvalidTranscript);
        }
        let expected_aux_width = align_width(
            airs[idx].aux_width() * EF::DIMENSION,
            params_snapshot.alignment,
        );
        if layout.air_widths.get(pos).map(|w| w.aux) != Some(expected_aux_width) {
            return Err(VerifierError::InvalidTranscript);
        }
    }

    // === Periodic tables (per air, unpermuted order) ===
    // Periodic tables are in AIR order and must match the AIR's column counts.
    let periodic_tables = read_periodic_tables::<F, _>(channel, layout.num_airs)?;
    for (idx, air) in airs.iter().enumerate() {
        if periodic_tables[idx].len() != air.periodic_table().len() {
            return Err(VerifierError::InvalidTranscript);
        }
    }

    // === Read commitments + sample randomness ===
    // Commitment order must mirror the prover: main, (sample randomness), aux,
    // (sample alphas, beta), quotient.
    let main_commit = *channel.receive_commitment()?;

    // === Randomness per air ===
    // Randomness is sampled per AIR (AIR order), and must match air.num_randomness().
    let mut randomness_per_air: Vec<Vec<EF>> = Vec::with_capacity(layout.num_airs);
    for (air, &num_r) in airs.iter().zip(&layout.num_randomness) {
        if air.num_randomness() != num_r {
            return Err(VerifierError::InvalidTranscript);
        }
        let randomness: Vec<EF> = (0..num_r)
            .map(|_| sample_ext::<F, EF, _>(channel))
            .collect();
        randomness_per_air.push(randomness);
    }

    let aux_commit = *channel.receive_commitment()?;

    // === Alpha per air ===
    // Alpha is sampled after aux commitment to bind folding to aux trace contents.
    let mut alphas = Vec::with_capacity(layout.num_airs);
    for _ in 0..layout.num_airs {
        alphas.push(sample_ext::<F, EF, _>(channel));
    }

    // Beta is used for the Horner combine across AIRs (permutation order).
    let beta: EF = sample_ext::<F, EF, _>(channel);

    let quotient_commit = *channel.receive_commitment()?;

    // === Zeta (OOD) ===
    // Rejection-sample zeta so it is not in H or gK (avoid divide-by-zero).
    // The loop is expected to run once with overwhelming probability.
    // Only sample zeta; derive zeta_next as zeta * h_max to stay on the max subgroup.
    let zeta: EF =
        sample_ood_zeta::<F, EF, _>(channel, layout.log_max_degree, layout.log_max_height);
    let h_max = F::two_adic_generator(layout.log_max_degree);
    let zeta_next = zeta * EF::from(h_max);

    // === PCS verification (DEEP + FRI) ===
    let (trace_widths, aux_widths): (Vec<usize>, Vec<usize>) = layout
        .air_widths
        .iter()
        .map(|widths| (widths.trace, widths.aux))
        .unzip();
    let commitments = vec![
        (main_commit, trace_widths),
        (aux_commit, aux_widths),
        (quotient_commit, layout.quotient_widths.clone()),
    ];

    let evals = verify_pcs_with_channel::<F, EF, L, _, 2>(
        &config.params,
        &config.lmcs,
        &commitments,
        layout.log_max_height,
        [zeta, zeta_next],
        channel,
    )?;

    let groups = evals.groups();
    if groups.len() != 3 {
        return Err(VerifierError::InvalidTranscript);
    }

    // Quotient value at zeta from the single quotient matrix.
    // The quotient matrix stores base-field coefficients for EF; recompose.
    let quotient_matrix = &groups[2][0];
    let chunk_coeffs = trim_row(quotient_matrix.row(0).unwrap(), EF::DIMENSION);
    let chunk_coeffs_base: Vec<F> = chunk_coeffs
        .iter()
        .map(|coeff| coeff.as_basis_coefficients_slice()[0])
        .collect();
    let quotient_zeta = EF::from_basis_coefficients_slice(&chunk_coeffs_base)
        .ok_or(VerifierError::InvalidAuxShape)?;

    // Combined constraint numerator at zeta (Horner in permutation order).
    let mut combined = EF::ZERO;

    for (pos, &idx) in layout.permutation.iter().enumerate() {
        // AIR i lives at position "pos" in the permutation ordering.
        let log_n = layout.log_degrees[idx];
        let n = 1usize << log_n;
        let log_r = layout.log_max_degree - log_n;
        let r = 1usize << log_r;

        let alpha = alphas[idx];
        let randomness = &randomness_per_air[idx];

        let main_matrix = &groups[0][pos];
        let aux_matrix = &groups[1][pos];

        // Trim padded widths down to the AIR's actual widths.
        let trace_width = airs[idx].width();
        let aux_width_base = airs[idx].aux_width() * EF::DIMENSION;

        let trace_local = trim_row(main_matrix.row(0).unwrap(), trace_width);
        let trace_next = trim_row(main_matrix.row(1).unwrap(), trace_width);

        let aux_local_base = trim_row(aux_matrix.row(0).unwrap(), aux_width_base);
        let aux_next_base = trim_row(aux_matrix.row(1).unwrap(), aux_width_base);
        let aux_local_base: Vec<F> = aux_local_base
            .iter()
            .map(|coeff| coeff.as_basis_coefficients_slice()[0])
            .collect();
        let aux_local = if aux_local_base.len().is_multiple_of(EF::DIMENSION) {
            EF::reconstitute_from_base(aux_local_base)
        } else {
            return Err(VerifierError::InvalidAuxShape);
        };
        let aux_next_base: Vec<F> = aux_next_base
            .iter()
            .map(|coeff| coeff.as_basis_coefficients_slice()[0])
            .collect();
        let aux_next = if aux_next_base.len().is_multiple_of(EF::DIMENSION) {
            EF::reconstitute_from_base(aux_next_base)
        } else {
            return Err(VerifierError::InvalidAuxShape);
        };

        let zeta_r = zeta.exp_u64(r as u64);
        let _zeta_next_r = (zeta * EF::from(h_max)).exp_u64(r as u64);
        // trace_next values correspond to evaluation at zeta_next^r via lifting

        let periodic_values = eval_periodic_values::<F, EF>(&periodic_tables[idx], n, zeta_r)
            .ok_or(VerifierError::InvalidTranscript)?;
        let public_values_ef: Vec<EF> = public_values[idx].iter().copied().map(EF::from).collect();

        let selectors = selectors_at::<F, EF>(zeta_r, n);

        let main_pair = row_pair_matrix(&trace_local, &trace_next);
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

        airs[idx].eval(&mut folder);
        let folded = folder.accumulator;

        combined = combined * beta + folded;
    }

    let n_max = 1usize << layout.log_max_degree;
    let inv_vanish_max = (zeta.exp_u64(n_max as u64) - EF::ONE).inverse();

    if combined * inv_vanish_max != quotient_zeta {
        return Err(VerifierError::ConstraintMismatch);
    }

    if !channel.is_empty() {
        return Err(VerifierError::TranscriptNotConsumed);
    }

    Ok(())
}

pub fn verify<F, EF, A, L, Dft, Ch>(
    config: &LiftedStarkConfig<F, L, Dft>,
    airs: &[A],
    proof: &Proof<F, L::Commitment>,
    public_values: &[Vec<F>],
    challenger: Ch,
) -> Result<(), VerifierError>
where
    F: TwoAdicField + PrimeCharacteristicRing + PrimeField64,
    EF: ExtensionField<F>,
    A: MidenAir<F, EF>,
    L: Lmcs<F = F>,
    L::Commitment: Copy,
    Dft: TwoAdicSubgroupDft<F>,
    Ch: GrindingChallenger<Witness = F>
        + CanObserve<F>
        + CanObserve<L::Commitment>
        + CanSample<F>
        + CanSampleBits<usize>,
{
    let mut init = InitTranscript::<F, L::Commitment, _>::new(challenger);
    observe_init_domain_sep(&mut init);
    observe_init_public_values(&mut init, public_values);
    let mut channel = init.into_verifier(&proof.transcript);
    verify_with_channel::<F, EF, A, L, Dft, _>(config, airs, public_values, &mut channel)
}
