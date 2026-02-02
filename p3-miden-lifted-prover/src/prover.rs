#![allow(dead_code, unused_imports)]
//! Exploratory lifted STARK prover (LMCS-based).
//!
//! Suggested long-term placement (mirrors original prototype comments):
//! - `config.rs`: public params (protocol-level) + LMCS/Dft config.
//! - `layout.rs`: instance layout (per-AIR degrees/widths/permutation).
//! - `prover.rs`: end-to-end proving flow (this file).
//! - `verifier.rs`: end-to-end verification flow.
//! - `folder.rs`: MidenAir builders (prover/verifier).
//! - `quotient.rs`: lifting + quotient combine/verify helpers.
//! - `periodic.rs`: periodic table encoding + eval helpers.

// -----------------------------------------------------------------------------
// Lifted STARK prover (exploratory, not wired into the workspace yet)
// -----------------------------------------------------------------------------
//
// Domain notes (nested cosets + lifting)
// --------------------------------------
// Let N = max trace height (power of two). Let H be the size-N subgroup.
// Let b = blowup factor, and K be the size N*b subgroup (so H = K^b).
// The largest trace is interpreted as evaluations over H, and we LDE to gK.
//
// For a trace that is r times smaller (n = N / r), its natural subgroup is H^r,
// and its LDE domain is (gK)^r (a nested coset). This is implemented by:
//   shift_r = g^r, domain size = n*b, generator = two_adic_generator(log(n*b)).
//
// Selectors are computed over the nested domain (gK)^r, with the usual formulas.
// For OOD checks, evaluate at zeta^r (periodicity over H^r). We rejection-sample
// zeta until zeta^N != 1 and zeta ∉ gK (avoid dividing by zero); this loop should
// run once with overwhelming probability. zeta_next is derived as zeta * h_max where
// h_max = generator(H), and for a size-N/r trace the "next" point is (zeta_next)^r.
//
// Quotient combination (lifting):
// - For each AIR, compute folded constraints C_i(x) on (gK)^r (numerator only).
// - Convert C_i to bit-reversed order, then upsample (repeat each value r times)
//   to lift into the max domain gK.
// - Combine across AIRs with a challenge beta (Horner in permutation order).
// - Divide once by X^N - 1 on the max domain to obtain the final quotient.
// -----------------------------------------------------------------------------

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::{CanObserve, CanSample, CanSampleBits, GrindingChallenger};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{
    BasedVectorSpace, ExtensionField, PrimeCharacteristicRing, PrimeField64, TwoAdicField,
};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_air::MidenAir;
use p3_miden_lifted_fri::prover::open_with_channel;
use p3_miden_lmcs::{Lmcs, LmcsTree};
use p3_miden_transcript::{InitTranscript, ProverChannel, ProverTranscript};
use p3_util::log2_strict_usize;

use p3_miden_lifted_stark::{
    ConstraintFolder, LayoutSnapshot, LiftedStarkConfig, ParamsSnapshot, Proof, TraceLayout,
    build_periodic_ldes, lde_matrix, observe_init_domain_sep, observe_init_public_values,
    pad_matrix, row_as_ext, row_pair_matrix, row_to_ext, sample_ext, sample_ood_zeta, selectors_at,
    shift_for_ratio, upsample_bitrev, vanishing_inv_bitrev, write_periodic_tables,
};

// -----------------------------------------------------------------------------
// Prover entrypoint
// -----------------------------------------------------------------------------
//
// Intern guide (repeat from notes.md for quick context):
// - The transcript is the protocol; verifier replays it byte-for-byte.
//   If you change ordering, you MUST change verifier parsing in lockstep.
// - All main/aux LDEs are committed on nested cosets (gK)^r, in bit-reversed
//   row order. The PCS expects that ordering when opening.
// - Zeta is rejection-sampled outside H and gK (loop ~1x); zeta_next is derived as zeta * h_max.
// - Quotient combination is "lifted": compute per-AIR numerators on (gK)^r,
//   bit-reverse, upsample into gK, Horner-combine with beta, then divide once
//   by X^N - 1 on the max domain.
// - Aux trace is required in this scaffold; preprocessed trace is ignored.
// - Periodic columns are written to the transcript and are used during
//   constraint evaluation (both prover and verifier).

pub fn prove_with_channel<F, EF, A, L, Dft, Ch>(
    config: &LiftedStarkConfig<F, L, Dft>,
    airs: &[A],
    traces: &[RowMajorMatrix<F>],
    public_values: &[Vec<F>],
    channel: &mut Ch,
) -> Option<()>
where
    F: TwoAdicField + PrimeField64,
    EF: ExtensionField<F>,
    A: MidenAir<F, EF>,
    L: Lmcs<F = F>,
    L::Commitment: Copy,
    Dft: TwoAdicSubgroupDft<F>,
    Ch: ProverChannel<F = F, Commitment = L::Commitment> + CanSample<F> + CanSampleBits<usize>,
{
    assert_eq!(airs.len(), traces.len());
    assert_eq!(airs.len(), public_values.len());
    for (idx, (air, values)) in airs.iter().zip(public_values).enumerate() {
        assert_eq!(
            traces[idx].width(),
            air.width(),
            "trace width mismatch for air index {idx}"
        );
        assert_eq!(
            values.len(),
            air.num_public_values(),
            "public values length mismatch for air index {idx}"
        );
    }

    // Build layout once. This decides:
    // - per-AIR log_degrees
    // - permutation order (sorted by trace height, stable)
    // - max degree/height + per-AIR ratios r = N / n
    // - aligned widths for transcript hints
    let layout =
        TraceLayout::new::<F, EF, A>(airs, traces, config.alignment, config.params.fri.log_blowup);

    // Transcript-backed challenger for the entire protocol.
    // === Public parameters ===
    let params_snapshot = ParamsSnapshot::from_config(config);
    params_snapshot
        .write_to_channel::<F, _>(channel)
        .expect("parameter values fit in transcript field elements");

    // === Instance layout ===
    let layout_snapshot = layout.snapshot();
    layout_snapshot
        .write_to_channel::<F, _>(channel)
        .expect("layout values fit in transcript field elements");

    // === Periodic tables (per air, unpermuted order) ===
    // IMPORTANT: periodic tables are written in AIR order (not permutation order).
    // Verifier validates lengths against AIR definitions in the same order.
    let periodic_tables: Vec<Vec<Vec<F>>> = airs.iter().map(|air| air.periodic_table()).collect();
    write_periodic_tables::<F, _>(channel, &periodic_tables)
        .expect("periodic table sizes fit in transcript field elements");

    // === Commit main traces (LDE on (gK)^r, bit-reversed) ===
    // We commit in permutation order so verifier can match openings to AIRs.
    // Each LDE is computed on its nested coset domain (gK)^r and then bit-reversed.
    let mut main_ldes = Vec::with_capacity(layout.num_airs);
    for &idx in &layout.permutation {
        let trace = &traces[idx];
        let r = layout.ratios[idx];
        let shift = shift_for_ratio::<F>(r);
        let lde = lde_matrix(
            &config.dft,
            trace,
            config.params.fri.log_blowup,
            shift,
            true,
        );
        let lde = pad_matrix(&lde, config.alignment);
        main_ldes.push(lde);
    }
    let main_tree = config.lmcs.build_tree(main_ldes);
    channel.send_commitment(main_tree.root());

    // === Aux randomness + aux traces ===
    // Sample per-AIR randomness in AIR order (not permutation order).
    // This matches how MidenAir exposes num_randomness and build_aux_trace.
    let mut randomness_per_air: Vec<Vec<EF>> = Vec::with_capacity(layout.num_airs);
    for (_air, &num_r) in airs.iter().zip(&layout.num_randomness) {
        let randomness: Vec<EF> = (0..num_r)
            .map(|_| sample_ext::<F, EF, _>(channel))
            .collect();
        randomness_per_air.push(randomness);
    }

    let mut aux_traces = Vec::with_capacity(layout.num_airs);
    for (idx, air) in airs.iter().enumerate() {
        let trace = &traces[idx];
        let randomness = &randomness_per_air[idx];
        let aux_trace = air
            .build_aux_trace(trace, randomness)
            .expect("aux trace required in this prototype");
        let expected_aux_width = air.aux_width() * EF::DIMENSION;
        assert_eq!(
            aux_trace.height(),
            trace.height(),
            "aux trace height mismatch for air index {idx}"
        );
        assert_eq!(
            aux_trace.width(),
            expected_aux_width,
            "aux trace width mismatch for air index {idx}"
        );
        aux_traces.push(aux_trace);
    }

    let mut aux_ldes = Vec::with_capacity(layout.num_airs);
    for &idx in &layout.permutation {
        let aux_trace = &aux_traces[idx];
        let r = layout.ratios[idx];
        let shift = shift_for_ratio::<F>(r);
        let lde = lde_matrix(
            &config.dft,
            aux_trace,
            config.params.fri.log_blowup,
            shift,
            true,
        );
        let lde = pad_matrix(&lde, config.alignment);
        aux_ldes.push(lde);
    }
    let aux_tree = config.lmcs.build_tree(aux_ldes);
    channel.send_commitment(aux_tree.root());

    // === Alpha per air ===
    // One alpha per AIR; used by the constraint folder to fold constraints.
    // Sampled after aux commitment so the fold is bound to aux trace contents.
    let mut alphas = Vec::with_capacity(layout.num_airs);
    for _ in 0..layout.num_airs {
        alphas.push(sample_ext::<F, EF, _>(channel));
    }

    // === Combine AIRs with beta ===
    // Beta drives the Horner combine across AIRs (in permutation order).
    let beta: EF = sample_ext::<F, EF, _>(channel);

    // === Constraint numerators per air (natural order), then bit-reverse + lift ===
    // We compute numerators in natural row order on (gK)^r, then convert to
    // bit-reversed order because the max domain commitment is bit-reversed.
    // Lifting repeats each value r times (upsample) to land on gK.
    let periodic_ldes = build_periodic_ldes::<F, EF, Dft>(
        &config.dft,
        &periodic_tables,
        traces,
        config.params.fri.log_blowup,
        &layout.ratios,
    );
    let max_lde_size = 1usize << layout.log_max_height;
    let mut combined = vec![EF::ZERO; max_lde_size];

    for &idx in &layout.permutation {
        let air = &airs[idx];
        let trace = &traces[idx];
        let randomness = &randomness_per_air[idx];
        let alpha = alphas[idx];
        let aux_trace = &aux_traces[idx];

        let r = layout.ratios[idx];
        let shift = shift_for_ratio::<F>(r);

        let trace_lde_nat = lde_matrix(
            &config.dft,
            trace,
            config.params.fri.log_blowup,
            shift,
            false,
        );
        let aux_lde_nat = lde_matrix(
            &config.dft,
            aux_trace,
            config.params.fri.log_blowup,
            shift,
            false,
        );

        let mut numerators = compute_constraint_numerators::<F, EF, A>(
            air,
            &trace_lde_nat,
            &aux_lde_nat,
            trace.height(),
            alpha,
            randomness,
            &public_values[idx],
            shift,
            &periodic_ldes[idx],
        );

        // Convert to bit-reversed order for lifting/upsampling.
        p3_miden_lifted_stark::reverse_slice_index_bits_in_place(&mut numerators);

        // Lift to max domain by repeating each value r times.
        let lifted = upsample_bitrev(&numerators, r);

        // Combine across AIRs (Horner in permutation order)
        combined.iter_mut().zip(lifted).for_each(|(acc, val)| {
            *acc = *acc * beta + val;
        });
    }

    // === Divide once by X^N - 1 on max domain (bit-reversed order) ===
    // Combined numerator is over gK (bit-reversed); divide by X^N - 1 once.
    let inv_vanish = vanishing_inv_bitrev::<F, EF>(layout.log_max_degree, layout.log_max_height);
    combined
        .iter_mut()
        .zip(inv_vanish.iter())
        .for_each(|(c, inv)| *c *= *inv);

    // === Commit combined quotient ===
    // Single quotient for now: one column of EF split into base-field coefficients.
    let chunk = RowMajorMatrix::new(
        <EF as BasedVectorSpace<F>>::flatten_to_base(combined),
        EF::DIMENSION,
    );
    let chunk = pad_matrix(&chunk, config.alignment);
    let quotient_tree = config.lmcs.build_tree(vec![chunk]);
    channel.send_commitment(quotient_tree.root());

    // === Zeta (OOD evaluation point) ===
    // Rejection-sample until zeta is not in H or gK (avoids divide-by-zero).
    // The loop is expected to run once with overwhelming probability.
    // Only zeta is sampled; zeta_next is derived as zeta * h_max.
    // This keeps the two points on the max subgroup, and smaller traces use zeta^r.
    let zeta: EF =
        sample_ood_zeta::<F, EF, _>(channel, layout.log_max_degree, layout.log_max_height);
    let h_max = F::two_adic_generator(layout.log_max_degree);
    let zeta_next = zeta * EF::from(h_max);

    // === PCS opening via lifted FRI ===
    // open_with_channel handles DEEP + FRI for all committed trees together.
    // The evaluation points are [zeta, zeta_next] on the max domain.
    open_with_channel::<F, EF, L, RowMajorMatrix<F>, _, 2>(
        &config.params,
        &config.lmcs,
        layout.log_max_height,
        [zeta, zeta_next],
        &[&main_tree, &aux_tree, &quotient_tree],
        channel,
    );

    Some(())
}

pub fn prove<F, EF, A, L, Dft, Ch>(
    config: &LiftedStarkConfig<F, L, Dft>,
    airs: &[A],
    traces: &[RowMajorMatrix<F>],
    public_values: &[Vec<F>],
    challenger: Ch,
) -> Proof<F, L::Commitment>
where
    F: TwoAdicField + PrimeField64,
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
    let mut channel = init.into_prover();
    prove_with_channel::<F, EF, A, L, Dft, _>(config, airs, traces, public_values, &mut channel)
        .expect("proof parameters must fit in transcript field elements");
    Proof {
        transcript: channel.into_data(),
    }
}

// -----------------------------------------------------------------------------
// Constraint numerators (prover-side folded constraints)
// -----------------------------------------------------------------------------

// Compute folded constraints (numerators only) on the LDE domain (gK)^r.
#[allow(clippy::too_many_arguments)]
fn compute_constraint_numerators<F, EF, A>(
    air: &A,
    trace_lde: &RowMajorMatrix<F>,
    aux_lde: &RowMajorMatrix<F>,
    trace_degree: usize,
    alpha: EF,
    randomness: &[EF],
    public_values: &[F],
    shift: F,
    periodic_ldes: &[Vec<EF>],
) -> Vec<EF>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F>,
    A: MidenAir<F, EF>,
{
    // trace_lde / aux_lde are in natural row order on (gK)^r.
    // quotient_size = n * blowup = size of (gK)^r.
    let quotient_size = trace_lde.height();
    let next_step = quotient_size / trace_degree;

    // Advance x over the nested coset (gK)^r. "shift" is g^r.
    let log_quotient = log2_strict_usize(quotient_size);
    let generator = F::two_adic_generator(log_quotient);
    let mut x = shift;

    let public_values_ef: Vec<EF> = public_values.iter().copied().map(EF::from).collect();

    let mut result = Vec::with_capacity(quotient_size);

    for i in 0..quotient_size {
        let next_i = (i + next_step) % quotient_size;

        // LDE rows are in base field; convert to EF vectors for the folder.
        let main_local_row: Vec<F> = trace_lde.row(i).unwrap().into_iter().collect();
        let main_next_row: Vec<F> = trace_lde.row(next_i).unwrap().into_iter().collect();
        let main_local = row_as_ext::<F, EF>(&main_local_row);
        let main_next = row_as_ext::<F, EF>(&main_next_row);

        let aux_local_row: Vec<F> = aux_lde.row(i).unwrap().into_iter().collect();
        let aux_next_row: Vec<F> = aux_lde.row(next_i).unwrap().into_iter().collect();
        let aux_local = row_to_ext::<F, EF>(&aux_local_row);
        let aux_next = row_to_ext::<F, EF>(&aux_next_row);

        let selectors = selectors_at::<F, EF>(EF::from(x), trace_degree);

        // Periodic values are precomputed LDEs; select by row index.
        let periodic_values: Vec<EF> = periodic_ldes
            .iter()
            .map(|col| {
                if col.is_empty() {
                    EF::ZERO
                } else {
                    col[i % col.len()]
                }
            })
            .collect();

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
