//! Helpers for building bit-reversed LDEs and LMCS commitments.
//!
//! These helpers keep coset handling explicit using `TwoAdicMultiplicativeCoset` and
//! the `exp_power_of_2` API to derive nested cosets `(gK)^r` without manual shifts.

use alloc::vec::Vec;

use p3_dft::TwoAdicSubgroupDft;
use p3_field::TwoAdicField;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_matrix::Matrix;
use p3_matrix::bitrev::BitReversibleMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_lmcs::Lmcs;
use p3_util::log2_strict_usize;

/// Return the max LDE coset `gK` of size `2^log_max_height`.
pub fn max_lde_coset<F: TwoAdicField>(log_max_height: usize) -> TwoAdicMultiplicativeCoset<F> {
    TwoAdicMultiplicativeCoset::new(F::GENERATOR, log_max_height)
        .expect("log_max_height within two-adicity")
}

/// Return the nested coset `(gK)^(2^log_ratio)`.
pub fn nested_coset_for_ratio<F: TwoAdicField>(
    base: TwoAdicMultiplicativeCoset<F>,
    log_ratio: usize,
) -> TwoAdicMultiplicativeCoset<F> {
    base.exp_power_of_2(log_ratio)
        .expect("log_ratio within coset size")
}

/// Compute an LDE mapping evaluations on `domain` to a target coset.
///
/// `coset_lde_batch` maps evaluations over `xH` to evaluations over `shift * x * K`.
/// To land on `target = tK`, use `shift = t / x` (implemented via `t * x^{-1}`).
/// This mirrors the logic used in the two-adic PCS commitment path.
pub fn lde_matrix_from_domain_to_coset<F, Dft>(
    dft: &Dft,
    evals: &RowMajorMatrix<F>,
    log_blowup: usize,
    domain: TwoAdicMultiplicativeCoset<F>,
    target: TwoAdicMultiplicativeCoset<F>,
    bit_reverse: bool,
) -> RowMajorMatrix<F>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    assert_eq!(domain.size(), evals.height(), "domain size mismatch");
    let log_height = log2_strict_usize(evals.height());
    let log_lde_height = log_height + log_blowup;
    assert_eq!(
        target.log_size(),
        log_lde_height,
        "target coset size mismatch for LDE"
    );

    let shift = target.shift() * domain.shift_inverse();
    let lde = dft
        .coset_lde_batch(evals.clone(), log_blowup, shift)
        .to_row_major_matrix();

    if bit_reverse {
        lde.bit_reverse_rows().to_row_major_matrix()
    } else {
        lde
    }
}

/// Compute an LDE over the provided coset (assumes the input domain shift is 1).
pub fn lde_matrix_on_coset<F, Dft>(
    dft: &Dft,
    evals: &RowMajorMatrix<F>,
    log_blowup: usize,
    coset: &TwoAdicMultiplicativeCoset<F>,
    bit_reverse: bool,
) -> RowMajorMatrix<F>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    let log_height = log2_strict_usize(evals.height());
    let domain = TwoAdicMultiplicativeCoset::new(F::ONE, log_height)
        .expect("trace height within two-adicity");
    lde_matrix_from_domain_to_coset(dft, evals, log_blowup, domain, *coset, bit_reverse)
}

/// Build bit-reversed LDEs on nested cosets and commit them via LMCS.
///
/// Matrices must be sorted by height (shortest to tallest). No explicit column
/// padding is added; LMCS alignment is a transcript formatting convention, and
/// callers that require zero padding must enforce it separately.
pub fn lde_commit_traces<F, Dft, L>(
    dft: &Dft,
    lmcs: &L,
    log_max_height: usize,
    log_blowup: usize,
    traces: &[RowMajorMatrix<F>],
) -> L::Tree<RowMajorMatrix<F>>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    L: Lmcs<F = F>,
{
    assert!(!traces.is_empty(), "at least one trace required");

    let base = max_lde_coset::<F>(log_max_height);
    let mut ldes = Vec::with_capacity(traces.len());
    let mut prev_height = 0usize;

    for (idx, trace) in traces.iter().enumerate() {
        let height = trace.height();
        assert!(
            height.is_power_of_two(),
            "non-power-of-two trace height at index {idx}"
        );
        assert!(height >= prev_height, "traces must be sorted by height");

        let log_trace_height = log2_strict_usize(height);
        let log_lde_height = log_trace_height + log_blowup;
        assert!(
            log_lde_height <= log_max_height,
            "trace LDE height exceeds max (index {idx})"
        );

        let log_ratio = log_max_height - log_lde_height;
        let coset = nested_coset_for_ratio(base, log_ratio);
        let lde = lde_matrix_on_coset(dft, trace, log_blowup, &coset, true);
        ldes.push(lde);
        prev_height = height;
    }

    lmcs.build_tree(ldes)
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::vec;

    use p3_dft::Radix2DFTSmallBatch;
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
    use p3_miden_lmcs::{Lmcs, LmcsConfig, LmcsTree};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    type TestLmcs =
        LmcsConfig<bb::P, bb::P, bb::Sponge, bb::Compress, { bb::WIDTH }, { bb::DIGEST }>;

    #[test]
    fn nested_coset_shift_matches_ratio() {
        let log_max_height = 6;
        let log_blowup = 1;
        let log_trace_height = 3;
        let log_ratio = log_max_height - (log_trace_height + log_blowup);

        let base = max_lde_coset::<bb::F>(log_max_height);
        let nested = nested_coset_for_ratio(base, log_ratio);

        assert_eq!(nested.log_size(), log_trace_height + log_blowup);
        assert_eq!(nested.shift(), bb::F::GENERATOR.exp_power_of_2(log_ratio));
    }

    #[test]
    fn lde_commit_traces_builds_expected_ldes() {
        let mut rng = SmallRng::seed_from_u64(42);
        let dft = Radix2DFTSmallBatch::<bb::F>::default();
        let (_, sponge, compress) = bb::test_components();
        let lmcs: TestLmcs = LmcsConfig::new(sponge, compress);

        let log_blowup = 1;
        let log_max_height = 4;

        let trace_small = RowMajorMatrix::rand(&mut rng, 1 << 2, 2);
        let trace_large = RowMajorMatrix::rand(&mut rng, 1 << 3, 2);
        let traces = vec![trace_small.clone(), trace_large.clone()];

        let tree: <TestLmcs as Lmcs>::Tree<RowMajorMatrix<bb::F>> =
            lde_commit_traces::<bb::F, _, _>(&dft, &lmcs, log_max_height, log_blowup, &traces);

        let leaves = tree.leaves();
        let base = max_lde_coset::<bb::F>(log_max_height);

        let small_log_ratio =
            log_max_height - (log2_strict_usize(trace_small.height()) + log_blowup);
        let small_coset = nested_coset_for_ratio(base, small_log_ratio);
        let expected_small =
            lde_matrix_on_coset(&dft, &trace_small, log_blowup, &small_coset, true);

        let large_log_ratio =
            log_max_height - (log2_strict_usize(trace_large.height()) + log_blowup);
        let large_coset = nested_coset_for_ratio(base, large_log_ratio);
        let expected_large =
            lde_matrix_on_coset(&dft, &trace_large, log_blowup, &large_coset, true);

        assert_eq!(leaves[0], expected_small);
        assert_eq!(leaves[1], expected_large);
    }
}
