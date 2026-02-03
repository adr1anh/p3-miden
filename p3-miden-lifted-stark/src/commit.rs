//! Trace commitment helpers (LDE + bit-reverse + LMCS).
//!
//! Provides helpers for committing traces with lifting support. Traces of different
//! heights are lifted to the max LDE domain using nested cosets `(gK)^r`.

use alloc::vec::Vec;

use p3_dft::TwoAdicSubgroupDft;
use p3_field::TwoAdicField;
use p3_matrix::Matrix;
use p3_matrix::bitrev::BitReversibleMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_lmcs::Lmcs;
use p3_util::log2_strict_usize;

use crate::StarkConfig;

/// Commit multiple trace matrices with lifting: LDE → bit-reverse → LMCS tree.
///
/// Traces must be sorted by height in ascending order. Each trace is lifted to
/// the max LDE domain using the appropriate nested coset shift.
///
/// The committed LDE matrices are accessible via `tree.leaves()`.
///
/// # Arguments
/// - `config`: STARK configuration containing PCS params, LMCS, and DFT
/// - `traces`: Trace matrices sorted by height (ascending)
///
/// # Panics
/// - If `traces` is empty
/// - If trace heights are not powers of two
/// - If traces are not sorted by height in ascending order
pub fn commit_traces<F, L, Dft>(
    config: &StarkConfig<L, Dft>,
    traces: Vec<RowMajorMatrix<F>>,
) -> L::Tree<RowMajorMatrix<F>>
where
    F: TwoAdicField,
    L: Lmcs<F = F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    assert!(!traces.is_empty(), "at least one trace required");

    // Validate traces are sorted by height
    assert!(
        traces.windows(2).all(|w| w[0].height() <= w[1].height()),
        "traces must be sorted by height in ascending order"
    );

    let log_blowup = config.pcs.fri.log_blowup;

    // Find max trace height and compute max LDE height
    let max_trace_height = traces.last().unwrap().height();
    let log_max_trace_height = log2_strict_usize(max_trace_height);
    let log_max_lde_height = log_max_trace_height + log_blowup;

    let ldes: Vec<_> = traces
        .into_iter()
        .enumerate()
        .map(|(idx, trace)| {
            let trace_height = trace.height();

            // Validate height is power of two
            assert!(
                trace_height.is_power_of_two(),
                "trace height must be power of two (index {idx})"
            );

            let log_trace_height = log2_strict_usize(trace_height);
            let log_lde_height = log_trace_height + log_blowup;

            // Compute lift ratio: how many times smaller this trace is vs max
            // r = max_height / trace_height = 2^(log_max - log_trace)
            let log_lift_ratio = log_max_lde_height - log_lde_height;

            // Coset shift for this trace: g^r where r = 2^log_lift_ratio
            // This places the LDE on the nested coset (gK)^r
            let coset_shift = F::GENERATOR.exp_power_of_2(log_lift_ratio);

            // Compute coset LDE and bit-reverse rows
            config
                .dft
                .coset_lde_batch(trace.clone(), log_blowup, coset_shift)
                .bit_reverse_rows()
                .to_row_major_matrix()
        })
        .collect();

    // Build aligned LMCS tree
    config.lmcs.build_aligned_tree(ldes)
}
