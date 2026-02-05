//! Miscellaneous helpers shared by prover/verifier.

use alloc::vec::Vec;

use p3_challenger::CanSample;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;

/// Sample an extension field element from a challenger.
pub fn sample_ext<F, EF, Ch>(channel: &mut Ch) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
    Ch: CanSample<F>,
{
    EF::from_basis_coefficients_fn(|_| channel.sample())
}

/// Sample an out-of-domain (OOD) evaluation point outside H and gK.
///
/// We reject:
/// - `zeta^N == 1` (zeta ∈ H), to avoid dividing by zero when inverting `X^N - 1`.
/// - `zeta ∈ gK`, the max LDE coset, to avoid division by zero in DEEP quotients.
///
/// The rejection probability is ~`1 / N + 1 / (N * blowup)`, so this loop is
/// expected to run once with overwhelming probability.
///
/// # Arguments
/// - `channel`: The challenger to sample from
/// - `log_trace_height`: Log₂ of the trace height (size of H)
/// - `log_lde_height`: Log₂ of the LDE height (size of gK)
pub fn sample_ood_zeta<F, EF, Ch>(
    channel: &mut Ch,
    log_trace_height: usize,
    log_lde_height: usize,
) -> EF
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    Ch: CanSample<F>,
{
    let trace_size = 1usize << log_trace_height;
    let lde_size = 1usize << log_lde_height;
    let shift_inv = F::GENERATOR.inverse();
    let shift_inv_ef = EF::from(shift_inv);

    loop {
        let zeta: EF = sample_ext::<F, EF, _>(channel);

        // Check zeta^N != 1 (not in trace subgroup H)
        if zeta.exp_u64(trace_size as u64) == EF::ONE {
            continue;
        }

        // Check zeta not in gK (LDE coset)
        // zeta ∈ gK iff (zeta/g)^|K| = 1
        let zeta_over_shift = zeta * shift_inv_ef;
        if zeta_over_shift.exp_u64(lde_size as u64) == EF::ONE {
            continue;
        }

        return zeta;
    }
}

/// Convert a base field row to extension field elements (one element per entry).
pub fn row_as_ext<F, EF>(row: &[F]) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    row.iter().copied().map(EF::from).collect()
}

/// Convert a base field row to extension field elements (packed representation).
///
/// The row length must be divisible by `EF::DIMENSION`. Each chunk of
/// `EF::DIMENSION` base field elements is packed into one extension element.
pub fn row_to_ext<F, EF>(row: &[F]) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let dim = EF::DIMENSION;
    assert_eq!(row.len() % dim, 0);
    row.chunks(dim)
        .map(|chunk| EF::from_basis_coefficients_slice(chunk).unwrap())
        .collect()
}

/// Build a 2-row matrix from local and next row values.
///
/// Used for constraint evaluation where we need access to consecutive rows.
pub fn row_pair_matrix<EF: Field>(row0: &[EF], row1: &[EF]) -> RowMajorMatrix<EF> {
    let mut combined = Vec::with_capacity(row0.len() + row1.len());
    combined.extend_from_slice(row0);
    combined.extend_from_slice(row1);
    RowMajorMatrix::new(combined, row0.len())
}
