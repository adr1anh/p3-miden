//! Miscellaneous helpers shared by prover/verifier.

use alloc::vec::Vec;

use p3_challenger::CanSample;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::bitrev::BitReversibleMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::{log2_strict_usize, reverse_slice_index_bits};

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
pub fn sample_ood_zeta<F, EF, Ch>(
    channel: &mut Ch,
    log_max_degree: usize,
    log_max_height: usize,
) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
    Ch: CanSample<F>,
{
    let n_max = 1usize << log_max_degree;
    let lde_size = 1usize << log_max_height;
    let shift_inv = F::GENERATOR.inverse();
    let shift_inv_ef = EF::from(shift_inv);
    loop {
        let zeta: EF = sample_ext::<F, EF, _>(channel);
        if zeta.exp_u64(n_max as u64) == EF::ONE {
            continue;
        }

        let zeta_over_shift = zeta * shift_inv_ef;
        if zeta_over_shift.exp_u64(lde_size as u64) == EF::ONE {
            continue;
        }

        return zeta;
    }
}

pub fn shift_for_ratio<F: TwoAdicField>(r: usize) -> F {
    F::GENERATOR.exp_u64(r as u64)
}

pub fn lde_matrix<F, Dft>(
    dft: &Dft,
    evals: &RowMajorMatrix<F>,
    log_blowup: usize,
    shift: F,
    bit_reverse: bool,
) -> RowMajorMatrix<F>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    let lde = dft
        .coset_lde_batch(evals.clone(), log_blowup, shift)
        .to_row_major_matrix();

    if bit_reverse {
        lde.bit_reverse_rows().to_row_major_matrix()
    } else {
        lde
    }
}

pub fn pad_matrix<F: Field>(matrix: &RowMajorMatrix<F>, alignment: usize) -> RowMajorMatrix<F> {
    if alignment <= 1 {
        return matrix.clone();
    }

    let width = matrix.width();
    let padded_width = width.next_multiple_of(alignment);
    if padded_width == width {
        return matrix.clone();
    }

    let height = matrix.height();
    let mut values = Vec::with_capacity(height * padded_width);
    for r in 0..height {
        let row = matrix.row(r).expect("row in range");
        values.extend(row);
        values.resize(values.len() + (padded_width - width), F::ZERO);
    }

    RowMajorMatrix::new(values, padded_width)
}

pub fn align_width(width: usize, alignment: usize) -> usize {
    if alignment <= 1 {
        width
    } else {
        width.next_multiple_of(alignment)
    }
}

pub fn upsample_bitrev<EF: Copy>(values: &[EF], r: usize) -> Vec<EF> {
    let mut out = Vec::with_capacity(values.len() * r);
    for &v in values {
        for _ in 0..r {
            out.push(v);
        }
    }
    out
}

pub fn vanishing_inv_bitrev<F, EF>(log_n: usize, log_lde: usize) -> Vec<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    let size = 1usize << log_lde;
    let generator = F::two_adic_generator(log_lde);
    let mut x = F::GENERATOR;

    let mut invs = Vec::with_capacity(size);
    for _ in 0..size {
        let x_n = EF::from(x).exp_u64(1u64 << log_n);
        let inv = (x_n - EF::ONE).inverse();
        invs.push(inv);
        x *= generator;
    }

    reverse_slice_index_bits(&mut invs);
    invs
}

pub fn row_as_ext<F, EF>(row: &[F]) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    row.iter().copied().map(EF::from).collect()
}

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

pub fn trim_row<F, I>(row: I, width: usize) -> Vec<F>
where
    F: Field,
    I: IntoIterator<Item = F>,
{
    row.into_iter().take(width).collect()
}

pub fn row_pair_matrix<EF: Field>(row0: &[EF], row1: &[EF]) -> RowMajorMatrix<EF> {
    let mut combined = Vec::with_capacity(row0.len() + row1.len());
    combined.extend_from_slice(row0);
    combined.extend_from_slice(row1);
    RowMajorMatrix::new(combined, row0.len())
}

pub fn reverse_slice_index_bits_in_place<T>(values: &mut [T])
where
    T: Copy + Send + Sync,
{
    reverse_slice_index_bits(values);
}
