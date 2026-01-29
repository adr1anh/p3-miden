//! Periodic column helpers.
//!
//! Periodic tables are treated as evaluations on a two-adic subgroup of size `p`.
//! Prover builds LDEs on a nested coset; verifier evaluates at
//! `(zeta^r)^(n/p)` using two-adic Lagrange interpolation.
//!
//! Intern notes (repeat from prover/verifier):
//! - Prover side: we lift periodic columns to an LDE on a coset of size p*b and
//!   index them by row (i % lde_len). This is simple but assumes periodicity is
//!   aligned to the trace subgroup structure.
//! - Verifier side: we reconstruct periodic values at the evaluation point using
//!   two-adic Lagrange interpolation over the subgroup of size p.
//! - If you change how periodic columns are encoded, update both sides together.

use alloc::vec::Vec;

use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;

use crate::utils::{lde_matrix, shift_for_ratio};

pub fn build_periodic_ldes<F, EF, Dft>(
    dft: &Dft,
    periodic_tables: &[Vec<Vec<F>>],
    traces: &[RowMajorMatrix<F>],
    log_blowup: usize,
    ratios: &[usize],
) -> Vec<Vec<Vec<EF>>>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    let mut per_air = Vec::with_capacity(periodic_tables.len());

    for (air_idx, columns) in periodic_tables.iter().enumerate() {
        let mut per_columns = Vec::with_capacity(columns.len());
        let n = traces[air_idx].height();
        let r = ratios[air_idx];

        for column in columns {
            let p = column.len();
            if p == 0 {
                per_columns.push(Vec::new());
                continue;
            }
            debug_assert!(n % p == 0);
            debug_assert!(p.is_power_of_two());

            // shift_exp = r * (n/p) gives g^(N/p) on the max domain; this keeps
            // periodicity aligned with the nested coset for this AIR.
            let shift_exp = r * (n / p);
            let shift = shift_for_ratio::<F>(shift_exp);

            let col_matrix = RowMajorMatrix::new_col(column.clone());
            let lde = lde_matrix(dft, &col_matrix, log_blowup, shift, false);

            let mut values = Vec::with_capacity(lde.height());
            for row in 0..lde.height() {
                let first = lde
                    .row(row)
                    .expect("row in range")
                    .into_iter()
                    .next()
                    .expect("column has one value");
                values.push(EF::from(first));
            }
            per_columns.push(values);
        }

        per_air.push(per_columns);
    }

    per_air
}

pub fn eval_periodic_values<F, EF>(
    tables: &[Vec<F>],
    trace_len: usize,
    zeta_r: EF,
) -> Option<Vec<EF>>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F>,
{
    let mut out = Vec::with_capacity(tables.len());
    for column in tables {
        let p = column.len();
        if p == 0 || !trace_len.is_multiple_of(p) || !p.is_power_of_two() {
            return None;
        }
        if p == 1 {
            out.push(EF::from(column[0]));
            continue;
        }
        // Evaluate the periodic column at y = (zeta^r)^(n/p), where the column
        // is defined over the subgroup of size p.
        let y = zeta_r.exp_u64((trace_len / p) as u64);
        out.push(lagrange_eval_two_adic::<F, EF>(column, y));
    }
    Some(out)
}

pub fn lagrange_eval_two_adic<F, EF>(values: &[F], x: EF) -> EF
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    let p = values.len();
    if p == 0 {
        return EF::ZERO;
    }
    if p == 1 {
        return EF::from(values[0]);
    }

    let log_p = log2_strict_usize(p);
    let generator = F::two_adic_generator(log_p);

    let mut roots = Vec::with_capacity(p);
    let mut cur = F::ONE;
    for _ in 0..p {
        roots.push(EF::from(cur));
        cur *= generator;
    }

    let mut result = EF::ZERO;
    for (i, xi) in roots.iter().enumerate() {
        let xi = *xi;
        let mut num = EF::ONE;
        let mut denom = EF::ONE;
        for (j, xj) in roots.iter().enumerate() {
            if i == j {
                continue;
            }
            let xj = *xj;
            num *= x - xj;
            denom *= xi - xj;
        }
        result += EF::from(values[i]) * num * denom.inverse();
    }

    result
}
