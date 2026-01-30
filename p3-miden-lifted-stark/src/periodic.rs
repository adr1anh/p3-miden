//! Periodic column helpers.
//!
//! Periodic tables are stored as polynomials (monomial coefficients) per AIR.
//! We construct them from subgroup evaluations using a naive DFT, then:
//! - Prover builds LDEs on a nested coset for constraint folding.
//! - Verifier evaluates at `(zeta^r)^(n/p)` using Horner on coefficients.
//!
//! Legacy helpers (`build_periodic_ldes`, `eval_periodic_values`) still accept
//! evaluation tables directly; they will be removed after integration.
//!
//! Intern notes (repeat from prover/verifier):
//! - Prover side: we lift periodic columns to an LDE on a coset of size p*b and
//!   index them by row (i % lde_len). This is simple but assumes periodicity is
//!   aligned to the trace subgroup structure.
//! - Verifier side: we reconstruct periodic values at the evaluation point using
//!   two-adic Lagrange interpolation over the subgroup of size p.
//! - If you change how periodic columns are encoded, update both sides together.

use alloc::vec;
use alloc::vec::Vec;

use p3_dft::{NaiveDft, TwoAdicSubgroupDft};
use p3_field::{ExtensionField, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_air::MidenAir;
use p3_util::log2_strict_usize;

use crate::utils::{lde_matrix, shift_for_ratio};

/// Periodic table for a single AIR, stored as polynomial coefficients.
#[derive(Clone, Debug)]
pub struct PeriodicTable<F> {
    polys: Vec<Vec<F>>,
}

impl<F: TwoAdicField> PeriodicTable<F> {
    /// Construct from subgroup evaluations (canonical order).
    /// Returns None if any column length is zero or not a power of two.
    pub fn from_evaluations(evals: Vec<Vec<F>>) -> Option<Self> {
        let dft = NaiveDft;
        let mut polys = Vec::with_capacity(evals.len());
        for column in evals {
            let p = column.len();
            if p == 0 || !p.is_power_of_two() {
                return None;
            }
            let coeffs = dft.idft_batch(RowMajorMatrix::new_col(column)).values;
            polys.push(coeffs);
        }
        Some(Self { polys })
    }

    /// Convert back to subgroup evaluations (canonical order).
    pub fn to_evaluations(&self) -> Vec<Vec<F>> {
        let dft = NaiveDft;
        self.polys
            .iter()
            .map(|coeffs| {
                dft.dft_batch(RowMajorMatrix::new_col(coeffs.clone()))
                    .values
            })
            .collect()
    }

    pub fn polys(&self) -> &[Vec<F>] {
        &self.polys
    }

    pub fn into_polys(self) -> Vec<Vec<F>> {
        self.polys
    }

    /// Build periodic LDEs for constraint folding (prover side).
    pub fn build_ldes<EF>(&self, trace_len: usize, ratio: usize, log_blowup: usize) -> Vec<Vec<EF>>
    where
        EF: ExtensionField<F>,
    {
        let dft = NaiveDft;
        let mut out = Vec::with_capacity(self.polys.len());
        for coeffs in &self.polys {
            let p = coeffs.len();
            debug_assert!(p != 0);
            debug_assert!(p.is_power_of_two());
            debug_assert!(trace_len.is_multiple_of(p));

            let shift_exp = ratio * (trace_len / p);
            let shift = shift_for_ratio::<F>(shift_exp);

            let lde_len = p << log_blowup;
            let mut padded = vec![F::ZERO; lde_len];
            padded[..p].copy_from_slice(coeffs);

            let lde = dft.coset_dft_batch(RowMajorMatrix::new_col(padded), shift);
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
            out.push(values);
        }
        out
    }

    /// Evaluate periodic polynomials at the point implied by trace_len.
    pub fn eval_at<EF>(&self, trace_len: usize, zeta_r: EF) -> Option<Vec<EF>>
    where
        EF: ExtensionField<F>,
    {
        let mut out = Vec::with_capacity(self.polys.len());
        for coeffs in &self.polys {
            let p = coeffs.len();
            if p == 0 || !p.is_power_of_two() || !trace_len.is_multiple_of(p) {
                return None;
            }
            let y = zeta_r.exp_u64((trace_len / p) as u64);
            out.push(eval_poly_horner::<F, EF>(coeffs, y));
        }
        Some(out)
    }
}

pub fn periodic_tables_from_airs<F, EF, A>(airs: &[A]) -> Option<Vec<PeriodicTable<F>>>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    A: MidenAir<F, EF>,
{
    let mut out = Vec::with_capacity(airs.len());
    for air in airs {
        out.push(PeriodicTable::from_evaluations(air.periodic_table())?);
    }
    Some(out)
}

fn eval_poly_horner<F, EF>(coeffs: &[F], x: EF) -> EF
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    let mut acc = EF::ZERO;
    for coeff in coeffs.iter().rev() {
        acc = acc * x + EF::from(*coeff);
    }
    acc
}

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
            debug_assert!(p != 0);
            debug_assert!(p.is_power_of_two());
            debug_assert!(n % p == 0);

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
