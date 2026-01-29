//! Transcript helper functions for serializing layout and periodic data.
//!
//! Transcript order (written by prover, replayed by verifier):
//! 1) ParamsSnapshot fields:
//!    - log_blowup, fold_log_arity, log_final_degree, fri_pow_bits,
//!      deep_pow_bits, num_queries, query_pow_bits, alignment.
//! 2) LayoutSnapshot fields:
//!    - num_airs
//!    - log_degrees[air], permutation[air]
//!    - log_max_degree, log_max_height
//!    - num_randomness[air]
//!    - trace_widths[air], aux_widths[air]
//!    - quotient_widths[0]
//! 3) Periodic tables (AIR order, not permutation order):
//!    - per AIR: num_cols, then per column: len, then len field elements.
//! 4) Commitments (LMCS roots): main, aux, quotient.
//! 5) Randomness per AIR (AIR order): num_randomness[i] extension elements.
//! 6) Alphas per AIR (AIR order): one extension element each.
//! 7) Beta (extension element): Horner combine across AIRs in permutation order.
//! 8) Zeta (extension element): OOD point; rejection-sample until `zeta^N != 1`
//!    and zeta is not in the max LDE coset gK. The first valid zeta is used
//!    (loop expected to run once with overwhelming probability), and zeta_next
//!    is derived as zeta * h_max.
//! 9) PCS transcript (lifted FRI: DEEP + FRI + query hints).
//!
//! If this order changes, update prover + verifier in lockstep.

use alloc::vec::Vec;

use p3_field::PrimeField64;
use p3_miden_transcript::{ProverChannel, VerifierChannel};

pub fn write_usize<F, Ch>(channel: &mut Ch, value: usize) -> Option<()>
where
    F: PrimeField64,
    Ch: ProverChannel<F = F>,
{
    let value = u32::try_from(value).ok()?;
    channel.send_u64(u64::from(value))
}

pub fn write_usize_list<F, Ch>(channel: &mut Ch, values: &[usize]) -> Option<()>
where
    F: PrimeField64,
    Ch: ProverChannel<F = F>,
{
    for &v in values {
        write_usize::<F, _>(channel, v)?;
    }
    Some(())
}

pub fn write_periodic_tables<F, Ch>(channel: &mut Ch, tables: &[Vec<Vec<F>>]) -> Option<()>
where
    F: PrimeField64,
    Ch: ProverChannel<F = F>,
{
    for air_table in tables {
        write_usize::<F, _>(channel, air_table.len())?;
        for column in air_table {
            write_usize::<F, _>(channel, column.len())?;
            channel.send_field_slice(column);
        }
    }
    Some(())
}

pub fn read_usize<F, Ch>(channel: &mut Ch) -> Option<usize>
where
    F: PrimeField64,
    Ch: VerifierChannel<F = F>,
{
    let value = channel.receive_u64()?;
    if value > u64::from(u32::MAX) {
        return None;
    }
    Some(value as usize)
}

pub fn read_usize_list<F, Ch>(channel: &mut Ch, count: usize) -> Option<Vec<usize>>
where
    F: PrimeField64,
    Ch: VerifierChannel<F = F>,
{
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(read_usize::<F, _>(channel)?);
    }
    Some(out)
}

pub fn read_periodic_tables<F, Ch>(channel: &mut Ch, num_airs: usize) -> Option<Vec<Vec<Vec<F>>>>
where
    F: PrimeField64,
    Ch: VerifierChannel<F = F>,
{
    let mut tables = Vec::with_capacity(num_airs);
    for _ in 0..num_airs {
        let num_cols = read_usize::<F, _>(channel)?;
        let mut cols = Vec::with_capacity(num_cols);
        for _ in 0..num_cols {
            let len = read_usize::<F, _>(channel)?;
            let mut col = Vec::with_capacity(len);
            for _ in 0..len {
                col.push(*channel.receive_field()?);
            }
            cols.push(col);
        }
        tables.push(cols);
    }
    Some(tables)
}

pub fn field_to_usize<F: PrimeField64>(value: F) -> usize {
    value.as_canonical_u64() as usize
}
