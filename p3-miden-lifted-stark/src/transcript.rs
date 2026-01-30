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
//!    - air_widths[air] (trace, aux in permutation order)
//!    - quotient_widths[0]
//! 3) Periodic tables (AIR order, not permutation order):
//!    - per AIR: num_cols, then per column: len, then len field elements.
//! 4) Main commitment (LMCS root).
//! 5) Randomness per AIR (AIR order): num_randomness[i] extension elements.
//! 6) Aux commitment (LMCS root).
//! 7) Alphas per AIR (AIR order): one extension element each.
//! 8) Beta (extension element): Horner combine across AIRs in permutation order.
//! 9) Quotient commitment (LMCS root).
//! 10) Zeta (extension element): OOD point; rejection-sample until `zeta^N != 1`
//!    and zeta is not in the max LDE coset gK. The first valid zeta is used
//!    (loop expected to run once with overwhelming probability), and zeta_next
//!    is derived as zeta * h_max.
//! 11) PCS transcript (lifted FRI: DEEP + FRI + query hints).
//!
//! If this order changes, update prover + verifier in lockstep.

use alloc::vec::Vec;

use p3_field::PrimeField64;
use p3_miden_transcript::{ProverChannel, VerifierChannel};

pub fn write_periodic_tables<F, Ch>(channel: &mut Ch, tables: &[Vec<Vec<F>>]) -> Option<()>
where
    F: PrimeField64,
    Ch: ProverChannel<F = F>,
{
    for air_table in tables {
        let num_cols = u64::try_from(air_table.len()).ok()?;
        channel.send_u64(num_cols)?;
        for column in air_table {
            let len = u64::try_from(column.len()).ok()?;
            channel.send_u64(len)?;
            channel.send_field_slice(column);
        }
    }
    Some(())
}

pub fn read_periodic_tables<F, Ch>(channel: &mut Ch, num_airs: usize) -> Option<Vec<Vec<Vec<F>>>>
where
    F: PrimeField64,
    Ch: VerifierChannel<F = F>,
{
    let mut tables = Vec::with_capacity(num_airs);
    for _ in 0..num_airs {
        let num_cols = usize::try_from(channel.receive_u64()?).ok()?;
        let mut cols = Vec::with_capacity(num_cols);
        for _ in 0..num_cols {
            let len = usize::try_from(channel.receive_u64()?).ok()?;
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
