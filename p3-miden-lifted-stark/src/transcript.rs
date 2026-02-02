//! Transcript helper functions for serializing layout and periodic data.
//!
//! Init observe (not serialized into the proof):
//! - Protocol domain separator "p3-miden-lifted-stark-v0" (bytes as field elements).
//! - Public values in AIR order (same values passed to AIR evaluation).
//!
//! Transcript order (written by prover, replayed by verifier) after init:
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
//!     and zeta is not in the max LDE coset gK. The first valid zeta is used
//!     (loop expected to run once with overwhelming probability), and zeta_next
//!     is derived as zeta * h_max.
//! 11) PCS transcript (lifted FRI: DEEP + FRI + query hints).
//!
//! If this order changes, update prover + verifier in lockstep.

use alloc::vec::Vec;

use p3_challenger::CanObserve;
use p3_field::PrimeField64;
use p3_field::integers::QuotientMap;
use p3_miden_transcript::{InitTranscript, ProverChannel, TranscriptError, VerifierChannel};

const PROTOCOL_DOMAIN_SEPARATOR: &[u8] = b"p3-miden-lifted-stark-v0";

pub fn observe_init_domain_sep<F, C, Ch>(init: &mut InitTranscript<F, C, Ch>)
where
    F: PrimeField64,
    C: Copy,
    Ch: CanObserve<F> + CanObserve<C>,
{
    for &byte in PROTOCOL_DOMAIN_SEPARATOR {
        let field = <F as QuotientMap<u8>>::from_int(byte);
        init.observe_init_field_element(field);
    }
}

pub fn observe_init_public_values<F, C, Ch>(
    init: &mut InitTranscript<F, C, Ch>,
    public_values: &[Vec<F>],
) where
    F: PrimeField64,
    C: Copy,
    Ch: CanObserve<F> + CanObserve<C>,
{
    for values in public_values {
        init.observe_init_field_slice(values);
    }
}

pub fn write_periodic_tables<F, Ch>(
    channel: &mut Ch,
    tables: &[Vec<Vec<F>>],
) -> Result<(), TranscriptError>
where
    F: PrimeField64,
    Ch: ProverChannel<F = F>,
{
    for air_table in tables {
        let num_cols =
            u64::try_from(air_table.len()).map_err(|_| TranscriptError::InvalidEncoding)?;
        channel.send_u64(num_cols);
        for column in air_table {
            let len = u64::try_from(column.len()).map_err(|_| TranscriptError::InvalidEncoding)?;
            channel.send_u64(len);
            channel.send_field_slice(column);
        }
    }
    Ok(())
}

pub fn read_periodic_tables<F, Ch>(
    channel: &mut Ch,
    num_airs: usize,
) -> Result<Vec<Vec<Vec<F>>>, TranscriptError>
where
    F: PrimeField64,
    Ch: VerifierChannel<F = F>,
{
    let mut tables = Vec::with_capacity(num_airs);
    for _ in 0..num_airs {
        let num_cols = usize::try_from(channel.receive_u64()?)
            .map_err(|_| TranscriptError::InvalidEncoding)?;
        let mut cols = Vec::with_capacity(num_cols);
        for _ in 0..num_cols {
            let len = usize::try_from(channel.receive_u64()?)
                .map_err(|_| TranscriptError::InvalidEncoding)?;
            let mut col = Vec::with_capacity(len);
            for _ in 0..len {
                col.push(*channel.receive_field()?);
            }
            cols.push(col);
        }
        tables.push(cols);
    }
    Ok(tables)
}

pub fn field_to_usize<F: PrimeField64>(value: F) -> usize {
    value.as_canonical_u64() as usize
}
