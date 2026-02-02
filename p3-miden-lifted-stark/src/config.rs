//! Shared configuration + public parameter snapshot.
//!
//! Notes:
//! - `alignment` is carried explicitly for now, but LMCS already exposes
//!   `alignment()`; consider removing this field once padding is centralized.
//! - This config intentionally avoids the heavy generic stack used by
//!   `p3-miden-uni-stark` and focuses on minimal types for the lifted protocol.

use core::marker::PhantomData;

use p3_field::PrimeField64;
use p3_miden_lifted_fri::PcsParams;
use p3_miden_lifted_fri::deep::DeepParams;
use p3_miden_lifted_fri::fri::{FriFold, FriParams};
use p3_miden_lmcs::Lmcs;
use p3_miden_transcript::{ProverChannel, TranscriptError, VerifierChannel};

#[derive(Clone)]
pub struct LiftedStarkConfig<F, L, Dft> {
    pub params: PcsParams,
    pub lmcs: L,
    pub dft: Dft,
    pub alignment: usize,
    pub _phantom: PhantomData<F>,
}

/// Snapshot of public parameters written into the transcript.
///
/// This mirrors the lifted-FRI PCS params plus the LMCS alignment hint.
#[derive(Clone, Debug)]
pub struct ParamsSnapshot {
    pub log_blowup: usize,
    pub fold_log_arity: usize,
    pub log_final_degree: usize,
    pub fri_pow_bits: usize,
    pub deep_pow_bits: usize,
    pub num_queries: usize,
    pub query_pow_bits: usize,
    pub alignment: usize,
}

impl ParamsSnapshot {
    pub fn from_config<F, L, Dft>(config: &LiftedStarkConfig<F, L, Dft>) -> Self {
        // Note: alignment is currently duplicated here and in Lmcs::alignment().
        // This is deliberate for now; once we trust LMCS alignment everywhere,
        // we can drop this field from the transcript.
        Self {
            log_blowup: config.params.fri.log_blowup,
            fold_log_arity: config.params.fri.fold.log_arity(),
            log_final_degree: config.params.fri.log_final_degree,
            fri_pow_bits: config.params.fri.proof_of_work_bits,
            deep_pow_bits: config.params.deep.proof_of_work_bits,
            num_queries: config.params.num_queries,
            query_pow_bits: config.params.query_proof_of_work_bits,
            alignment: config.alignment,
        }
    }

    pub fn write_to_channel<F, Ch>(&self, channel: &mut Ch) -> Result<(), TranscriptError>
    where
        F: PrimeField64,
        Ch: ProverChannel<F = F>,
    {
        let mut send_len = |value: usize| -> Result<(), TranscriptError> {
            let value = u64::try_from(value).map_err(|_| TranscriptError::InvalidEncoding)?;
            channel.send_u64(value);
            Ok(())
        };

        send_len(self.log_blowup)?;
        send_len(self.fold_log_arity)?;
        send_len(self.log_final_degree)?;
        send_len(self.fri_pow_bits)?;
        send_len(self.deep_pow_bits)?;
        send_len(self.num_queries)?;
        send_len(self.query_pow_bits)?;
        send_len(self.alignment)?;
        Ok(())
    }

    pub fn read_from_channel<F, Ch>(channel: &mut Ch) -> Result<Self, TranscriptError>
    where
        F: PrimeField64,
        Ch: VerifierChannel<F = F>,
    {
        let mut read_len = || -> Result<usize, TranscriptError> {
            let value = channel.receive_u64()?;
            let value = usize::try_from(value).map_err(|_| TranscriptError::InvalidEncoding)?;
            Ok(value)
        };

        Ok(Self {
            log_blowup: read_len()?,
            fold_log_arity: read_len()?,
            log_final_degree: read_len()?,
            fri_pow_bits: read_len()?,
            deep_pow_bits: read_len()?,
            num_queries: read_len()?,
            query_pow_bits: read_len()?,
            alignment: read_len()?,
        })
    }

    pub fn matches_config<F, L, Dft>(&self, config: &LiftedStarkConfig<F, L, Dft>) -> bool {
        self.log_blowup == config.params.fri.log_blowup
            && self.fold_log_arity == config.params.fri.fold.log_arity()
            && self.log_final_degree == config.params.fri.log_final_degree
            && self.fri_pow_bits == config.params.fri.proof_of_work_bits
            && self.deep_pow_bits == config.params.deep.proof_of_work_bits
            && self.num_queries == config.params.num_queries
            && self.query_pow_bits == config.params.query_proof_of_work_bits
            && self.alignment == config.alignment
    }

    pub fn to_pcs_params(&self) -> Option<PcsParams> {
        let fold = FriFold::new(self.fold_log_arity)?;
        Some(PcsParams {
            fri: FriParams {
                log_blowup: self.log_blowup,
                fold,
                log_final_degree: self.log_final_degree,
                proof_of_work_bits: self.fri_pow_bits,
            },
            deep: DeepParams {
                proof_of_work_bits: self.deep_pow_bits,
            },
            num_queries: self.num_queries,
            query_proof_of_work_bits: self.query_pow_bits,
        })
    }
}
