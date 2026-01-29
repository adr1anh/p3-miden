//! Shared configuration + public parameter snapshot.
//!
//! Notes:
//! - `alignment` is carried explicitly for now, but LMCS already exposes
//!   `alignment()`; consider removing this field once padding is centralized.
//! - This config intentionally avoids the heavy generic stack used by
//!   `p3-miden-uni-stark` and focuses on minimal types for the lifted protocol.

use core::marker::PhantomData;

use p3_dft::TwoAdicSubgroupDft;
use p3_field::PrimeCharacteristicRing;
use p3_miden_lifted_fri::deep::DeepParams;
use p3_miden_lifted_fri::fri::{FriFold, FriParams};
use p3_miden_lifted_fri::PcsParams;
use p3_miden_lmcs::Lmcs;
use p3_miden_transcript::{ProverChannel, VerifierChannel};

use crate::transcript::{read_usize, write_usize};

#[derive(Clone)]
pub struct LiftedStarkConfig<F, L, Dft, Ch> {
    pub params: PcsParams,
    pub lmcs: L,
    pub dft: Dft,
    pub alignment: usize,
    pub challenger: Ch,
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
    pub fn from_config<F, L, Dft, Ch>(config: &LiftedStarkConfig<F, L, Dft, Ch>) -> Self
    where
        L: Lmcs,
        Dft: TwoAdicSubgroupDft<L::F>,
    {
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

    pub fn write_to_channel<F, Ch>(&self, channel: &mut Ch)
    where
        F: PrimeCharacteristicRing,
        Ch: ProverChannel<F = F>,
    {
        write_usize::<F, _>(channel, self.log_blowup);
        write_usize::<F, _>(channel, self.fold_log_arity);
        write_usize::<F, _>(channel, self.log_final_degree);
        write_usize::<F, _>(channel, self.fri_pow_bits);
        write_usize::<F, _>(channel, self.deep_pow_bits);
        write_usize::<F, _>(channel, self.num_queries);
        write_usize::<F, _>(channel, self.query_pow_bits);
        write_usize::<F, _>(channel, self.alignment);
    }

    pub fn read_from_channel<F, Ch>(channel: &mut Ch) -> Option<Self>
    where
        F: PrimeCharacteristicRing,
        Ch: VerifierChannel<F = F>,
    {
        Some(Self {
            log_blowup: read_usize::<F, _>(channel)?,
            fold_log_arity: read_usize::<F, _>(channel)?,
            log_final_degree: read_usize::<F, _>(channel)?,
            fri_pow_bits: read_usize::<F, _>(channel)?,
            deep_pow_bits: read_usize::<F, _>(channel)?,
            num_queries: read_usize::<F, _>(channel)?,
            query_pow_bits: read_usize::<F, _>(channel)?,
            alignment: read_usize::<F, _>(channel)?,
        })
    }

    pub fn matches_config<F, L, Dft, Ch>(&self, config: &LiftedStarkConfig<F, L, Dft, Ch>) -> bool
    where
        L: Lmcs,
        Dft: TwoAdicSubgroupDft<L::F>,
    {
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
