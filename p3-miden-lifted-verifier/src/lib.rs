#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

mod constraints;
mod periodic;
mod proof;
mod verifier;

pub use p3_miden_lifted_stark::{
    AirInstance, DeepParams, FriFold, FriParams, Lmcs, LmcsConfig, PcsParams, StarkConfig,
    TranscriptData,
};
pub use p3_miden_transcript::{VerifierChannel, VerifierTranscript};
pub use proof::Proof;
pub use verifier::{VerifierError, verify_multi, verify_single};
