//! Transcript channels for Fiat-Shamir protocols with raw field/commitment storage.
//!
//! This crate provides:
//! - [`ProverTranscript`] and [`ProverChannel`] for prover-side recording.
//! - [`VerifierTranscript`] and [`VerifierChannel`] for verifier-side reading.
//! - [`InitTranscript`] for init-only observation before starting transcripts.

#![no_std]

extern crate alloc;

mod data;
mod error;
mod init;
mod prover;
mod verifier;

// Public API re-exports.
pub use data::TranscriptData;
pub use error::TranscriptError;
pub use init::InitTranscript;
pub use prover::{ProverChannel, ProverTranscript};
pub use verifier::{TranscriptError, VerifierChannel, VerifierTranscript};
