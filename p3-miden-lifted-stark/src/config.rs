//! Minimal STARK configuration.
//!
//! This configuration wraps the PCS parameters from lifted-FRI along with
//! the LMCS instance and DFT implementation needed for proving/verification.

use p3_miden_lifted_fri::PcsParams;

/// Minimal STARK configuration.
///
/// Encapsulates all parameters needed for the lifted STARK prover and verifier:
/// - `pcs`: PCS parameters (DEEP + FRI settings)
/// - `lmcs`: LMCS instance for Merkle commitments
/// - `dft`: DFT implementation for LDE computation
#[derive(Clone)]
pub struct StarkConfig<L, Dft> {
    /// PCS parameters (DEEP + FRI).
    pub pcs: PcsParams,
    /// LMCS instance for commitments.
    pub lmcs: L,
    /// DFT for LDE computation.
    pub dft: Dft,
}
