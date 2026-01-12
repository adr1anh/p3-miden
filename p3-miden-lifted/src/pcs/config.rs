//! PCS configuration.

use crate::deep::DeepParams;
use crate::fri::FriParams;

/// Complete PCS configuration combining DEEP and FRI parameters.
///
/// Groups all parameters needed for `open` and `verify` into a single struct,
/// reducing the number of function arguments and ensuring consistent configuration.
#[derive(Clone, Copy, Debug)]
pub struct PcsConfig {
    /// DEEP quotient parameters (alignment, grinding).
    pub deep: DeepParams,

    /// FRI protocol parameters (blowup, folding, final degree, grinding).
    pub fri: FriParams,

    /// Number of query repetitions for soundness amplification.
    ///
    /// Each query provides ~`log_blowup` bits of security.
    pub num_queries: usize,

    /// Number of bits for proof-of-work grinding before query sampling.
    ///
    /// Set to 0 to disable grinding. Higher values increase prover work but improve
    /// soundness by preventing grinding attacks on query indices.
    pub query_proof_of_work_bits: usize,
}
