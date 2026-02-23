//! Proof container.

use p3_miden_transcript::TranscriptData;
use serde::Serialize;

/// Proof container wrapping raw transcript data.
///
/// This type intentionally does not include the statement (AIR, trace heights, public
/// values, etc.). Callers are expected to provide the statement out-of-band and ensure
/// the Fiat-Shamir challenger is bound to it.
#[derive(Clone, Debug, Serialize)]
#[serde(bound(serialize = "F: Serialize, C: Serialize"))]
pub struct Proof<F, C> {
    pub transcript: TranscriptData<F, C>,
}
