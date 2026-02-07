//! Proof container.

use p3_miden_transcript::TranscriptData;

/// Proof container wrapping transcript data.
#[derive(Clone, Debug)]
pub struct Proof<F, C> {
    pub transcript: TranscriptData<F, C>,
}
