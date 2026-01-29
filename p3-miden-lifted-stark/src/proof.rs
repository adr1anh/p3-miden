//! Proof container (transcript data only for now).

use p3_miden_transcript::TranscriptData;

#[derive(Clone, Debug)]
pub struct Proof<F, C> {
    pub transcript: TranscriptData<F, C>,
}
