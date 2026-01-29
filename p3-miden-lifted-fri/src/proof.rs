//! PCS transcript data structures.

use crate::PcsParams;
use crate::deep::DeepTranscript;
use crate::fri::FriTranscript;
use alloc::vec::Vec;
use p3_challenger::{CanSample, CanSampleBits};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_miden_lmcs::Lmcs;
use p3_miden_lmcs::utils::aligned_len;
use p3_miden_transcript::VerifierChannel;

/// Structured transcript view for the full PCS interaction.
///
/// Captures observed transcript data plus parsed LMCS batch openings for inspection.
pub struct PcsTranscript<EF, L>
where
    L: Lmcs,
    L::F: Field,
    EF: ExtensionField<L::F>,
{
    /// DEEP transcript data (evals, PoW witness, challenges).
    pub deep_transcript: DeepTranscript<L::F, EF>,
    /// FRI transcript data (round commitments/challenges, final polynomial).
    pub fri_transcript: FriTranscript<L::F, EF, L::Commitment>,
    /// Proof-of-work witness for query sampling.
    pub query_pow_witness: L::F,
    /// Query indices sampled for openings.
    pub indices: Vec<usize>,
    /// Batch openings per trace tree, aligned with `indices`.
    pub deep_openings: Vec<L::BatchProof>,
    /// Batch openings per FRI round, aligned with per-round indices.
    pub fri_openings: Vec<L::BatchProof>,
}

impl<EF, L> PcsTranscript<EF, L>
where
    L: Lmcs,
    L::F: TwoAdicField,
    EF: ExtensionField<L::F>,
{
    /// Parse a PCS transcript view from a verifier channel.
    ///
    /// Commitment widths must already include any alignment padding, and all
    /// commitments are expected to be lifted to the same `log_lde_height`.
    ///
    /// `log_lde_height` is the log₂ of the LDE evaluation domain height (i.e. the height of
    /// the committed LDE matrices). When a trace degree is known, it is typically
    /// `log_trace_height + params.fri.log_blowup` (plus any extension used by the caller).
    pub fn from_verifier_channel<Ch, const N: usize>(
        params: &PcsParams,
        lmcs: &L,
        commitments: &[(L::Commitment, Vec<usize>)],
        log_lde_height: usize,
        eval_points: [EF; N],
        channel: &mut Ch,
    ) -> Option<Self>
    where
        Ch: VerifierChannel<F = L::F, Commitment = L::Commitment>
            + CanSample<L::F>
            + CanSampleBits<usize>,
    {
        if commitments.is_empty() {
            return None;
        }

        let deep_transcript = DeepTranscript::from_verifier_channel::<Ch>(
            &params.deep,
            commitments,
            eval_points.len(),
            channel,
        )?;

        let fri_transcript =
            FriTranscript::from_verifier_channel(&params.fri, log_lde_height, channel)?;

        let query_pow_witness = channel.grind(params.query_proof_of_work_bits)?;

        let indices: Vec<usize> = (0..params.num_queries)
            .map(|_| channel.sample_bits(log_lde_height))
            .collect();

        let deep_openings: Vec<_> = commitments
            .iter()
            .map(|(_commitment, widths)| {
                lmcs.read_batch_proof_from_channel(widths, log_lde_height, &indices, channel)
                    .ok()
            })
            .collect::<Option<Vec<_>>>()?;

        let log_arity = params.fri.fold.log_arity();
        let arity = params.fri.fold.arity();
        let num_rounds = params.fri.num_rounds(log_lde_height);

        let mut fri_openings = Vec::with_capacity(num_rounds);
        for round in 0..num_rounds {
            let log_num_rows = log_lde_height.saturating_sub(log_arity * (round + 1));
            let round_indices: Vec<usize> = indices
                .iter()
                .map(|&idx| idx >> (log_arity * (round + 1)))
                .collect();
            let base_width = arity * EF::DIMENSION;
            let aligned_width = aligned_len(base_width, lmcs.alignment());
            let widths = [aligned_width];
            let batch = lmcs
                .read_batch_proof_from_channel(&widths, log_num_rows, &round_indices, channel)
                .ok()?;
            fri_openings.push(batch);
        }

        Some(Self {
            deep_transcript,
            fri_transcript,
            query_pow_witness,
            indices,
            deep_openings,
            fri_openings,
        })
    }
}
