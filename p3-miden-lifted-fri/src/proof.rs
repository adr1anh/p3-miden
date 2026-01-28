//! PCS transcript data structures.

use crate::PcsParams;
use crate::deep::DeepTranscript;
use crate::fri::FriTranscript;
use alloc::vec::Vec;
use p3_challenger::{CanSample, CanSampleBits};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_miden_lmcs::Lmcs;
use p3_miden_transcript::VerifierChannel;

/// Ordered per-index proofs, aligned with the sampled query indices.
pub struct OrderedProofs<P>(pub Vec<P>);

/// Structured transcript view for the full PCS interaction.
///
/// Captures observed transcript data plus reconstructed LMCS proofs for inspection.
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
    /// Per-index Merkle proofs per trace tree, ordered by query indices.
    pub deep_openings: Vec<OrderedProofs<L::SingleProof>>,
    /// Per-index Merkle proofs per FRI round, ordered by query indices.
    pub fri_openings: Vec<OrderedProofs<L::SingleProof>>,
}

impl<EF, L> PcsTranscript<EF, L>
where
    L: Lmcs,
    L::F: TwoAdicField,
    EF: ExtensionField<L::F>,
{
    /// Parse a PCS transcript view from a verifier channel.
    pub fn from_verifier_channel<Ch, const N: usize>(
        params: &PcsParams,
        lmcs: &L,
        commitments: &[(L::Commitment, Vec<usize>)],
        log_max_height: usize,
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
            FriTranscript::from_verifier_channel(&params.fri, log_max_height, channel)?;

        let query_pow_witness = channel.grind(params.query_proof_of_work_bits)?;

        let indices: Vec<usize> = (0..params.num_queries)
            .map(|_| channel.sample_bits(log_max_height))
            .collect();

        let deep_openings: Vec<_> = commitments
            .iter()
            .map(|(_commitment, widths)| {
                lmcs.read_batch_from_channel(widths, log_max_height, &indices, channel)
                    .ok()
                    .map(OrderedProofs)
            })
            .collect::<Option<Vec<_>>>()?;

        let log_arity = params.fri.fold.log_arity();
        let arity = params.fri.fold.arity();
        let num_rounds = params.fri.num_rounds(log_max_height);

        let mut fri_openings = Vec::with_capacity(num_rounds);
        for round in 0..num_rounds {
            let log_num_rows = log_max_height.saturating_sub(log_arity * (round + 1));
            let round_indices: Vec<usize> = indices
                .iter()
                .map(|&idx| idx >> (log_arity * (round + 1)))
                .collect();
            let widths = [arity * EF::DIMENSION];
            let proofs = lmcs
                .read_batch_from_channel(&widths, log_num_rows, &round_indices, channel)
                .ok()
                .map(OrderedProofs)?;
            fri_openings.push(proofs);
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
