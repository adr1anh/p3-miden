//! Single-opening proof structures and transcript parsing.
//!
//! - [`Proof`]: Single-opening proof with rows, optional salt, and authentication path.
//!
//! For batched openings via transcript hints, see [`Lmcs`](crate::Lmcs)::open_batch.
//! The [`Proof`] type stores opened rows (and optional salt) alongside its Merkle path.

use crate::utils::digest_rows_and_salt;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use p3_miden_stateful_hasher::StatefulHasher;
use p3_miden_transcript::VerifierChannel;
use p3_symmetric::{Hash, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};

/// Single-opening Merkle proof with rows and authentication path.
///
/// Contains the opening (rows + salt) and siblings (bottom-to-top) for a single leaf.
///
/// # Type Parameters
///
/// - `F`: Field element type.
/// - `D`: Digest element type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, D: Serialize, [F; SALT_ELEMS]: Serialize, [D; DIGEST_ELEMS]: Serialize",
    deserialize = "F: Deserialize<'de>, D: Deserialize<'de>, [F; SALT_ELEMS]: Deserialize<'de>, [D; DIGEST_ELEMS]: Deserialize<'de>"
))]
pub struct Proof<F, D, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize = 0> {
    /// Opened rows for this query.
    pub rows: Vec<Vec<F>>,
    /// Salt for this leaf (zero-sized when the commitment is non-hiding).
    pub salt: [F; SALT_ELEMS],
    /// Sibling digests from leaf level to root (bottom-to-top).
    pub siblings: Vec<[D; DIGEST_ELEMS]>,
}

impl<F, D, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    Proof<F, D, DIGEST_ELEMS, SALT_ELEMS>
{
    /// Read a batch opening from a transcript channel and reconstruct per-index proofs.
    ///
    /// This only parses hints; it does not verify against a commitment.
    pub fn read_batch_from_channel<H, C, Ch, const WIDTH: usize>(
        sponge: &H,
        compress: &C,
        widths: &[usize],
        log_max_height: usize,
        indices: &[usize],
        channel: &mut Ch,
    ) -> Option<Vec<Self>>
    where
        F: Default + Copy,
        D: Default + Copy + PartialEq,
        H: StatefulHasher<F, [D; DIGEST_ELEMS], State = [D; WIDTH]>,
        C: PseudoCompressionFunction<[D; DIGEST_ELEMS], 2>,
        Ch: VerifierChannel<F = F, Commitment = Hash<F, D, DIGEST_ELEMS>>,
    {
        let max_height = 1 << log_max_height;
        // Keep proofs in query order for callers that align openings with indices.
        let mut proofs = Vec::with_capacity(indices.len());
        // Track known nodes by (depth, index) to reconstruct sibling paths from a hint stream.
        let mut tree: BTreeMap<(usize, usize), [D; DIGEST_ELEMS]> = BTreeMap::new();

        for &index in indices {
            if index >= max_height {
                return None;
            }

            // Read hinted rows/salt for each queried leaf; these are not observed by the FS challenger.
            let rows = widths
                .iter()
                .map(|&width| channel.receive_hint_field_slice(width).map(Vec::from))
                .collect::<Option<Vec<_>>>()?;

            let salt: [F; SALT_ELEMS] = channel
                .receive_hint_field_slice(SALT_ELEMS)?
                .try_into()
                .unwrap();

            // Compute the leaf digest so we can rebuild authentication paths deterministically.
            let digest = digest_rows_and_salt(
                sponge,
                rows.iter().map(|row| row.as_slice()),
                salt.as_slice(),
            );

            proofs.push(Proof {
                rows,
                salt,
                siblings: Vec::with_capacity(log_max_height),
            });

            if tree
                .insert((0, index), digest)
                .is_some_and(|existing_digest| existing_digest != digest)
            {
                return None;
            }
        }

        for current_depth in 0..log_max_height {
            // Walk a level at a time; consume missing siblings from the hint stream
            // in a canonical left-to-right order to avoid ambiguity.
            let nodes_at_depth: Vec<(usize, [D; DIGEST_ELEMS])> = tree
                .range((current_depth, 0)..=(current_depth, usize::MAX))
                .map(|(&(_, idx), digest)| (idx, *digest))
                .collect();

            let mut nodes_iter = nodes_at_depth.into_iter().peekable();
            while let Some((index, digest)) = nodes_iter.next() {
                let sibling_index = index ^ 1;
                let sibling_digest = match nodes_iter
                    .next_if(|(next_index, _)| *next_index == sibling_index)
                {
                    Some((_, digest)) => digest,
                    None => {
                        // Sibling not known from queries, so it must be provided by the transcript.
                        let digest = channel.receive_hint_commitment().copied().map(Into::into)?;
                        tree.insert((current_depth, sibling_index), digest);
                        digest
                    }
                };

                let is_left_child = index & 1 == 0;
                let (left, right) = if is_left_child {
                    (digest, sibling_digest)
                } else {
                    (sibling_digest, digest)
                };

                let parent_depth = current_depth + 1;
                let parent_index = index / 2;
                let parent_digest = compress.compress([left, right]);
                tree.insert((parent_depth, parent_index), parent_digest);
            }
        }

        // add authentication paths from the tree
        for (proof, &index) in proofs.iter_mut().zip(indices.iter()) {
            let mut current_index = index;
            for current_depth in 0..log_max_height {
                let sibling_index = current_index ^ 1;
                let digest = tree.get(&(current_depth, sibling_index)).copied()?;
                proof.siblings.push(digest);
                current_index >>= 1;
            }
        }

        Some(proofs)
    }
}
