//! Single-opening proof structures and transcript parsing helpers.
//!
//! - [`Proof`]: Single-opening proof with rows, optional salt, and authentication path.
//! - [`BatchProof`]: Parsed batch opening containing rows/salt plus hinted siblings.
//!
//! For batched openings via transcript hints in this crate, see
//! [`LmcsConfig`](crate::LmcsConfig) and [`LiftedMerkleTree`](crate::LiftedMerkleTree).
//! [`BatchProof`] parses hints without hashing, and can be turned into per-index
//! [`Proof`] objects once the hashing context is available.

use crate::utils::digest_rows_and_salt;
use alloc::collections::{BTreeMap, BTreeSet};
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

/// Opened rows and optional salt for a single leaf.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, [F; SALT_ELEMS]: Serialize",
    deserialize = "F: Deserialize<'de>, [F; SALT_ELEMS]: Deserialize<'de>"
))]
pub struct LeafOpening<F, const SALT_ELEMS: usize = 0> {
    /// Opened rows for this query.
    pub rows: Vec<Vec<F>>,
    /// Salt for this leaf (zero-sized when the commitment is non-hiding).
    pub salt: [F; SALT_ELEMS],
}

/// Batch opening parsed from transcript hints, without hashing.
///
/// Stores per-index openings plus the hinted siblings needed to reconstruct
/// authentication paths. Siblings are indexed by `(depth, index)` where depth 0
/// is the leaf level.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, D: Serialize, [F; SALT_ELEMS]: Serialize, [D; DIGEST_ELEMS]: Serialize",
    deserialize = "F: Deserialize<'de>, D: Deserialize<'de>, [F; SALT_ELEMS]: Deserialize<'de>, [D; DIGEST_ELEMS]: Deserialize<'de>"
))]
pub struct BatchProof<F, D, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize = 0> {
    /// Openings keyed by leaf index.
    pub openings: BTreeMap<usize, LeafOpening<F, SALT_ELEMS>>,
    /// Hinted siblings keyed by `(depth, index)`.
    pub siblings: BTreeMap<(usize, usize), [D; DIGEST_ELEMS]>,
}

impl<F, D, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    BatchProof<F, D, DIGEST_ELEMS, SALT_ELEMS>
{
    /// Read a batch opening from a transcript channel without hashing.
    ///
    /// This parses rows/salt for each queried index, then consumes exactly the
    /// hinted sibling digests implied by the query set and tree depth.
    /// Widths must already include any alignment padding.
    pub fn read_from_channel<Ch>(
        widths: &[usize],
        log_max_height: usize,
        indices: &[usize],
        channel: &mut Ch,
    ) -> Option<Self>
    where
        F: Copy,
        D: Copy + PartialEq,
        Ch: VerifierChannel<F = F, Commitment = Hash<F, D, DIGEST_ELEMS>>,
    {
        if indices.is_empty() {
            return Some(Self {
                openings: BTreeMap::new(),
                siblings: BTreeMap::new(),
            });
        }

        let max_height = 1 << log_max_height;
        let mut openings = BTreeMap::new();

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

            openings.entry(index).or_insert(LeafOpening { rows, salt });
        }

        let mut siblings: BTreeMap<(usize, usize), [D; DIGEST_ELEMS]> = BTreeMap::new();
        // Consume sibling hints in the same canonical order the prover emits them.
        for (current_depth, missing_pos) in
            required_siblings(indices.iter().copied(), log_max_height)
        {
            let digest = channel.receive_hint_commitment().copied().map(Into::into)?;

            if siblings
                .insert((current_depth, missing_pos), digest)
                .is_some_and(|existing| existing != digest)
            {
                return None;
            }
        }

        Some(Self { openings, siblings })
    }

    /// Reconstruct per-index proofs by hashing rows/salt and rebuilding paths.
    ///
    /// Returns a map keyed by leaf index; duplicate indices are coalesced.
    /// This does not verify against a commitment.
    pub fn single_proofs<H, C, const WIDTH: usize>(
        &self,
        sponge: &H,
        compress: &C,
        widths: &[usize],
        log_max_height: usize,
    ) -> Option<BTreeMap<usize, Proof<F, D, DIGEST_ELEMS, SALT_ELEMS>>>
    where
        F: Copy,
        D: Default + Copy + PartialEq,
        H: StatefulHasher<F, [D; DIGEST_ELEMS], State = [D; WIDTH]>,
        C: PseudoCompressionFunction<[D; DIGEST_ELEMS], 2>,
    {
        let max_height = 1 << log_max_height;
        let mut proofs: BTreeMap<usize, Proof<F, D, DIGEST_ELEMS, SALT_ELEMS>> = BTreeMap::new();
        // Track known nodes by (depth, index) to reconstruct sibling paths deterministically.
        let mut tree: BTreeMap<(usize, usize), [D; DIGEST_ELEMS]> = BTreeMap::new();

        for (&index, opening) in self.openings.iter() {
            if index >= max_height {
                return None;
            }

            if opening.rows.len() != widths.len() {
                return None;
            }
            for (row, &width) in opening.rows.iter().zip(widths.iter()) {
                if row.len() != width {
                    return None;
                }
            }

            let digest = digest_rows_and_salt(
                sponge,
                opening.rows.iter().map(|row| row.as_slice()),
                opening.salt.as_slice(),
            );

            proofs.entry(index).or_insert_with(|| Proof {
                rows: opening.rows.clone(),
                salt: opening.salt,
                siblings: Vec::with_capacity(log_max_height),
            });

            if tree
                .insert((0, index), digest)
                .is_some_and(|existing_digest| existing_digest != digest)
            {
                return None;
            }
        }

        // Preload hinted siblings so combining pairs can assume adjacency.
        for (current_depth, missing_pos) in
            required_siblings(self.openings.keys().copied(), log_max_height)
        {
            let digest = *self.siblings.get(&(current_depth, missing_pos))?;
            if tree
                .insert((current_depth, missing_pos), digest)
                .is_some_and(|existing| existing != digest)
            {
                return None;
            }
        }

        for current_depth in 0..log_max_height {
            // BTreeMap ordering yields left-to-right pairing at this depth.
            let nodes_at_depth: Vec<(usize, [D; DIGEST_ELEMS])> = tree
                .range((current_depth, 0)..=(current_depth, usize::MAX))
                .map(|(&(_, idx), digest)| (idx, *digest))
                .collect();

            let mut nodes_iter = nodes_at_depth.into_iter().peekable();
            while let Some((index, digest)) = nodes_iter.next() {
                let sibling_index = index ^ 1;
                let sibling_digest =
                    match nodes_iter.next_if(|(next_index, _)| *next_index == sibling_index) {
                        Some((_, digest)) => digest,
                        None => return None,
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

        // Add authentication paths from the reconstructed tree.
        for (&index, proof) in proofs.iter_mut() {
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

/// Determine which sibling positions must be supplied, in canonical order.
fn required_siblings<I>(indices: I, log_max_height: usize) -> Vec<(usize, usize)>
where
    I: IntoIterator<Item = usize>,
{
    let mut missing = Vec::new();
    // Track known nodes per level; BTreeSet keeps canonical left-to-right iteration.
    let mut known: BTreeSet<usize> = indices.into_iter().collect();

    for current_depth in 0..log_max_height {
        let mut parents = BTreeSet::new();

        for &pos in &known {
            let parent_pos = pos / 2;
            if !parents.insert(parent_pos) {
                continue;
            }

            let left_pos = parent_pos * 2;
            let right_pos = left_pos + 1;
            let have_left = known.contains(&left_pos);
            let have_right = known.contains(&right_pos);

            // Only emit a sibling when exactly one child is known.
            let missing_pos = match (have_left, have_right) {
                (true, false) => right_pos,
                (false, true) => left_pos,
                _ => continue,
            };

            missing.push((current_depth, missing_pos));
        }

        known = parents;
    }

    missing
}
