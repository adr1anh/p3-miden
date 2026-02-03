//! Single-opening proof structures and transcript parsing helpers.
//!
//! - [`Proof`]: Single-opening proof with rows, optional salt, and authentication path.
//! - [`BatchProof`]: Parsed batch opening containing rows/salt plus hinted siblings.
//!
//! For batched openings via transcript hints in this crate, see
//! [`LmcsConfig`](crate::LmcsConfig) and [`LiftedMerkleTree`](crate::LiftedMerkleTree).
//! [`BatchProof`] parses hints without hashing, and can be turned into per-index
//! [`Proof`] objects once the hashing context is available.

use crate::Lmcs;
use alloc::collections::btree_map::Entry;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use p3_miden_transcript::{TranscriptError, VerifierChannel};
use serde::{Deserialize, Serialize};

/// Single-opening Merkle proof with rows and authentication path.
///
/// Contains the opening (rows + salt) and siblings (bottom-to-top) for a single leaf.
///
/// # Type Parameters
///
/// - `F`: Field element type.
/// - `C`: Hash type (also used for commitments).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, C: Serialize, [F; SALT_ELEMS]: Serialize",
    deserialize = "F: Deserialize<'de>, C: Deserialize<'de>, [F; SALT_ELEMS]: Deserialize<'de>"
))]
pub struct Proof<F, C, const SALT_ELEMS: usize = 0> {
    /// Opened rows for this query.
    pub rows: Vec<Vec<F>>,
    /// Salt for this leaf (zero-sized when the configuration is non-hiding).
    pub salt: [F; SALT_ELEMS],
    /// Sibling hashes from leaf level to root (bottom-to-top).
    pub siblings: Vec<C>,
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
    /// Salt for this leaf (zero-sized when the configuration is non-hiding).
    pub salt: [F; SALT_ELEMS],
}

/// Batch opening parsed from transcript hints, without hashing.
///
/// Stores per-index openings plus the hinted siblings needed to reconstruct
/// authentication paths. Siblings are indexed by `(depth, index)` where depth 0
/// is the leaf level.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, C: Serialize, [F; SALT_ELEMS]: Serialize",
    deserialize = "F: Deserialize<'de>, C: Deserialize<'de>, [F; SALT_ELEMS]: Deserialize<'de>"
))]
pub struct BatchProof<F, C, const SALT_ELEMS: usize = 0> {
    /// Openings keyed by leaf index.
    pub openings: BTreeMap<usize, LeafOpening<F, SALT_ELEMS>>,
    /// Hinted sibling hashes keyed by `(depth, index)`.
    pub siblings: BTreeMap<(usize, usize), C>,
}

impl<F, C, const SALT_ELEMS: usize> BatchProof<F, C, SALT_ELEMS> {
    /// Read a batch opening from a transcript channel without hashing.
    ///
    /// Parses rows/salt for each unique queried index in sorted (ascending) order,
    /// matching the order in which [`LmcsTree::prove_batch`](crate::LmcsTree::prove_batch)
    /// writes them. Consumes exactly the hinted sibling hashes implied by the query
    /// set and tree depth.
    ///
    /// Assumes all indices are in `0..2^log_max_height`; out-of-range indices
    /// produce an invalid proof that will fail verification.
    pub fn read_from_channel<Ch>(
        widths: &[usize],
        log_max_height: usize,
        indices: &[usize],
        channel: &mut Ch,
    ) -> Result<Self, TranscriptError>
    where
        F: Copy,
        C: Copy + PartialEq,
        Ch: VerifierChannel<F = F, Commitment = C>,
    {
        // Collect and sort indices to match prover's write order (BTreeSet iteration).
        let unique_indices: BTreeSet<usize> = indices.iter().copied().collect();
        let mut openings: BTreeMap<usize, LeafOpening<F, SALT_ELEMS>> = BTreeMap::new();

        // Read openings in sorted order, matching prove_batch's write order.
        for index in unique_indices.iter().copied() {
            let Entry::Vacant(entry) = openings.entry(index) else {
                unreachable!("unique_indices is deduplicated");
            };

            let mut rows = Vec::with_capacity(widths.len());
            for &width in widths {
                let row = channel.receive_hint_field_slice(width)?;
                rows.push(Vec::from(row));
            }

            let salt_slice = channel.receive_hint_field_slice(SALT_ELEMS)?;
            let salt: [F; SALT_ELEMS] = salt_slice.try_into().unwrap();

            entry.insert(LeafOpening { rows, salt });
        }

        // Consume sibling hints in the same canonical order the prover emits them.
        let siblings: BTreeMap<(usize, usize), C> =
            required_siblings(openings.keys().copied(), log_max_height)
                .into_iter()
                .map(|key| Ok((key, *channel.receive_hint_commitment()?)))
                .collect::<Result<_, TranscriptError>>()?;

        Ok(Self { openings, siblings })
    }

    /// Reconstruct per-index proofs by hashing rows/salt and rebuilding paths.
    ///
    /// Returns a map keyed by leaf index; duplicate indices are coalesced.
    /// Does not verify against a commitment.
    ///
    /// Assumes all indices are in `0..2^log_max_height`; returns `None` if
    /// widths mismatch or sibling reconstruction fails.
    pub fn single_proofs<L>(
        &self,
        lmcs: &L,
        widths: &[usize],
        log_max_height: usize,
    ) -> Option<BTreeMap<usize, Proof<F, C, SALT_ELEMS>>>
    where
        F: Copy,
        C: Copy + PartialEq,
        L: Lmcs<F = F, Commitment = C>,
    {
        let mut proofs: BTreeMap<usize, Proof<F, C, SALT_ELEMS>> = BTreeMap::new();
        // Track known nodes by (depth, index) to reconstruct sibling paths deterministically.
        let mut tree: BTreeMap<(usize, usize), C> = BTreeMap::new();

        for (&index, opening) in self.openings.iter() {
            if opening.rows.len() != widths.len() {
                return None;
            }
            for (row, &width) in opening.rows.iter().zip(widths.iter()) {
                if row.len() != width {
                    return None;
                }
            }

            let leaf_hash = lmcs.hash(
                opening
                    .rows
                    .iter()
                    .map(|row| row.as_slice())
                    .chain(core::iter::once(opening.salt.as_slice())),
            );

            proofs.entry(index).or_insert_with(|| Proof {
                rows: opening.rows.clone(),
                salt: opening.salt,
                siblings: Vec::with_capacity(log_max_height),
            });

            if tree
                .insert((0, index), leaf_hash)
                .is_some_and(|existing_hash| existing_hash != leaf_hash)
            {
                return None;
            }
        }

        // Preload hinted siblings so combining pairs can assume adjacency.
        for (depth, index) in required_siblings(self.openings.keys().copied(), log_max_height) {
            tree.insert((depth, index), *self.siblings.get(&(depth, index))?);
        }

        for current_depth in 0..log_max_height {
            // BTreeMap ordering yields left-to-right pairing at this depth.
            let nodes_at_depth: Vec<(usize, C)> = tree
                .range((current_depth, 0)..=(current_depth, usize::MAX))
                .map(|(&(_, idx), hash)| (idx, *hash))
                .collect();

            let mut nodes_iter = nodes_at_depth.into_iter().peekable();
            while let Some((index, hash)) = nodes_iter.next() {
                let sibling_index = index ^ 1;
                let sibling_hash =
                    match nodes_iter.next_if(|(next_index, _)| *next_index == sibling_index) {
                        Some((_, hash)) => hash,
                        None => return None,
                    };

                let is_left_child = index & 1 == 0;
                let (left, right) = if is_left_child {
                    (hash, sibling_hash)
                } else {
                    (sibling_hash, hash)
                };

                let parent_depth = current_depth + 1;
                let parent_index = index / 2;
                let parent_hash = lmcs.compress(left, right);
                tree.insert((parent_depth, parent_index), parent_hash);
            }
        }

        // Add authentication paths from the reconstructed tree.
        for (&index, proof) in proofs.iter_mut() {
            let mut current_index = index;
            for current_depth in 0..log_max_height {
                let sibling_index = current_index ^ 1;
                let sibling_hash = tree.get(&(current_depth, sibling_index)).copied()?;
                proof.siblings.push(sibling_hash);
                current_index >>= 1;
            }
        }

        Some(proofs)
    }
}

/// Sibling positions that must be supplied, in canonical (depth, left-to-right) order.
///
/// Returns unique `(depth, index)` pairs where depth 0 is the leaf level.
/// Callers can insert directly without checking for duplicates.
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
