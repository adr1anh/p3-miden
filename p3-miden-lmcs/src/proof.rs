//! Single-opening proof structures for Merkle trees.
//!
//! - [`Proof`]: Single-opening proof with opening and authentication path.
//!
//! For batch multi-opening proofs, see [`crate::batch_proof::BatchProof`].
//! For opening data structures, see [`crate::opening::Opening`].

use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use p3_miden_stateful_hasher::StatefulHasher;
use p3_symmetric::{Hash, PseudoCompressionFunction};

use crate::LmcsError;
use crate::opening::Opening;

// ============================================================================
// Public Types
// ============================================================================

/// Single-opening Merkle proof with opening data and authentication path.
///
/// Contains the opening (rows + salt) and siblings (bottom-to-top) for a single leaf.
/// Use `open()` to verify against a commitment and retrieve the opened rows.
///
/// # Type Parameters
///
/// - `F`: Field element type.
/// - `D`: Digest element type.
/// - `DIGEST_ELEMS`: Number of elements in each digest.
/// - `SALT_ELEMS`: Number of salt elements. Use `0` for non-hiding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, [D; DIGEST_ELEMS]: Serialize, [F; SALT_ELEMS]: Serialize",
    deserialize = "F: Deserialize<'de>, [D; DIGEST_ELEMS]: Deserialize<'de>, [F; SALT_ELEMS]: Deserialize<'de>"
))]
pub struct Proof<F, D, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize = 0> {
    /// The opened row data (rows + salt).
    pub opening: Opening<F, SALT_ELEMS>,
    /// Sibling digests from leaf level to root (bottom-to-top).
    pub siblings: Vec<[D; DIGEST_ELEMS]>,
}

impl<F, D, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize> Proof<F, D, DIGEST_ELEMS, SALT_ELEMS>
where
    F: Default + Copy + PartialEq,
    D: Default + Copy + PartialEq,
{
    /// Open this proof against a commitment.
    ///
    /// Returns references to the opened rows on success.
    pub fn open<'a, H, C, const WIDTH: usize>(
        &'a self,
        sponge: &H,
        compress: &C,
        commitment: &Hash<F, D, DIGEST_ELEMS>,
        widths: &[usize],
        index: usize,
    ) -> Result<Vec<&'a [F]>, LmcsError>
    where
        H: StatefulHasher<F, [D; DIGEST_ELEMS], State = [D; WIDTH]>,
        C: PseudoCompressionFunction<[D; DIGEST_ELEMS], 2>,
    {
        let leaf_digest = self
            .opening
            .digest::<D, H, WIDTH, DIGEST_ELEMS>(sponge, widths)?;

        let computed_root = self.compute_root(index, leaf_digest, compress);

        if Hash::from(computed_root) != *commitment {
            return Err(LmcsError::RootMismatch);
        }

        Ok(self.opening.rows().collect())
    }

    /// Compute Merkle root from leaf digest and this path.
    pub fn compute_root<C>(
        &self,
        index: usize,
        leaf_digest: [D; DIGEST_ELEMS],
        compress: &C,
    ) -> [D; DIGEST_ELEMS]
    where
        C: PseudoCompressionFunction<[D; DIGEST_ELEMS], 2>,
    {
        let mut current = leaf_digest;
        let mut pos = index;

        for sibling in &self.siblings {
            let is_left = pos & 1 == 0;
            current = if is_left {
                compress.compress([current, *sibling])
            } else {
                compress.compress([*sibling, current])
            };
            pos >>= 1;
        }

        current
    }
}
