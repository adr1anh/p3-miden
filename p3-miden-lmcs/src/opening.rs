//! Opening data for Merkle tree proofs.
//!
//! Contains the per-query row data with optional salt, and methods for computing leaf digests.

use alloc::vec::Vec;
use core::iter::zip;

use p3_miden_stateful_hasher::StatefulHasher;
use serde::{Deserialize, Serialize};

use crate::LmcsError;

/// Per-query row data with optional salt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, [F; SALT_ELEMS]: Serialize",
    deserialize = "F: Deserialize<'de>, [F; SALT_ELEMS]: Deserialize<'de>"
))]
pub struct Opening<F, const SALT_ELEMS: usize = 0> {
    /// Opened rows: `rows[matrix_idx]` = row data for that matrix.
    pub(crate) rows: Vec<Vec<F>>,
    /// Salt for this leaf. Zero-sized when `SALT_ELEMS = 0`.
    pub(crate) salt: [F; SALT_ELEMS],
}

impl<F, const SALT_ELEMS: usize> Opening<F, SALT_ELEMS> {
    /// Get references to the opened rows.
    #[inline]
    pub fn rows(&self) -> impl Iterator<Item = &[F]> {
        self.rows.iter().map(|r| r.as_slice())
    }
}

impl<F, const SALT_ELEMS: usize> Opening<F, SALT_ELEMS>
where
    F: Default + Copy,
{
    /// Compute the leaf digest from this opening's rows and salt.
    ///
    /// # Errors
    ///
    /// - `WrongMatrixCount`: Number of rows doesn't match number of widths.
    /// - `WrongWidth`: A row's width doesn't match its expected width.
    pub fn digest<D, H, const WIDTH: usize, const DIGEST_ELEMS: usize>(
        &self,
        sponge: &H,
        widths: &[usize],
    ) -> Result<[D; DIGEST_ELEMS], LmcsError>
    where
        D: Default + Copy,
        H: StatefulHasher<F, [D; DIGEST_ELEMS], State = [D; WIDTH]>,
    {
        if self.rows.len() != widths.len() {
            return Err(LmcsError::WrongMatrixCount {
                expected: widths.len(),
                actual: self.rows.len(),
            });
        }

        let mut state = [D::default(); WIDTH];
        for (idx, (row, &width)) in zip(&self.rows, widths).enumerate() {
            if row.len() != width {
                return Err(LmcsError::WrongWidth { matrix: idx });
            }
            sponge.absorb_into(&mut state, row.iter().copied());
        }

        // Absorb salt
        if !self.salt.is_empty() {
            sponge.absorb_into(&mut state, self.salt.iter().copied());
        }

        Ok(sponge.squeeze(&state))
    }
}
