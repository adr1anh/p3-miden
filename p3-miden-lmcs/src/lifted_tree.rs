use alloc::vec;
use alloc::vec::Vec;
use core::{array, mem};

use crate::utils::PackedValueExt;
use crate::{LmcsTree, Proof};
use p3_field::PackedValue;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_miden_stateful_hasher::StatefulHasher;
use p3_miden_transcript::ProverChannel;
use p3_symmetric::{Hash, PseudoCompressionFunction};
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};

/// A uniform binary Merkle tree whose leaves are constructed from matrices with power-of-two heights.
///
/// # Type Parameters
///
/// * `F` – scalar field element type used in both matrices and digests.
/// * `D` – digest element type.
/// * `M` – matrix type. Must implement [`Matrix<F>`].
/// * `DIGEST_ELEMS` – number of elements in one digest.
/// * `SALT_ELEMS` – number of salt elements per leaf (0 = non-hiding, >0 = hiding).
///
/// Unlike the standard `MerkleTree`, this uniform variant requires:
/// - **All matrix heights must be powers of two**
/// - **Matrices must be sorted by height** (shortest to tallest)
/// - Uses incremental hashing via [`StatefulHasher`] instead of one-shot hashing
///
/// The per-leaf row composition uses nearest-neighbor upsampling: each matrix `M_i` is virtually
/// extended to height `N` (width unchanged) by repeating each row `r_i = N / n_i` times
/// contiguously. For leaf index `j`, the sponge absorbs the `j`-th row from each lifted matrix
/// in sequence (with per-matrix zero padding to a multiple of the hasher's padding width for
/// absorption).
///
/// Equivalent single-matrix view: this commitment is equivalent to first forming a single
/// height-`N` matrix by (a) lifting every input matrix to height `N`, (b) padding each lifted
/// matrix horizontally with zero columns so each width is a multiple of the hasher's padding
/// width, and (c) concatenating the results side-by-side. The leaf digest at index `j` is then
/// the sponge of that single concatenated matrix's row `j`. From the verifier's perspective,
/// the two constructions are indistinguishable: verification absorbs the same padded row
/// segments in the same order and checks the same Merkle path.
///
/// Since [`StatefulHasher`] operates on a single field type, this tree uses the same type `F`
/// for both matrix elements and digest words, unlike `MerkleTree` which can hash `F → W`.
///
/// Use [`root`](Self::root) to fetch the final digest once the tree is built.
///
/// ## Transcript Hints
///
/// `prove_batch` streams transcript hints in the format expected by
/// [`LmcsConfig::open_batch`](crate::LmcsConfig::open_batch):
/// - For each query index (in caller order): one row per matrix (in leaf order), then
///   `SALT_ELEMS` field elements of salt.
/// - After all indices: missing sibling hashes, level-by-level, left-to-right, bottom-to-top.
/// Hints are not observed into the Fiat-Shamir challenger.
///
/// This generally shouldn't be used directly. If you're using a Merkle tree as an MMCS,
/// see the MMCS wrapper types.
#[derive(Debug, Serialize, Deserialize)]
pub struct LiftedMerkleTree<F, D, M, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize = 0> {
    /// All leaf matrices in insertion order.
    ///
    /// Matrices must be sorted by height (shortest to tallest) and all heights must be
    /// powers of two. Each matrix's rows are absorbed into sponge states that are
    /// maintained and upsampled across matrices of increasing height.
    ///
    /// This vector is retained for inspection or re-opening of the tree; it is not used
    /// after construction time.
    pub(crate) leaves: Vec<M>,

    /// All intermediate digest layers, index 0 being the leaf digest layer
    /// and the last layer containing exactly one root digest.
    ///
    /// Every inner vector holds contiguous digests. Higher layers are built by
    /// compressing pairs from the previous layer.
    #[serde(bound(
        serialize = "[D; DIGEST_ELEMS]: Serialize",
        deserialize = "[D; DIGEST_ELEMS]: Deserialize<'de>"
    ))]
    pub(crate) digest_layers: Vec<Vec<[D; DIGEST_ELEMS]>>,

    /// Salt matrix for hiding commitment. Each row contains `SALT_ELEMS` random field elements.
    /// `None` when `SALT_ELEMS = 0` (non-hiding mode).
    pub(crate) salt: Option<RowMajorMatrix<F>>,
}

impl<F, D, M, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    LmcsTree<F, Hash<F, D, DIGEST_ELEMS>, M> for LiftedMerkleTree<F, D, M, DIGEST_ELEMS, SALT_ELEMS>
where
    F: Copy + Default + PartialEq + Send + Sync,
    D: Copy + Default + PartialEq + Send + Sync,
    M: Matrix<F>,
{
    fn root(&self) -> Hash<F, D, DIGEST_ELEMS> {
        self.digest_layers.last().unwrap()[0].into()
    }

    fn height(&self) -> usize {
        self.leaves.last().unwrap().height()
    }

    fn leaves(&self) -> &[M] {
        &self.leaves
    }

    fn rows(&self, index: usize) -> Vec<Vec<F>> {
        let max_height = self.height();

        self.leaves
            .iter()
            .map(|m| {
                let height = m.height();
                let log_scaling_factor = log2_strict_usize(max_height / height);
                let row_index = index >> log_scaling_factor;
                m.row_slice(row_index)
                    .expect("row_index must be valid after upsampling")
                    .to_vec()
            })
            .collect()
    }

    fn prove_batch<Ch>(&self, indices: &[usize], channel: &mut Ch)
    where
        Ch: ProverChannel<F = F, Commitment = Hash<F, D, DIGEST_ELEMS>>,
    {
        use alloc::collections::BTreeSet;

        let final_height = self.leaves.last().unwrap().height();
        let depth = log2_strict_usize(final_height);

        // Validate indices
        for &index in indices {
            assert!(
                index < final_height,
                "index {index} out of range {final_height}"
            );
        }

        for &idx in indices {
            for m in &self.leaves {
                let height = m.height();
                let log_scaling_factor = log2_strict_usize(final_height / height);
                let row_index = idx >> log_scaling_factor;
                let row = m
                    .row_slice(row_index)
                    .expect("row_index must be valid after upsampling");
                channel.hint_field_slice(&row);
            }
            if SALT_ELEMS > 0 {
                let salt = self.salt(idx);
                channel.hint_field_slice(&salt);
            }
        }

        let mut known: BTreeSet<usize> = indices.iter().copied().collect();

        // Walk up the tree level by level
        for layer_idx in 0..depth {
            let mut parents = BTreeSet::new();

            // BTreeSet iterates in sorted order (left-to-right)
            for &pos in &known {
                let parent_pos = pos / 2;
                if !parents.insert(parent_pos) {
                    continue; // Already processed this pair
                }

                let left_pos = parent_pos * 2;
                let right_pos = left_pos + 1;
                let have_left = known.contains(&left_pos);
                let have_right = known.contains(&right_pos);

                // Add sibling hash if exactly one child is known
                if have_left && !have_right {
                    channel.hint_commitment(Hash::from(self.digest_layers[layer_idx][right_pos]));
                } else if !have_left && have_right {
                    channel.hint_commitment(Hash::from(self.digest_layers[layer_idx][left_pos]));
                }
            }

            known = parents;
        }
    }
}

impl<F, D, M, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    LiftedMerkleTree<F, D, M, DIGEST_ELEMS, SALT_ELEMS>
where
    F: Copy + Default + PartialEq + Send + Sync,
    D: Copy + Default + PartialEq + Send + Sync,
    M: Matrix<F>,
{
    /// Internal builder for creating trees with optional salt.
    pub(crate) fn build<PF, PD, H, C, const WIDTH: usize>(
        h: &H,
        c: &C,
        leaves: Vec<M>,
        salt: Option<RowMajorMatrix<F>>,
    ) -> Self
    where
        PF: PackedValue<Value = F>,
        PD: PackedValue<Value = D>,
        H: StatefulHasher<F, [D; DIGEST_ELEMS], State = [D; WIDTH]>
            + StatefulHasher<PF, [PD; DIGEST_ELEMS], State = [PD; WIDTH]>
            + Sync,
        C: PseudoCompressionFunction<[D; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PD; DIGEST_ELEMS], 2>
            + Sync,
    {
        assert!(!leaves.is_empty(), "cannot commit empty batch");

        // Build leaf states from matrices using the sponge
        let mut leaf_states: Vec<[PD::Value; WIDTH]> =
            build_leaf_states_upsampled::<PF, PD, M, H, WIDTH, DIGEST_ELEMS>(&leaves, h);

        // Absorb salt into states using SIMD-parallelized path (no-op when salt is None)
        if let Some(ref salt_matrix) = salt {
            debug_assert_eq!(salt_matrix.height(), leaf_states.len());
            debug_assert_eq!(salt_matrix.width(), SALT_ELEMS);
            absorb_matrix::<PF, PD, _, _, WIDTH, DIGEST_ELEMS>(&mut leaf_states, salt_matrix, h);
        }

        // Squeeze the final digests from the states
        let leaf_digests: Vec<[PD::Value; DIGEST_ELEMS]> = leaf_states
            .into_par_iter()
            .map(|state| h.squeeze(&state))
            .collect();

        // Build digest layers by repeatedly compressing until we reach the root
        let mut digest_layers = vec![leaf_digests];

        loop {
            let prev_layer = digest_layers.last().unwrap();
            if prev_layer.len() == 1 {
                break;
            }

            let next_layer = compress_uniform::<PD, C, DIGEST_ELEMS>(prev_layer, c);
            digest_layers.push(next_layer);
        }

        Self {
            leaves,
            digest_layers,
            salt,
        }
    }

    /// Build a full opening proof for a single leaf index.
    pub fn single_proof(&self, index: usize) -> Proof<F, D, DIGEST_ELEMS, SALT_ELEMS> {
        let mut siblings = Vec::with_capacity(self.digest_layers.len().saturating_sub(1));
        let mut layer_index = index;
        for layer in &self.digest_layers {
            if layer.len() == 1 {
                break;
            }
            let sibling = layer[layer_index ^ 1];
            siblings.push(sibling);
            layer_index >>= 1;
        }

        Proof {
            rows: self.rows(index),
            salt: self.salt(index),
            siblings,
        }
    }

    /// Extract the salt for the given leaf index.
    ///
    /// # Panics
    ///
    /// Panics if `SALT_ELEMS > 0` but the tree was constructed without salt.
    pub fn salt(&self, index: usize) -> [F; SALT_ELEMS] {
        match &self.salt {
            Some(salt_matrix) => {
                let row = salt_matrix.row_slice(index).expect("index must be valid");
                // Tree construction guarantees salt width == SALT_ELEMS
                array::from_fn(|i| row[i])
            }
            None => {
                // For SALT_ELEMS == 0, this returns an empty array.
                // For SALT_ELEMS > 0, this should never be reached if using safe constructors.
                debug_assert!(
                    SALT_ELEMS == 0,
                    "tree constructed without salt but SALT_ELEMS > 0"
                );
                [F::default(); SALT_ELEMS]
            }
        }
    }
}

/// Build leaf states using the upsampled view (nearest-neighbor upsampling).
///
/// Returns the sponge states after absorbing all matrix rows but **before squeezing**.
/// Callers must squeeze the states to obtain final leaf digests.
///
/// Conceptually, each matrix is virtually extended to height `H` by repeating each row
/// `L = H / h` times (width unchanged), and the leaf `r` absorbs the `r`-th row from each
/// extended matrix in order. Each absorbed row is virtually padded with zeros to a multiple of the
/// hasher's padding width for absorption; see [`LiftedMerkleTree`](crate::LiftedMerkleTree) docs
/// for the equivalent single-matrix view.
///
/// # Preconditions
/// - `matrices` is non-empty and sorted by non-decreasing power-of-two heights.
/// - `P::WIDTH` is a power of two.
///
/// Panics in debug builds if preconditions are violated.
fn build_leaf_states_upsampled<PF, PD, M, H, const WIDTH: usize, const DIGEST_ELEMS: usize>(
    matrices: &[M],
    sponge: &H,
) -> Vec<[PD::Value; WIDTH]>
where
    PF: PackedValue,
    PD: PackedValue,
    M: Matrix<PF::Value>,
    H: StatefulHasher<PF::Value, [PD::Value; DIGEST_ELEMS], State = [PD::Value; WIDTH]>
        + StatefulHasher<PF, [PD; DIGEST_ELEMS], State = [PD; WIDTH]>
        + Sync,
{
    const { assert!(PF::WIDTH.is_power_of_two()) };
    const { assert!(PD::WIDTH.is_power_of_two()) };
    let final_height = validate_heights(matrices.iter().map(|d| d.dimensions().height));

    // Memory buffers:
    // - states: Per-leaf scalar states (one per final row), maintained across matrices.
    // - scratch_states: Temporary buffer used when duplicating states during upsampling.
    let default_state = [PD::Value::default(); WIDTH];
    let mut states = vec![default_state; final_height];
    let mut scratch_states = vec![default_state; final_height];

    let mut active_height = matrices.first().unwrap().height();

    for matrix in matrices {
        let height = matrix.height();

        // Upsample states when height increases (applies to both scalar and packed paths).
        // Duplicate each existing state to fill the expanded height.
        // E.g., [s0, s1] with scaling_factor=2 → [s0, s0, s1, s1]
        if height > active_height {
            let scaling_factor = height / active_height;

            // Copy `states` into `scratch_states`, repeating each entry `scaling_factor` times
            // so we keep the accumulated sponge states aligned with the taller matrix.
            scratch_states[..height]
                .par_chunks_mut(scaling_factor)
                .zip(states[..active_height].par_iter())
                .for_each(|(chunk, state)| chunk.fill(*state));

            // Copy upsampled states back to canonical buffer
            mem::swap(&mut scratch_states, &mut states);
        }

        // Absorb the rows of the matrix into the extended state vector
        absorb_matrix::<PF, PD, _, _, _, _>(&mut states[..height], matrix, sponge);

        active_height = height;
    }

    states
}

/// Incorporate one matrix's row-wise contribution into the running per-leaf states.
///
/// Semantics: given `states` of length `h = matrix.height()`, for each row index `r ∈ [0, h)`
/// update `states[r]` by absorbing the matrix row `r` into that state. In the overall tree
/// construction, callers ensure that `states` is the correct lifted view for the current matrix
/// (either the "nearest-neighbor" duplication or the "modulo" duplication across the final
/// height). This helper performs exactly one absorption round for that matrix and returns with the
/// states mutated; it does not change the lifting shape or squeeze digests.
fn absorb_matrix<PF, PD, M, H, const WIDTH: usize, const DIGEST_ELEMS: usize>(
    states: &mut [[PD::Value; WIDTH]],
    matrix: &M,
    sponge: &H,
) where
    PF: PackedValue,
    PD: PackedValue,
    M: Matrix<PF::Value>,
    H: StatefulHasher<PF::Value, [PD::Value; DIGEST_ELEMS], State = [PD::Value; WIDTH]>
        + StatefulHasher<PF, [PD; DIGEST_ELEMS], State = [PD; WIDTH]>
        + Sync,
{
    let height = matrix.height();
    assert_eq!(height, states.len());

    if height < PF::WIDTH || PF::WIDTH == 1 {
        // Scalar path: walk every final leaf state and absorb the wrapped row for this matrix.
        states
            .par_iter_mut()
            .zip(matrix.par_rows())
            .for_each(|(state, row)| {
                sponge.absorb_into(state, row);
            });
    } else {
        // SIMD path: gather → absorb wrapped packed row → scatter per chunk.
        states
            .par_chunks_mut(PF::WIDTH)
            .enumerate()
            .for_each(|(packed_idx, states_chunk)| {
                let mut packed_state: [PD; WIDTH] = PD::pack_columns(states_chunk);
                let row_idx = packed_idx * PF::WIDTH;
                let row = matrix.vertically_packed_row::<PF>(row_idx);
                sponge.absorb_into(&mut packed_state, row);
                PD::unpack_into(&packed_state, states_chunk);
            });
    }
}

/// Compress a layer of digests in a uniform Merkle tree.
///
/// Takes a layer of digests and compresses pairs into a new layer with half as many elements.
/// The layer length must be a power of two.
///
/// When the result would be smaller than the packing width, uses a pure scalar path.
/// Otherwise uses SIMD parallelization. Since both the result length and packing width are
/// powers of two, the result is always a multiple of the packing width in the SIMD path,
/// requiring no scalar fallback for remainders.
fn compress_uniform<
    P: PackedValue,
    C: PseudoCompressionFunction<[P::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[P; DIGEST_ELEMS], 2>
        + Sync,
    const DIGEST_ELEMS: usize,
>(
    prev_layer: &[[P::Value; DIGEST_ELEMS]],
    c: &C,
) -> Vec<[P::Value; DIGEST_ELEMS]> {
    assert!(
        prev_layer.len().is_power_of_two(),
        "previous layer length must be a power of 2"
    );

    let next_len = prev_layer.len() / 2;
    let default_digest = [P::Value::default(); DIGEST_ELEMS];
    let mut next_digests = vec![default_digest; next_len];

    // Use scalar path when output is too small for packing
    if next_len < P::WIDTH || P::WIDTH == 1 {
        let (prev_layer_pairs, _) = prev_layer.as_chunks::<2>();
        next_digests
            .par_iter_mut()
            .zip(prev_layer_pairs.par_iter())
            .for_each(|(next_digest, prev_layer_pair)| {
                *next_digest = c.compress(*prev_layer_pair);
            });
    } else {
        // Packed path: since next_len and P::WIDTH are both powers of 2,
        // next_len is a multiple of P::WIDTH, so no remainder handling needed.
        next_digests
            .par_chunks_exact_mut(P::WIDTH)
            .enumerate()
            .for_each(|(packed_chunk_idx, digests_chunk)| {
                let chunk_idx = packed_chunk_idx * P::WIDTH;
                let left: [P; DIGEST_ELEMS] =
                    array::from_fn(|j| P::from_fn(|k| prev_layer[2 * (chunk_idx + k)][j]));
                let right: [P; DIGEST_ELEMS] =
                    array::from_fn(|j| P::from_fn(|k| prev_layer[2 * (chunk_idx + k) + 1][j]));
                let packed_digest = c.compress([left, right]);
                P::unpack_into(&packed_digest, digests_chunk);
            });
    }
    next_digests
}

/// Validate a sequence of matrix heights for LMCS.
///
/// Requirements enforced:
/// - Non-empty sequence (at least one matrix).
/// - Every height is a power of two and non-zero.
/// - Heights are in non-decreasing order (sorted by height), so the last height is the maximum
///   `H` used by lifting.
///
/// # Panics
/// Panics if any requirement is violated.
fn validate_heights(heights: impl IntoIterator<Item = usize>) -> usize {
    let mut active_height = 0;

    for (matrix, height) in heights.into_iter().enumerate() {
        assert_ne!(height, 0, "zero height at matrix {matrix}");
        assert!(
            height.is_power_of_two(),
            "non-power-of-two height at matrix {matrix}"
        );
        assert!(height >= active_height, "matrices must be sorted by height");
        active_height = height;
    }

    assert_ne!(active_height, 0, "empty batch");
    active_height
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_miden_stateful_hasher::StatefulHasher;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use crate::tests::{
        DIGEST, F, P, RATE, Sponge, build_leaves_single, components, concatenate_matrices,
        matrix_scenarios,
    };
    use crate::utils::upsample_matrix;

    fn build_leaves_upsampled(matrices: &[RowMajorMatrix<F>], sponge: &Sponge) -> Vec<[F; DIGEST]> {
        let mut states = super::build_leaf_states_upsampled::<P, P, _, _, _, _>(matrices, sponge);
        states.iter_mut().map(|s| sponge.squeeze(s)).collect()
    }

    /// Test that upsampled lifting produces correct results:
    /// 1. Incremental lifting equals explicit lifting
    /// 2. Explicit lifting equals single-matrix concatenation baseline
    #[test]
    fn upsampled_equivalence() {
        let (sponge, _compressor) = components();
        let mut rng = SmallRng::seed_from_u64(42);

        for scenario in matrix_scenarios() {
            let matrices: Vec<RowMajorMatrix<F>> = scenario
                .into_iter()
                .map(|(h, w)| RowMajorMatrix::rand(&mut rng, h, w))
                .collect();

            let max_height = matrices.last().unwrap().height();

            // Upsampled path equivalence vs explicit upsampled lifting and single-concat baseline
            let leaves = build_leaves_upsampled(&matrices, &sponge);

            let matrices_upsampled: Vec<_> = matrices
                .iter()
                .map(|m: &RowMajorMatrix<F>| upsample_matrix(m, max_height))
                .collect();
            let leaves_lifted = build_leaves_upsampled(&matrices_upsampled, &sponge);
            assert_eq!(leaves, leaves_lifted);

            let matrix_single = concatenate_matrices::<_, RATE>(&matrices_upsampled);
            let leaves_single = build_leaves_single(&matrix_single, &sponge);
            assert_eq!(leaves, leaves_single);
        }
    }
}
