use alloc::vec;
use alloc::vec::Vec;
use core::{array, mem};

use crate::utils::{PackedValueExt, aligned_widths, pad_row_to_alignment, pad_rows_to_alignment};
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
/// * `F` – scalar field element type used in both matrices and hash words.
/// * `D` – hash word element type.
/// * `M` – matrix type. Must implement [`Matrix<F>`].
/// * `DIGEST_ELEMS` – number of elements in one hash.
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
/// in sequence. The sponge applies its own padding semantics during absorption; LMCS alignment
/// only affects transcript hints.
///
/// Note: alignment padding is a convention for transcript openings and does not affect the
/// commitment. It is independent of the sponge's absorption alignment. LMCS does not enforce
/// that padded columns are zero; verifiers cannot distinguish zero padding from arbitrary values
/// unless they check those columns or constrain them elsewhere.
///
/// Equivalent single-matrix view: this commitment is equivalent to first forming a single
/// height-`N` matrix by (a) lifting every input matrix to height `N`, (b) padding each lifted
/// matrix horizontally with zero columns to reflect the sponge's absorption alignment (if any),
/// and (c) concatenating the results side-by-side. The leaf hash at index `j` is then the
/// sponge of that single concatenated matrix's row `j`. This is a conceptual view: LMCS does
/// not enforce that those padded columns are zero unless the caller checks them.
///
/// Since [`StatefulHasher`] operates on a single field type, this tree uses the same type `F`
/// for both matrix elements and hash words, unlike `MerkleTree` which can hash `F → W`.
///
/// Use [`root`](Self::root) to fetch the final commitment once the tree is built.
///
/// ## Transcript Hints
///
/// `prove_batch` streams transcript hints in the format expected by
/// [`Lmcs::open_batch`](crate::Lmcs::open_batch):
/// - For each unique query index **in sorted tree index order** (ascending, deduplicated): one
///   row per matrix (in leaf order), then `SALT_ELEMS` field elements of salt.
/// - Each row is padded with explicit zeros to the LMCS alignment.
///   This allows verifiers to absorb fixed-size chunks without special-casing
///   the final partial chunk; padding is not enforced to be zero.
/// - After all indices: missing sibling hashes, level-by-level, left-to-right, bottom-to-top.
///
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

    /// All intermediate hash layers (digest arrays), index 0 being the leaf hash layer
    /// and the last layer containing exactly one root hash.
    ///
    /// Every inner vector holds contiguous hashes. Higher layers are built by
    /// compressing pairs from the previous layer.
    ///
    /// When `log_extension > 0` (non-salted target height), `digest_layers[0]` contains
    /// `N_nat` entries (self-compressed digests), not `N` entries. The conceptual tree
    /// height is `digest_layers[0].len() << log_extension`.
    #[serde(bound(
        serialize = "[D; DIGEST_ELEMS]: Serialize",
        deserialize = "[D; DIGEST_ELEMS]: Deserialize<'de>"
    ))]
    pub(crate) digest_layers: Vec<Vec<[D; DIGEST_ELEMS]>>,

    /// Virtual self-compression layers below the stored tree.
    ///
    /// When `log_extension > 0`, this contains `log_extension` layers, each with `N_nat`
    /// entries. `virtual_layers[0]` holds the original squeezed digests, and
    /// `virtual_layers[λ]` holds `compress_self^λ(d_j)`. These are used to compute
    /// sibling hashes during `prove_batch` for the virtual levels.
    ///
    /// Empty when `log_extension == 0`.
    #[serde(bound(
        serialize = "[D; DIGEST_ELEMS]: Serialize",
        deserialize = "[D; DIGEST_ELEMS]: Deserialize<'de>"
    ))]
    pub(crate) virtual_layers: Vec<Vec<[D; DIGEST_ELEMS]>>,

    /// Number of virtual self-compression levels below the stored tree.
    ///
    /// The conceptual tree height is `digest_layers[0].len() << log_extension`.
    /// When 0, the tree is standard with no virtual extension.
    pub(crate) log_extension: usize,

    /// Salt matrix for hiding commitment. Each row contains `SALT_ELEMS` random field elements.
    /// `None` when `SALT_ELEMS = 0` (non-hiding mode).
    pub(crate) salt: Option<RowMajorMatrix<F>>,
    /// Column alignment used for transcript proofs.
    pub(crate) alignment: usize,
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
        self.digest_layers[0].len() << self.log_extension
    }

    fn leaves(&self) -> &[M] {
        &self.leaves
    }

    /// Return the upsampled rows for `index`, padded to `alignment`.
    ///
    /// Padding uses `Default::default()` and is not enforced by verification; callers
    /// that require zero padding must check these columns explicitly.
    ///
    /// Panics if `index` is out of range for the tree height.
    fn rows(&self, index: usize) -> Vec<Vec<F>> {
        let max_height = self.height();
        let alignment = self.alignment;

        let rows = self
            .leaves
            .iter()
            .map(|m| {
                let height = m.height();
                let log_scaling_factor = log2_strict_usize(max_height / height);
                let row_index = index >> log_scaling_factor;
                m.row_slice(row_index)
                    .expect("row_index must be valid after upsampling")
                    .to_vec()
            })
            .collect();

        pad_rows_to_alignment(rows, alignment)
    }

    /// Prove a batch opening and stream it into a transcript channel.
    ///
    /// Panics if any index is out of range. Rows are padded to `alignment` and those
    /// padding values are not validated by verification; callers that require zero
    /// padding must check the opened rows explicitly.
    ///
    /// Leaf openings are written in **sorted tree index order** (ascending, deduplicated).
    fn prove_batch<Ch>(&self, indices: impl IntoIterator<Item = usize>, channel: &mut Ch)
    where
        Ch: ProverChannel<F = F, Commitment = Hash<F, D, DIGEST_ELEMS>>,
    {
        use alloc::collections::BTreeSet;

        let tree_height = self.height();
        let alignment = self.alignment;

        // Collect and deduplicate indices. BTreeSet iteration yields sorted order,
        // which is critical for transcript determinism: both prover and verifier
        // must process indices in the same order.
        let unique_indices: BTreeSet<usize> = indices.into_iter().collect();

        // Stream leaf openings in sorted tree index order.
        for &index in &unique_indices {
            assert!(
                index < tree_height,
                "index {index} out of range {tree_height}"
            );
            for m in self.leaves.iter() {
                let height = m.height();
                let log_scaling_factor = log2_strict_usize(tree_height / height);
                let row_index = index >> log_scaling_factor;
                let row = m
                    .row_slice(row_index)
                    .expect("row_index must be valid after upsampling")
                    .to_vec();
                let row = pad_row_to_alignment(row, alignment);
                channel.hint_field_slice(&row);
            }
            if SALT_ELEMS > 0 {
                let salt = self.salt(index);
                channel.hint_field_slice(&salt);
            }
        }

        // Use the same sorted set for sibling traversal
        let mut known = unique_indices;
        let total_depth = log2_strict_usize(tree_height);

        // Walk up the tree level by level using the deduplicated set.
        // Virtual levels (0..log_extension) use self-compression layers;
        // stored levels (log_extension..total_depth) use digest_layers.
        for level in 0..total_depth {
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
                let sibling_pos = if have_left && !have_right {
                    Some(right_pos)
                } else if !have_left && have_right {
                    Some(left_pos)
                } else {
                    None
                };

                if let Some(sib) = sibling_pos {
                    let hash = if level < self.log_extension {
                        // Virtual level: positions in groups of 2^(k-level) share the
                        // same compress_self^level hash from the original digest.
                        let j = sib >> (self.log_extension - level);
                        self.virtual_layers[level][j]
                    } else {
                        let stored_level = level - self.log_extension;
                        self.digest_layers[stored_level][sib]
                    };
                    channel.hint_commitment(Hash::from(hash));
                }
            }

            known = parents;
        }
    }

    fn alignment(&self) -> usize {
        self.alignment
    }

    fn widths(&self) -> Vec<usize> {
        let alignment = self.alignment;
        aligned_widths(self.leaves.iter().map(|m| m.width()), alignment)
    }
}

impl<F, D, M, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    LiftedMerkleTree<F, D, M, DIGEST_ELEMS, SALT_ELEMS>
where
    F: Copy + Default + PartialEq + Send + Sync,
    D: Copy + Default + PartialEq + Send + Sync,
    M: Matrix<F>,
{
    /// Builder for creating trees with optional salt, explicit alignment, and optional target
    /// height.
    ///
    /// Preconditions:
    /// - `leaves` is non-empty and heights are powers of two.
    /// - Matrices are sorted by height (shortest to tallest).
    /// - If `log_target_height` is `Some(h)`, then `1 << h >= max_matrix_height`.
    ///
    /// When `log_target_height` is `Some(h)`, the resulting Merkle tree has `1 << h` leaves,
    /// which may be larger than the tallest matrix. The upsampling strategy depends on whether
    /// salt is present:
    ///
    /// - **With salt**: leaf sponge states are upsampled (nearest-neighbor) from the natural
    ///   height to the target height *before* absorbing salt. The salt matrix must have
    ///   `target_height` rows. This ensures that duplicated states receive independent salt,
    ///   producing unique leaf hashes.
    ///
    /// - **Without salt**: leaf states are squeezed at the natural height, producing `N_nat`
    ///   digests. Each digest undergoes `k = log2(target_height / N_nat)` rounds of
    ///   self-compression (`compress(d, d)`), yielding the base of a stored tree with `N_nat`
    ///   nodes. The bottom `k` levels of the conceptual `target_height`-leaf tree are virtual:
    ///   every sibling equals its partner, so they need not be materialized. This avoids the
    ///   cost of building a full `target_height`-leaf Merkle tree.
    ///
    /// When `log_target_height` is `None`, the tree height equals the tallest matrix height
    /// (existing behavior).
    ///
    /// `alignment` controls transcript padding only; it does not affect the commitment.
    /// LMCS does not enforce that padded columns are zero.
    ///
    /// Panics if `leaves` is empty or if the target height is smaller than the max matrix height.
    pub(crate) fn build_with_alignment<PF, PD, H, C, const WIDTH: usize>(
        h: &H,
        c: &C,
        leaves: Vec<M>,
        salt: Option<RowMajorMatrix<F>>,
        alignment: usize,
        log_target_height: Option<usize>,
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
        const { assert!(PF::WIDTH == PD::WIDTH) }
        assert!(!leaves.is_empty(), "cannot commit empty batch");
        debug_assert!(alignment > 0, "alignment must be non-zero");

        let natural_height = leaves.last().unwrap().height();
        let target_height = log_target_height
            .map(|lth| {
                let th = 1usize << lth;
                assert!(
                    th >= natural_height,
                    "target height {th} must be >= max matrix height {natural_height}"
                );
                th
            })
            .unwrap_or(natural_height);

        // Build leaf states from matrices using the sponge
        let mut leaf_states: Vec<[PD::Value; WIDTH]> =
            build_leaf_states_upsampled::<PF, PD, M, H, WIDTH, DIGEST_ELEMS>(&leaves, h);

        let (leaf_digests, log_extension, virtual_layers) = if salt.is_some() {
            // Salt path: upsample states to target height, then absorb salt, then squeeze.
            // Upsampling before salt ensures that duplicated states receive independent salt
            // values, producing unique leaf hashes. No virtual layers needed.
            if target_height > natural_height {
                leaf_states = upsample_vec(leaf_states, target_height);
            }
            if let Some(ref salt_matrix) = salt {
                debug_assert_eq!(salt_matrix.height(), leaf_states.len());
                debug_assert_eq!(salt_matrix.width(), SALT_ELEMS);
                absorb_matrix::<PF, PD, _, _, WIDTH, DIGEST_ELEMS>(
                    &mut leaf_states,
                    salt_matrix,
                    h,
                );
            }
            let digests = leaf_states
                .into_par_iter()
                .map(|state| h.squeeze(&state))
                .collect();
            (digests, 0, vec![])
        } else {
            // No-salt path: squeeze at the natural height, then use iterated
            // self-compression to virtually extend to the target height.
            //
            // Given k = log2(target_height / natural_height) extension levels:
            //   virtual_layers[λ][j] = compress_self^λ(d_j)  for λ in 0..k
            //   digest_layers[0][j]  = compress_self^k(d_j)
            //
            // The stored tree has N_nat = natural_height nodes. The bottom k levels
            // are virtual: every sibling equals its partner (self-compression property),
            // so they need not be materialized as a full N-leaf tree.
            let original_digests: Vec<[PD::Value; DIGEST_ELEMS]> = leaf_states
                .into_par_iter()
                .map(|state| h.squeeze(&state))
                .collect();

            let k = if target_height > natural_height {
                log2_strict_usize(target_height / natural_height)
            } else {
                0
            };

            if k > 0 {
                // Build virtual layers: virtual_layers[λ][j] = compress_self^λ(d_j)
                let mut vl = Vec::with_capacity(k);
                vl.push(original_digests);
                for _ in 1..k {
                    let prev = vl.last().unwrap();
                    let next: Vec<_> = prev
                        .par_iter()
                        .map(|d| c.compress([*d, *d]))
                        .collect();
                    vl.push(next);
                }

                // digest_layers[0] = compress_self^k(d_j) for each j
                let base: Vec<_> = vl
                    .last()
                    .unwrap()
                    .par_iter()
                    .map(|d| c.compress([*d, *d]))
                    .collect();

                (base, k, vl)
            } else {
                (original_digests, 0, vec![])
            }
        };

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
            virtual_layers,
            log_extension,
            salt,
            alignment: alignment.max(1),
        }
    }

    /// Build a full opening proof for a single leaf index.
    ///
    /// Rows are padded to `alignment` and LMCS does not enforce that padding is zero.
    /// Panics if `index` is out of range for the tree height.
    pub fn single_proof(&self, index: usize) -> Proof<F, Hash<F, D, DIGEST_ELEMS>, SALT_ELEMS> {
        let total_depth = log2_strict_usize(self.height());
        let mut siblings = Vec::with_capacity(total_depth);
        let mut pos = index;

        // Virtual levels: sibling = compress_self^level(d_j)
        for level in 0..self.log_extension {
            let sibling_pos = pos ^ 1;
            let j = sibling_pos >> (self.log_extension - level);
            siblings.push(Hash::from(self.virtual_layers[level][j]));
            pos >>= 1;
        }

        // Stored levels
        for layer in &self.digest_layers {
            if layer.len() == 1 {
                break;
            }
            let sibling = layer[pos ^ 1];
            siblings.push(Hash::from(sibling));
            pos >>= 1;
        }

        Proof {
            rows: self.rows(index),
            salt: self.salt(index),
            siblings,
        }
    }

    /// Column alignment used when streaming openings.
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Extract the salt for the given leaf index.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of range, or if `SALT_ELEMS > 0` but the tree was
    /// constructed without salt.
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
/// Callers must squeeze the states to obtain final leaf hashes.
///
/// Conceptually, each matrix is virtually extended to height `H` by repeating each row
/// `L = H / h` times (width unchanged), and the leaf `r` absorbs the `r`-th row from each
/// extended matrix in order. Each absorbed row is virtually padded with zeros to a multiple of the
/// hasher's padding width for absorption; see [`LiftedMerkleTree`](crate::LiftedMerkleTree) docs
/// for the equivalent single-matrix view.
///
/// Padding is implicit and not checked; callers that require zero padding must enforce
/// it elsewhere.
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
/// states mutated; it does not change the lifting shape or squeeze hashes.
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

/// Compress a layer of hashes in a uniform Merkle tree.
///
/// Takes a layer of hashes and compresses pairs into a new layer with half as many elements.
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

/// Upsample a vector to `target_len` via nearest-neighbor repetition.
///
/// Each element is repeated `target_len / vec.len()` times contiguously.
/// Requires `target_len >= vec.len()` and `target_len` to be a multiple of `vec.len()`.
fn upsample_vec<T: Copy>(vec: Vec<T>, target_len: usize) -> Vec<T> {
    let natural_len = vec.len();
    debug_assert!(target_len >= natural_len);
    debug_assert!(
        target_len % natural_len == 0,
        "target_len must be a multiple of natural_len"
    );

    if target_len == natural_len {
        return vec;
    }

    let factor = target_len / natural_len;
    let mut result = Vec::with_capacity(target_len);
    for item in vec {
        for _ in 0..factor {
            result.push(item);
        }
    }
    result
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
    use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
    use p3_miden_stateful_hasher::StatefulHasher;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use crate::tests::{DIGEST, F, P, RATE, Sponge, build_leaves_single, concatenate_matrices};
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
        let (_, sponge, _compressor) = bb::test_components();
        let mut rng = SmallRng::seed_from_u64(42);

        for scenario in p3_miden_dev_utils::fixtures::matrix_scenarios::<P>(RATE) {
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
