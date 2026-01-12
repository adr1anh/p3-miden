//! Common test fixtures for the lifted PCS crate.
//!
//! This module provides shared type aliases, constants, and helper functions
//! used across all test modules to reduce duplication and ensure consistency.

use alloc::vec;
use alloc::vec::Vec;

use p3_baby_bear::BabyBear;
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, PackedValue};
use p3_matrix::bitrev::BitReversibleMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use p3_miden_stateful_hasher::{StatefulHasher, StatefulSponge};
use p3_symmetric::TruncatedPermutation;
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;

use crate::merkle_tree::MerkleTreeLmcs;

// ============================================================================
// Type Aliases
// ============================================================================

/// Base field: BabyBear (p = 2^31 - 2^27 + 1).
pub(crate) type F = BabyBear;

/// Extension field: degree-4 extension of BabyBear for ~128-bit security.
pub(crate) type EF = BinomialExtensionField<F, 4>;

/// Packed base field for SIMD operations.
pub(crate) type P = <F as p3_field::Field>::Packing;

// ============================================================================
// Constants
// ============================================================================

/// Poseidon2 permutation width.
pub(crate) const WIDTH: usize = 16;

/// Sponge rate (elements absorbed per permutation).
pub(crate) const RATE: usize = 8;

/// Digest size in field elements.
pub(crate) const DIGEST: usize = 8;

/// Standard seed for reproducible tests.
pub(crate) const TEST_SEED: u64 = 2025;

// ============================================================================
// Cryptographic Component Types
// ============================================================================

/// Poseidon2 permutation over BabyBear.
pub(crate) type Perm = p3_baby_bear::Poseidon2BabyBear<WIDTH>;

/// Stateful sponge for hashing.
pub(crate) type Sponge = StatefulSponge<Perm, WIDTH, RATE, DIGEST>;

/// Truncated permutation for 2-to-1 compression.
pub(crate) type Compress = TruncatedPermutation<Perm, 2, DIGEST, WIDTH>;

/// Base Merkle tree LMCS over packed BabyBear.
pub(crate) type BaseLmcs = MerkleTreeLmcs<P, P, Sponge, Compress, WIDTH, DIGEST>;

/// Duplex challenger for Fiat-Shamir.
pub(crate) type Challenger = DuplexChallenger<F, Perm, WIDTH, RATE>;

/// FRI MMCS for extension field elements.
pub(crate) type FriMmcs = ExtensionMmcs<F, EF, BaseLmcs>;

// ============================================================================
// Fixture Functions
// ============================================================================

/// Create a standard challenger for Fiat-Shamir.
pub(crate) fn challenger() -> Challenger {
    let (perm, _, _) = test_components();
    Challenger::new(perm)
}

/// Create standard test components with a consistent seed.
///
/// Returns the permutation, sponge, and compressor for Merkle tree construction.
pub(crate) fn test_components() -> (Perm, Sponge, Compress) {
    let mut rng = SmallRng::seed_from_u64(TEST_SEED);
    let perm = Perm::new_from_rng_128(&mut rng);
    let sponge = Sponge::new(perm.clone());
    let compress = Compress::new(perm.clone());
    (perm, sponge, compress)
}

/// Create standard base LMCS for testing.
pub(crate) fn base_lmcs() -> BaseLmcs {
    let (_, sponge, compress) = test_components();
    MerkleTreeLmcs::new(sponge, compress)
}

/// Create sponge and compressor for Merkle tree tests.
pub(crate) fn components() -> (Sponge, Compress) {
    let (_, sponge, compress) = test_components();
    (sponge, compress)
}

/// Concatenate matrices horizontally, padding each to a multiple of RATE.
///
/// All matrices are lifted to the maximum height first.
pub(crate) fn concatenate_matrices<const R: usize>(
    matrices: &[RowMajorMatrix<F>],
) -> RowMajorMatrix<F> {
    use p3_field::PrimeCharacteristicRing;

    let max_height = matrices.last().unwrap().height();
    let width: usize = matrices.iter().map(|m| m.width().next_multiple_of(R)).sum();

    let concatenated_data: Vec<_> = (0..max_height)
        .flat_map(|idx| {
            matrices.iter().flat_map(move |m| {
                let mut row = m.row_slice(idx).unwrap().to_vec();
                let padded_width = row.len().next_multiple_of(R);
                row.resize(padded_width, F::ZERO);
                row
            })
        })
        .collect();
    RowMajorMatrix::new(concatenated_data, width)
}

/// Build leaf digests for a single matrix (used for equivalence testing).
pub(crate) fn build_leaves_single(matrix: &RowMajorMatrix<F>, sponge: &Sponge) -> Vec<[F; DIGEST]> {
    use p3_field::PrimeCharacteristicRing;

    matrix
        .rows()
        .map(|row| {
            let mut state = [F::ZERO; WIDTH];
            sponge.absorb_into(&mut state, row);
            sponge.squeeze(&state)
        })
        .collect()
}

/// Explicitly lift a matrix to the target height using nearest-neighbor upsampling.
///
/// Used for testing equivalence between incremental and explicit lifting.
pub(crate) fn lift_matrix(matrix: &RowMajorMatrix<F>, max_height: usize) -> RowMajorMatrix<F> {
    use p3_util::log2_strict_usize;

    let Dimensions { height, width } = matrix.dimensions();
    let log_scaling_factor = log2_strict_usize(max_height / height);
    let data = (0..max_height)
        .flat_map(|index| {
            let mapped_index = index >> log_scaling_factor;
            matrix.row(mapped_index).unwrap()
        })
        .collect();
    RowMajorMatrix::new(data, width)
}

/// Common matrix group scenarios for testing lifting with varying heights.
///
/// Each scenario is a list of (height, width) pairs, sorted by ascending height.
/// These cover various edge cases:
/// - Single matrices of various sizes
/// - Multiple matrices with different lift factors
/// - Edge cases around packing width boundaries
pub(crate) fn matrix_scenarios() -> Vec<Vec<(usize, usize)>> {
    // Use max(1, ...) to ensure heights are always at least 1, even when P::WIDTH=1
    // (no SIMD) on platforms like ubuntu/windows CI without AVX.
    let pack_width = P::WIDTH.max(2);
    vec![
        // Single matrices
        vec![(1, 1)],
        vec![(1, RATE - 1)],
        // Multiple heights (must be ascending)
        vec![(2, 3), (4, 5), (8, RATE)],
        vec![(1, 5), (1, 3), (2, 7), (4, 1), (8, RATE + 1)],
        // Packing boundary tests
        vec![
            (pack_width / 2, RATE - 1),
            (pack_width, RATE),
            (pack_width * 2, RATE + 3),
        ],
        vec![(pack_width, RATE + 5), (pack_width * 2, 25)],
        vec![
            (1, RATE * 2),
            (pack_width / 2, RATE * 2 - 1),
            (pack_width, RATE * 2),
            (pack_width * 2, RATE * 3 - 2),
        ],
        // Same-height matrices
        vec![(4, RATE - 1), (4, RATE), (8, RATE + 3), (8, RATE * 2)],
        // Single tall matrix
        vec![(pack_width * 2, RATE - 1)],
    ]
}

/// Create standard FRI MMCS for testing.
pub(crate) fn fri_mmcs() -> FriMmcs {
    ExtensionMmcs::new(base_lmcs())
}

/// Generate a matrix of LDE evaluations for random low-degree polynomials.
///
/// Each column is a polynomial of degree `poly_degree`, evaluated on the coset gK
/// in bit-reversed order, where g = F::GENERATOR and K is a subgroup of order `lde_size`.
///
/// The coset evaluation is computed by scaling coefficients: for f(X) = Σ c_j X^j,
/// the coset evaluations f(gX) = Σ (c_j g^j) X^j are obtained by DFT of scaled coefficients.
pub(crate) fn random_lde_matrix<V>(
    rng: &mut SmallRng,
    log_poly_degree: usize,
    log_blowup: usize,
    num_columns: usize,
    shift: F,
) -> RowMajorMatrix<V>
where
    V: BasedVectorSpace<F> + Clone + Send + Sync + Default,
    StandardUniform: Distribution<V>,
{
    let poly_degree = 1 << log_poly_degree;
    let dft = Radix2DFTSmallBatch::<F>::default();

    let evals = RowMajorMatrix::rand(rng, poly_degree, num_columns);
    let lde = dft.coset_lde_algebra_batch(evals, log_blowup, shift);
    lde.bit_reverse_rows().to_row_major_matrix()
}
