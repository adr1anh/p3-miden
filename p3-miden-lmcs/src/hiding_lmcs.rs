//! Hiding LMCS configuration types.

use alloc::vec::Vec;
use core::cell::RefCell;
use core::marker::PhantomData;
use p3_field::PackedValue;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_stateful_hasher::StatefulHasher;
use p3_miden_transcript::VerifierChannel;
use p3_symmetric::{Hash, PseudoCompressionFunction};
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

use crate::{LiftedMerkleTree, Lmcs, LmcsConfig, LmcsError, Proof};

/// Configuration for hiding LMCS with random salt.
///
/// This type wraps a [`LmcsConfig`] and adds an RNG for generating salt
/// during tree construction. The RNG is stored in a `RefCell` to allow
/// salt generation without `&mut self` (required by `Mmcs::commit`).
///
/// # Type Parameters
///
/// - `PF`: Packed field element type for SIMD operations.
/// - `PD`: Packed digest element type.
/// - `H`: Stateful hasher/sponge type.
/// - `C`: 2-to-1 compression function type.
/// - `R`: Random number generator type.
/// - `WIDTH`: State width for the hasher.
/// - `DIGEST`: Number of elements in a digest.
/// - `SALT`: Number of salt elements per leaf (must be > 0).
///
/// # Example
///
/// ```ignore
/// use p3_miden_lmcs::{HidingLmcsConfig, Lmcs, LmcsTree};
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// let rng = StdRng::seed_from_u64(42);
/// let config = HidingLmcsConfig::<PF, PD, _, _, _, WIDTH, DIGEST, 4>::new(sponge, compress, rng);
///
/// let tree = config.build_tree(matrices);
/// let root = tree.root();
/// ```
#[derive(Clone, Debug)]
pub struct HidingLmcsConfig<
    PF,
    PD,
    H,
    C,
    R,
    const WIDTH: usize,
    const DIGEST: usize,
    const SALT: usize,
> {
    /// Inner non-hiding config with sponge and compression.
    pub inner: LmcsConfig<PF, PD, H, C, WIDTH, DIGEST, SALT>,
    /// RNG for salt generation. Uses `RefCell` for interior mutability.
    rng: RefCell<R>,
}

impl<PF, PD, H, C, R, const WIDTH: usize, const DIGEST: usize, const SALT: usize>
    HidingLmcsConfig<PF, PD, H, C, R, WIDTH, DIGEST, SALT>
{
    /// Create a new hiding LMCS configuration.
    ///
    /// # Compile-time Error
    ///
    /// Fails to compile if `SALT == 0`. Use [`LmcsConfig`] for non-hiding commitments.
    #[inline]
    pub fn new(sponge: H, compress: C, rng: R) -> Self {
        const { assert!(SALT > 0) }
        Self {
            inner: LmcsConfig {
                sponge,
                compress,
                _phantom: PhantomData,
            },
            rng: RefCell::new(rng),
        }
    }
}

impl<PF, PD, H, C, R, const WIDTH: usize, const DIGEST: usize, const SALT: usize> Lmcs
    for HidingLmcsConfig<PF, PD, H, C, R, WIDTH, DIGEST, SALT>
where
    PF: PackedValue + Default,
    PD: PackedValue + Default,
    R: Rng + Clone,
    StandardUniform: Distribution<PF::Value>,
    H: StatefulHasher<PF::Value, [PD::Value; DIGEST], State = [PD::Value; WIDTH]>
        + StatefulHasher<PF, [PD; DIGEST], State = [PD; WIDTH]>
        + Sync,
    C: PseudoCompressionFunction<[PD::Value; DIGEST], 2>
        + PseudoCompressionFunction<[PD; DIGEST], 2>
        + Sync,
{
    type F = PF::Value;
    type Commitment = Hash<PF::Value, PD::Value, DIGEST>;
    type SingleProof = Proof<PF::Value, PD::Value, DIGEST, SALT>;
    type Tree<M: Matrix<PF::Value>> = LiftedMerkleTree<PF::Value, PD::Value, M, DIGEST, SALT>;

    fn build_tree<M: Matrix<Self::F>>(&self, leaves: Vec<M>) -> Self::Tree<M> {
        let tree_height = leaves.last().map(|m| m.height()).unwrap_or(0);
        let salt = RowMajorMatrix::rand(&mut *self.rng.borrow_mut(), tree_height, SALT);

        LiftedMerkleTree::build::<PF, PD, H, C, WIDTH>(
            &self.inner.sponge,
            &self.inner.compress,
            leaves,
            Some(salt),
        )
    }

    fn open_batch<Ch>(
        &self,
        commitment: &Self::Commitment,
        widths: &[usize],
        log_max_height: usize,
        indices: &[usize],
        channel: &mut Ch,
    ) -> Result<Vec<Vec<Vec<Self::F>>>, LmcsError>
    where
        Ch: VerifierChannel<F = Self::F, Commitment = Self::Commitment>,
    {
        self.inner
            .open_batch(commitment, widths, log_max_height, indices, channel)
    }

    fn read_batch_from_channel<Ch>(
        &self,
        widths: &[usize],
        log_max_height: usize,
        indices: &[usize],
        channel: &mut Ch,
    ) -> Result<Vec<Self::SingleProof>, LmcsError>
    where
        Ch: VerifierChannel<F = Self::F, Commitment = Self::Commitment>,
    {
        self.inner
            .read_batch_from_channel(widths, log_max_height, indices, channel)
    }
}
