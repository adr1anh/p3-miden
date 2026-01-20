//! Shared utilities for LMCS benchmarks.

use p3_commit::Mmcs;
use p3_keccak::KeccakF;
use p3_miden_dev_utils::configs::baby_bear_keccak as bb_keccak;
use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
use p3_miden_dev_utils::configs::goldilocks_keccak as gl_keccak;
use p3_miden_dev_utils::configs::goldilocks_poseidon2 as gl;
use p3_miden_dev_utils::{
    BabyBearKeccak, BabyBearPoseidon2, BenchScenario, GoldilocksKeccak, GoldilocksPoseidon2,
};
use p3_miden_lmcs::LmcsMmcs;
use p3_miden_stateful_hasher::{SerializingStatefulSponge, StatefulSponge};

// =============================================================================
// Poseidon2 LMCS types (re-exported from dev-utils)
// =============================================================================

pub type BabyBearLmcs = bb::BaseLmcs;
pub type GoldilocksLmcs = gl::BaseLmcs;

// =============================================================================
// Keccak LMCS types
// =============================================================================

type KeccakStatefulSponge = StatefulSponge<KeccakF, 25, 17, 4>;

pub type BabyBearKeccakLmcs = LmcsMmcs<
    bb_keccak::F,
    u64,
    SerializingStatefulSponge<KeccakStatefulSponge>,
    bb_keccak::KeccakCompress,
    25,
    4,
>;

pub type GoldilocksKeccakLmcs = LmcsMmcs<
    gl_keccak::F,
    u64,
    SerializingStatefulSponge<KeccakStatefulSponge>,
    gl_keccak::KeccakCompress,
    25,
    4,
>;

// =============================================================================
// LmcsScenario trait
// =============================================================================

/// Trait for creating LMCS instances from BenchScenario.
pub trait LmcsScenario: BenchScenario {
    type Lmcs: Mmcs<Self::F>;

    fn lmcs() -> Self::Lmcs;
}

// =============================================================================
// Poseidon2 implementations
// =============================================================================

impl LmcsScenario for BabyBearPoseidon2 {
    type Lmcs = BabyBearLmcs;

    fn lmcs() -> Self::Lmcs {
        bb::base_lmcs()
    }
}

impl LmcsScenario for GoldilocksPoseidon2 {
    type Lmcs = GoldilocksLmcs;

    fn lmcs() -> Self::Lmcs {
        gl::base_lmcs()
    }
}

// =============================================================================
// Keccak implementations
// =============================================================================

impl LmcsScenario for BabyBearKeccak {
    type Lmcs = BabyBearKeccakLmcs;

    fn lmcs() -> Self::Lmcs {
        let stateful = KeccakStatefulSponge::new(KeccakF {});
        let mmcs_sponge = bb_keccak::KeccakMmcsSponge::new(KeccakF {});
        LmcsMmcs::new(
            SerializingStatefulSponge::new(stateful),
            bb_keccak::KeccakCompress::new(mmcs_sponge),
        )
    }
}

impl LmcsScenario for GoldilocksKeccak {
    type Lmcs = GoldilocksKeccakLmcs;

    fn lmcs() -> Self::Lmcs {
        let stateful = KeccakStatefulSponge::new(KeccakF {});
        let mmcs_sponge = gl_keccak::KeccakMmcsSponge::new(KeccakF {});
        LmcsMmcs::new(
            SerializingStatefulSponge::new(stateful),
            gl_keccak::KeccakCompress::new(mmcs_sponge),
        )
    }
}
