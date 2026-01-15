//! Common test fixtures for the lifted PCS crate.
//!
//! Re-exports test fixtures from `p3_miden_dev_utils` for use in tests.

pub use p3_miden_dev_utils::configs::baby_bear_poseidon2::{
    base_lmcs as test_lmcs, test_fri_mmcs, *,
};
pub use p3_miden_dev_utils::matrix::random_lde_matrix;
