//! PCS comparison benchmarks: Lifted PCS vs Workspace TwoAdicFriPcs.
//!
//! Compares the complete open operation for both PCS implementations
//! using multiple trace groups with different heights (simulating real STARK scenarios).
//!
//! Setup uses `RELATIVE_SPECS` from bench_utils which defines 3 groups:
//! - Group 0: Main trace columns at various heights
//! - Group 1: Auxiliary/permutation columns
//! - Group 2: Quotient polynomial chunks
//!
//! Runs benchmarks for BabyBear and Goldilocks fields with Poseidon2.
//!
//! Run with:
//! ```bash
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench pcs
//!
//! # With parallelism
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench pcs --features parallel
//!
//! # Filter by field
//! cargo bench --bench pcs -- babybear
//! cargo bench --bench pcs -- goldilocks
//! ```

mod utils;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{ExtensionMmcs, Mmcs, Pcs};
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::Field;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::Matrix;
use p3_matrix::bitrev::BitReversibleMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_dev_utils::{
    BenchScenario, LOG_HEIGHTS, PARALLEL_STR, PcsScenario, RELATIVE_SPECS, criterion_config_long,
    generate_matrices_from_specs, total_elements,
};
use p3_miden_lifted::deep::DeepParams;
use p3_miden_lifted::fri::{FriFold, FriParams};
use p3_miden_lifted::pcs::{PcsConfig, prover as lifted_prover};
use utils::LmcsScenario;

// =============================================================================
// Constants
// =============================================================================

/// Log blowup factor for FRI.
const LOG_BLOWUP: usize = 2;

/// Number of FRI queries.
const NUM_QUERIES: usize = 30;

/// Log degree of final polynomial.
const LOG_FINAL_DEGREE: usize = 8;

// =============================================================================
// Scenario-specific benchmark runner
// =============================================================================

/// Run PCS benchmarks for a specific scenario.
///
/// This macro handles the complex type aliasing required by TwoAdicFriPcs and
/// the lifted PCS, while keeping the benchmark logic readable.
///
/// The macro expands to a function that benchmarks both workspace and lifted PCS
/// implementations for the given scenario type.
macro_rules! bench_scenario {
    ($scenario:ty, $lmcs_fn:expr) => {{
        |c: &mut Criterion| {
            // Type aliases for this scenario
            type S = $scenario;
            type F = <S as BenchScenario>::F;
            type EF = <S as BenchScenario>::EF;
            type ValMmcs = <S as BenchScenario>::Mmcs;
            type Challenger = <S as PcsScenario>::Challenger;
            type Dft = Radix2DitParallel<F>;
            type ChallengeMmcs = ExtensionMmcs<F, EF, ValMmcs>;
            type WorkspacePcs = TwoAdicFriPcs<F, Dft, ValMmcs, ChallengeMmcs>;

            // Lifted PCS types
            type Lmcs = <S as LmcsScenario>::Lmcs;
            type FriMmcs = ExtensionMmcs<F, EF, Lmcs>;

            let dft = Dft::default();
            let shift = F::GENERATOR;

            for &log_max_height in LOG_HEIGHTS {
                let max_lde_size = 1usize << log_max_height;
                let group_name = format!(
                    "PCS_Open/{}/{}/{}/{}",
                    max_lde_size,
                    S::FIELD_NAME,
                    S::HASH_NAME,
                    PARALLEL_STR
                );
                let mut group = c.benchmark_group(&group_name);

                // Generate test matrices
                let matrix_groups: Vec<Vec<RowMajorMatrix<F>>> =
                    generate_matrices_from_specs(RELATIVE_SPECS, log_max_height);
                group.throughput(Throughput::Elements(total_elements(&matrix_groups)));

                // =============================================================
                // Workspace TwoAdicFriPcs benchmark
                // =============================================================
                {
                    let val_mmcs = S::mmcs();
                    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
                    let fri_params = FriParameters {
                        log_blowup: LOG_BLOWUP,
                        log_final_poly_len: LOG_FINAL_DEGREE,
                        num_queries: NUM_QUERIES,
                        commit_proof_of_work_bits: 0,
                        query_proof_of_work_bits: 0,
                        mmcs: challenge_mmcs,
                    };
                    let workspace_pcs = WorkspacePcs::new(Dft::default(), val_mmcs, fri_params);

                    let commits_and_data: Vec<_> = matrix_groups
                        .iter()
                        .map(|matrices| {
                            let domains_and_evals = matrices.iter().map(|m| {
                                let domain =
                                    <WorkspacePcs as Pcs<EF, Challenger>>::natural_domain_for_degree(
                                        &workspace_pcs,
                                        m.height(),
                                    );
                                (domain, m.clone())
                            });
                            <WorkspacePcs as Pcs<EF, Challenger>>::commit(
                                &workspace_pcs,
                                domains_and_evals,
                            )
                        })
                        .collect();

                    let base_challenger = S::challenger();

                    group.bench_function(BenchmarkId::from_parameter("workspace"), |b| {
                        b.iter(|| {
                            let mut challenger = base_challenger.clone();
                            for (commitment, _) in &commits_and_data {
                                challenger.observe(*commitment);
                            }
                            let z1: EF = challenger.sample_algebra_element();
                            let z2: EF = challenger.sample_algebra_element();

                            let data_and_points: Vec<_> = commits_and_data
                                .iter()
                                .enumerate()
                                .map(|(i, (_, prover_data))| {
                                    let num_matrices = matrix_groups[i].len();
                                    let points = if i < 2 {
                                        vec![vec![z1, z2]; num_matrices]
                                    } else {
                                        vec![vec![z1]; num_matrices]
                                    };
                                    (prover_data, points)
                                })
                                .collect();

                            let (_openings, proof) =
                                <WorkspacePcs as Pcs<EF, Challenger>>::open(
                                    &workspace_pcs,
                                    black_box(data_and_points),
                                    &mut challenger,
                                );
                            black_box(proof)
                        });
                    });
                }

                // =============================================================
                // Lifted PCS benchmarks (arity 2 and 4)
                // =============================================================
                {
                    let lmcs: Lmcs = $lmcs_fn();
                    let fri_mmcs: FriMmcs = ExtensionMmcs::new($lmcs_fn());

                    // Compute LDE matrices
                    let lde_groups: Vec<Vec<_>> = matrix_groups
                        .iter()
                        .map(|matrices| {
                            matrices
                                .iter()
                                .map(|m| {
                                    let lde = dft.coset_lde_batch(m.clone(), LOG_BLOWUP, shift);
                                    lde.bit_reverse_rows().to_row_major_matrix()
                                })
                                .collect()
                        })
                        .collect();

                    let commits_and_data: Vec<_> = lde_groups
                        .iter()
                        .map(|matrices| lmcs.commit(matrices.clone()))
                        .collect();

                    let base_challenger = S::challenger();

                    for (name, fold) in [
                        ("lifted/arity2", FriFold::ARITY_2),
                        ("lifted/arity4", FriFold::ARITY_4),
                    ] {
                        let config = PcsConfig {
                            deep: DeepParams {
                                alignment: S::RATE,
                                proof_of_work_bits: 0,
                            },
                            fri: FriParams {
                                log_blowup: LOG_BLOWUP,
                                fold,
                                log_final_degree: LOG_FINAL_DEGREE,
                                proof_of_work_bits: 0,
                            },
                            num_queries: NUM_QUERIES,
                            query_proof_of_work_bits: 0,
                        };

                        group.bench_function(BenchmarkId::from_parameter(name), |b| {
                            b.iter(|| {
                                let mut challenger = base_challenger.clone();
                                for (commitment, _) in &commits_and_data {
                                    challenger.observe(*commitment);
                                }
                                let z1: EF = challenger.sample_algebra_element();
                                let z2: EF = challenger.sample_algebra_element();

                                let prover_data_refs: Vec<_> =
                                    commits_and_data.iter().map(|(_, data)| data).collect();
                                let proof = lifted_prover::open::<F, EF, _, _, _, Challenger, 2>(
                                    &lmcs,
                                    &fri_mmcs,
                                    &config,
                                    [z1, z2],
                                    prover_data_refs,
                                    &mut challenger,
                                );
                                black_box(proof)
                            });
                        });
                    }
                }

                group.finish();
            }
        }
    }};
}

// =============================================================================
// Entry point
// =============================================================================

fn bench_pcs(c: &mut Criterion) {
    use p3_miden_dev_utils::configs::{baby_bear_poseidon2 as bb, goldilocks_poseidon2 as gl};
    use p3_miden_dev_utils::{BabyBearPoseidon2, GoldilocksPoseidon2};

    // BabyBear + Poseidon2
    bench_scenario!(BabyBearPoseidon2, bb::base_lmcs)(c);

    // Goldilocks + Poseidon2
    bench_scenario!(GoldilocksPoseidon2, gl::base_lmcs)(c);
}

criterion_group! {
    name = benches;
    config = criterion_config_long();
    targets = bench_pcs
}
criterion_main!(benches);
