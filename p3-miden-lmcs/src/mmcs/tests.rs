//! MMCS integration tests.

use alloc::vec::Vec;

use p3_commit::{BatchOpeningRef, Mmcs};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use p3_miden_dev_utils::configs::baby_bear_poseidon2::{
    Compress, DIGEST, F, P, Sponge, WIDTH, test_challenger, test_components,
};
use p3_miden_transcript::{ProverTranscript, VerifierTranscript};
use p3_util::log2_strict_usize;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::{BatchProof, HidingLmcsConfig, Lmcs, LmcsConfig, LmcsError, LmcsTree};

type BaseMmcs = LmcsConfig<P, P, Sponge, Compress, WIDTH, DIGEST>;
type RowMatrix = RowMajorMatrix<F>;
const SALT: usize = 4;
type HidingMmcs = HidingLmcsConfig<P, P, Sponge, Compress, SmallRng, WIDTH, DIGEST, SALT>;
type BaseTree = <BaseMmcs as Lmcs>::Tree<RowMatrix>;
type HidingTree = <HidingMmcs as Lmcs>::Tree<RowMatrix>;

const BASE_SHAPES: &[(usize, usize)] = &[(4, 5), (8, 3)];

fn mmcs() -> BaseMmcs {
    let (_, sponge, compress) = test_components();
    LmcsConfig::new_aligned(sponge, compress)
}

fn hiding_mmcs(rng: SmallRng) -> HidingMmcs {
    let (_, sponge, compress) = test_components();
    HidingLmcsConfig::new_aligned(sponge, compress, rng)
}

fn components() -> (Sponge, Compress) {
    let (_, sponge, compress) = test_components();
    (sponge, compress)
}

fn random_matrices(rng: &mut SmallRng, shapes: &[(usize, usize)]) -> Vec<RowMatrix> {
    shapes
        .iter()
        .map(|&(h, w)| RowMajorMatrix::rand(rng, h, w))
        .collect()
}

fn dimensions_from_tree<C, T>(tree: &T) -> Vec<Dimensions>
where
    T: LmcsTree<F, C, RowMatrix>,
{
    tree.leaves()
        .iter()
        .zip(tree.widths())
        .map(|(m, width)| Dimensions {
            width,
            height: m.height(),
        })
        .collect()
}

fn tree_context<C, T>(tree: &T) -> (C, Vec<Dimensions>, usize)
where
    T: LmcsTree<F, C, RowMatrix>,
{
    let commitment = tree.root();
    let dimensions = dimensions_from_tree(tree);
    let index = tree.height() / 2;
    (commitment, dimensions, index)
}

fn base_tree(seed: u64, shapes: &[(usize, usize)]) -> (BaseMmcs, BaseTree) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let matrices = random_matrices(&mut rng, shapes);
    let mmcs = mmcs();
    let tree = mmcs.build_tree(matrices);
    (mmcs, tree)
}

fn hiding_tree(seed: u64, shapes: &[(usize, usize)], salt_seed: u64) -> (HidingMmcs, HidingTree) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let matrices = random_matrices(&mut rng, shapes);
    let mmcs = hiding_mmcs(SmallRng::seed_from_u64(salt_seed));
    let tree = mmcs.build_tree(matrices);
    (mmcs, tree)
}

#[test]
fn extract_proofs_roundtrip() {
    let (sponge, compress) = components();
    let mmcs = mmcs();

    let test = |seed: u64, matrices: &[(usize, usize)], indices: &[usize]| {
        let mut rng = SmallRng::seed_from_u64(seed);
        let matrices = random_matrices(&mut rng, matrices);

        let tree = mmcs.build_tree(matrices);
        let widths = tree.widths();
        let log_max_height = log2_strict_usize(tree.height());
        let (commitment, dimensions, _) = tree_context(&tree);

        let mut prover_channel = ProverTranscript::new(test_challenger());
        tree.prove_batch(indices, &mut prover_channel);
        let transcript = prover_channel.into_data();

        let mut verifier_channel = VerifierTranscript::from_data(test_challenger(), &transcript);
        let batch = BatchProof::<F, F, DIGEST>::read_from_channel(
            &widths,
            log_max_height,
            indices,
            &mut verifier_channel,
        )
        .expect("batch proof should parse from transcript");
        let proofs = batch
            .single_proofs::<Sponge, Compress, WIDTH>(&sponge, &compress, &widths, log_max_height)
            .expect("batch proof should reconstruct proofs");
        assert_eq!(proofs.len(), indices.len());

        for (pos, &idx) in indices.iter().enumerate() {
            let proof = proofs.get(&idx).expect("proof for index");
            let proof_expected = tree.single_proof(idx);
            assert_eq!(
                proof, &proof_expected,
                "path mismatch for index {idx} at position {pos}"
            );

            let opening_proof = (proof.salt, proof.siblings.clone());
            let batch_opening = BatchOpeningRef {
                opened_values: &proof.rows,
                opening_proof: &opening_proof,
            };
            Mmcs::verify_batch(&mmcs, &commitment, &dimensions, idx, batch_opening)
                .expect("proof should verify");

            let expected_rows = tree.rows(idx);
            for (matrix_idx, expected_row) in expected_rows.iter().enumerate() {
                assert_eq!(
                    proof.rows[matrix_idx].as_slice(),
                    expected_row.as_slice(),
                    "row mismatch for index {idx}, matrix {matrix_idx}"
                );
            }
        }
    };

    test(42, &[(4, 3), (8, 5)], &[0, 1, 5]); // adjacent + non-adjacent
    test(55, &[(4, 2), (16, 3)], &[0, 5, 10, 15]); // larger tree
}

#[test]
fn mmcs_roundtrip_non_hiding() {
    let (mmcs, tree) = base_tree(10, BASE_SHAPES);
    let (commitment, dimensions, index) = tree_context(&tree);

    let batch_opening = Mmcs::open_batch(&mmcs, index, &tree);
    Mmcs::verify_batch(
        &mmcs,
        &commitment,
        &dimensions,
        index,
        (&batch_opening).into(),
    )
    .expect("mmcs verify should succeed");

    let expected_rows = tree.rows(index);
    for (row, expected_row) in batch_opening.opened_values.iter().zip(expected_rows.iter()) {
        assert_eq!(row.as_slice(), expected_row.as_slice());
    }
}

#[test]
fn mmcs_roundtrip_hiding() {
    let (mmcs, tree) = hiding_tree(11, BASE_SHAPES, 12);
    let (commitment, dimensions, index) = tree_context(&tree);

    let batch_opening = Mmcs::open_batch(&mmcs, index, &tree);
    Mmcs::verify_batch(
        &mmcs,
        &commitment,
        &dimensions,
        index,
        (&batch_opening).into(),
    )
    .expect("mmcs verify should succeed");
}

#[test]
fn mmcs_verify_rejects_wrong_row_count() {
    let (mmcs, tree) = base_tree(20, BASE_SHAPES);
    let (commitment, dimensions, index) = tree_context(&tree);

    let batch_opening = Mmcs::open_batch(&mmcs, index, &tree);
    let opened_values = &batch_opening.opened_values[..batch_opening.opened_values.len() - 1];
    let batch_opening_ref = BatchOpeningRef {
        opened_values,
        opening_proof: &batch_opening.opening_proof,
    };

    assert_eq!(
        Mmcs::verify_batch(&mmcs, &commitment, &dimensions, index, batch_opening_ref),
        Err(LmcsError::InvalidProof)
    );
}

#[test]
fn mmcs_verify_rejects_wrong_row_width() {
    let (mmcs, tree) = base_tree(21, BASE_SHAPES);
    let (commitment, dimensions, index) = tree_context(&tree);

    let batch_opening = Mmcs::open_batch(&mmcs, index, &tree);
    let mut bad_rows = batch_opening.opened_values.clone();
    bad_rows[0].pop();

    let opening_proof = batch_opening.opening_proof.clone();
    let batch_opening_ref = BatchOpeningRef {
        opened_values: &bad_rows,
        opening_proof: &opening_proof,
    };

    assert_eq!(
        Mmcs::verify_batch(&mmcs, &commitment, &dimensions, index, batch_opening_ref),
        Err(LmcsError::InvalidProof)
    );
}

#[test]
fn mmcs_verify_rejects_wrong_siblings_len() {
    let (mmcs, tree) = base_tree(22, BASE_SHAPES);
    let (commitment, dimensions, index) = tree_context(&tree);

    let batch_opening = Mmcs::open_batch(&mmcs, index, &tree);
    let mut opening_proof = batch_opening.opening_proof.clone();
    opening_proof.1.pop();

    let batch_opening_ref = BatchOpeningRef {
        opened_values: &batch_opening.opened_values,
        opening_proof: &opening_proof,
    };

    assert_eq!(
        Mmcs::verify_batch(&mmcs, &commitment, &dimensions, index, batch_opening_ref),
        Err(LmcsError::InvalidProof)
    );
}

#[test]
fn mmcs_verify_rejects_root_mismatch() {
    let (mmcs, tree) = base_tree(23, BASE_SHAPES);
    let (commitment, dimensions, index) = tree_context(&tree);

    let batch_opening = Mmcs::open_batch(&mmcs, index, &tree);
    let mut bad_rows = batch_opening.opened_values.clone();
    bad_rows[0][0] += F::ONE;

    let opening_proof = batch_opening.opening_proof.clone();
    let batch_opening_ref = BatchOpeningRef {
        opened_values: &bad_rows,
        opening_proof: &opening_proof,
    };

    assert_eq!(
        Mmcs::verify_batch(&mmcs, &commitment, &dimensions, index, batch_opening_ref),
        Err(LmcsError::RootMismatch)
    );
}
