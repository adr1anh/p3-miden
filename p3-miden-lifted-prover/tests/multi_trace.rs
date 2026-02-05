//! Tests for multi-trace functionality in lifted STARK prover.
//!
//! Tests the cyclic extension and numerator accumulation helpers,
//! as well as the prove_multi function.

use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
use p3_miden_lifted_prover::{accumulate_numerators, cyclic_extend};

type F = bb::F;
type EF = bb::EF;

// ============================================================================
// Cyclic Extension Tests
// ============================================================================

#[test]
fn test_cyclic_extend_identity() {
    // Same size should return a copy
    let values = vec![
        F::from_u32(1),
        F::from_u32(2),
        F::from_u32(3),
        F::from_u32(4),
    ];
    let extended = cyclic_extend(&values, 4);
    assert_eq!(extended, values);
}

#[test]
fn test_cyclic_extend_2x() {
    // 4 -> 8 should cycle twice
    let values = vec![
        F::from_u32(1),
        F::from_u32(2),
        F::from_u32(3),
        F::from_u32(4),
    ];
    let extended = cyclic_extend(&values, 8);
    assert_eq!(extended.len(), 8);
    assert_eq!(extended[0], F::from_u32(1));
    assert_eq!(extended[1], F::from_u32(2));
    assert_eq!(extended[2], F::from_u32(3));
    assert_eq!(extended[3], F::from_u32(4));
    assert_eq!(extended[4], F::from_u32(1)); // Cycle starts again
    assert_eq!(extended[5], F::from_u32(2));
    assert_eq!(extended[6], F::from_u32(3));
    assert_eq!(extended[7], F::from_u32(4));
}

#[test]
fn test_cyclic_extend_4x() {
    // 2 -> 8 should cycle 4 times
    let values = vec![F::from_u32(10), F::from_u32(20)];
    let extended = cyclic_extend(&values, 8);
    assert_eq!(extended.len(), 8);
    for i in 0..8 {
        assert_eq!(extended[i], values[i % 2]);
    }
}

#[test]
#[should_panic(expected = "must be a multiple")]
fn test_cyclic_extend_non_multiple_panics() {
    let values = vec![F::ONE, F::TWO, F::from_u32(3)];
    // 3 does not divide 8
    let _ = cyclic_extend(&values, 8);
}

// ============================================================================
// Accumulate Numerators Tests
// ============================================================================

#[test]
fn test_accumulate_empty() {
    let numerators: Vec<Vec<EF>> = vec![];
    let beta = EF::from(F::from_u32(7));
    let result = accumulate_numerators(numerators, beta);
    assert!(result.is_empty());
}

#[test]
fn test_accumulate_single() {
    // Single numerator should just be returned as-is
    let num = vec![
        EF::from(F::from_u32(1)),
        EF::from(F::from_u32(2)),
        EF::from(F::from_u32(3)),
        EF::from(F::from_u32(4)),
    ];
    let beta = EF::from(F::from_u32(7));
    let result = accumulate_numerators(vec![num.clone()], beta);
    assert_eq!(result, num);
}

#[test]
fn test_accumulate_two_same_size() {
    // Two numerators of same size: acc = acc * beta + num_1
    let num_0 = vec![EF::from(F::from_u32(1)), EF::from(F::from_u32(2))];
    let num_1 = vec![EF::from(F::from_u32(10)), EF::from(F::from_u32(20))];
    let beta = EF::from(F::from_u32(3));

    let result = accumulate_numerators(vec![num_0.clone(), num_1.clone()], beta);
    assert_eq!(result.len(), 2);

    // result[i] = num_0[i] * beta + num_1[i]
    for i in 0..2 {
        let expected = num_0[i] * beta + num_1[i];
        assert_eq!(result[i], expected);
    }
}

#[test]
fn test_accumulate_two_different_sizes() {
    // First numerator size 2, second size 4
    // First should be extended to size 4, then folded
    let num_0 = vec![EF::from(F::from_u32(1)), EF::from(F::from_u32(2))];
    let num_1 = vec![
        EF::from(F::from_u32(10)),
        EF::from(F::from_u32(20)),
        EF::from(F::from_u32(30)),
        EF::from(F::from_u32(40)),
    ];
    let beta = EF::from(F::from_u32(5));

    let result = accumulate_numerators(vec![num_0.clone(), num_1.clone()], beta);
    assert_eq!(result.len(), 4);

    // Extended num_0 is [1, 2, 1, 2]
    // result[i] = extended_0[i] * beta + num_1[i]
    let extended_0 = cyclic_extend(&num_0, 4);
    for i in 0..4 {
        let expected = extended_0[i] * beta + num_1[i];
        assert_eq!(result[i], expected);
    }
}

#[test]
fn test_accumulate_three_ascending() {
    // Three numerators: size 2, size 4, size 8
    let num_0 = vec![EF::from(F::ONE), EF::from(F::TWO)];
    let num_1 = vec![
        EF::from(F::from_u32(10)),
        EF::from(F::from_u32(20)),
        EF::from(F::from_u32(30)),
        EF::from(F::from_u32(40)),
    ];
    let num_2 = vec![
        EF::from(F::from_u32(100)),
        EF::from(F::from_u32(200)),
        EF::from(F::from_u32(300)),
        EF::from(F::from_u32(400)),
        EF::from(F::from_u32(500)),
        EF::from(F::from_u32(600)),
        EF::from(F::from_u32(700)),
        EF::from(F::from_u32(800)),
    ];
    let beta = EF::from(F::from_u32(2));

    let result = accumulate_numerators(vec![num_0.clone(), num_1.clone(), num_2.clone()], beta);
    assert_eq!(result.len(), 8);

    // Step 1: acc = num_0 (size 2)
    // Step 2: acc = cyclic_extend(acc, 4) * beta + num_1 (size 4)
    let extended_0 = cyclic_extend(&num_0, 4);
    let acc_after_1: Vec<EF> = extended_0
        .iter()
        .zip(num_1.iter())
        .map(|(&a, &n)| a * beta + n)
        .collect();

    // Step 3: acc = cyclic_extend(acc, 8) * beta + num_2 (size 8)
    let extended_1 = cyclic_extend(&acc_after_1, 8);
    let expected: Vec<EF> = extended_1
        .iter()
        .zip(num_2.iter())
        .map(|(&a, &n)| a * beta + n)
        .collect();

    assert_eq!(result, expected);
}

// ============================================================================
// Extension Field Dimension Test
// ============================================================================

#[test]
fn test_ef_dimension() {
    // Verify EF dimension is 4 for BabyBear quartic extension
    assert_eq!(<EF as BasedVectorSpace<F>>::DIMENSION, 4);
}
