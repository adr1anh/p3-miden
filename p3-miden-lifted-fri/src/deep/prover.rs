use alloc::vec::Vec;
use core::iter::zip;

use super::DeepParams;
use super::interpolate::PointQuotients;
use crate::deep::BatchedEvals;
use crate::utils::PackedFieldExtensionExt;
use p3_challenger::CanSample;
use p3_field::{
    ExtensionField, Field, FieldArray, PackedFieldExtension, PackedValue, TwoAdicField, dot_product,
};
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_miden_transcript::ProverChannel;

/// The DEEP quotient `Q(X)` evaluated over the LDE domain.
///
/// Combines all polynomial evaluation claims into a single low-degree polynomial.
pub struct DeepPoly<EF> {
    /// The DEEP quotient polynomial evaluated over the domain.
    /// `deep_evals[i]` is the evaluation at the i-th domain point (bit-reversed order).
    pub(crate) deep_evals: Vec<EF>,
}

impl<EF> DeepPoly<EF> {
    /// Construct `Q(X)` from committed matrices and batched evaluations at N opening points.
    ///
    /// # Arguments
    /// - `batched_evals`: Indexed as `batched_evals[group_idx][matrix_idx]` with `FieldArray<N>` per column.
    /// - `quotient`: Precomputed `1/(zⱼ - xᵢ)` for all opening points zⱼ and domain points xᵢ.
    pub fn new<F, M, const N: usize, Ch>(
        params: &DeepParams,
        matrices_groups: &[Vec<&M>],
        batched_evals: &BatchedEvals<EF, N>,
        quotient: &PointQuotients<F, EF, N>,
        channel: &mut Ch,
    ) -> Self
    where
        F: TwoAdicField,
        EF: ExtensionField<F>,
        M: Matrix<F>,
        Ch: ProverChannel<F = F> + CanSample<F>,
    {
        // Validate: one eval group per matrix group
        let batched_groups = batched_evals.groups();
        assert_eq!(
            batched_groups.len(),
            matrices_groups.len(),
            "batched_evals should have one group per matrix group"
        );

        // Validate: each group's matrix count matches
        for (evals_group, matrices) in zip(batched_groups, matrices_groups) {
            assert_eq!(
                evals_group.num_matrices(),
                matrices.len(),
                "batched_evals matrix count must match matrices count"
            );
        }

        // 1. Observe evaluations into transcript in point-major order.
        batched_evals.write_to_channel(params.alignment, channel);

        // 2. Grind for proof-of-work witness
        let _pow_witness = channel.grind(params.proof_of_work_bits);

        // 3. Sample DEEP challenges
        let challenge_columns: EF = channel.sample_algebra_element();
        let challenge_points: EF = channel.sample_algebra_element();

        let w = F::Packing::WIDTH;
        let point_quotient = &quotient.point_quotient;
        let n = point_quotient.len();

        let group_sizes: Vec<usize> = matrices_groups.iter().map(|g| g.len()).collect();
        let widths: Vec<usize> = matrices_groups
            .iter()
            .flat_map(|g| g.iter().map(|m| m.width()))
            .collect();

        let coeffs_columns: Vec<Vec<EF>> =
            derive_coeffs_from_challenge(&widths, challenge_columns, params.alignment);

        // Negate coefficients so inner loop computes f_reduced(z) - f_reduced(X) via addition
        let neg_column_coeffs: Vec<Vec<EF>> = coeffs_columns
            .iter()
            .map(|c| c.iter().copied().map(EF::neg).collect())
            .collect();

        // Compute -f_reduced(X) = -Σᵢ αⁱ · fᵢ(X) over the LDE domain.
        let mut neg_column_coeffs_iter = neg_column_coeffs.iter();
        let neg_f_reduced = zip(matrices_groups, &group_sizes)
            .map(|(matrices_group, &size)| {
                let group_coeffs: Vec<&Vec<EF>> =
                    neg_column_coeffs_iter.by_ref().take(size).collect();
                accumulate_matrices(matrices_group, &group_coeffs)
            })
            .reduce(|mut acc, next| {
                debug_assert_eq!(acc.len(), next.len());
                acc.par_chunks_mut(w).zip(next.par_chunks(w)).for_each(
                    |(acc_chunk, next_chunk)| {
                        EF::add_slices(acc_chunk, next_chunk);
                    },
                );
                acc
            })
            .unwrap_or_else(|| EF::zero_vec(n));

        // Pre-compute f_reduced(zⱼ) for all N points
        // Structure: batched_evals.groups()[group_idx][matrix_idx][col_idx] is FieldArray<EF, N>
        let f_reduced_at_points: FieldArray<EF, N> = {
            let coeffs_flat = coeffs_columns
                .iter()
                .flatten()
                .copied()
                .map(FieldArray::from);
            let evals_flat = batched_evals
                .groups()
                .iter()
                .flat_map(|group| group.iter_evals())
                .copied();
            dot_product(coeffs_flat, evals_flat)
        };

        // Pre-compute βʲ for all N points
        let point_coeffs: [EF; N] = core::array::from_fn(|j| {
            if j == 0 {
                EF::ONE
            } else {
                challenge_points.exp_u64(j as u64)
            }
        });

        // Q(X) = Σⱼ βʲ · (f_reduced(zⱼ) - f_reduced(X)) · 1/(zⱼ - X)
        let mut deep_evals = EF::zero_vec(n);

        if w == 1 {
            // Scalar path: use par_iter directly to avoid chunking overhead
            deep_evals
                .par_iter_mut()
                .zip(neg_f_reduced.par_iter())
                .zip(point_quotient.par_iter())
                .for_each(|((acc, &neg), q)| {
                    // First point (j=0) has coefficient β⁰ = 1
                    let mut result = q[0] * (f_reduced_at_points[0] + neg);
                    // Remaining points multiply by βʲ
                    for j in 1..N {
                        result += point_coeffs[j] * q[j] * (f_reduced_at_points[j] + neg);
                    }
                    *acc = result;
                });
        } else {
            // Packed path: use chunks for SIMD vectorization
            // Pre-broadcast scalars to packed values (done once, outside hot loop)
            let f_reduced_packed: [EF::ExtensionPacking; N] =
                f_reduced_at_points.0.map(EF::ExtensionPacking::from);
            let point_coeffs_packed: [EF::ExtensionPacking; N] =
                point_coeffs.map(EF::ExtensionPacking::from);

            deep_evals
                .par_chunks_exact_mut(w)
                .zip(neg_f_reduced.par_chunks_exact(w))
                .zip(point_quotient.par_chunks_exact(w))
                .for_each(|((acc_chunk, neg_chunk), q_chunk)| {
                    let neg_p = EF::ExtensionPacking::from_ext_slice(neg_chunk);

                    // Transpose quotients: q_chunk[lane][point] -> q_packed[point] packs all lanes
                    let q_packed: [EF::ExtensionPacking; N] =
                        EF::ExtensionPacking::pack_ext_columns(FieldArray::as_raw_slice(q_chunk));

                    // First point (j=0) has coefficient β⁰ = 1, compute directly
                    let mut result_p = q_packed[0] * (f_reduced_packed[0] + neg_p);

                    // Remaining points (j>0) multiply by βʲ
                    for j in 1..N {
                        result_p +=
                            point_coeffs_packed[j] * q_packed[j] * (f_reduced_packed[j] + neg_p);
                    }
                    result_p.to_ext_slice(acc_chunk);
                });
        }

        Self { deep_evals }
    }
}

/// Accumulate `f_reduced(X) = Σᵢ αⁱ · fᵢ(X)` across matrices of varying heights.
///
/// In bit-reversed order, lifting `f(X)` to `f(Xʳ)` repeats each value r times.
/// We exploit this: when crossing a height boundary, upsample by repeating entries,
/// then continue accumulating. Matrices must be sorted by ascending height.
fn accumulate_matrices<F: Field, EF: ExtensionField<F>, M: Matrix<F>, C: AsRef<[EF]>>(
    matrices: &[&M],
    coeffs: &[C],
) -> Vec<EF> {
    let n = matrices.last().unwrap().height();

    let mut acc = EF::zero_vec(n);
    let mut scratch = EF::zero_vec(n);

    let mut active_height = matrices.first().unwrap().height();

    for (&matrix, coeffs) in zip(matrices, coeffs) {
        let coeffs = coeffs.as_ref();
        let height = matrix.height();
        debug_assert!(
            height.is_power_of_two(),
            "matrix height must be a power of two"
        );
        debug_assert!(
            matrix.width() <= coeffs.len(),
            "matrix width {} exceeds coeffs length {}",
            matrix.width(),
            coeffs.len()
        );

        // Upsample: [a, b] → [a, a, b, b] when height doubles
        if height > active_height {
            let scaling_factor = height / active_height;
            scratch[..height]
                .par_chunks_mut(scaling_factor)
                .zip(acc[..active_height].par_iter())
                .for_each(|(chunk, &val)| chunk.fill(val));
            acc[..height].swap_with_slice(&mut scratch[..height]);
        }

        // SIMD path using horizontal packing
        // Pack coefficients: group WIDTH coefficients into each ExtensionPacking
        let w = F::Packing::WIDTH;
        let packed_coeffs: Vec<EF::ExtensionPacking> = coeffs
            .chunks(w)
            .map(|chunk| {
                if chunk.len() == w {
                    EF::ExtensionPacking::from_ext_slice(chunk)
                } else {
                    // Pad with zeros for the last chunk
                    let mut padded = EF::zero_vec(w);
                    padded[..chunk.len()].copy_from_slice(chunk);
                    EF::ExtensionPacking::from_ext_slice(&padded)
                }
            })
            .collect();

        matrix
            .rowwise_packed_dot_product::<EF>(&packed_coeffs)
            .zip(acc[..height].par_iter_mut())
            .for_each(|(dot_result, acc_val)| {
                *acc_val += dot_result;
            });

        active_height = height;
    }

    acc
}

/// Derive coefficients `[αⁿ⁻¹, ..., α, 1]` (reversed) for batching.
///
/// Reversed order enables Horner evaluation: the verifier processes values
/// left-to-right computing `α·acc + val`, which produces `Σ αⁿ⁻¹⁻ⁱ·vᵢ`.
///
/// # Alignment
///
/// Each matrix's coefficient range is padded to a multiple of `alignment`.
/// This is equivalent to an implementation that:
/// 1. Pads each row with zeros to the alignment width
/// 2. Computes the linear combination including those zeros
///
/// The zeros don't affect the sum, but they do affect coefficient indexing.
/// By aligning indices, we ensure the prover's explicit coefficients match
/// the verifier's Horner reduction (which skips the implicit zeros via `α^gap`).
fn derive_coeffs_from_challenge<EF: Field>(
    widths: &[usize],
    challenge: EF,
    alignment: usize,
) -> Vec<Vec<EF>> {
    let total: usize = widths.iter().map(|w| w.next_multiple_of(alignment)).sum();
    let all_powers: Vec<EF> = challenge.powers().collect_n(total);
    let rev_powers_iter = &mut all_powers.into_iter().rev();

    widths
        .iter()
        .map(|&width| {
            let padded = width.next_multiple_of(alignment);
            let mut coeffs: Vec<EF> = rev_powers_iter.take(padded).collect();
            coeffs.truncate(width); // drop alignment padding
            coeffs
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::derive_coeffs_from_challenge;
    use crate::deep::utils::reduce_with_powers_from;
    use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, dot_product};

    use crate::tests::{EF, F};

    /// `reduce_with_powers` (Horner) must match explicit `derive_coeffs` + dot product.
    #[test]
    fn reduce_evals_matches_reduce_with_powers() {
        let c: EF = EF::from_u64(2);
        let alignment = 3;
        let widths = [2usize, 3];
        let rows: Vec<Vec<F>> = vec![
            vec![F::from_u64(1), F::from_u64(2)],
            vec![F::from_u64(3), F::from_u64(4), F::from_u64(5)],
        ];

        let coeffs = derive_coeffs_from_challenge(&widths, c, alignment);

        // Explicit coefficient sum: Σᵢ coeffs[i] · rows[i]
        let explicit: EF = dot_product(
            coeffs.iter().flatten().copied(),
            rows.iter().flatten().copied(),
        );

        // Horner using reduce_with_powers (same as used in verifier)
        let horner: EF = reduce_with_powers(rows.iter().map(|r| r.as_slice()), c, alignment);

        assert_eq!(explicit, horner);
    }

    /// Alignment: coeffs match Horner for various width/alignment combos.
    #[test]
    fn derive_coeffs_alignment() {
        let c: EF = EF::from_u64(7);
        let alignment = 4;
        let widths = [3usize, 5, 2];

        let coeffs = derive_coeffs_from_challenge(&widths, c, alignment);

        // Verify lengths match widths
        assert_eq!(coeffs[0].len(), 3);
        assert_eq!(coeffs[1].len(), 5);
        assert_eq!(coeffs[2].len(), 2);

        // Verify this matches Horner reduction with arbitrary test data
        let rows: Vec<Vec<F>> = vec![
            vec![F::from_u64(10), F::from_u64(20), F::from_u64(30)],
            vec![
                F::from_u64(1),
                F::from_u64(2),
                F::from_u64(3),
                F::from_u64(4),
                F::from_u64(5),
            ],
            vec![F::from_u64(100), F::from_u64(200)],
        ];

        let explicit: EF = dot_product(
            coeffs.iter().flatten().copied(),
            rows.iter().flatten().copied(),
        );
        let horner: EF = reduce_with_powers(rows.iter().map(|r| r.as_slice()), c, alignment);

        assert_eq!(explicit, horner);
    }

    /// Horner reduction: computes `Σᵢ αⁿ⁻¹⁻ⁱ · vᵢ` via left-to-right accumulation.
    ///
    /// For each value v, computes `acc = α·acc + v`. The reversed coefficient order
    /// (from [`super::prover::derive_coeffs_from_challenge`]) makes this produce the
    /// same result as explicit `Σᵢ coeffs[i] · vals[i]`.
    ///
    /// # Alignment
    ///
    /// After each slice, multiplies by `α^gap` where `gap` pads the slice length to
    /// the next multiple of `alignment`. This is equivalent to:
    /// - Padding each row with zeros to the alignment width
    /// - Including those zeros in the Horner accumulation
    ///
    /// An alternative implementation could materialize zero-padded rows; this approach
    /// achieves the same result without allocating the padding.
    pub fn reduce_with_powers<'a, F, EF>(
        slices: impl IntoIterator<Item = &'a [F]>,
        challenge: EF,
        alignment: usize,
    ) -> EF
    where
        F: Field + 'a,
        EF: ExtensionField<F>,
    {
        reduce_with_powers_from(EF::ZERO, slices, challenge, alignment)
    }
}
