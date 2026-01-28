use alloc::vec::Vec;
use core::ops::Deref;

use p3_field::{ExtensionField, Field, FieldArray, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_transcript::{ProverChannel, VerifierChannel};

use crate::deep::utils::alignment_padding;

/// Out-of-domain evaluations organized per commitment group and matrix.
///
/// Structure: `groups[group_idx][matrix_idx]` is a row-major matrix where rows are points.
#[derive(Clone, Debug)]
pub struct DeepEvals<EF: Field> {
    groups: Vec<Vec<RowMajorMatrix<EF>>>,
}

impl<EF: Field> DeepEvals<EF> {
    /// Create `DeepEvals` from grouped row-major matrices.
    pub const fn new(groups: Vec<Vec<RowMajorMatrix<EF>>>) -> Self {
        Self { groups }
    }

    /// Grouped row-major matrices, one per commitment group.
    pub fn groups(&self) -> &[Vec<RowMajorMatrix<EF>>] {
        &self.groups
    }

    /// Consume and return the grouped row-major matrices.
    pub fn into_groups(self) -> Vec<Vec<RowMajorMatrix<EF>>> {
        self.groups
    }

    /// Number of evaluation points (rows per matrix).
    pub fn num_points(&self) -> usize {
        self.groups
            .first()
            .and_then(|group| group.first())
            .map(|matrix| matrix.height())
            .unwrap_or(0)
    }

    pub(crate) fn reduce_all(&self, challenge: EF, alignment: usize) -> Vec<EF> {
        let num_points = self.num_points();
        let mut reduced = Vec::with_capacity(num_points);
        for point_idx in 0..num_points {
            reduced.push(self.reduce_point(point_idx, challenge, alignment));
        }
        reduced
    }

    fn reduce_point(&self, point_idx: usize, challenge: EF, alignment: usize) -> EF {
        self.groups
            .iter()
            .flat_map(move |group| {
                group
                    .iter()
                    .map(move |matrix| matrix.row_slice(point_idx).expect("point index in range"))
            })
            .fold(EF::ZERO, |acc, row| {
                let row = row.deref();
                let acc = row.iter().fold(acc, |a, &val| a * challenge + val);
                acc * challenge.exp_u64(alignment_padding(row.len(), alignment) as u64)
            })
    }

    pub(crate) fn read_from_channel<F, Ch>(
        widths: &[&[usize]],
        num_points: usize,
        alignment: usize,
        channel: &mut Ch,
    ) -> Option<Self>
    where
        F: TwoAdicField,
        EF: ExtensionField<F>,
        Ch: VerifierChannel<F = F>,
    {
        let mut values: Vec<Vec<Vec<EF>>> = widths
            .iter()
            .map(|group_widths| {
                group_widths
                    .iter()
                    .map(|&width| Vec::with_capacity(num_points * width))
                    .collect()
            })
            .collect();

        for _ in 0..num_points {
            for (group_idx, group_widths) in widths.iter().enumerate() {
                let group_values = &mut values[group_idx];
                for (matrix_idx, &width) in group_widths.iter().enumerate() {
                    let matrix_values = &mut group_values[matrix_idx];
                    for _ in 0..width {
                        let val = channel.receive_algebra_element::<EF>()?;
                        matrix_values.push(val);
                    }
                    for _ in 0..alignment_padding(width, alignment) {
                        channel.receive_algebra_element::<EF>()?;
                    }
                }
            }
        }

        let groups = values
            .into_iter()
            .zip(widths)
            .map(|(group_values, group_widths)| {
                group_values
                    .into_iter()
                    .zip(*group_widths)
                    .map(|(matrix_values, &width)| RowMajorMatrix::new(matrix_values, width))
                    .collect()
            })
            .collect();

        Some(Self { groups })
    }
}

/// Batched evaluations for a single commitment group.
///
/// Structure: `matrices[matrix_idx][col_idx]` is a `FieldArray<EF, N>`.
#[derive(Clone, Debug)]
pub struct BatchedGroupEvals<EF: Field, const N: usize> {
    matrices: Vec<Vec<FieldArray<EF, N>>>,
}

impl<EF: Field, const N: usize> BatchedGroupEvals<EF, N> {
    pub const fn new(matrices: Vec<Vec<FieldArray<EF, N>>>) -> Self {
        Self { matrices }
    }

    pub fn matrices(&self) -> &[Vec<FieldArray<EF, N>>] {
        &self.matrices
    }

    pub fn matrix(&self, idx: usize) -> Option<&[FieldArray<EF, N>]> {
        self.matrices.get(idx).map(|v| v.as_slice())
    }

    pub fn num_matrices(&self) -> usize {
        self.matrices.len()
    }

    pub fn iter_matrices(&self) -> impl Iterator<Item = &[FieldArray<EF, N>]> {
        self.matrices.iter().map(|v| v.as_slice())
    }

    pub fn iter_evals(&self) -> impl Iterator<Item = &FieldArray<EF, N>> {
        self.matrices.iter().flatten()
    }
}

/// Batched evaluations at N points, grouped by commitment.
#[derive(Clone, Debug)]
pub struct BatchedEvals<EF: Field, const N: usize> {
    groups: Vec<BatchedGroupEvals<EF, N>>,
}

impl<EF: Field, const N: usize> BatchedEvals<EF, N> {
    pub const fn new(groups: Vec<BatchedGroupEvals<EF, N>>) -> Self {
        Self { groups }
    }

    pub fn groups(&self) -> &[BatchedGroupEvals<EF, N>] {
        &self.groups
    }

    pub fn write_to_channel<F, Ch>(&self, alignment: usize, channel: &mut Ch)
    where
        F: TwoAdicField,
        EF: ExtensionField<F>,
        Ch: ProverChannel<F = F>,
    {
        for point_idx in 0..N {
            for group in &self.groups {
                for matrix_evals in group.iter_matrices() {
                    for eval in matrix_evals {
                        channel.send_algebra_element(eval[point_idx]);
                    }
                    for _ in 0..alignment_padding(matrix_evals.len(), alignment) {
                        channel.send_algebra_element(EF::ZERO);
                    }
                }
            }
        }
    }
}
