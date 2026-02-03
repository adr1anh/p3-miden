use alloc::vec::Vec;

use p3_field::{ExtensionField, Field, FieldArray, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_lmcs::utils::pad_row_to_alignment;
use p3_miden_transcript::{ProverChannel, TranscriptError, VerifierChannel};

use crate::utils::horner;

/// Out-of-domain evaluations organized per commitment group and matrix.
///
/// Structure: `groups[group_idx][matrix_idx]` is a row-major matrix where rows are points.
/// All non-empty matrices must have the same height; widths are expected to be aligned already.
#[derive(Clone, Debug)]
pub struct DeepEvals<EF: Field> {
    groups: Vec<Vec<RowMajorMatrix<EF>>>,
    num_points: usize,
}

impl<EF: Field> DeepEvals<EF> {
    pub(crate) fn read_from_channel<F, Ch>(
        widths: &[&[usize]],
        num_points: usize,
        channel: &mut Ch,
    ) -> Result<Self, TranscriptError>
    where
        F: TwoAdicField,
        EF: ExtensionField<F>,
        Ch: VerifierChannel<F = F>,
    {
        // Widths are expected to include any alignment padding.
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
                    let point_values = channel.receive_algebra_slice::<EF>(width)?;
                    matrix_values.extend(point_values);
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

        Ok(Self { groups, num_points })
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
        self.num_points
    }

    pub(crate) fn reduce_point(&self, point_idx: usize, challenge: EF) -> EF {
        let values = self.groups.iter().flat_map(move |group| {
            group
                .iter()
                .flat_map(move |matrix| matrix.row(point_idx).expect("point index in range"))
        });
        horner(challenge, values)
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
///
/// Use [`Self::aligned`] to pad each matrix with zero columns so serialization
/// no longer needs an explicit alignment parameter.
#[derive(Clone, Debug)]
pub struct BatchedEvals<EF: Field, const N: usize> {
    groups: Vec<BatchedGroupEvals<EF, N>>,
}

impl<EF: Field, const N: usize> BatchedEvals<EF, N> {
    pub const fn new(groups: Vec<BatchedGroupEvals<EF, N>>) -> Self {
        Self { groups }
    }

    /// Pad each matrix with zero columns so widths are aligned.
    pub fn aligned(mut self, alignment: usize) -> Self {
        for group in &mut self.groups {
            for matrix in &mut group.matrices {
                let row = core::mem::take(matrix);
                *matrix = pad_row_to_alignment(row, alignment);
            }
        }
        self
    }

    pub fn groups(&self) -> &[BatchedGroupEvals<EF, N>] {
        &self.groups
    }

    pub(crate) fn reduce(&self, challenge: EF) -> FieldArray<EF, N> {
        let values = self
            .groups
            .iter()
            .flat_map(|group| group.iter_evals())
            .copied();
        horner(challenge, values)
    }

    pub fn write_to_channel<F, Ch>(&self, channel: &mut Ch)
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
                }
            }
        }
    }
}
