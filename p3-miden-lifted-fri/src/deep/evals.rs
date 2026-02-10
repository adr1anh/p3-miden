use alloc::vec::Vec;

use p3_field::{ExtensionField, Field, FieldArray, TwoAdicField};
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
    /// Parse out-of-domain evaluations from a verifier channel without validation.
    ///
    /// Reads in point-major order: for each evaluation point, then for each group,
    /// then for each matrix, we receive `width` extension field elements (one row).
    /// Widths must include any alignment padding applied during commitment.
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
        // Accumulate rows into flat buffers; convert to RowMajorMatrix at the end.
        let mut buffers: Vec<Vec<Vec<EF>>> = widths
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(|&w| Vec::with_capacity(num_points * w))
                    .collect()
            })
            .collect();

        // Reads point-by-point: each point contributes one row to every matrix.
        for _ in 0..num_points {
            for (group, group_widths) in buffers.iter_mut().zip(widths.iter()) {
                for (buf, &width) in group.iter_mut().zip(group_widths.iter()) {
                    buf.extend(channel.receive_algebra_slice::<EF>(width)?);
                }
            }
        }

        let groups = buffers
            .into_iter()
            .zip(widths)
            .map(|(group, widths)| {
                group
                    .into_iter()
                    .zip(*widths)
                    .map(|(values, &width)| RowMajorMatrix::new(values, width))
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
}

/// Batched evaluations for a single commitment group.
///
/// Structure: `matrices[matrix_idx][col_idx]` is a `FieldArray<EF, N>`.
#[derive(Clone, Debug)]
pub struct BatchedGroupEvals<EF: Field, const N: usize> {
    pub matrices: Vec<Vec<FieldArray<EF, N>>>,
}

/// Batched evaluations at N points, grouped by commitment.
///
/// Use [`Self::aligned`] to pad each matrix with zero columns so serialization
/// no longer needs an explicit alignment parameter.
#[derive(Clone, Debug)]
pub struct BatchedEvals<EF: Field, const N: usize> {
    pub groups: Vec<BatchedGroupEvals<EF, N>>,
}

impl<EF: Field, const N: usize> BatchedEvals<EF, N> {
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

    pub fn reduce(&self, challenge: EF) -> FieldArray<EF, N> {
        let values = self
            .groups
            .iter()
            .flat_map(|group| group.matrices.iter().flatten())
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
                for matrix in &group.matrices {
                    for eval in matrix {
                        channel.send_algebra_element(eval[point_idx]);
                    }
                }
            }
        }
    }
}
