use alloc::vec::Vec;

use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_miden_lmcs::RowList;
use p3_miden_transcript::{TranscriptError, VerifierChannel};

/// Out-of-domain (OOD) evaluations in point-major layout.
///
/// DEEP asks the prover for evaluations of committed polynomials at a small set of
/// random points `z₀, z₁, ...` that lie outside the evaluation domain. These claims
/// are then folded into a single low-degree test (FRI).
///
/// `points[point_idx]` is a `RowList<EF>` with one row per committed matrix (across all
/// commitment groups). Point-major layout matches the verifier's access pattern: it
/// reduces all column values for one point at a time via Horner, so
/// `point(idx).iter_values()` yields exactly the streaming order used by DEEP.
#[derive(Clone, Debug)]
pub struct DeepEvals<EF> {
    points: Vec<RowList<EF>>,
}

impl<EF: Field> DeepEvals<EF> {
    /// Parse out-of-domain evaluations from a verifier channel.
    ///
    /// Reads in point-major order: for each evaluation point, then for each matrix
    /// width, we receive that many extension field elements.
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
        let flat_widths: Vec<usize> = widths.iter().flat_map(|s| s.iter()).copied().collect();
        let total: usize = flat_widths.iter().sum();

        let mut points = Vec::with_capacity(num_points);
        for _ in 0..num_points {
            // All column evaluations for this point; RowList recovers per-matrix structure.
            let elems = channel.receive_algebra_slice::<EF>(total)?;
            points.push(RowList::new(elems, &flat_widths));
        }

        Ok(Self { points })
    }

    /// Get the `RowList` for a specific evaluation point.
    pub fn point(&self, idx: usize) -> &RowList<EF> {
        &self.points[idx]
    }

    /// Number of evaluation points.
    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    /// Split into one `DeepEvals` per commitment group.
    ///
    /// `group_sizes[g]` is the number of matrices in group `g`. The sum of all
    /// group sizes must equal the number of rows in each point's `RowList`.
    ///
    /// The PCS transcript returns evaluations as one flat stream. The STARK verifier
    /// often wants the original grouping (e.g. main, aux, quotient), so this helper
    /// recovers that structure.
    pub fn split_by_groups(self, group_sizes: &[usize]) -> Vec<Self> {
        assert_eq!(
            group_sizes.iter().sum::<usize>(),
            self.points.first().map_or(0, |p| p.num_rows()),
            "group_sizes sum must equal number of rows per point"
        );

        let num_groups = group_sizes.len();
        let mut groups: Vec<Vec<RowList<EF>>> = (0..num_groups)
            .map(|_| Vec::with_capacity(self.points.len()))
            .collect();

        for row_list in self.points {
            let mut rows = row_list.iter_rows();
            for (g, &size) in group_sizes.iter().enumerate() {
                let group_rows: Vec<&[EF]> = (&mut rows).take(size).collect();
                groups[g].push(RowList::from_rows(group_rows));
            }
            assert!(
                rows.next().is_none(),
                "group_sizes sum does not match row count"
            );
        }

        groups.into_iter().map(|points| Self { points }).collect()
    }
}
