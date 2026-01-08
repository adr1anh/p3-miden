use crate::Matrix;
use crate::row_index_mapped::{RowIndexMap, RowIndexMappedView};

/// A vertical row-mapping strategy that selects every `stride`-th row from an inner matrix,
/// starting at a fixed `offset`.
///
/// This enables vertical striding like selecting rows: `offset`, `offset + stride`, etc.
#[derive(Debug)]
pub struct VerticallyStridedRowIndexMap {
    /// The number of rows in the resulting view.
    height: usize,
    /// The step size between selected rows in the inner matrix.
    stride: usize,
    /// The offset to start the stride from.
    offset: usize,
}

pub type VerticallyStridedMatrixView<Inner> =
    RowIndexMappedView<VerticallyStridedRowIndexMap, Inner>;

impl VerticallyStridedRowIndexMap {
    /// Create a new vertically strided view over a matrix.
    ///
    /// This selects rows in the inner matrix starting from `offset`, and then every `stride` rows after.
    ///
    /// # Arguments
    /// - `inner`: The inner matrix to view.
    /// - `stride`: The number of rows between each selected row.
    /// - `offset`: The initial row to start from.
    pub fn new_view<T: Send + Sync + Clone, Inner: Matrix<T>>(
        inner: Inner,
        stride: usize,
        offset: usize,
    ) -> VerticallyStridedMatrixView<Inner> {
        let h = inner.height();
        let full_strides = h / stride;
        let remainder = h % stride;
        let final_stride = offset < remainder;
        let height = full_strides + final_stride as usize;
        RowIndexMappedView {
            index_map: Self {
                height,
                stride,
                offset,
            },
            inner,
        }
    }
}

impl RowIndexMap for VerticallyStridedRowIndexMap {
    fn height(&self) -> usize {
        self.height
    }

    fn map_row_index(&self, r: usize) -> usize {
        r * self.stride + self.offset
    }
}
