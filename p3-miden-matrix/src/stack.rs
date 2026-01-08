use core::ops::Deref;

use crate::Matrix;
use crate::dense::RowMajorMatrixView;

/// A type alias representing a vertical composition of two row-major matrix views.
///
/// `ViewPair` combines two [`RowMajorMatrixView`]'s with the same element type `T`
/// and lifetime `'a` into a single virtual matrix stacked vertically.
///
/// Both views must have the same width; the resulting view has a height equal
/// to the sum of the two original heights.
pub type ViewPair<'a, T> = VerticalPair<RowMajorMatrixView<'a, T>, RowMajorMatrixView<'a, T>>;

/// A matrix composed by stacking two matrices vertically, one on top of the other.
///
/// Both matrices must have the same `width`.
/// The resulting matrix has dimensions:
/// - `width`: The same as the inputs.
/// - `height`: The sum of the `heights` of the input matrices.
///
/// Element access and iteration will first access the rows of the top matrix,
/// followed by the rows of the bottom matrix.
#[derive(Copy, Clone, Debug)]
pub struct VerticalPair<Top, Bottom> {
    /// The top matrix in the vertical composition.
    pub top: Top,
    /// The bottom matrix in the vertical composition.
    pub bottom: Bottom,
}

/// A matrix composed by placing two matrices side-by-side horizontally.
///
/// Both matrices must have the same `height`.
/// The resulting matrix has dimensions:
/// - `width`: The sum of the `widths` of the input matrices.
/// - `height`: The same as the inputs.
///
/// Element access and iteration for a given row `i` will first access the elements in the `i`'th row of the left matrix,
/// followed by elements in the `i'`th row of the right matrix.
#[derive(Copy, Clone, Debug)]
pub struct HorizontalPair<Left, Right> {
    /// The left matrix in the horizontal composition.
    pub left: Left,
    /// The right matrix in the horizontal composition.
    pub right: Right,
}

impl<Top, Bottom> VerticalPair<Top, Bottom> {
    /// Create a new `VerticalPair` by stacking two matrices vertically.
    ///
    /// # Panics
    /// Panics if the two matrices do not have the same width (i.e., number of columns),
    /// since vertical composition requires column alignment.
    ///
    /// # Returns
    /// A `VerticalPair` that represents the combined matrix.
    pub fn new<T>(top: Top, bottom: Bottom) -> Self
    where
        T: Send + Sync + Clone,
        Top: Matrix<T>,
        Bottom: Matrix<T>,
    {
        assert_eq!(top.width(), bottom.width());
        Self { top, bottom }
    }
}

impl<Left, Right> HorizontalPair<Left, Right> {
    /// Create a new `HorizontalPair` by joining two matrices side by side.
    ///
    /// # Panics
    /// Panics if the two matrices do not have the same height (i.e., number of rows),
    /// since horizontal composition requires row alignment.
    ///
    /// # Returns
    /// A `HorizontalPair` that represents the combined matrix.
    pub fn new<T>(left: Left, right: Right) -> Self
    where
        T: Send + Sync + Clone,
        Left: Matrix<T>,
        Right: Matrix<T>,
    {
        assert_eq!(left.height(), right.height());
        Self { left, right }
    }
}

impl<T: Send + Sync + Clone, Top: Matrix<T>, Bottom: Matrix<T>> Matrix<T>
    for VerticalPair<Top, Bottom>
{
    fn width(&self) -> usize {
        self.top.width()
    }

    fn height(&self) -> usize {
        self.top.height() + self.bottom.height()
    }

    unsafe fn get_unchecked(&self, r: usize, c: usize) -> T {
        unsafe {
            // Safety: The caller must ensure that r < self.height() and c < self.width()
            if r < self.top.height() {
                self.top.get_unchecked(r, c)
            } else {
                self.bottom.get_unchecked(r - self.top.height(), c)
            }
        }
    }

    unsafe fn row_unchecked(
        &self,
        r: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        unsafe {
            // Safety: The caller must ensure that r < self.height()
            if r < self.top.height() {
                EitherRow::Left(self.top.row_unchecked(r).into_iter())
            } else {
                EitherRow::Right(self.bottom.row_unchecked(r - self.top.height()).into_iter())
            }
        }
    }

    unsafe fn row_subseq_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        unsafe {
            // Safety: The caller must ensure that r < self.height() and start <= end <= self.width()
            if r < self.top.height() {
                EitherRow::Left(self.top.row_subseq_unchecked(r, start, end).into_iter())
            } else {
                EitherRow::Right(
                    self.bottom
                        .row_subseq_unchecked(r - self.top.height(), start, end)
                        .into_iter(),
                )
            }
        }
    }

    unsafe fn row_slice_unchecked(&self, r: usize) -> impl Deref<Target = [T]> {
        unsafe {
            // Safety: The caller must ensure that r < self.height()
            if r < self.top.height() {
                EitherRow::Left(self.top.row_slice_unchecked(r))
            } else {
                EitherRow::Right(self.bottom.row_slice_unchecked(r - self.top.height()))
            }
        }
    }

    unsafe fn row_subslice_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl Deref<Target = [T]> {
        unsafe {
            // Safety: The caller must ensure that r < self.height() and start <= end <= self.width()
            if r < self.top.height() {
                EitherRow::Left(self.top.row_subslice_unchecked(r, start, end))
            } else {
                EitherRow::Right(self.bottom.row_subslice_unchecked(
                    r - self.top.height(),
                    start,
                    end,
                ))
            }
        }
    }
}

impl<T: Send + Sync + Clone, Left: Matrix<T>, Right: Matrix<T>> Matrix<T>
    for HorizontalPair<Left, Right>
{
    fn width(&self) -> usize {
        self.left.width() + self.right.width()
    }

    fn height(&self) -> usize {
        self.left.height()
    }

    unsafe fn get_unchecked(&self, r: usize, c: usize) -> T {
        unsafe {
            // Safety: The caller must ensure that r < self.height() and c < self.width()
            if c < self.left.width() {
                self.left.get_unchecked(r, c)
            } else {
                self.right.get_unchecked(r, c - self.left.width())
            }
        }
    }

    unsafe fn row_unchecked(
        &self,
        r: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        unsafe {
            // Safety: The caller must ensure that r < self.height()
            self.left
                .row_unchecked(r)
                .into_iter()
                .chain(self.right.row_unchecked(r))
        }
    }
}

/// We use this to wrap both the row iterator and the row slice.
#[derive(Debug)]
pub enum EitherRow<L, R> {
    Left(L),
    Right(R),
}

impl<T, L, R> Iterator for EitherRow<L, R>
where
    L: Iterator<Item = T>,
    R: Iterator<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Left(l) => l.next(),
            Self::Right(r) => r.next(),
        }
    }
}

impl<T, L, R> Deref for EitherRow<L, R>
where
    L: Deref<Target = [T]>,
    R: Deref<Target = [T]>,
{
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Left(l) => l,
            Self::Right(r) => r,
        }
    }
}
