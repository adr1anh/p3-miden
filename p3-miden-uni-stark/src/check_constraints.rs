use p3_air::{AirBuilder, RowWindow};
use p3_field::Field;

#[cfg(debug_assertions)]
use p3_air::Air;
#[cfg(debug_assertions)]
use p3_matrix::Matrix;
#[cfg(debug_assertions)]
use p3_matrix::dense::RowMajorMatrix;
#[cfg(debug_assertions)]
use tracing::instrument;

/// Runs constraint checks using a given [`Air`] implementation and trace matrix.
///
/// Iterates over every row in `main`, providing both the current and next row
/// (with wraparound) to the [`Air`] logic. Also injects public values into the
/// [`DebugConstraintBuilder`] for first/last row assertions.
///
/// # Arguments
/// - `air`: The [`Air`] logic to run.
/// - `main`: The [`RowMajorMatrix`] containing witness rows.
/// - `public_values`: Public values provided to the builder.
#[cfg(debug_assertions)]
#[instrument(skip_all)]
pub(crate) fn check_constraints<F, A>(air: &A, main: &RowMajorMatrix<F>, public_values: &[F])
where
    F: Field,
    A: for<'a> Air<DebugConstraintBuilder<'a, F>>,
{
    let height = main.height();
    let preprocessed = air.preprocessed_trace();

    (0..height).for_each(|row_index| {
        let row_index_next = (row_index + 1) % height;

        let local = unsafe { main.row_slice_unchecked(row_index) };
        let next = unsafe { main.row_slice_unchecked(row_index_next) };
        let main_window = RowWindow::from_two_rows(&local, &next);

        let preprocessed_window = if let Some(prep) = preprocessed.as_ref() {
            let prep_local = unsafe { prep.row_slice_unchecked(row_index) };
            let prep_next = unsafe { prep.row_slice_unchecked(row_index_next) };
            RowWindow::from_two_rows(
                // SAFETY: slices are valid for the duration of this closure
                unsafe { core::slice::from_raw_parts(prep_local.as_ptr(), prep_local.len()) },
                unsafe { core::slice::from_raw_parts(prep_next.as_ptr(), prep_next.len()) },
            )
        } else {
            RowWindow::from_two_rows(&[], &[])
        };

        let mut builder = DebugConstraintBuilder {
            row_index,
            main: main_window,
            preprocessed: preprocessed_window,
            public_values,
            is_first_row: F::from_bool(row_index == 0),
            is_last_row: F::from_bool(row_index == height - 1),
            is_transition: F::from_bool(row_index != height - 1),
        };

        air.eval(&mut builder);
    });
}

/// A builder that runs constraint assertions during testing.
///
/// Used in conjunction with `check_constraints` to simulate
/// an execution trace and verify that the [`Air`] logic enforces all constraints.
#[derive(Debug)]
pub struct DebugConstraintBuilder<'a, F: Field> {
    /// The index of the row currently being evaluated.
    row_index: usize,
    /// A two-row window over the current and next main trace rows.
    main: RowWindow<'a, F>,
    /// A two-row window over the current and next preprocessed trace rows.
    preprocessed: RowWindow<'a, F>,
    /// The public values provided for constraint validation (e.g. inputs or outputs).
    public_values: &'a [F],
    /// A flag indicating whether this is the first row.
    is_first_row: F,
    /// A flag indicating whether this is the last row.
    is_last_row: F,
    /// A flag indicating whether this is a transition row (not the last row).
    is_transition: F,
}

impl<'a, F> AirBuilder for DebugConstraintBuilder<'a, F>
where
    F: Field,
{
    type F = F;
    type Expr = F;
    type Var = F;
    type M = RowWindow<'a, F>;
    type PublicVar = Self::F;

    fn main(&self) -> Self::M {
        self.main
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    /// # Panics
    /// This function panics if `size` is not `2`.
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        assert_eq!(
            x.into(),
            F::ZERO,
            "constraints had nonzero value on row {}",
            self.row_index
        );
    }

    fn assert_eq<I1: Into<Self::Expr>, I2: Into<Self::Expr>>(&mut self, x: I1, y: I2) {
        let x = x.into();
        let y = y.into();
        assert_eq!(
            x, y,
            "values didn't match on row {}: {} != {}",
            self.row_index, x, y
        );
    }

    fn preprocessed(&self) -> &Self::M {
        &self.preprocessed
    }

    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
}
