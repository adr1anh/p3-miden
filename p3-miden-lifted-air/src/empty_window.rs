//! A zero-width window that statically prevents access to preprocessed columns.
//!
//! Lifted AIRs have no preprocessed trace. [`EmptyWindow`] encodes this invariant
//! at the type level: any code path that calls [`WindowAccess`] methods on it will
//! fail to compile, catching misuse early rather than at runtime.

use core::marker::PhantomData;

use p3_air::WindowAccess;

/// A window type for traces that must never be accessed.
///
/// Satisfies the `WindowAccess<T> + Clone` bound required by
/// [`AirBuilder::PreprocessedWindow`](p3_air::AirBuilder::PreprocessedWindow),
/// but uses inline `const { panic!() }` blocks to turn any access into a
/// compile-time error.
#[derive(Debug, Clone, Copy)]
pub struct EmptyWindow<T>(PhantomData<T>);

impl<T> EmptyWindow<T> {
    /// Static reference to an empty window.
    ///
    /// Safe because `EmptyWindow` is a ZST — no actual `T` is stored,
    /// so the `'static` lifetime is always valid.
    pub fn empty_ref() -> &'static Self {
        &EmptyWindow(PhantomData)
    }
}

impl<T> WindowAccess<T> for EmptyWindow<T> {
    fn current_slice(&self) -> &[T] {
        const { panic!("EmptyWindow: preprocessed trace does not exist in lifted AIRs") }
    }

    fn next_slice(&self) -> &[T] {
        const { panic!("EmptyWindow: preprocessed trace does not exist in lifted AIRs") }
    }
}
