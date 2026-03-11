use crate::PrimeCharacteristicRing;

/// Trait for expression types that support boolean property tagging.
///
/// When implemented for a symbolic expression type, `tag_as_bool` wraps the expression
/// in a way that records the boolean property for later analysis. For concrete field
/// types (used in prover/verifier), the implementation is a no-op identity.
///
/// # Motivation
///
/// In the symbolic air, knowing which sub-expressions are boolean enables analysis
/// passes to:
/// - Detect redundant boolean constraints
/// - Simplify `bool * bool → bool` and `1 - bool → bool`
/// - Verify that boolean operations are applied to actually-boolean operands
///
/// # Coherence Constraints
///
/// We cannot provide a blanket `impl<T: PrimeCharacteristicRing> BoolTaggable for T`
/// because it would conflict with the specific `SymbolicExpression` implementation
/// (both match `SymbolicExpression<F>` which implements `PrimeCharacteristicRing`).
/// Stable Rust does not support specialization.
///
/// Instead, implementations must be provided explicitly:
/// - `SymbolicExpression<F>`: tags via the `Bool` variant
/// - Concrete field/packed types: no-op (implement with default methods)
///
/// To reduce boilerplate, use the [`impl_bool_taggable_noop!`] macro for types
/// where tagging is a no-op.
pub trait BoolTaggable: PrimeCharacteristicRing + Sized {
    /// Mark this expression as boolean. Returns a (possibly wrapped) expression
    /// that carries the boolean property.
    ///
    /// For symbolic expressions, this adds metadata. For concrete field elements,
    /// this is an identity operation.
    fn tag_as_bool(self) -> Self {
        self
    }

    /// Check if this expression has been marked as boolean.
    ///
    /// Returns `false` by default (for concrete types where tagging is a no-op).
    fn is_tagged_bool(&self) -> bool {
        false
    }
}

/// Implement [`BoolTaggable`] as a no-op for one or more types.
///
/// This is intended for concrete field types (e.g., `BabyBear`, packed fields)
/// where boolean tagging has no runtime effect.
///
/// # Example
///
/// ```rust,ignore
/// impl_bool_taggable_noop!(BabyBear, PackedBabyBear);
/// ```
#[macro_export]
macro_rules! impl_bool_taggable_noop {
    ($($ty:ty),+ $(,)?) => {
        $(impl $crate::BoolTaggable for $ty {})+
    };
}
