use crate::{BoolExpr, BoolTaggable, MidenAirBuilder, PrimeCharacteristicRing};

/// Extension trait for [`MidenAirBuilder`] that provides boolean expression support.
///
/// This trait adds `to_bool` methods that combine constraint assertion with type-safe
/// wrapping. It requires `Self::Expr: BoolTaggable` so that symbolic expressions can
/// be tagged with the boolean property.
///
/// # Usage in AIR constraints
///
/// ```rust,ignore
/// fn eval<AB: BoolExprBuilder>(&self, builder: &mut AB) {
///     let main = builder.main();
///     let flag = main.row_slice(0).unwrap()[0].clone();
///
///     // Assert flag is boolean and get a typed wrapper
///     let flag = builder.to_bool(flag);
///
///     // Use boolean operations
///     let not_flag = flag.not();
///
///     // Use in conditional expressions
///     let result = flag.select(value_a, value_b);
///     builder.assert_eq(output, result);
///
///     // BoolExpr participates in arithmetic directly
///     builder.assert_zero(flag * some_expr);
/// }
/// ```
///
/// # Symbolic Tagging
///
/// When `AB::Expr` is a `SymbolicExpression`, `to_bool` calls `tag_as_bool()` on the
/// expression, wrapping it in the `Bool` variant for symbolic analysis. For concrete
/// expression types, `tag_as_bool()` is a no-op.
pub trait BoolExprBuilder: MidenAirBuilder
where
    Self::Expr: BoolTaggable,
{
    /// Assert that `x` is boolean and return it as a [`BoolExpr`].
    ///
    /// This adds the constraint `x * (x - 1) = 0` and wraps the expression in
    /// `BoolExpr` for type safety. For symbolic expressions, the inner expression
    /// is also tagged as boolean.
    fn to_bool<I: Into<Self::Expr>>(&mut self, x: I) -> BoolExpr<Self::Expr> {
        let expr = x.into();
        self.assert_zero(expr.clone().bool_check());
        BoolExpr::from_unchecked(expr.tag_as_bool())
    }

    /// Assert that each element of an array is boolean and return them as [`BoolExpr`]s.
    fn to_bools<const N: usize, I: Into<Self::Expr>>(
        &mut self,
        array: [I; N],
    ) -> [BoolExpr<Self::Expr>; N] {
        array.map(|x| self.to_bool(x))
    }
}

/// Blanket implementation: any `MidenAirBuilder` whose `Expr` supports `BoolTaggable`
/// automatically gets `BoolExprBuilder`.
impl<AB> BoolExprBuilder for AB
where
    AB: MidenAirBuilder,
    AB::Expr: BoolTaggable,
{
}
