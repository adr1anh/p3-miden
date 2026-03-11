use core::ops::{Add, Mul, Sub};

use crate::PrimeCharacteristicRing;

/// A wrapper around an expression that is known to be boolean (i.e., evaluates to 0 or 1).
///
/// `BoolExpr` provides type-level safety for boolean expressions in AIR constraints.
/// Once an expression is wrapped in `BoolExpr`, it can be used in boolean-specific
/// operations ([`not`](BoolExpr::not), [`and`](BoolExpr::and), [`or`](BoolExpr::or),
/// [`select`](BoolExpr::select)) that produce correct results only for boolean inputs.
///
/// # Creation
///
/// There are two ways to create a `BoolExpr`:
///
/// 1. **Via the builder** (recommended): Use [`BoolExprBuilder::to_bool`] which
///    automatically adds the boolean constraint `x * (x - 1) = 0`.
///
/// 2. **Unchecked** (advanced): Use [`BoolExpr::from_unchecked`] when you already
///    know an expression is boolean (e.g., it comes from a selector like `is_first_row`).
///
/// # Arithmetic
///
/// `BoolExpr<E>` implements `Add`, `Sub`, `Mul` so it can be combined with other
/// expressions without manual unwrapping. Use [`into_inner`](BoolExpr::into_inner)
/// when you need the raw expression.
///
/// # Design Note: Symbolic Expression Tagging
///
/// When used with a symbolic air builder, the `to_bool` method can wrap the symbolic
/// expression in a dedicated `Bool` variant. This allows symbolic analysis passes to
/// know which sub-expressions are boolean, enabling potential optimizations.
///
/// For concrete builders (prover, verifier), no tagging is needed — the expression
/// is stored as-is inside the `BoolExpr` wrapper, and the type-level guarantee is
/// sufficient.
///
/// ## Future: Expression-Level Property Trait
///
/// Ideally, we'd add a `BoolTaggable` trait bound on `AB::Expr`:
///
/// ```rust,ignore
/// pub trait BoolTaggable: PrimeCharacteristicRing {
///     fn tag_as_bool(self) -> Self;
///     fn is_bool(&self) -> bool;
/// }
/// ```
///
/// This would allow `expr.tag_as_bool()` to work generically, with
/// `SymbolicExpression` recording the property and concrete field types being no-ops.
/// However, this requires implementing the trait for all upstream packed field and
/// extension field types (from `p3-field`), which is fragile. A blanket impl for
/// `PrimeCharacteristicRing` would conflict with the specific `SymbolicExpression`
/// impl without specialization (unstable in Rust).
///
/// The current design keeps tagging at the builder level (via
/// [`BoolExprBuilder::to_bool`]) rather than requiring an `Expr`-level trait bound.
#[derive(Clone, Debug)]
pub struct BoolExpr<E>(E);

impl<E> BoolExpr<E> {
    /// Create a `BoolExpr` without adding any constraints.
    ///
    /// # Safety (logical)
    ///
    /// The caller must guarantee that the expression evaluates to 0 or 1 on every
    /// row where it is used. Misuse can lead to unsound constraints.
    ///
    /// Typical safe uses:
    /// - Selectors: `is_first_row()`, `is_last_row()`, `is_transition()`
    /// - Expressions already constrained to be boolean elsewhere
    #[inline]
    pub fn from_unchecked(expr: E) -> Self {
        BoolExpr(expr)
    }

    /// Consume the wrapper and return the inner expression.
    #[inline]
    pub fn into_inner(self) -> E {
        self.0
    }

    /// Borrow the inner expression.
    #[inline]
    pub fn as_inner(&self) -> &E {
        &self.0
    }
}

impl<E: PrimeCharacteristicRing + Clone> BoolExpr<E> {
    /// Boolean NOT: `1 - self`.
    #[inline]
    pub fn not(&self) -> BoolExpr<E> {
        BoolExpr(E::ONE - self.0.clone())
    }

    /// Boolean AND: `self * other`.
    #[inline]
    pub fn and(&self, other: &BoolExpr<E>) -> BoolExpr<E> {
        BoolExpr(self.0.clone() * other.0.clone())
    }

    /// Boolean OR: `self + other - self * other`.
    #[inline]
    pub fn or(&self, other: &BoolExpr<E>) -> BoolExpr<E> {
        BoolExpr(self.0.clone() + other.0.clone() - self.0.clone() * other.0.clone())
    }

    /// Conditional select: when `self` is 1 returns `when_true`, when 0 returns `when_false`.
    ///
    /// Computes `self * when_true + (1 - self) * when_false`, which simplifies to
    /// `self * (when_true - when_false) + when_false`.
    #[inline]
    pub fn select<T>(&self, when_true: T, when_false: T) -> E
    where
        T: Into<E>,
    {
        let t = when_true.into();
        let f = when_false.into();
        self.0.clone() * (t - f.clone()) + f
    }
}

// ---------------------------------------------------------------------------
// Arithmetic: BoolExpr<E> can participate directly in expression arithmetic
// ---------------------------------------------------------------------------

impl<E, Rhs> Add<Rhs> for BoolExpr<E>
where
    E: Add<Rhs>,
{
    type Output = E::Output;

    #[inline]
    fn add(self, rhs: Rhs) -> Self::Output {
        self.0 + rhs
    }
}

impl<E, Rhs> Sub<Rhs> for BoolExpr<E>
where
    E: Sub<Rhs>,
{
    type Output = E::Output;

    #[inline]
    fn sub(self, rhs: Rhs) -> Self::Output {
        self.0 - rhs
    }
}

impl<E, Rhs> Mul<Rhs> for BoolExpr<E>
where
    E: Mul<Rhs>,
{
    type Output = E::Output;

    #[inline]
    fn mul(self, rhs: Rhs) -> Self::Output {
        self.0 * rhs
    }
}
