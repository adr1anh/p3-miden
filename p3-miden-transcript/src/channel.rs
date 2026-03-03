//! Shared `Channel` trait and `TranscriptChallenger` supertrait.

use p3_challenger::{CanObserve, CanSample, CanSampleBits, GrindingChallenger};
use p3_field::{BasedVectorSpace, Field};

/// Bundle of challenger bounds required by transcript channels.
///
/// Any challenger that satisfies `CanObserve<F>`, `CanObserve<C>`, `CanSample<F>`,
/// `CanSampleBits<usize>`, and `GrindingChallenger<Witness = F>` automatically
/// implements this trait via a blanket impl.
pub trait TranscriptChallenger<F: Field, C>:
    CanObserve<F>
    + CanObserve<C>
    + CanSample<F>
    + CanSampleBits<usize>
    + GrindingChallenger<Witness = F>
{
}

impl<F, C, Ch> TranscriptChallenger<F, C> for Ch
where
    F: Field,
    Ch: CanObserve<F>
        + CanObserve<C>
        + CanSample<F>
        + CanSampleBits<usize>
        + GrindingChallenger<Witness = F>,
{
}

/// Shared base trait for [`ProverChannel`](crate::ProverChannel) and
/// [`VerifierChannel`](crate::VerifierChannel).
///
/// Provides sampling methods common to both sides of the transcript.
pub trait Channel {
    type F: Field;
    type Commitment: Clone;
    type Challenger: TranscriptChallenger<Self::F, Self::Commitment>;

    /// Sample a random field element from the challenger.
    fn sample(&mut self) -> Self::F;

    /// Sample a random `bits`-bit integer from the challenger.
    fn sample_bits(&mut self, bits: usize) -> usize;

    /// Sample a random algebra element (e.g. extension field) from the challenger.
    fn sample_algebra_element<A: BasedVectorSpace<Self::F>>(&mut self) -> A {
        A::from_basis_coefficients_fn(|_| self.sample())
    }
}
