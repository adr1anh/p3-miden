//! Verifier-side transcript channel.

use alloc::vec::Vec;

use p3_challenger::{
    CanObserve, CanSample, CanSampleBits, CanSampleUniformBits, GrindingChallenger,
};
use p3_field::{BasedVectorSpace, Field, PrimeField64};

use crate::TranscriptData;

/// Verifier channel that reads transcript data and observes into the challenger.
#[derive(Clone, Debug)]
pub struct VerifierTranscript<'a, F, C, Ch> {
    challenger: Ch,
    fields: &'a [F],
    commitments: &'a [C],
}

impl<'a, F, C, Ch> VerifierTranscript<'a, F, C, Ch> {
    /// Creates a new verifier transcript backed by the provided challenger.
    pub fn new(challenger: Ch, fields: &'a [F], commitments: &'a [C]) -> Self {
        Self {
            challenger,
            fields,
            commitments,
        }
    }

    /// Creates a verifier transcript backed by a `TranscriptData` container.
    pub fn from_data(challenger: Ch, data: &'a TranscriptData<F, C>) -> Self {
        let (fields, commitments) = data.as_slices();
        Self::new(challenger, fields, commitments)
    }
}

/// Verifier-side channel interface for transcript operations.
pub trait VerifierChannel {
    type F: Field;
    type Commitment: Copy;

    fn receive_field_slice(&mut self, count: usize) -> Option<&[Self::F]>;

    fn receive_commitment_slice(&mut self, count: usize) -> Option<&[Self::Commitment]>;

    fn receive_field(&mut self) -> Option<&Self::F> {
        self.receive_field_slice(1)
            .and_then(|values| values.first())
    }

    fn receive_algebra_element<A>(&mut self) -> Option<A>
    where
        Self::F: Field,
        A: BasedVectorSpace<Self::F>,
    {
        let coeffs = self.receive_field_slice(A::DIMENSION)?;
        Some(A::from_basis_coefficients_slice(coeffs).unwrap())
    }

    fn receive_algebra_slice<A>(&mut self, count: usize) -> Option<Vec<A>>
    where
        Self::F: Field,
        A: BasedVectorSpace<Self::F>,
    {
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            let coeffs = self.receive_field_slice(A::DIMENSION)?;
            values.push(A::from_basis_coefficients_slice(coeffs).unwrap());
        }
        Some(values)
    }

    fn receive_commitment(&mut self) -> Option<&Self::Commitment> {
        self.receive_commitment_slice(1)
            .and_then(|values| values.first())
    }

    fn receive_u64(&mut self) -> Option<u64>
    where
        Self::F: PrimeField64,
    {
        self.receive_field().map(|value| value.as_canonical_u64())
    }

    fn receive_hint_field_slice(&mut self, count: usize) -> Option<&[Self::F]>;

    fn receive_hint_commitment_slice(&mut self, count: usize) -> Option<&[Self::Commitment]>;

    fn receive_hint_field(&mut self) -> Option<&Self::F> {
        self.receive_hint_field_slice(1)
            .and_then(|values| values.first())
    }

    fn receive_hint_commitment(&mut self) -> Option<&Self::Commitment> {
        self.receive_hint_commitment_slice(1)
            .and_then(|values| values.first())
    }

    fn grind(&mut self, bits: usize) -> Option<Self::F>;

    fn is_empty(&self) -> bool;

    fn sample_algebra_element<A: BasedVectorSpace<Self::F>>(&mut self) -> A
    where
        Self: CanSample<Self::F>,
    {
        A::from_basis_coefficients_fn(|_| self.sample())
    }
}

impl<'a, F, C, Ch> VerifierChannel for VerifierTranscript<'a, F, C, Ch>
where
    F: Field,
    C: Copy,
    Ch: CanObserve<F> + CanObserve<C> + GrindingChallenger<Witness = F>,
{
    type F = F;
    type Commitment = C;

    // === Observed data ===
    fn receive_field_slice(&mut self, count: usize) -> Option<&'a [Self::F]> {
        let values = pop_slice(&mut self.fields, count)?;
        self.challenger.observe_slice(values);
        Some(values)
    }

    fn receive_commitment_slice(&mut self, count: usize) -> Option<&'a [Self::Commitment]> {
        let values = pop_slice(&mut self.commitments, count)?;
        self.challenger.observe_slice(values);
        Some(values)
    }

    fn receive_hint_field_slice(&mut self, count: usize) -> Option<&'a [Self::F]> {
        pop_slice(&mut self.fields, count)
    }

    fn receive_hint_commitment_slice(&mut self, count: usize) -> Option<&'a [Self::Commitment]> {
        pop_slice(&mut self.commitments, count)
    }

    fn grind(&mut self, bits: usize) -> Option<Self::F> {
        let (witness, rest) = self.fields.split_first()?;
        self.fields = rest;
        if self.challenger.check_witness(bits, *witness) {
            Some(*witness)
        } else {
            None
        }
    }

    fn is_empty(&self) -> bool {
        self.fields.is_empty() && self.commitments.is_empty()
    }
}

impl<'a, F, C, Ch, T> CanSample<T> for VerifierTranscript<'a, F, C, Ch>
where
    Ch: CanSample<T>,
{
    #[inline]
    fn sample(&mut self) -> T {
        self.challenger.sample()
    }
}

impl<'a, F, C, Ch> CanSampleBits<usize> for VerifierTranscript<'a, F, C, Ch>
where
    Ch: CanSampleBits<usize>,
{
    #[inline]
    fn sample_bits(&mut self, bits: usize) -> usize {
        self.challenger.sample_bits(bits)
    }
}

impl<'a, F, C, Ch> CanSampleUniformBits<F> for VerifierTranscript<'a, F, C, Ch>
where
    Ch: CanSampleUniformBits<F>,
{
    #[inline]
    fn sample_uniform_bits<const RESAMPLE: bool>(
        &mut self,
        bits: usize,
    ) -> Result<usize, p3_challenger::ResamplingError> {
        self.challenger.sample_uniform_bits::<RESAMPLE>(bits)
    }
}

fn pop_slice<'a, T>(values: &mut &'a [T], count: usize) -> Option<&'a [T]> {
    if values.len() < count {
        return None;
    }
    let (slice, rest) = values.split_at(count);
    *values = rest;
    Some(slice)
}
