//! Init-only transcript wrapper.

use core::marker::PhantomData;

use p3_challenger::CanObserve;
use p3_field::{BasedVectorSpace, Field};

use crate::{ProverTranscript, TranscriptData, VerifierTranscript};

/// Init-only transcript that can observe shared data before starting.
#[derive(Clone, Debug)]
pub struct InitTranscript<F, C, Ch> {
    challenger: Ch,
    _marker: PhantomData<(F, C)>,
}

impl<F, C, Ch> InitTranscript<F, C, Ch>
where
    F: Field,
    C: Copy,
    Ch: CanObserve<F> + CanObserve<C>,
{
    /// Create a new init transcript from a challenger.
    pub fn new(challenger: Ch) -> Self {
        Self {
            challenger,
            _marker: PhantomData,
        }
    }

    pub fn observe_init_field_slice(&mut self, values: &[F]) {
        self.challenger.observe_slice(values);
    }

    pub fn observe_init_commitment_slice(&mut self, values: &[C]) {
        self.challenger.observe_slice(values);
    }

    pub fn observe_init_field_element(&mut self, value: F) {
        self.observe_init_field_slice(core::slice::from_ref(&value))
    }

    pub fn observe_init_algebra_element<A>(&mut self, value: A)
    where
        A: BasedVectorSpace<F>,
    {
        self.observe_init_field_slice(value.as_basis_coefficients_slice())
    }

    pub fn observe_init_algebra_slice<A>(&mut self, values: &[A])
    where
        A: BasedVectorSpace<F>,
    {
        for value in values {
            self.observe_init_field_slice(value.as_basis_coefficients_slice());
        }
    }

    pub fn observe_init_commitment(&mut self, value: C) {
        self.observe_init_commitment_slice(core::slice::from_ref(&value))
    }

    /// Transition into a started prover transcript.
    pub fn into_prover(self) -> ProverTranscript<F, C, Ch> {
        ProverTranscript::new(self.challenger)
    }

    /// Transition into a started verifier transcript over provided data.
    pub fn into_verifier<'a>(
        self,
        data: &'a TranscriptData<F, C>,
    ) -> VerifierTranscript<'a, F, C, Ch> {
        VerifierTranscript::new(self.challenger, data)
    }
}
