use alloc::vec::Vec;
use core::iter::zip;
use core::marker::PhantomData;

use super::DeepParams;
use super::utils::{observe_evals, reduce_with_powers};
use crate::deep::proof::DeepProof;
use crate::utils::MatrixGroupEvals;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, TwoAdicField};
use p3_miden_lmcs::{Lmcs, LmcsError};
use p3_util::reverse_bits_len;
use thiserror::Error;

/// Verifier's view of the DEEP quotient as a point-query oracle.
///
/// Stores commitments and the prover's reduced claims `(zⱼ, f_reduced(zⱼ))`.
/// At query time, verifies Merkle openings and reconstructs `Q(X)` at that point:
///
/// ```text
/// Q(X) = Σⱼ βʲ · (f_reduced(zⱼ) - f_reduced(X)) / (zⱼ - X)
/// ```
///
/// where `f_reduced = Σᵢ αⁱ · fᵢ` batches all polynomial columns.
///
/// From the verifier's perspective, all opened columns appear to have the same height—
/// lifting is transparent. The prover evaluates `fᵢ(zʳ)` for degree-d polynomials
/// (where r is the lift factor), but the verifier sees this as `fᵢ'(z)` where
/// `fᵢ'(X) = fᵢ(Xʳ)` is the lifted polynomial on the full domain.
///
/// An alternative implementation could open rows padded with zeros to the alignment
/// width, allowing the hasher to process fixed-size chunks. This implementation
/// uses alignment > 1 to support such padding virtually (without materializing zeros).
pub struct DeepOracle<F: TwoAdicField, EF: ExtensionField<F>, L: Lmcs<F = F>> {
    /// Trace commitments with their widths (one per trace tree).
    commitments: Vec<(L::Commitment, Vec<usize>)>,

    /// Log2 of the universal domain height (tree has 2^log_max_height leaves).
    /// May be larger than any actual trace height (for domain alignment).
    log_max_height: usize,

    /// Reduced openings: pairs of `(zⱼ, f_reduced(zⱼ))` from the prover's claims.
    reduced_openings: Vec<(EF, EF)>,

    /// Challenge `α` for batching columns into `f_reduced`.
    challenge_columns: EF,
    /// Challenge `β` for batching opening points.
    challenge_points: EF,

    /// Alignment width for Horner reduction (typically hasher's rate).
    /// When alignment > 1, coefficient indices are padded to multiples of this value,
    /// equivalent to virtually appending zeros to each row before hashing.
    alignment: usize,

    _marker: PhantomData<F>,
}

impl<F: TwoAdicField, EF: ExtensionField<F>, L: Lmcs<F = F>> DeepOracle<F, EF, L> {
    /// Construct by observing evaluations, checking PoW, and sampling challenges.
    pub fn new<Challenger, const N: usize>(
        params: &DeepParams,
        eval_points: [EF; N],
        evals: &[Vec<MatrixGroupEvals<EF>>],
        commitments: Vec<(L::Commitment, Vec<usize>)>,
        log_max_height: usize,
        challenger: &mut Challenger,
        proof: &DeepProof<Challenger::Witness>,
    ) -> Result<Self, DeepError>
    where
        Challenger: FieldChallenger<F> + GrindingChallenger,
    {
        // 1. Observe evaluations into transcript
        observe_evals::<F, EF, Challenger>(evals, challenger, params.alignment);

        // 2. Check grinding witness
        if !challenger.check_witness(params.proof_of_work_bits, proof.pow_witness) {
            return Err(DeepError::InvalidPowWitness);
        }

        // 3. Sample DEEP challenges
        let challenge_columns: EF = challenger.sample_algebra_element();
        let challenge_points: EF = challenger.sample_algebra_element();

        // Validate structure: evals[point_idx] should have `commitments.len()` groups
        // Structure: evals[point_idx][commit_idx][matrix_idx][col_idx]
        for point_evals in evals {
            if point_evals.len() != commitments.len() {
                return Err(DeepError::StructureMismatch);
            }
            for (eval_group, (_, widths)) in zip(point_evals, &commitments) {
                if eval_group.num_matrices() != widths.len() {
                    return Err(DeepError::StructureMismatch);
                }
                for (matrix_evals, width) in zip(eval_group.iter_matrices(), widths.iter().copied())
                {
                    if matrix_evals.len() != width {
                        return Err(DeepError::StructureMismatch);
                    }
                }
            }
        }

        // Validate number of eval points matches
        if evals.len() != N {
            return Err(DeepError::StructureMismatch);
        }

        // Reduce each opening's evaluations via Horner: (z_j, f_reduced(z_j))
        let reduced_openings: Vec<(EF, EF)> = zip(&eval_points, evals)
            .map(|(&point, point_evals)| {
                let slices = point_evals.iter().flat_map(|g| g.iter_matrices());
                let reduced_eval = reduce_with_powers(slices, challenge_columns, params.alignment);
                (point, reduced_eval)
            })
            .collect();

        Ok(Self {
            commitments,
            log_max_height,
            reduced_openings,
            challenge_columns,
            challenge_points,
            alignment: params.alignment,
            _marker: PhantomData,
        })
    }

    /// Open the oracle at given indices, verifying proofs and computing DEEP evaluations.
    ///
    /// Returns a vector of DEEP evaluations, one per query index.
    pub fn open_batch(
        &self,
        lmcs: &L,
        indices: &[usize],
        proofs: &[L::Proof],
    ) -> Result<Vec<EF>, DeepError> {
        // Verify each commitment's proof and collect opened rows
        // all_opened_rows[commit_idx][query_idx] = Vec<&[F]> (rows for that commitment)
        let all_opened_rows: Vec<Vec<Vec<&[F]>>> = zip(&self.commitments, proofs)
            .map(|((commit, widths), p)| {
                lmcs.open_batch(commit, widths, self.log_max_height, indices, p)
                    .map_err(DeepError::LmcsError)
            })
            .collect::<Result<_, _>>()?;

        // Use stored log_max_height for domain reconstruction
        let generator = F::two_adic_generator(self.log_max_height);
        let shift = F::GENERATOR;

        // Compute DEEP evaluation for each query
        let evals: Vec<EF> = indices
            .iter()
            .enumerate()
            .map(|(query_idx, &index)| {
                // Collect rows from all commitments for this query index
                // Rows are combined in order: commitment 0's rows, then commitment 1's rows, etc.
                let rows_iter = all_opened_rows
                    .iter()
                    .flat_map(|commit_rows| commit_rows[query_idx].iter().copied());

                // Reconstruct the domain point X from the query index.
                // The LDE domain is the coset gK in bit-reversed order where:
                //   g = F::GENERATOR (coset shift, avoids subgroup)
                //   K = <ω> with ω = primitive 2^log_n root of unity
                // In bit-reversed order: X = g · ω^{bit_rev(index)}
                let index_bit_rev = reverse_bits_len(index, self.log_max_height);
                let row_point = shift * generator.exp_u64(index_bit_rev as u64);

                let reduced_row =
                    reduce_with_powers(rows_iter, self.challenge_columns, self.alignment);

                zip(&self.reduced_openings, self.challenge_points.powers())
                    .map(|((point, reduced_eval), coeff_point)| {
                        coeff_point * (*reduced_eval - reduced_row) / (*point - row_point)
                    })
                    .sum()
            })
            .collect();

        Ok(evals)
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during DEEP oracle construction or verification.
#[derive(Debug, Error)]
pub enum DeepError {
    #[error("evaluation structure doesn't match commitment")]
    StructureMismatch,
    #[error("invalid proof-of-work witness")]
    InvalidPowWitness,
    #[error("LMCS error: {0}")]
    LmcsError(#[from] LmcsError),
}
