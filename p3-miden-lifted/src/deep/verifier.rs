use alloc::vec::Vec;
use core::iter::zip;
use core::marker::PhantomData;

use super::DeepParams;
use super::utils::{observe_evals, reduce_with_powers};
use crate::deep::proof::DeepProof;
use crate::utils::MatrixGroupEvals;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::Dimensions;
use p3_miden_lmcs::{Lmcs, LmcsError};
use p3_util::{log2_strict_usize, reverse_bits_len};
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
    /// Trace commitments with their dimensions (one per trace tree).
    commitments: Vec<(L::Commitment, Vec<Dimensions>)>,

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
    ///
    /// This method handles the complete transcript flow:
    /// 1. Observes evaluations into the Fiat-Shamir transcript
    /// 2. Checks the proof-of-work witness
    /// 3. Samples DEEP batching challenges (α for columns, β for points)
    /// 4. Validates proof structure and computes reduced openings
    ///
    /// # Arguments
    /// - `proof`: DEEP proof containing PoW witness
    /// - `evals`: Claimed evaluations at each opening point
    /// - `eval_points`: The out-of-domain evaluation points
    /// - `commitments`: The trace commitments with their dimensions (one per trace tree)
    /// - `challenger`: The Fiat-Shamir challenger
    /// - `params`: DEEP parameters (alignment and proof_of_work_bits)
    ///
    /// We reduce each opening's evaluations to `f_reduced(zⱼ) = Σᵢ αⁱ · fᵢ(zⱼʳ)` eagerly.
    /// This optimization is possible because all columns share the same opening points—
    /// at query time, we only compute one Horner reduction per query, not per-column.
    ///
    /// # Errors
    ///
    /// Returns `DeepError` if the proof structure is invalid or PoW witness is invalid.
    pub fn new<Challenger, const N: usize>(
        params: &DeepParams,
        eval_points: [EF; N],
        evals: &[Vec<MatrixGroupEvals<EF>>],
        commitments: Vec<(L::Commitment, Vec<Dimensions>)>,
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
            for (eval_group, (_, dims)) in zip(point_evals, &commitments) {
                if eval_group.num_matrices() != dims.len() {
                    return Err(DeepError::StructureMismatch);
                }
                for (matrix_evals, matrix_dims) in zip(eval_group.iter_matrices(), dims) {
                    if matrix_evals.len() != matrix_dims.width {
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
            reduced_openings,
            challenge_columns,
            challenge_points,
            alignment: params.alignment,
            _marker: PhantomData,
        })
    }

    /// Verify Merkle openings and compute `Q(X)` at the queried domain points.
    ///
    /// Reduces opened row values via Horner to get `f_reduced(X)`, then computes
    /// `Σⱼ βʲ · (f_reduced(zⱼ) - f_reduced(X)) / (zⱼ - X)` for each queried index.
    ///
    /// Returns a vector of DEEP evaluations, one per query index.
    ///
    /// # Arguments
    ///
    /// - `lmcs`: The LMCS instance for verification
    /// - `indices`: Query indices in the evaluation domain
    /// - `proofs`: Compact multi-opening proofs, one per trace tree
    pub fn query(
        &self,
        lmcs: &L,
        indices: &[usize],
        proofs: &[L::Proof],
    ) -> Result<Vec<EF>, DeepError> {
        // Verify each commitment's proof and collect opened rows
        // all_opened_rows[commit_idx][query_idx] = Vec<&[F]> (rows for that commitment)
        let all_opened_rows: Vec<Vec<Vec<&[F]>>> = zip(&self.commitments, proofs)
            .map(|((commit, dims), p)| {
                lmcs.verify(commit, dims, indices, p)
                    .map_err(DeepError::LmcsError)
            })
            .collect::<Result<_, _>>()?;

        // Compute max height across all commitments for domain reconstruction
        let max_height = self
            .commitments
            .iter()
            .flat_map(|(_, dims)| dims.iter().map(|d| d.height))
            .max()
            .unwrap();
        let log_max_height = log2_strict_usize(max_height);
        let generator = F::two_adic_generator(log_max_height);
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
                let index_bit_rev = reverse_bits_len(index, log_max_height);
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
    /// Claimed evaluations don't match commitment structure.
    ///
    /// This can mean wrong number of evaluation groups, matrices, or columns.
    #[error("evaluation structure doesn't match commitment")]
    StructureMismatch,
    /// Proof-of-work witness verification failed.
    #[error("invalid proof-of-work witness")]
    InvalidPowWitness,
    /// LMCS verification failed.
    #[error("LMCS error: {0}")]
    LmcsError(#[from] LmcsError),
}
