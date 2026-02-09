use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::iter::zip;
use core::marker::PhantomData;

use super::{DeepEvals, DeepParams};
use crate::utils::horner_acc;
use p3_challenger::CanSample;
use p3_field::{ExtensionField, TwoAdicField};
use p3_miden_lmcs::{Lmcs, LmcsError};
use p3_miden_transcript::{TranscriptError, VerifierChannel};
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
pub struct DeepOracle<F: TwoAdicField, EF: ExtensionField<F>, L: Lmcs<F = F>> {
    /// Trace commitments with their widths (one per trace tree).
    ///
    /// Widths must match the committed rows (including any alignment padding if
    /// `build_aligned_tree` was used).
    commitments: Vec<(L::Commitment, Vec<usize>)>,

    /// Log2 of the LDE domain height (tree has 2^log_lde_height leaves).
    /// Verifier expects all commitments to be lifted to this same LDE height.
    log_lde_height: usize,

    /// Reduced openings: pairs of `(zⱼ, f_reduced(zⱼ))` from the prover's claims.
    reduced_openings: Vec<(EF, EF)>,

    /// Challenge `α` for batching columns into `f_reduced`.
    challenge_columns: EF,
    /// Challenge `β` for batching opening points.
    challenge_points: EF,

    _marker: PhantomData<F>,
}

impl<F: TwoAdicField, EF: ExtensionField<F>, L: Lmcs<F = F>> DeepOracle<F, EF, L> {
    /// Construct by reading evaluations, checking PoW, and sampling challenges.
    ///
    /// Commitment widths must match the committed rows (including any alignment padding).
    /// All commitments are expected to be lifted to the same `log_lde_height`.
    ///
    /// Preconditions: `eval_points` must be distinct and lie outside the trace subgroup `H`
    /// and LDE evaluation coset `gK`. The outer protocol is expected to enforce this.
    ///
    /// `log_lde_height` is the log₂ of the LDE evaluation domain height (i.e. the height of
    /// the committed LDE matrices). When a trace degree is known, it is typically
    /// `log_trace_height + params.fri.log_blowup` (plus any extension used by the caller).
    pub fn new<Ch>(
        params: &DeepParams,
        eval_points: &[EF],
        commitments: Vec<(L::Commitment, Vec<usize>)>,
        log_lde_height: usize,
        channel: &mut Ch,
    ) -> Result<(Self, DeepEvals<EF>), DeepError>
    where
        Ch: VerifierChannel<F = F, Commitment = L::Commitment> + CanSample<F>,
    {
        let widths: Vec<&[usize]> = commitments
            .iter()
            .map(|(_, widths)| widths.as_slice())
            .collect();
        let evals = DeepEvals::read_from_channel::<F, Ch>(&widths, eval_points.len(), channel)?;

        // 1. Check grinding witness
        channel.grind(params.deep_pow_bits)?;

        // 2. Sample DEEP challenges
        let challenge_columns: EF = channel.sample_algebra_element();
        let challenge_points: EF = channel.sample_algebra_element();

        // Reduce each opening's evaluations via Horner: (z_j, f_reduced(z_j))
        let reduced_openings: Vec<(EF, EF)> = eval_points
            .iter()
            .enumerate()
            .map(|(point_idx, point)| (*point, evals.reduce_point(point_idx, challenge_columns)))
            .collect();

        let oracle = Self {
            commitments,
            log_lde_height,
            reduced_openings,
            challenge_columns,
            challenge_points,
            _marker: PhantomData,
        };

        Ok((oracle, evals))
    }

    /// Open the oracle at given tree indices by reading proofs from a verifier channel.
    ///
    /// `tree_indices` are bit-reversed positions (sorted, deduplicated).
    /// Returns a map from tree index to DEEP evaluation at that point.
    pub fn open_batch<Ch>(
        &self,
        lmcs: &L,
        tree_indices: &BTreeSet<usize>,
        channel: &mut Ch,
    ) -> Result<BTreeMap<usize, EF>, DeepError>
    where
        Ch: VerifierChannel<F = F, Commitment = L::Commitment>,
    {
        let mut reduced_rows: BTreeMap<usize, EF> = BTreeMap::new();

        for (commit, widths) in &self.commitments {
            let opened_rows = lmcs
                .open_batch(
                    commit,
                    widths,
                    self.log_lde_height,
                    tree_indices.iter().copied(),
                    channel,
                )
                .map_err(DeepError::LmcsError)?;

            // Reduce opened rows via Horner: f_reduced(X) = Σᵢ αⁱ · fᵢ(X).
            //
            // `horner_acc` continues the running accumulation across commitment groups:
            // group 0's columns get the highest powers, group 1's continue from where
            // group 0 left off. This matches the prover's coefficient ordering and
            // `reduce_point`'s iteration order in `evals.rs`.
            for &tree_idx in tree_indices {
                let rows_for_query = &opened_rows[&tree_idx];
                let acc = reduced_rows.entry(tree_idx).or_insert(EF::ZERO);
                *acc = horner_acc(
                    *acc,
                    self.challenge_columns,
                    rows_for_query.iter().flatten().copied(),
                );
            }
        }

        let generator = F::two_adic_generator(self.log_lde_height);
        let shift = F::GENERATOR;

        // Compute DEEP evaluation for each query
        let evals: BTreeMap<usize, EF> = reduced_rows
            .into_iter()
            .map(|(tree_idx, reduced_row)| {
                // Recover domain point X = g·ω^{exp} from tree index (bit-reversed position)
                let exp = reverse_bits_len(tree_idx, self.log_lde_height);
                let row_point = shift * generator.exp_u64(exp as u64);

                // DEEP quotient: Q(X) = Σⱼ βʲ · (f_reduced(zⱼ) - f_reduced(X)) / (zⱼ - X)
                let deep_eval: EF = zip(&self.reduced_openings, self.challenge_points.powers())
                    .map(|((point, reduced_eval), coeff_point)| {
                        coeff_point * (*reduced_eval - reduced_row) / (*point - row_point)
                    })
                    .sum();
                (tree_idx, deep_eval)
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
    #[error("LMCS error: {0}")]
    LmcsError(#[from] LmcsError),
    #[error("transcript error: {0}")]
    TranscriptError(#[from] TranscriptError),
}
