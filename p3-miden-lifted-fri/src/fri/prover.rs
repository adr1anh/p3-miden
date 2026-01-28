use alloc::vec::Vec;
use core::ops::Deref;

use p3_challenger::CanSample;
use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::extension::FlatMatrixView;
use p3_maybe_rayon::prelude::*;
use p3_miden_lmcs::{Lmcs, LmcsTree};
use p3_miden_transcript::ProverChannel;
use p3_util::{log2_strict_usize, reverse_slice_index_bits};

use crate::fri::FriParams;

/// Tree type for FRI folding rounds.
///
/// Stores extension field evaluations flattened to base field via `FlatMatrixView`.
type FoldedTree<F, EF, L> = <L as Lmcs>::Tree<FlatMatrixView<F, EF, RowMajorMatrix<EF>>>;

// ============================================================================
// Prover Data Structure
// ============================================================================

/// Prover's state from the FRI commit phase.
///
/// Contains the data needed to answer queries (LMCS trees).
/// Commitments, PoW witnesses, and the final polynomial are written to the
/// transcript channel during construction.
///
/// Uses a single base-field LMCS. Extension field evaluations are flattened
/// to base field before commitment.
pub struct FriPolys<F, EF, L>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    L: Lmcs<F = F>,
{
    /// Trees for each folding round, used to open multiple query indices at once.
    /// Stores flattened base-field matrices (EF elements flattened to F via FlatMatrixView).
    folded_trees: Vec<FoldedTree<F, EF, L>>,
}

// ============================================================================
// Commit Phase (Prover)
// ============================================================================
//
// The FRI commit phase iteratively folds a polynomial until it reaches a
// target degree, committing to intermediate evaluations along the way.
//
// ## Algorithm
//
// Given polynomial f of degree d with evaluations on domain D of size n = d·blowup:
//
// 1. Reshape evaluations into matrix M with `arity` columns
//    - Row i contains the coset {f(s·ωʲ) : j ∈ [0, arity)} where s = g^{bitrev(i)}
//
// 2. Commit to M via Merkle tree
//
// 3. Sample folding challenge β from verifier
//
// 4. Fold each row: for coset evaluations [y₀, y₁, ...], compute f(β)
//    - This reduces degree by factor of `arity`
//    - New evaluations live on domain D' of size n/arity
//
// 5. Repeat until degree ≤ final_degree
//
// 6. Send final polynomial coefficients to verifier
//
// ## Coset Structure in Bit-Reversed Order
//
// For domain D = g·H where H = ⟨ω⟩ has order n:
//   - Row i contains evaluations at s·⟨ω_arity⟩ where s = g·ω^{bitrev(i)}
//   - Adjacent rows have s values that are negatives (for arity=2)
//   - After folding, row i maps to row i in the halved domain

impl<F, EF, L> FriPolys<F, EF, L>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    L: Lmcs<F = F>,
{
    /// Execute the FRI commit phase.
    ///
    pub fn new<Ch>(params: &FriParams, lmcs: &L, evals: Vec<EF>, channel: &mut Ch) -> Self
    where
        Ch: ProverChannel<F = F, Commitment = L::Commitment> + CanSample<F>,
    {
        let log_arity = params.fold.log_arity();
        let arity = params.fold.arity();

        let mut folded_trees = Vec::new();

        let mut domain_size = evals.len();
        let log_domain_size = log2_strict_usize(domain_size); // needed for two_adic_generator
        let final_poly_degree = params.final_poly_degree(log_domain_size);
        let final_domain_size = final_poly_degree << params.log_blowup;

        // ─────────────────────────────────────────────────────────────────────────
        // Precompute s⁻¹ for all cosets
        // ─────────────────────────────────────────────────────────────────────────
        // Evaluations are in bit-reversed order: evals[i] = f(g^{bitrev(i)})
        // Row k contains [evals[k*arity], evals[k*arity+1], ...] which correspond
        // to evaluations at points forming a coset s·⟨ω⟩ where:
        //   - s = g^{bitrev(k*arity, log_domain_size)} = g^{bitrev(k, log_num_rows)}
        //     (because bitrev(k*arity, log_domain_size) = bitrev(k, log_num_rows)
        //      when arity = 2^log_arity)
        //   - ω is a primitive arity-th root of unity
        //
        // We compute s_inv for each row k, where s = g^{bitrev(k, log_num_rows)}
        // and g has order 2^log_domain_size.
        //
        // We generate sequential powers of g_inv and bit-reverse to get s_inv values
        // in the correct order for each row.
        let mut num_rows = domain_size >> log_arity;

        let g_inv = F::two_adic_generator(log_domain_size).inverse();
        let mut s_invs: Vec<F> = g_inv.powers().take(num_rows).collect();
        reverse_slice_index_bits(&mut s_invs);

        let mut folded_evals = evals;
        while domain_size > final_domain_size {
            // ─────────────────────────────────────────────────────────────────────
            // Reshape into matrix and wrap with FlatMatrixView for commitment
            // ─────────────────────────────────────────────────────────────────────
            // FlatMatrixView presents EF matrix as F matrix without copying.
            let matrix = RowMajorMatrix::new(folded_evals, arity);
            let flat_view = FlatMatrixView::new(matrix);
            let tree = lmcs.build_tree(alloc::vec![flat_view]);
            let commitment = tree.root();
            channel.send_commitment(commitment.clone());

            // ─────────────────────────────────────────────────────────────────────
            // Grind and sample folding challenge β
            // ─────────────────────────────────────────────────────────────────────
            let _pow_witness = channel.grind(params.proof_of_work_bits);
            let beta: EF = channel.sample_algebra_element();

            // ─────────────────────────────────────────────────────────────────────
            // Fold all rows: f(β) = interpolate coset evaluations at β
            // ─────────────────────────────────────────────────────────────────────
            // Get the underlying EF matrix from the FlatMatrixView via Deref for folding.
            let flat_view_ref = &tree.leaves()[0];
            let ef_matrix: &RowMajorMatrix<EF> = flat_view_ref.deref();
            folded_evals = params.fold.fold_matrix(ef_matrix.as_view(), &s_invs, beta);
            // No bit-reversal needed: folded evals maintain bit-reversed order
            // because s_invs are already bit-reversed to match

            folded_trees.push(tree);

            domain_size /= arity;

            // ─────────────────────────────────────────────────────────────────────
            // Update s⁻¹ for next round
            // ─────────────────────────────────────────────────────────────────────
            // After folding, domain shrinks by `arity`. The new generator g' = g^arity.
            // We need: s'_inv[k] = g'^{-bitrev(k, L')} = g^{-arity * bitrev(k, L')}
            //
            // Using the identity bitrev(k, L-log_arity) = bitrev(k*arity, L):
            //   s'_inv[k] = g^{-arity * bitrev(k*arity, L)}
            //             = (g^{-bitrev(k*arity, L)})^arity
            //             = s_inv[k*arity]^arity
            //
            // So we select every `arity`-th element and raise to power `arity`.
            // After domain_size update, new num_rows = domain_size / arity.
            num_rows = domain_size >> log_arity;
            s_invs = (0..num_rows)
                .into_par_iter()
                .map(|k| s_invs[k * arity].exp_power_of_2(log_arity))
                .collect();
        }

        // ─────────────────────────────────────────────────────────────────────────
        // Extract final polynomial coefficients
        // ─────────────────────────────────────────────────────────────────────────
        // The remaining evaluations are on a domain of size `final_domain_size`.
        // The polynomial degree is `final_poly_degree = final_domain_size / blowup`.
        // We need coefficients of degree < final_poly_degree, so we:
        // 1. Take the first `final_poly_degree` evaluations (others are redundant due to blowup)
        // 2. Convert from bit-reversed to standard order
        // 3. Apply inverse DFT to get coefficients
        folded_evals.truncate(final_poly_degree);
        reverse_slice_index_bits(&mut folded_evals);

        let final_poly = Radix2DFTSmallBatch::default().idft_algebra(folded_evals);

        // Observe final polynomial coefficients for Fiat-Shamir
        channel.send_algebra_slice(&final_poly);

        Self { folded_trees }
    }

    /// Stream all FRI query proofs into a transcript channel.
    pub fn prove_queries<Ch>(&self, params: &FriParams, indices: &[usize], channel: &mut Ch)
    where
        Ch: ProverChannel<F = F, Commitment = L::Commitment>,
    {
        let log_arity = params.fold.log_arity();

        // For each round, compute the corresponding indices (shifted from previous round)
        for (round, tree) in self.folded_trees.iter().enumerate() {
            // Compute indices for this round: shift each index by log_arity * (round + 1)
            // The +1 accounts for the initial fold from evaluation domain to first round
            let round_indices: Vec<usize> = indices
                .iter()
                .map(|&idx| idx >> (log_arity * (round + 1)))
                .collect();
            tree.prove_batch(&round_indices, channel);
        }
    }
}
