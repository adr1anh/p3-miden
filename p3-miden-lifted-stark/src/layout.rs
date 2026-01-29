//! Instance layout helpers (per-AIR degrees/widths/permutation).
//!
//! The layout is serialized into the transcript to make verification replayable.
//! See the note on `LayoutSnapshot` for long-term cleanup of redundant fields.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, PrimeField64};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_air::MidenAir;
use p3_miden_transcript::{ProverChannel, VerifierChannel};
use p3_util::log2_strict_usize;

use crate::transcript::{read_usize, read_usize_list, write_usize, write_usize_list};
use crate::utils::align_width;

/// Prover-side layout (contains derived data like ratios).
///
/// Intern notes:
/// - `permutation` sorts AIRs by trace height (ascending) with stable tie-breaks.
///   The prover commits/openings in this order; the verifier uses the same order
///   when reconstructing constraint checks.
/// - `ratios[i] = N / n_i` where N is the max trace height and n_i is this AIR's
///   trace height. These ratios determine nested coset domains (gK)^r.
/// - `trace_widths` / `aux_widths` are padded to `alignment` for LMCS hints.
///   When checking constraints, we trim back to the AIR's real widths.
/// - `quotient_widths` is currently a single aligned width for the combined
///   quotient (EF flattened to base field).
#[derive(Clone, Debug)]
pub struct TraceLayout {
    pub num_airs: usize,
    pub log_degrees: Vec<usize>,
    pub permutation: Vec<usize>,
    pub log_max_degree: usize,
    pub log_max_height: usize,
    pub ratios: Vec<usize>,
    pub num_randomness: Vec<usize>,
    pub trace_widths: Vec<usize>,
    pub aux_widths: Vec<usize>,
    pub quotient_widths: Vec<usize>,
}

impl TraceLayout {
    pub fn new<F, EF, A>(
        airs: &[A],
        traces: &[RowMajorMatrix<F>],
        alignment: usize,
        log_blowup: usize,
    ) -> Self
    where
        F: Field + PrimeCharacteristicRing,
        EF: ExtensionField<F>,
        A: MidenAir<F, EF>,
    {
        let num_airs = airs.len();
        let mut log_degrees = Vec::with_capacity(num_airs);
        let mut heights = Vec::with_capacity(num_airs);
        let mut num_randomness = Vec::with_capacity(num_airs);

        for (idx, (air, trace)) in airs.iter().zip(traces).enumerate() {
            let n = trace.height();
            assert!(
                n.is_power_of_two(),
                "trace heights must be powers of two (air index {idx})"
            );
            let log_n = log2_strict_usize(n);
            log_degrees.push(log_n);
            heights.push(n);
            num_randomness.push(air.num_randomness());
        }

        let mut permutation: Vec<usize> = (0..num_airs).collect();
        permutation.sort_by_key(|&i| (heights[i], i));

        let log_max_degree = *log_degrees.iter().max().unwrap_or(&0);
        let log_max_height = log_max_degree + log_blowup;

        let mut ratios = Vec::with_capacity(num_airs);
        for &log_n in &log_degrees {
            let log_r = log_max_degree - log_n;
            ratios.push(1usize << log_r);
        }

        let mut trace_widths = Vec::with_capacity(num_airs);
        let mut aux_widths = Vec::with_capacity(num_airs);
        for &idx in &permutation {
            let air = &airs[idx];
            trace_widths.push(align_width(air.width(), alignment));
            aux_widths.push(align_width(air.aux_width() * EF::DIMENSION, alignment));
        }

        // Single combined quotient for now.
        let quotient_widths = vec![align_width(EF::DIMENSION, alignment)];

        Self {
            num_airs,
            log_degrees,
            permutation,
            log_max_degree,
            log_max_height,
            ratios,
            num_randomness,
            trace_widths,
            aux_widths,
            quotient_widths,
        }
    }

    pub fn snapshot(&self) -> LayoutSnapshot {
        LayoutSnapshot {
            num_airs: self.num_airs,
            log_degrees: self.log_degrees.clone(),
            permutation: self.permutation.clone(),
            log_max_degree: self.log_max_degree,
            log_max_height: self.log_max_height,
            num_randomness: self.num_randomness.clone(),
            trace_widths: self.trace_widths.clone(),
            aux_widths: self.aux_widths.clone(),
            quotient_widths: self.quotient_widths.clone(),
        }
    }
}

/// Layout snapshot serialized into the transcript.
///
/// Intern notes:
/// - This is intentionally verbose for now. We expect to split it later into
///   "public params" vs "instance data", and drop derivable fields like
///   `log_max_degree` and `log_max_height`.
#[derive(Clone, Debug)]
// NOTE: This should eventually be split between PublicParams (protocol-level, fixed across instances)
// and instance-specific data. Anything derivable from PublicParams + per-AIR log_degrees should not be
// serialized here. For example, log_max_degree/log_max_height/ratios and quotient widths can be derived
// from log_degrees + log_blowup (in PublicParams). Only irreducible instance data (num_airs, log_degrees,
// permutation, widths, num_randomness, periodic tables) should remain in the transcript snapshot.
pub struct LayoutSnapshot {
    pub num_airs: usize,
    pub log_degrees: Vec<usize>,
    pub permutation: Vec<usize>,
    pub log_max_degree: usize,
    pub log_max_height: usize,
    pub num_randomness: Vec<usize>,
    pub trace_widths: Vec<usize>,
    pub aux_widths: Vec<usize>,
    pub quotient_widths: Vec<usize>,
}

impl LayoutSnapshot {
    pub fn write_to_channel<F, Ch>(&self, channel: &mut Ch) -> Option<()>
    where
        F: PrimeField64,
        Ch: ProverChannel<F = F>,
    {
        write_usize::<F, _>(channel, self.num_airs)?;
        write_usize_list::<F, _>(channel, &self.log_degrees)?;
        write_usize_list::<F, _>(channel, &self.permutation)?;
        write_usize::<F, _>(channel, self.log_max_degree)?;
        write_usize::<F, _>(channel, self.log_max_height)?;
        write_usize_list::<F, _>(channel, &self.num_randomness)?;
        write_usize_list::<F, _>(channel, &self.trace_widths)?;
        write_usize_list::<F, _>(channel, &self.aux_widths)?;
        write_usize_list::<F, _>(channel, &self.quotient_widths)?;
        Some(())
    }

    pub fn read_from_channel<F, Ch>(channel: &mut Ch) -> Option<Self>
    where
        F: PrimeField64,
        Ch: VerifierChannel<F = F>,
    {
        let num_airs = read_usize::<F, _>(channel)?;
        let log_degrees = read_usize_list::<F, _>(channel, num_airs)?;
        let permutation = read_usize_list::<F, _>(channel, num_airs)?;
        let log_max_degree = read_usize::<F, _>(channel)?;
        let log_max_height = read_usize::<F, _>(channel)?;
        let num_randomness = read_usize_list::<F, _>(channel, num_airs)?;
        let trace_widths = read_usize_list::<F, _>(channel, num_airs)?;
        let aux_widths = read_usize_list::<F, _>(channel, num_airs)?;
        let quotient_widths = read_usize_list::<F, _>(channel, 1)?;

        Some(Self {
            num_airs,
            log_degrees,
            permutation,
            log_max_degree,
            log_max_height,
            num_randomness,
            trace_widths,
            aux_widths,
            quotient_widths,
        })
    }
}
