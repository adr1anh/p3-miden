//! Lifted STARK shared scaffolding (LMCS-based).
//!
//! This crate contains shared types, layout logic, transcript helpers, periodic
//! column utilities, and constraint folder machinery used by the lifted STARK
//! prover and verifier crates.
//!
//! See `notes` (module doc comments) and `notes.md` for exploration notes and
//! outstanding design questions.

#![no_std]
#![allow(dead_code, unused_imports)]

extern crate alloc;

mod config;
mod folder;
mod layout;
mod notes;
mod periodic;
mod proof;
mod selectors;
mod transcript;
mod utils;

pub use config::*;
pub use folder::*;
pub use layout::*;
pub use periodic::*;
pub use proof::*;
pub use selectors::*;
pub use transcript::*;
pub use utils::*;
