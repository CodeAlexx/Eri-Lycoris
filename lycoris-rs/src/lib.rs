//! LyCORIS-RS: Rust port of LyCORIS library for LoRA algorithms
//!
//! Implements LoCon, LoHa, LoKr, and Full adapters on top of `flame-core`.
//!
//! # Features
//! - BF16 storage with FP32 compute for numerical stability
//! - NHWC ↔ NCHW layout adapters
//! - safetensors loader with auto-detected adapter type
//! - Weight-merge `apply_to` API for inference-time fusion

pub mod algorithms;
pub mod ops;
pub mod error;
pub mod dtype;
pub mod layout;
pub mod tensor_utils;
pub mod loader;

pub use error::{Error, Result};
pub use dtype::DType;
pub use layout::{TensorLayout, LayoutConverter};

// Re-export core Flame types
pub use flame_core::{Tensor, Shape, Device};

// Re-export adapter structs
pub use algorithms::full::FullAdapter;
pub use algorithms::locon::LoConModule as LoconAdapter;
pub use algorithms::loha::LoHaModule as LohaAdapter;
pub use algorithms::lokr::LoKrModule as LokrAdapter;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use cudarc::driver::CudaDevice;

/// Module trait for all LyCORIS modules
pub trait LycorisModule {
    /// Forward pass through the module
    fn forward(&self, x: &Tensor) -> Result<Tensor>;

    /// Get the differential weight ΔW
    fn get_diff_weight(&self) -> Result<Tensor>;

    /// Merge LoRA weights into base weights
    fn merge_to(&mut self, multiplier: f32) -> Result<()>;
}

/// Top-level adapter variant — one entry per Kohya/LyCORIS prefix in a
/// safetensors checkpoint.
pub enum LycorisAdapter {
    LoCon(LoconAdapter),
    LoHa(LohaAdapter),
    LoKr(LokrAdapter),
    Full(FullAdapter),
}

impl LycorisAdapter {
    /// Returns the unscaled ΔW for this adapter (alpha/rank already applied
    /// inside the math). Caller multiplies by `strength` and adds to base.
    pub fn delta_weight(&self) -> Result<Tensor> {
        match self {
            LycorisAdapter::LoCon(m) => m.get_diff_weight(),
            LycorisAdapter::LoHa(m)  => m.get_diff_weight(),
            LycorisAdapter::LoKr(m)  => m.get_diff_weight(),
            LycorisAdapter::Full(m)  => m.delta_weight(1.0),
        }
    }
}

/// A collection of LyCORIS adapters keyed by Kohya prefix
/// (e.g. `lora_unet_down_blocks_0_attentions_0_proj_in`).
pub struct LycorisCollection {
    pub adapters: HashMap<String, LycorisAdapter>,
}

impl LycorisCollection {
    /// Load a LyCORIS safetensors file and auto-detect each adapter's type.
    pub fn load(path: &Path, device: Arc<CudaDevice>) -> anyhow::Result<Self> {
        loader::load(path, device)
    }

    /// Weight-merge mode. For each adapter, compute ΔW, reshape to base
    /// weight shape, and add `strength * ΔW` to the base tensor in place
    /// (replaces the entry in `weights`).
    ///
    /// `name_mapper` translates a LyCORIS adapter prefix into the caller's
    /// weight-dict key. Returning `None` skips that adapter.
    pub fn apply_to(
        &self,
        weights: &mut HashMap<String, Tensor>,
        strength: f32,
        name_mapper: impl Fn(&str) -> Option<String>,
    ) -> anyhow::Result<()> {
        loader::apply_collection(self, weights, strength, name_mapper)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_basic() {
        // Basic sanity test
        assert_eq!(2 + 2, 4);
    }
}
