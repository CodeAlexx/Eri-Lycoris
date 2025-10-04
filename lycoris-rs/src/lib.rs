//! LyCORIS-RS: Rust port of LyCORIS library for LoRA algorithms
//!
//! Implements LoRA, LoHa, and LoKr with BF16 storage and FP32 compute
//! using the Flame tensor library.
//!
//! # Features
//! - BF16 storage with FP32 compute for numerical stability
//! - Stream-aware GPU operations
//! - NHWC ↔ NCHW layout adapters
//! - Comprehensive testing suite

pub mod algorithms;
pub mod ops;
pub mod kernels;
pub mod error;
pub mod dtype;
pub mod layout;
pub mod tensor_utils;

pub use error::{Error, Result};
pub use dtype::DType;
pub use layout::{TensorLayout, LayoutConverter};

// Re-export core Flame types
pub use flame_core::{Tensor, Shape, Device};

/// Module trait for all LyCORIS modules
pub trait LycorisModule {
    /// Forward pass through the module
    fn forward(&self, x: &Tensor) -> Result<Tensor>;

    /// Get the differential weight ΔW
    fn get_diff_weight(&self) -> Result<Tensor>;

    /// Merge LoRA weights into base weights
    fn merge_to(&mut self, multiplier: f32) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        // Basic sanity test
        assert_eq!(2 + 2, 4);
    }
}
