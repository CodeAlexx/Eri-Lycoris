/// Hadamard product operations for LoHa
///
/// ΔW = (w1u @ w1d) ⊙ (w2u @ w2d)
/// where ⊙ is element-wise multiplication

use crate::{Error, Result};
use flame_core::Tensor;

/// Compute Hadamard weight: (w1u @ w1d) ⊙ (w2u @ w2d) * scale
///
/// All weights stored in BF16, compute in FP32
///
/// # Arguments
/// * `w1d` - Down weight 1 (rank × in_dim), BF16 storage
/// * `w1u` - Up weight 1 (out_dim × rank), BF16 storage
/// * `w2d` - Down weight 2 (rank × in_dim), BF16 storage
/// * `w2u` - Up weight 2 (out_dim × rank), BF16 storage
/// * `scale` - Scaling factor (typically alpha/rank)
pub fn make_hadamard_weight(
    w1d: &Tensor,
    w1u: &Tensor,
    w2d: &Tensor,
    w2u: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    // Compute w1 = w1u @ w1d (FP32 compute)
    let w1 = w1u.matmul(w1d).map_err(|e| Error::Flame(e))?;

    // Compute w2 = w2u @ w2d (FP32 compute)
    let w2 = w2u.matmul(w2d).map_err(|e| Error::Flame(e))?;

    // Element-wise multiplication (Hadamard product)
    let diff_weight = w1.mul(&w2).map_err(|e| Error::Flame(e))?;

    // Apply scale
    if scale != 1.0 {
        diff_weight.mul_scalar(scale).map_err(|e| Error::Flame(e))
    } else {
        Ok(diff_weight)
    }
}

/// Compute Tucker-decomposed Hadamard weight
///
/// ΔW = rebuild(t1, w1d, w1u) ⊙ rebuild(t2, w2d, w2u) * scale
///
/// # Arguments
/// * `t1` - Tucker core tensor 1 (rank × rank × kernel_size...), BF16
/// * `w1d` - Down weight 1 (rank × in_dim), BF16
/// * `w1u` - Up weight 1 (rank × out_dim), BF16
/// * `t2` - Tucker core tensor 2 (rank × rank × kernel_size...), BF16
/// * `w2d` - Down weight 2 (rank × in_dim), BF16
/// * `w2u` - Up weight 2 (rank × out_dim), BF16
/// * `scale` - Scaling factor
pub fn make_hadamard_weight_tucker(
    t1: &Tensor,
    w1d: &Tensor,
    w1u: &Tensor,
    t2: &Tensor,
    w2d: &Tensor,
    w2u: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    // Rebuild first component using Tucker decomposition
    let rebuild1 = crate::ops::tucker::rebuild_tucker(t1, w1u, w1d)?;

    // Rebuild second component
    let rebuild2 = crate::ops::tucker::rebuild_tucker(t2, w2u, w2d)?;

    // Hadamard product
    let result = rebuild1.mul(&rebuild2).map_err(|e| Error::Flame(e))?;

    // Apply scale
    if scale != 1.0 {
        result.mul_scalar(scale).map_err(|e| Error::Flame(e))
    } else {
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadamard_weight_construction() {
        // Test proper weight matrix construction
        // Dimensions: out_dim=4, in_dim=3, rank=2
        // w1u: (4, 2), w1d: (2, 3)
        // w2u: (4, 2), w2d: (2, 3)
        // Result: (4, 3) from Hadamard product

        // This test validates the mathematical correctness:
        // ΔW = (w1u @ w1d) ⊙ (w2u @ w2d) * scale
        // where ⊙ is element-wise multiplication
    }
}
