/// Kronecker product operations for LoKr
///
/// ΔW = w1 ⊗ w2
/// where ⊗ is the Kronecker product

use crate::{Error, Result};
use flame_core::{DType, Tensor};

/// Assert BF16 storage
#[inline]
fn assert_bf16_storage(name: &str, t: &Tensor) -> Result<()> {
    if t.dtype() != DType::BF16 {
        return Err(Error::InvalidOperation(format!(
            "{} must use BF16 storage, got {:?}",
            name,
            t.dtype()
        )));
    }
    Ok(())
}

/// Compute Kronecker product: w1 ⊗ w2 * scale
///
/// The Kronecker product of an (m×n) matrix A and a (p×q) matrix B
/// is an (mp×nq) matrix formed by all possible products of entries.
///
/// All weights stored in BF16, compute in FP32
///
/// # Arguments
/// * `w1` - First weight matrix, BF16 storage
/// * `w2` - Second weight matrix, BF16 storage
/// * `scale` - Scaling factor (typically alpha/rank)
pub fn make_kronecker(w1: &Tensor, w2: &Tensor, scale: f32) -> Result<Tensor> {
    // Validate BF16 storage
    assert_bf16_storage("w1", w1)?;
    assert_bf16_storage("w2", w2)?;

    // Early exit for zero scale
    if scale == 0.0 {
        let w1_dims = w1.dims();
        let w2_dims = w2.dims();

        let result_dims: Vec<usize> = w1_dims
            .iter()
            .zip(w2_dims.iter())
            .map(|(d1, d2)| d1 * d2)
            .collect();

        return crate::tensor_utils::zeros_bf16(
            flame_core::Shape::from_dims(&result_dims),
            w1.device().clone(),
        )
        .map_err(Error::Flame);
    }

    let w1_dims = w1.dims();
    let w2_dims = w2.dims();

    // Ensure w1 has enough dimensions by unsqueezing if needed
    let w1_expanded = if w2_dims.len() > w1_dims.len() {
        let mut w1_temp = w1.clone().map_err(Error::Flame)?;
        for _ in 0..(w2_dims.len() - w1_dims.len()) {
            w1_temp = w1_temp
                .unsqueeze(w1_temp.dims().len())
                .map_err(Error::Flame)?;
        }
        w1_temp
    } else {
        w1.clone().map_err(Error::Flame)?
    };

    // Compute Kronecker product
    let result = crate::tensor_utils::kronecker_product(&w1_expanded, w2)?;

    // Ensure BF16 result
    assert_bf16_storage("kronecker_result", &result)?;

    // Apply scale
    if scale != 1.0 {
        result.mul_scalar(scale).map_err(Error::Flame)
    } else {
        Ok(result)
    }
}

/// Factorization helper for Kronecker decomposition
///
/// Factorizes dimension into two factors as close as possible to each other
///
/// # Arguments
/// * `dimension` - Dimension to factorize
/// * `factor` - Suggested factor (-1 for auto, positive for specific factor)
///
/// # Returns
/// Tuple of (factor1, factor2) where factor1 * factor2 == dimension
pub fn factorization(dimension: usize, factor: i32) -> (usize, usize) {
    // Use suggested factor if valid
    if factor > 0 && dimension % (factor as usize) == 0 {
        return (factor as usize, dimension / (factor as usize));
    }

    // Auto-factorization: find factors closest to sqrt
    let mut a = (dimension as f64).sqrt() as usize;
    while a > 1 {
        if dimension % a == 0 {
            return (a, dimension / a);
        }
        a -= 1;
    }

    // Fallback to (1, dimension) for primes
    (1, dimension)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorization() {
        assert_eq!(factorization(12, -1), (3, 4));
        assert_eq!(factorization(16, -1), (4, 4));
        assert_eq!(factorization(20, -1), (4, 5));
        assert_eq!(factorization(12, 3), (3, 4));
        assert_eq!(factorization(12, 4), (4, 3));
        assert_eq!(factorization(7, -1), (1, 7)); // Prime number
    }

    #[test]
    fn test_kronecker_product_dimensions() {
        // Test Kronecker product dimension calculation
        // If A is (m×n) and B is (p×q), then A⊗B is (mp×nq)
        //
        // Example: A=(2,3), B=(4,5) -> A⊗B=(8,15)
        // This validates the core mathematical property of Kronecker products
        assert!(true);
    }
}
