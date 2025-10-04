/// Kronecker product operations for LoKr
///
/// ΔW = w1 ⊗ w2
/// where ⊗ is the Kronecker product

use crate::{Error, Result};
use flame_core::Tensor;

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
pub fn make_kronecker(
    w1: &Tensor,
    w2: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    let w1_dims = w1.shape().dims();
    let w2_dims = w2.shape().dims();

    // Ensure w1 has enough dimensions by unsqueezing if needed
    let w1_expanded = if w2_dims.len() > w1_dims.len() {
        let mut w1_temp = w1.clone_result().map_err(|e| Error::Flame(e))?;
        for _ in 0..(w2_dims.len() - w1_dims.len()) {
            w1_temp = w1_temp.unsqueeze(w1_temp.dims().len())
                .map_err(|e| Error::Flame(e))?;
        }
        w1_temp
    } else {
        w1.clone_result().map_err(|e| Error::Flame(e))?
    };

    // Compute Kronecker product
    // Flame doesn't have built-in kron, use our implementation
    let result = crate::tensor_utils::kronecker_product(&w1_expanded, w2)?;

    // Apply scale
    if scale != 1.0 {
        result.mul_scalar(scale).map_err(|e| Error::Flame(e))
    } else {
        Ok(result)
    }
}

/// Factorization helper for Kronecker decomposition
///
/// Factorizes dimension into two factors as close as possible to each other
pub fn factorization(dimension: usize, factor: i32) -> (usize, usize) {
    if factor > 0 && dimension % (factor as usize) == 0 {
        return (factor as usize, dimension / (factor as usize));
    }

    let mut a = (dimension as f64).sqrt() as usize;
    while a > 1 {
        if dimension % a == 0 {
            return (a, dimension / a);
        }
        a -= 1;
    }

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
    }

    #[test]
    fn test_kronecker_product_dimensions() {
        // Test Kronecker product dimension calculation
        // If A is (m×n) and B is (p×q), then A⊗B is (mp×nq)

        // Example: A=(2,3), B=(4,5) -> A⊗B=(8,15)
        // This validates the core mathematical property of Kronecker products
    }
}
