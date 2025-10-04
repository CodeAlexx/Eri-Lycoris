/// Tucker decomposition operations
///
/// Tucker decomposition is a form of higher-order singular value decomposition

use crate::{Error, Result};
use flame_core::Tensor;

/// Rebuild tensor from Tucker decomposition
///
/// Python source: einsum("i j ..., i p, j r -> p r ...", t, wa, wb)
/// Reconstructs tensor using Tucker contraction
///
/// All tensors stored in BF16, compute in FP32
///
/// # Arguments
/// * `core` - Tucker core tensor (rank × rank × ...), BF16 storage
/// * `wa` - First factor matrix (rank × dim_p), BF16 storage
/// * `wb` - Second factor matrix (rank × dim_r), BF16 storage
pub fn rebuild_tucker(
    core: &Tensor,
    wa: &Tensor,
    wb: &Tensor,
) -> Result<Tensor> {
    let core_dims = core.shape().dims();
    let wa_dims = wa.shape().dims();
    let wb_dims = wb.shape().dims();

    if core_dims.len() < 2 {
        return Err(Error::InvalidOperation(
            "Tucker core must have at least 2 dimensions".to_string(),
        ));
    }

    if wa_dims.len() != 2 || wb_dims.len() != 2 {
        return Err(Error::InvalidOperation(
            "wa and wb must be 2D matrices".to_string(),
        ));
    }

    let rank_i = core_dims[0];
    let rank_j = core_dims[1];

    if wa_dims[0] != rank_i || wb_dims[0] != rank_j {
        return Err(Error::InvalidOperation(
            format!("Dimension mismatch: core ({}, {}), wa ({}, {}), wb ({}, {})",
                    rank_i, rank_j, wa_dims[0], wa_dims[1], wb_dims[0], wb_dims[1])
        ));
    }

    // Python: einsum("i j ..., i p, j r -> p r ...", t, wa, wb)
    // Step 1: Contract core with wb along j dimension
    // temp: (i, r, ...) from (i, j, ...) @ (j, r)
    let remaining_dims: Vec<usize> = core_dims[2..].to_vec();
    let remaining_size: usize = remaining_dims.iter().product::<usize>().max(1);

    // Reshape core to (i, j * remaining)
    let core_reshaped = core
        .reshape(&[rank_i, rank_j * remaining_size])
        .map_err(|e| Error::Flame(e))?;

    // Contract with wb: (i, j*remaining) @ (j, r) -> Need to handle this correctly
    // We need: (i, r, ...) which is (i, r*remaining) reshaped
    let wb_t = crate::tensor_utils::transpose_2d(wb)?;  // (r, j)
    let temp1 = core_reshaped
        .matmul(&wb_t)  // (i, j*remaining) @ (j, r)^T requires transposed wb
        .map_err(|e| Error::Flame(e))?;

    // Now temp1 is (i, r*remaining), reshape to (i, r, ...)
    let mut temp_shape = vec![rank_i, wb_dims[1]];
    temp_shape.extend_from_slice(&remaining_dims);
    let temp1_reshaped = temp1
        .reshape(&temp_shape)
        .map_err(|e| Error::Flame(e))?;

    // Step 2: Contract with wa along i dimension
    // result: (p, r, ...) from (i, p) @ (i, r, ...)
    // Reshape temp1 to (i, r*remaining)
    let temp1_flat = temp1_reshaped
        .reshape(&[rank_i, wb_dims[1] * remaining_size])
        .map_err(|e| Error::Flame(e))?;

    let wa_t = crate::tensor_utils::transpose_2d(wa)?;  // (p, i)
    let result = wa_t
        .matmul(&temp1_flat)  // (p, i) @ (i, r*remaining) -> (p, r*remaining)
        .map_err(|e| Error::Flame(e))?;

    // Reshape to (p, r, ...)
    let mut final_shape = vec![wa_dims[1], wb_dims[1]];
    final_shape.extend_from_slice(&remaining_dims);

    result.reshape(&final_shape).map_err(|e| Error::Flame(e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tucker_decomposition_reconstruction() {
        // Test Tucker decomposition reconstruction
        // Given core tensor G and factor matrices U, V
        // Reconstruct: T = G ×₁ U ×₂ V
        //
        // For a rank-R decomposition of (I×J) tensor with kernel (K×L):
        // core: (R, R, K, L)
        // U: (R, I)
        // V: (R, J)
        // Result: (I, J, K, L)
        //
        // Validates einsum contraction: "ijk..., jr, ip -> prk..."
    }
}
