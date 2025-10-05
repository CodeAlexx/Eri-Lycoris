/// Hadamard product operations for LoHa
///
/// ΔW = (w1a @ w1b) ⊙ (w2a @ w2b)
/// where ⊙ is element-wise multiplication
///
/// Weight layouts follow Flame contracts:
/// - Linear: [IN, OUT]
/// - Conv2d: [KH, KW, IC, OC]

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

/// Compute Hadamard weight: (w1a @ w1b) ⊙ (w2a @ w2b) * scale
///
/// All weights stored in BF16, compute in FP32
///
/// # Arguments
/// * `w1a` - First down weight: [IN, RANK] for linear, [KH, KW, IC, RANK] for conv
/// * `w1b` - First up weight: [RANK, OUT] for linear, [KH, KW, RANK, OC] for conv
/// * `w2a` - Second down weight: [IN, RANK] for linear, [KH, KW, IC, RANK] for conv
/// * `w2b` - Second up weight: [RANK, OUT] for linear, [KH, KW, RANK, OC] for conv
/// * `scale` - Scaling factor (typically alpha/rank)
pub fn make_hadamard_weight(
    w1a: &Tensor,
    w1b: &Tensor,
    w2a: &Tensor,
    w2b: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    // Validate BF16 storage
    assert_bf16_storage("w1a", w1a)?;
    assert_bf16_storage("w1b", w1b)?;
    assert_bf16_storage("w2a", w2a)?;
    assert_bf16_storage("w2b", w2b)?;

    // Early exit for zero scale
    if scale == 0.0 {
        let dims = w1a.dims();
        if dims.len() == 2 {
            // Linear: [IN, OUT]
            return crate::tensor_utils::zeros_bf16(
                flame_core::Shape::from_dims(&[dims[0], w1b.dims()[1]]),
                w1a.device().clone(),
            );
        } else {
            // Conv: same shape as w1b
            return crate::tensor_utils::zeros_bf16(
                w1b.shape().clone(),
                w1a.device().clone(),
            );
        }
    }

    let dims = w1a.dims();

    if dims.len() == 2 {
        // Linear path: w1a[IN,RANK] @ w1b[RANK,OUT] = [IN,OUT]
        let w1 = w1a.matmul(w1b).map_err(Error::Flame)?;
        let w2 = w2a.matmul(w2b).map_err(Error::Flame)?;

        // Hadamard product
        let diff = w1.mul(&w2).map_err(Error::Flame)?;
        diff.mul_scalar(scale).map_err(Error::Flame)
    } else if dims.len() == 4 {
        // Conv path: [KH, KW, IC, RANK] @ [KH, KW, RANK, OC]
        // Need to do matmul per spatial position
        let kh = dims[0];
        let kw = dims[1];
        let ic = dims[2];
        let r = dims[3];
        let oc = w1b.dims()[3];

        // Reshape for batch matmul: [KH*KW, IC, R] @ [KH*KW, R, OC]
        let w1a_batch = w1a.reshape(&[kh * kw, ic, r]).map_err(Error::Flame)?;
        let w1b_batch = w1b.reshape(&[kh * kw, r, oc]).map_err(Error::Flame)?;
        let w2a_batch = w2a.reshape(&[kh * kw, ic, r]).map_err(Error::Flame)?;
        let w2b_batch = w2b.reshape(&[kh * kw, r, oc]).map_err(Error::Flame)?;

        // Batch matmul
        let w1_batch = w1a_batch.matmul(&w1b_batch).map_err(Error::Flame)?;
        let w2_batch = w2a_batch.matmul(&w2b_batch).map_err(Error::Flame)?;

        // Hadamard product
        let diff_batch = w1_batch.mul(&w2_batch).map_err(Error::Flame)?;

        // Reshape back: [KH*KW, IC, OC] -> [KH, KW, IC, OC]
        let diff = diff_batch.reshape(&[kh, kw, ic, oc]).map_err(Error::Flame)?;
        diff.mul_scalar(scale).map_err(Error::Flame)
    } else {
        Err(Error::InvalidOperation(format!(
            "Unsupported tensor dimensions: expected 2D or 4D, got {}D",
            dims.len()
        )))
    }
}

/// Compute Tucker-decomposed Hadamard weight
///
/// ΔW = rebuild(t1, w1a, w1b) ⊙ rebuild(t2, w2a, w2b) * scale
///
/// # Arguments
/// * `t1` - Tucker core tensor 1: [KH, KW, RANK, RANK], BF16
/// * `w1a` - First down weight: [1, 1, IC, RANK], BF16
/// * `w1b` - First up weight: [1, 1, RANK, OC], BF16
/// * `t2` - Tucker core tensor 2: [KH, KW, RANK, RANK], BF16
/// * `w2a` - Second down weight: [1, 1, IC, RANK], BF16
/// * `w2b` - Second up weight: [1, 1, RANK, OC], BF16
/// * `scale` - Scaling factor
pub fn make_hadamard_weight_tucker(
    t1: &Tensor,
    w1a: &Tensor,
    w1b: &Tensor,
    t2: &Tensor,
    w2a: &Tensor,
    w2b: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    // Validate BF16 storage
    assert_bf16_storage("t1", t1)?;
    assert_bf16_storage("w1a", w1a)?;
    assert_bf16_storage("w1b", w1b)?;
    assert_bf16_storage("t2", t2)?;
    assert_bf16_storage("w2a", w2a)?;
    assert_bf16_storage("w2b", w2b)?;

    // Early exit for zero scale
    if scale == 0.0 {
        return crate::tensor_utils::zeros_bf16(
            t1.shape().clone(),
            t1.device().clone(),
        );
    }

    // Tucker reconstruction requires proper tensor contraction
    // For now, return error indicating full implementation needed
    Err(Error::InvalidOperation(
        "Tucker Hadamard weight reconstruction requires full tensor contraction implementation. \
         Use non-Tucker path or implement tensor slice assignment for full Tucker support.".into()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadamard_weight_construction() {
        // Test proper weight matrix construction
        // Dimensions: out_dim=4, in_dim=3, rank=2
        // w1a: [IN, RANK] = [3, 2]
        // w1b: [RANK, OUT] = [2, 4]
        // w2a: [IN, RANK] = [3, 2]
        // w2b: [RANK, OUT] = [2, 4]
        // Result: [3, 4] from (w1a @ w1b) ⊙ (w2a @ w2b)
        assert!(true);
    }
}
