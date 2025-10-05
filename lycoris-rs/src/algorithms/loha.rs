/// LoHa (LoRA with Hadamard Product) Module
///
/// ΔW = (w1a @ w1b) ⊙ (w2a @ w2b) * scale
/// where ⊙ is element-wise (Hadamard) product
///
/// Weight layouts follow Flame contracts:
/// - Linear: [IN, OUT]
/// - Conv2d: [KH, KW, IC, OC]

use crate::{tensor_utils, Error, LycorisModule, Result};
use cudarc::driver::CudaDevice;
use flame_core::{DType, Shape, Tensor};
use std::sync::Arc;

pub struct LoHaModule {
    /// First down projection (w1a)
    /// Linear: [IN, RANK], Conv: [KH, KW, IC, RANK] or [1, 1, IC, RANK]
    /// BF16 storage
    pub w1a: Tensor,

    /// First up projection (w1b)
    /// Linear: [RANK, OUT], Conv: [KH, KW, RANK, OC] or [1, 1, RANK, OC]
    /// BF16 storage
    pub w1b: Tensor,

    /// Second down projection (w2a)
    /// Linear: [IN, RANK], Conv: [KH, KW, IC, RANK] or [1, 1, IC, RANK]
    /// BF16 storage
    pub w2a: Tensor,

    /// Second up projection (w2b)
    /// Linear: [RANK, OUT], Conv: [KH, KW, RANK, OC] or [1, 1, RANK, OC]
    /// BF16 storage
    pub w2b: Tensor,

    /// Tucker core 1 (optional): [KH, KW, RANK, RANK]
    /// BF16 storage
    pub t1: Option<Tensor>,

    /// Tucker core 2 (optional): [KH, KW, RANK, RANK]
    /// BF16 storage
    pub t2: Option<Tensor>,

    /// Rank of the decomposition
    pub rank: usize,

    /// Alpha parameter for scaling
    pub alpha: f32,

    /// Device
    pub device: Arc<CudaDevice>,

    /// Whether this is for a convolution layer
    pub is_conv: bool,
}

// Helper functions
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

impl LoHaModule {
    /// Create a new LoHa module for Linear layer
    ///
    /// # Arguments
    /// * `in_features` - Input dimension
    /// * `out_features` - Output dimension
    /// * `rank` - Rank of decomposition
    /// * `alpha` - Scaling parameter (if None, uses rank)
    /// * `device` - CUDA device
    pub fn new_linear(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: Option<f32>,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let alpha = alpha.unwrap_or(rank as f32);

        // w1a: [IN, RANK], initialized with normal(0, 1)
        let w1a = tensor_utils::randn_bf16(
            Shape::from_dims(&[in_features, rank]),
            0.0,
            1.0,
            device.clone(),
        )?;

        // w1b: [RANK, OUT], initialized with zeros
        let w1b = tensor_utils::zeros_bf16(
            Shape::from_dims(&[rank, out_features]),
            device.clone(),
        )?;

        // w2a: [IN, RANK], initialized with normal(0, 1)
        let w2a = tensor_utils::randn_bf16(
            Shape::from_dims(&[in_features, rank]),
            0.0,
            1.0,
            device.clone(),
        )?;

        // w2b: [RANK, OUT], initialized with normal(0, 0.1)
        let w2b = tensor_utils::randn_bf16(
            Shape::from_dims(&[rank, out_features]),
            0.0,
            0.1,
            device.clone(),
        )?;

        assert_bf16_storage("w1a", &w1a)?;
        assert_bf16_storage("w1b", &w1b)?;
        assert_bf16_storage("w2a", &w2a)?;
        assert_bf16_storage("w2b", &w2b)?;

        Ok(Self {
            w1a,
            w1b,
            w2a,
            w2b,
            t1: None,
            t2: None,
            rank,
            alpha,
            device,
            is_conv: false,
        })
    }

    /// Create a new LoHa module for Conv2d layer with optional Tucker decomposition
    ///
    /// # Arguments
    /// * `in_channels` - Input channels
    /// * `out_channels` - Output channels
    /// * `kernel_size` - Convolution kernel size (h, w)
    /// * `rank` - Rank of decomposition
    /// * `alpha` - Scaling parameter
    /// * `use_tucker` - Whether to use Tucker decomposition
    /// * `device` - CUDA device
    pub fn new_conv2d(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        rank: usize,
        alpha: Option<f32>,
        use_tucker: bool,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let alpha = alpha.unwrap_or(rank as f32);
        let (kh, kw) = kernel_size;

        // Follow Flame conv layout: [KH, KW, IC, OC]
        let (w1a, w1b, w2a, w2b, t1, t2) = if kh == 1 && kw == 1 {
            // 1×1 convolution
            let w1a = tensor_utils::randn_bf16(
                Shape::from_dims(&[1, 1, in_channels, rank]),
                0.0,
                1.0,
                device.clone(),
            )?;
            let w1b = tensor_utils::zeros_bf16(
                Shape::from_dims(&[1, 1, rank, out_channels]),
                device.clone(),
            )?;
            let w2a = tensor_utils::randn_bf16(
                Shape::from_dims(&[1, 1, in_channels, rank]),
                0.0,
                1.0,
                device.clone(),
            )?;
            let w2b = tensor_utils::randn_bf16(
                Shape::from_dims(&[1, 1, rank, out_channels]),
                0.0,
                0.1,
                device.clone(),
            )?;
            (w1a, w1b, w2a, w2b, None, None)
        } else if use_tucker {
            // Tucker decomposition: spatial kernels in t1/t2
            let w1a = tensor_utils::randn_bf16(
                Shape::from_dims(&[1, 1, in_channels, rank]),
                0.0,
                1.0,
                device.clone(),
            )?;
            let w1b = tensor_utils::zeros_bf16(
                Shape::from_dims(&[1, 1, rank, out_channels]),
                device.clone(),
            )?;
            let w2a = tensor_utils::randn_bf16(
                Shape::from_dims(&[1, 1, in_channels, rank]),
                0.0,
                1.0,
                device.clone(),
            )?;
            let w2b = tensor_utils::randn_bf16(
                Shape::from_dims(&[1, 1, rank, out_channels]),
                0.0,
                0.1,
                device.clone(),
            )?;

            let t1 = tensor_utils::randn_bf16(
                Shape::from_dims(&[kh, kw, rank, rank]),
                0.0,
                0.1,
                device.clone(),
            )?;
            let t2 = tensor_utils::randn_bf16(
                Shape::from_dims(&[kh, kw, rank, rank]),
                0.0,
                0.1,
                device.clone(),
            )?;

            (w1a, w1b, w2a, w2b, Some(t1), Some(t2))
        } else {
            // Standard spatial convolution
            let w1a = tensor_utils::randn_bf16(
                Shape::from_dims(&[kh, kw, in_channels, rank]),
                0.0,
                1.0,
                device.clone(),
            )?;
            let w1b = tensor_utils::zeros_bf16(
                Shape::from_dims(&[kh, kw, rank, out_channels]),
                device.clone(),
            )?;
            let w2a = tensor_utils::randn_bf16(
                Shape::from_dims(&[kh, kw, in_channels, rank]),
                0.0,
                1.0,
                device.clone(),
            )?;
            let w2b = tensor_utils::randn_bf16(
                Shape::from_dims(&[kh, kw, rank, out_channels]),
                0.0,
                0.1,
                device.clone(),
            )?;
            (w1a, w1b, w2a, w2b, None, None)
        };

        assert_bf16_storage("w1a", &w1a)?;
        assert_bf16_storage("w1b", &w1b)?;
        assert_bf16_storage("w2a", &w2a)?;
        assert_bf16_storage("w2b", &w2b)?;
        if let Some(ref t) = t1 {
            assert_bf16_storage("t1", t)?;
        }
        if let Some(ref t) = t2 {
            assert_bf16_storage("t2", t)?;
        }

        Ok(Self {
            w1a,
            w1b,
            w2a,
            w2b,
            t1,
            t2,
            rank,
            alpha,
            device,
            is_conv: true,
        })
    }

    /// Get the scaling factor (alpha / rank), returns 0.0 if rank==0
    #[inline]
    pub fn scale(&self) -> f32 {
        if self.rank == 0 {
            0.0
        } else {
            self.alpha / self.rank as f32
        }
    }

    /// Merge into base weight tensor
    ///
    /// Returns new merged tensor (Flame doesn't support in-place add)
    pub fn merge_into(&self, base_weight: &Tensor, multiplier: f32) -> Result<Tensor> {
        let delta = self
            .get_diff_weight()?
            .mul_scalar(multiplier)
            .map_err(Error::Flame)?;
        // Add: result = base + delta
        base_weight.add(&delta).map_err(Error::Flame)
    }
}

impl LycorisModule for LoHaModule {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let scale = self.scale();

        // Early exit for zero rank
        if scale == 0.0 {
            return tensor_utils::zeros_bf16(Shape::from_dims(x.dims()), self.device.clone());
        }

        // Compute w1 and w2 with proper operations
        if self.is_conv {
            // Conv path: use conv2d operations
            let h1 = if let Some(ref t1) = self.t1 {
                // Tucker: w1a → t1 → w1b
                let temp = crate::ops::conv2d::conv2d(
                    x,
                    &self.w1a,
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                    crate::ops::conv2d::Layout::NHWC,
                )
                ?;
                let temp = crate::ops::conv2d::conv2d(
                    &temp,
                    t1,
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                    crate::ops::conv2d::Layout::NHWC,
                )
                ?;
                crate::ops::conv2d::conv2d(
                    &temp,
                    &self.w1b,
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                    crate::ops::conv2d::Layout::NHWC,
                )
                ?
            } else {
                // Direct: w1a → w1b
                let temp = crate::ops::conv2d::conv2d(
                    x,
                    &self.w1a,
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                    crate::ops::conv2d::Layout::NHWC,
                )
                ?;
                crate::ops::conv2d::conv2d(
                    &temp,
                    &self.w1b,
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                    crate::ops::conv2d::Layout::NHWC,
                )
                ?
            };

            let h2 = if let Some(ref t2) = self.t2 {
                // Tucker: w2a → t2 → w2b
                let temp = crate::ops::conv2d::conv2d(
                    x,
                    &self.w2a,
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                    crate::ops::conv2d::Layout::NHWC,
                )
                ?;
                let temp = crate::ops::conv2d::conv2d(
                    &temp,
                    t2,
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                    crate::ops::conv2d::Layout::NHWC,
                )
                ?;
                crate::ops::conv2d::conv2d(
                    &temp,
                    &self.w2b,
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                    crate::ops::conv2d::Layout::NHWC,
                )
                ?
            } else {
                // Direct: w2a → w2b
                let temp = crate::ops::conv2d::conv2d(
                    x,
                    &self.w2a,
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                    crate::ops::conv2d::Layout::NHWC,
                )
                ?;
                crate::ops::conv2d::conv2d(
                    &temp,
                    &self.w2b,
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                    crate::ops::conv2d::Layout::NHWC,
                )
                ?
            };

            // Hadamard product and scale
            let result = h1.mul(&h2)?;
            result.mul_scalar(scale).map_err(Error::Flame)
        } else {
            // Linear path: w1 = w1a @ w1b, w2 = w2a @ w2b
            let w1 = self.w1a.matmul(&self.w1b)?;
            let w2 = self.w2a.matmul(&self.w2b)?;

            // Hadamard product
            let diff_w = w1.mul(&w2)?;
            let scaled_diff = diff_w.mul_scalar(scale)?;

            // Apply to input: x @ diff_w
            x.matmul(&scaled_diff).map_err(Error::Flame)
        }
    }

    fn get_diff_weight(&self) -> Result<Tensor> {
        let scale = self.scale();

        // Early exit for zero scale
        if scale == 0.0 {
            return if self.is_conv {
                tensor_utils::zeros_bf16(self.w1b.shape().clone(), self.device.clone())
            } else {
                tensor_utils::zeros_bf16(
                    Shape::from_dims(&[self.w1a.dims()[0], self.w1b.dims()[1]]),
                    self.device.clone(),
                )
            };
        }

        if self.is_conv {
            // Conv path
            if let (Some(ref t1), Some(ref t2)) = (&self.t1, &self.t2) {
                // Tucker path: need full reconstruction
                // For now, use simplified approach via hadamard op
                crate::ops::hadamard::make_hadamard_weight_tucker(
                    t1, &self.w1a, &self.w1b, t2, &self.w2a, &self.w2b, scale,
                )
            } else {
                // Standard conv: compute kernel via hadamard
                let dims = self.w1a.dims();
                if dims[0] == 1 && dims[1] == 1 {
                    // 1×1 case: can use linear math
                    let ic = dims[2];
                    let r = dims[3];
                    let oc = self.w1b.dims()[3];

                    let w1a_lin = self.w1a.reshape(&[ic, r])?;
                    let w1b_lin = self.w1b.reshape(&[r, oc])?;
                    let w2a_lin = self.w2a.reshape(&[ic, r])?;
                    let w2b_lin = self.w2b.reshape(&[r, oc])?;

                    let w1 = w1a_lin.matmul(&w1b_lin)?;
                    let w2 = w2a_lin.matmul(&w2b_lin)?;
                    let diff = w1.mul(&w2)?;
                    let k = diff.reshape(&[1, 1, ic, oc])?;
                    k.mul_scalar(scale).map_err(Error::Flame)
                } else {
                    // Spatial case: use hadamard op
                    crate::ops::hadamard::make_hadamard_weight(
                        &self.w1a, &self.w1b, &self.w2a, &self.w2b, scale,
                    )
                }
            }
        } else {
            // Linear: w1 = w1a @ w1b, w2 = w2a @ w2b, diff = w1 ⊙ w2
            let w1 = self.w1a.matmul(&self.w1b)?;
            let w2 = self.w2a.matmul(&self.w2b)?;
            let diff = w1.mul(&w2)?;
            diff.mul_scalar(scale).map_err(Error::Flame)
        }
    }

    fn merge_to(&mut self, multiplier: f32) -> Result<()> {
        // Deprecated in favor of merge_into()
        let _scaled = self
            .get_diff_weight()?
            .mul_scalar(multiplier)
            .map_err(Error::Flame)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loha_creation() {
        // Placeholder - requires CUDA device initialization
        assert!(true);
    }

    #[test]
    fn test_scale_zero_rank() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let module = LoHaModule {
            w1a: tensor_utils::zeros_bf16(Shape::from_dims(&[4, 0]), device.clone()).unwrap(),
            w1b: tensor_utils::zeros_bf16(Shape::from_dims(&[0, 8]), device.clone()).unwrap(),
            w2a: tensor_utils::zeros_bf16(Shape::from_dims(&[4, 0]), device.clone()).unwrap(),
            w2b: tensor_utils::zeros_bf16(Shape::from_dims(&[0, 8]), device.clone()).unwrap(),
            t1: None,
            t2: None,
            rank: 0,
            alpha: 1.0,
            device,
            is_conv: false,
        };

        assert_eq!(module.scale(), 0.0);
    }
}
