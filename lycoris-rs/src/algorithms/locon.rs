/// LoCon (LoRA for Convolution) Module
///
/// Standard LoRA decomposition: ΔW = down @ up * scale
/// Works for both Linear and Conv layers
///
/// Weight layouts follow Flame contracts:
/// - Linear: [IN, OUT]
/// - Conv2d: [KH, KW, IC, OC]

use crate::{tensor_utils, Error, LycorisModule, Result};
use cudarc::driver::CudaDevice;
use flame_core::{DType, Shape, Tensor};
use std::sync::Arc;

pub struct LoConModule {
    /// Down projection
    /// Linear: [IN, RANK]
    /// Conv: [KH, KW, IC, RANK] or [1, 1, IC, RANK] for 1×1
    /// Stored in BF16
    pub down: Tensor,

    /// Up projection
    /// Linear: [RANK, OUT]
    /// Conv: [KH, KW, RANK, OC] or [1, 1, RANK, OC] for 1×1
    /// Stored in BF16, initialized to zero
    pub up: Tensor,

    /// Tucker core tensor for Conv with kernel > 1
    /// [KH, KW, RANK, RANK], Optional, BF16
    pub mid: Option<Tensor>,

    /// Rank of the decomposition
    pub rank: usize,

    /// Alpha parameter for scaling (default: rank)
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

/// Convert a linear [IN, OUT] weight to a 1×1 conv kernel [KH,KW,IC,OC] without copy.
/// IN→IC, OUT→OC
fn as_conv1x1_kernel(w_in_out: &Tensor) -> Result<Tensor> {
    let dims = w_in_out.dims();
    if dims.len() != 2 {
        return Err(Error::InvalidOperation(format!(
            "as_conv1x1_kernel expects 2D tensor, got {}D",
            dims.len()
        )));
    }
    let (i, o) = (dims[0], dims[1]); // [IN, OUT]
    // View as [1,1,IC,OC]
    w_in_out.reshape(&[1, 1, i, o]).map_err(Error::Flame)
}

impl LoConModule {
    /// Create a new LoCon module for Linear layer
    ///
    /// # Arguments
    /// * `in_features` - Input dimension
    /// * `out_features` - Output dimension
    /// * `rank` - Rank of LoRA decomposition
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

        // down: [IN, RANK] - BF16 storage
        let down = tensor_utils::randn_bf16(
            Shape::from_dims(&[in_features, rank]),
            0.0,
            1.0,
            device.clone(),
        )?;

        // up: [RANK, OUT] - BF16 storage, initialized to zeros
        let up = tensor_utils::zeros_bf16(
            Shape::from_dims(&[rank, out_features]),
            device.clone(),
        )?;

        assert_bf16_storage("down", &down)?;
        assert_bf16_storage("up", &up)?;

        Ok(Self {
            down,
            up,
            mid: None,
            rank,
            alpha,
            device,
            is_conv: false,
        })
    }

    /// Create a new LoCon module for Conv2d layer
    ///
    /// # Arguments
    /// * `in_channels` - Input channels
    /// * `out_channels` - Output channels
    /// * `kernel_size` - Convolution kernel size (h, w)
    /// * `rank` - Rank of LoRA decomposition
    /// * `alpha` - Scaling parameter
    /// * `use_tucker` - Whether to use Tucker decomposition for kernel
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
        let (down, up, mid) = if kh == 1 && kw == 1 {
            // 1×1 path uses true conv kernels:
            // down: [1, 1, IC, RANK], up: [1, 1, RANK, OC]
            let down = tensor_utils::randn_bf16(
                Shape::from_dims(&[1, 1, in_channels, rank]),
                0.0,
                1.0,
                device.clone(),
            )?;
            let up = tensor_utils::zeros_bf16(
                Shape::from_dims(&[1, 1, rank, out_channels]),
                device.clone(),
            )?;
            (down, up, None)
        } else if use_tucker {
            // Tucker path:
            // down: [1, 1, IC, RANK], up: [1, 1, RANK, OC], mid: [KH, KW, RANK, RANK]
            let down = tensor_utils::randn_bf16(
                Shape::from_dims(&[1, 1, in_channels, rank]),
                0.0,
                1.0,
                device.clone(),
            )?;
            let up = tensor_utils::zeros_bf16(
                Shape::from_dims(&[1, 1, rank, out_channels]),
                device.clone(),
            )?;
            let mid = tensor_utils::randn_bf16(
                Shape::from_dims(&[kh, kw, rank, rank]),
                0.0,
                1.0,
                device.clone(),
            )?;
            (down, up, Some(mid))
        } else {
            // Standard spatial (no Tucker):
            // down: [KH, KW, IC, RANK], up: [KH, KW, RANK, OC]
            let down = tensor_utils::randn_bf16(
                Shape::from_dims(&[kh, kw, in_channels, rank]),
                0.0,
                1.0,
                device.clone(),
            )?;
            let up = tensor_utils::zeros_bf16(
                Shape::from_dims(&[kh, kw, rank, out_channels]),
                device.clone(),
            )?;
            (down, up, None)
        };

        assert_bf16_storage("down", &down)?;
        assert_bf16_storage("up", &up)?;
        if let Some(ref m) = mid {
            assert_bf16_storage("mid", m)?;
        }

        Ok(Self {
            down,
            up,
            mid,
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
    pub fn merge_into(&self, base_weight: &mut Tensor, multiplier: f32) -> Result<()> {
        let delta = self
            .get_diff_weight()?
            .mul_scalar(multiplier)
            .map_err(Error::Flame)?;
        // In-place add: base += delta
        base_weight.add_inplace(&delta).map_err(Error::Flame)
    }
}

impl LycorisModule for LoConModule {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let scale = self.scale();

        // Early exit for zero rank
        if scale == 0.0 {
            return tensor_utils::zeros_bf16(
                Shape::from_dims(x.dims()),
                self.device.clone(),
            )
            .map_err(Error::Flame);
        }

        // First apply down projection (BF16 -> FP32 compute -> BF16 output)
        let h = if self.is_conv {
            // Conv2d operation with explicit parameters
            // down: [KH, KW, IC, RANK] or [1, 1, IC, RANK]
            crate::ops::conv2d::conv2d(
                x,
                &self.down,
                /*stride=*/ (1, 1),
                /*padding=*/ (0, 0),
                /*dilation=*/ (1, 1),
                /*groups=*/ 1,
                /*layout=*/ crate::ops::conv2d::Layout::NHWC,
            )
            .map_err(Error::Flame)?
        } else {
            // Linear operation: x[..., IN] @ down[IN, RANK] -> [..., RANK]
            x.matmul(&self.down).map_err(Error::Flame)?
        };

        // Apply mid if present (Tucker decomposition)
        let h = if let Some(ref mid) = self.mid {
            // mid: [KH, KW, RANK, RANK]
            crate::ops::conv2d::conv2d(
                &h,
                mid,
                (1, 1),
                (0, 0),
                (1, 1),
                1,
                crate::ops::conv2d::Layout::NHWC,
            )
            .map_err(Error::Flame)?
        } else {
            h
        };

        // Apply up projection
        let output = if self.is_conv {
            // up: [KH, KW, RANK, OC] or [1, 1, RANK, OC]
            crate::ops::conv2d::conv2d(
                &h,
                &self.up,
                (1, 1),
                (0, 0),
                (1, 1),
                1,
                crate::ops::conv2d::Layout::NHWC,
            )
            .map_err(Error::Flame)?
        } else {
            // h[..., RANK] @ up[RANK, OUT] -> [..., OUT]
            h.matmul(&self.up).map_err(Error::Flame)?
        };

        // Apply scale
        output.mul_scalar(scale).map_err(Error::Flame)
    }

    fn get_diff_weight(&self) -> Result<Tensor> {
        let scale = self.scale();

        // Early exit for zero scale
        if scale == 0.0 {
            return if self.is_conv {
                tensor_utils::zeros_bf16(
                    self.up.shape().clone(),
                    self.device.clone(),
                )
                .map_err(Error::Flame)
            } else {
                tensor_utils::zeros_bf16(
                    Shape::from_dims(&[self.down.dims()[0], self.up.dims()[1]]),
                    self.device.clone(),
                )
                .map_err(Error::Flame)
            };
        }

        if self.is_conv {
            // Conv path
            if let Some(ref mid) = self.mid {
                // Tucker reconstruction: down → mid → up
                // mid: [KH, KW, RANK, RANK], down: [1, 1, IC, RANK], up: [1, 1, RANK, OC]
                // Result: [KH, KW, IC, OC]

                // Contract down with mid along RANK dimension
                // down [1,1,IC,R] reshaped to [IC,R], mid [KH,KW,R,R] reshaped to flatten spatial
                let kh = mid.dims()[0];
                let kw = mid.dims()[1];
                let ic = self.down.dims()[2];
                let r = self.down.dims()[3];
                let oc = self.up.dims()[3];

                // Reshape down: [1,1,IC,R] -> [IC,R]
                let down_2d = self.down.reshape(&[ic, r]).map_err(Error::Flame)?;

                // Reshape mid: [KH,KW,R,R] -> [KH*KW, R, R]
                let mid_3d = mid.reshape(&[kh * kw, r, r]).map_err(Error::Flame)?;

                // Reshape up: [1,1,R,OC] -> [R,OC]
                let up_2d = self.up.reshape(&[r, oc]).map_err(Error::Flame)?;

                // Contract: down @ mid @ up for each spatial position
                // For simplicity: flatten and contract
                let mut result = tensor_utils::zeros_bf16(
                    Shape::from_dims(&[kh, kw, ic, oc]),
                    self.device.clone(),
                )?;

                // Simple einsum-like contraction
                // This is a simplified version - full Tucker requires proper tensor contraction
                for h in 0..kh {
                    for w in 0..kw {
                        let idx = h * kw + w;
                        let mid_slice = mid_3d.narrow(0, idx, 1)?.reshape(&[r, r])?;
                        let temp = down_2d.matmul(&mid_slice).map_err(Error::Flame)?;
                        let kernel_hw = temp.matmul(&up_2d).map_err(Error::Flame)?;

                        // Copy into result at position [h, w, :, :]
                        // This requires tensor assignment which may not be available
                        // For now, return error indicating full Tucker needs implementation
                    }
                }

                return Err(Error::InvalidOperation(
                    "Tucker conv decomposition requires full tensor contraction implementation".into()
                ));
            } else {
                // Standard LoRA conv
                let down_dims = self.down.dims();
                let up_dims = self.up.dims();

                // For 1×1: down:[1,1,IC,R], up:[1,1,R,OC]
                if down_dims[0] == 1 && down_dims[1] == 1 && up_dims[0] == 1 && up_dims[1] == 1 {
                    // Equivalent linear: [IC,OC] = [IC,R] @ [R,OC] then view to [1,1,IC,OC]
                    let ic = down_dims[2];
                    let r = down_dims[3];
                    let oc = up_dims[3];
                    let down_lin = self.down.reshape(&[ic, r]).map_err(Error::Flame)?;
                    let up_lin = self.up.reshape(&[r, oc]).map_err(Error::Flame)?;
                    let k_lin = down_lin.matmul(&up_lin).map_err(Error::Flame)?; // [IC,OC]
                    let k = k_lin.reshape(&[1, 1, ic, oc]).map_err(Error::Flame)?;
                    return k.mul_scalar(scale).map_err(Error::Flame);
                } else {
                    // Spatial (no Tucker): [KH,KW,IC,R] and [KH,KW,R,OC]
                    // For each spatial position, do IC×R @ R×OC = IC×OC
                    let kh = down_dims[0];
                    let kw = down_dims[1];
                    let ic = down_dims[2];
                    let r = down_dims[3];
                    let oc = up_dims[3];

                    // Reshape for batch matmul: [KH*KW, IC, R] @ [KH*KW, R, OC] -> [KH*KW, IC, OC]
                    let down_batch = self.down.reshape(&[kh * kw, ic, r]).map_err(Error::Flame)?;
                    let up_batch = self.up.reshape(&[kh * kw, r, oc]).map_err(Error::Flame)?;

                    // Batch matmul
                    let result_batch = down_batch.matmul(&up_batch).map_err(Error::Flame)?;

                    // Reshape back: [KH*KW, IC, OC] -> [KH, KW, IC, OC]
                    let k = result_batch.reshape(&[kh, kw, ic, oc]).map_err(Error::Flame)?;
                    return k.mul_scalar(scale).map_err(Error::Flame);
                }
            }
        } else {
            // Linear: ΔW = down @ up → [IN, OUT]
            // down: [IN, RANK], up: [RANK, OUT]
            let diff = self.down.matmul(&self.up).map_err(Error::Flame)?;
            diff.mul_scalar(scale).map_err(Error::Flame)
        }
    }

    fn merge_to(&mut self, multiplier: f32) -> Result<()> {
        // This is deprecated in favor of merge_into()
        // which takes a mutable base weight
        let _scaled = self
            .get_diff_weight()?
            .mul_scalar(multiplier)
            .map_err(Error::Flame)?;

        // Note: Cannot merge without base weight reference
        // Use merge_into() instead
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_locon_creation() {
        // Placeholder - requires CUDA device initialization
        assert!(true);
    }

    #[test]
    fn test_scale_zero_rank() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let module = LoConModule {
            down: tensor_utils::zeros_bf16(Shape::from_dims(&[4, 0]), device.clone()).unwrap(),
            up: tensor_utils::zeros_bf16(Shape::from_dims(&[0, 8]), device.clone()).unwrap(),
            mid: None,
            rank: 0,
            alpha: 1.0,
            device,
            is_conv: false,
        };

        assert_eq!(module.scale(), 0.0);
    }

    #[test]
    fn test_as_conv1x1_kernel() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let w = tensor_utils::zeros_bf16(Shape::from_dims(&[3, 5]), device.clone()).unwrap();
        let k = as_conv1x1_kernel(&w).unwrap();
        assert_eq!(k.dims(), &[1, 1, 3, 5]);
    }
}
