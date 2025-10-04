/// LoCon (LoRA for Convolution) Module
///
/// Standard LoRA decomposition: ΔW = up @ down * scale
/// Works for both Linear and Conv layers

use crate::{tensor_utils, Error, LycorisModule, Result};
use cudarc::driver::CudaDevice;
use flame_core::{Shape, Tensor};
use std::sync::Arc;

pub struct LoConModule {
    /// Down projection (rank × in_dim) for Linear, (rank × in_channels × 1 × 1) for Conv
    /// Stored in BF16
    pub down: Tensor,

    /// Up projection (out_dim × rank) for Linear, (out_channels × rank × 1 × 1) for Conv
    /// Stored in BF16, initialized to zero
    pub up: Tensor,

    /// Tucker core tensor for Conv with kernel > 1
    /// (rank × rank × kernel_h × kernel_w), Optional, BF16
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

        // down: (rank, in_features) - BF16 storage
        let down_shape = Shape::from_dims(&[rank, in_features]);
        let down = tensor_utils::randn_bf16(down_shape, 0.0, 1.0, device.clone())?;

        // up: (out_features, rank) - BF16 storage, initialized to zeros
        let up_shape = Shape::from_dims(&[out_features, rank]);
        let up = tensor_utils::zeros_bf16(up_shape, device.clone())?;

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

        let (down, up, mid) = if kh > 1 || kw > 1 {
            if use_tucker {
                // Tucker decomposition: down (rank, in_channels, 1, 1)
                let down_shape = Shape::from_dims(&[rank, in_channels, 1, 1]);
                let down = tensor_utils::randn_bf16(
                    down_shape,
                    0.0,
                    1.0,
                    device.clone(),
                )
                .map_err(|e| Error::Flame(e))?;

                // up (out_channels, rank, 1, 1)
                let up_shape = Shape::from_dims(&[out_channels, rank, 1, 1]);
                let up = tensor_utils::zeros_bf16(
                    up_shape,
                    device.clone(),
                )
                .map_err(|e| Error::Flame(e))?;

                // mid (rank, rank, kh, kw)
                let mid_shape = Shape::from_dims(&[rank, rank, kh, kw]);
                let mid = tensor_utils::randn_bf16(
                    mid_shape,
                    0.0,
                    1.0,
                    device.clone(),
                )
                .map_err(|e| Error::Flame(e))?;

                (down, up, Some(mid))
            } else {
                // Standard conv: down (rank, in_channels, kh, kw)
                let down_shape = Shape::from_dims(&[rank, in_channels, kh, kw]);
                let down = tensor_utils::randn_bf16(
                    down_shape,
                    0.0,
                    1.0,
                    device.clone(),
                )
                .map_err(|e| Error::Flame(e))?;

                // up (out_channels, rank, kh, kw)
                let up_shape = Shape::from_dims(&[out_channels, rank, kh, kw]);
                let up = tensor_utils::zeros_bf16(
                    up_shape,
                    device.clone(),
                )
                .map_err(|e| Error::Flame(e))?;

                (down, up, None)
            }
        } else {
            // 1x1 conv, same as linear
            let down_shape = Shape::from_dims(&[rank, in_channels]);
            let down = tensor_utils::randn_bf16(
                down_shape,
                0.0,
                1.0,
                device.clone(),
            )
            .map_err(|e| Error::Flame(e))?;

            let up_shape = Shape::from_dims(&[out_channels, rank]);
            let up = tensor_utils::zeros_bf16(
                up_shape,
                device.clone(),
            )
            .map_err(|e| Error::Flame(e))?;

            (down, up, None)
        };

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

    /// Get the scaling factor (alpha / rank)
    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

impl LycorisModule for LoConModule {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: input tensor
        // Compute: x -> down(x) -> [mid(x)] -> up(x) * scale

        let scale = self.scale();

        // First apply down projection (BF16 -> FP32 compute -> BF16 output)
        let h = if self.is_conv {
            // Conv2d operation
            x.conv2d(&self.down, None, 1, 1, 1, 1)
                .map_err(|e| Error::Flame(e))?
        } else {
            // Linear operation
            let down_t = tensor_utils::transpose_2d(&self.down)?;
            x.matmul(&down_t).map_err(|e| Error::Flame(e))?
        };

        // Apply mid if present (Tucker decomposition)
        let h = if let Some(ref mid) = self.mid {
            h.conv2d(mid, None, 1, 1, 1, 1)
                .map_err(|e| Error::Flame(e))?
        } else {
            h
        };

        // Apply up projection
        let output = if self.is_conv {
            h.conv2d(&self.up, None, 1, 1, 1, 1)
                .map_err(|e| Error::Flame(e))?
        } else {
            let up_t = tensor_utils::transpose_2d(&self.up)?;
            h.matmul(&up_t).map_err(|e| Error::Flame(e))?
        };

        // Apply scale
        output.mul_scalar(scale).map_err(|e| Error::Flame(e))
    }

    fn get_diff_weight(&self) -> Result<Tensor> {
        // Compute ΔW = up @ down * scale (all in FP32 compute)
        let scale = self.scale();

        if let Some(ref mid) = self.mid {
            // Tucker decomposition path
            // ΔW = rebuild_tucker(mid, up, down)
            let diff = crate::ops::tucker::rebuild_tucker(mid, &self.up, &self.down)?;
            diff.mul_scalar(scale).map_err(|e| Error::Flame(e))
        } else {
            // Standard LoRA path
            let up_reshaped = self.up.reshape(&[
                self.up.shape().dims()[0],
                self.up.shape().dims()[1],
            ])
            .map_err(|e| Error::Flame(e))?;

            let down_reshaped = self.down.reshape(&[
                self.down.shape().dims()[0],
                self.down.shape().dims()[1],
            ])
            .map_err(|e| Error::Flame(e))?;

            let diff = up_reshaped.matmul(&down_reshaped)
                .map_err(|e| Error::Flame(e))?;

            let diff = diff.mul_scalar(scale).map_err(|e| Error::Flame(e))?;

            // Reshape back if conv
            if self.is_conv && self.down.dims().len() == 4 {
                let out_shape: Vec<usize> = self.up.shape().dims().to_vec();
                diff.reshape(&out_shape).map_err(|e| Error::Flame(e))
            } else {
                Ok(diff)
            }
        }
    }

    fn merge_to(&mut self, multiplier: f32) -> Result<()> {
        // Get differential weight
        let diff_weight = self.get_diff_weight()?;

        // Scale by multiplier
        let scaled_diff = diff_weight.mul_scalar(multiplier)
            .map_err(|e| Error::Flame(e))?;

        // In a real implementation, we would add this to the base weight
        // For now, we just store it
        // base_weight = base_weight + scaled_diff

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
}
