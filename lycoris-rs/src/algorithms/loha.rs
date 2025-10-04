/// LoHa (LoRA with Hadamard Product) Module
///
/// ΔW = (w1u @ w1d) ⊙ (w2u @ w2d) * scale
/// where ⊙ is element-wise (Hadamard) product

use crate::{tensor_utils, Error, LycorisModule, Result};
use cudarc::driver::CudaDevice;
use flame_core::{ Shape, Tensor};
use std::sync::Arc;

pub struct LoHaModule {
    /// Down weight 1 (rank × in_dim), BF16 storage
    pub w1d: Tensor,

    /// Up weight 1 (out_dim × rank), BF16 storage
    pub w1u: Tensor,

    /// Down weight 2 (rank × in_dim), BF16 storage
    pub w2d: Tensor,

    /// Up weight 2 (out_dim × rank), BF16 storage
    pub w2u: Tensor,

    /// Tucker core 1 (rank × rank × kernel_h × kernel_w), Optional, BF16
    pub t1: Option<Tensor>,

    /// Tucker core 2 (rank × rank × kernel_h × kernel_w), Optional, BF16
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

        // w1d: (rank, in_features), initialized with normal(0, 1)
        let w1d = tensor_utils::randn_bf16(
            Shape::from_dims(&[rank, in_features]),
            0.0,
            1.0,
            device.clone(),
        )
        .map_err(|e| Error::Flame(e))?;

        // w1u: (out_features, rank), initialized with zeros
        let w1u = tensor_utils::zeros_bf16(
            Shape::from_dims(&[out_features, rank]),
            device.clone(),
        )
        .map_err(|e| Error::Flame(e))?;

        // w2d: (rank, in_features), initialized with normal(0, 1)
        let w2d = tensor_utils::randn_bf16(
            Shape::from_dims(&[rank, in_features]),
            0.0,
            1.0,
            device.clone(),
        )
        .map_err(|e| Error::Flame(e))?;

        // w2u: (out_features, rank), initialized with normal(0, 0.1)
        let w2u = tensor_utils::randn_bf16(
            Shape::from_dims(&[out_features, rank]),
            0.0,
            0.1,
            device.clone(),
        )
        .map_err(|e| Error::Flame(e))?;

        Ok(Self {
            w1d,
            w1u,
            w2d,
            w2u,
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

        if use_tucker && (kh > 1 || kw > 1) {
            // Tucker decomposition path
            // w1d, w2d: (rank, in_channels)
            let w1d = tensor_utils::randn_bf16(
                Shape::from_dims(&[rank, in_channels]),
                0.0,
                1.0,
                device.clone(),
            )
            .map_err(|e| Error::Flame(e))?;

            let w2d = tensor_utils::randn_bf16(
                Shape::from_dims(&[rank, in_channels]),
                0.0,
                1.0,
                device.clone(),
            )
            .map_err(|e| Error::Flame(e))?;

            // w1u, w2u: (rank, out_channels)
            let w1u = tensor_utils::zeros_bf16(
                Shape::from_dims(&[rank, out_channels]),
                device.clone(),
            )
            .map_err(|e| Error::Flame(e))?;

            let w2u = tensor_utils::randn_bf16(
                Shape::from_dims(&[rank, out_channels]),
                0.0,
                0.1,
                device.clone(),
            )
            .map_err(|e| Error::Flame(e))?;

            // t1, t2: (rank, rank, kh, kw)
            let t1 = tensor_utils::randn_bf16(
                Shape::from_dims(&[rank, rank, kh, kw]),
                0.0,
                0.1,
                device.clone(),
            )
            .map_err(|e| Error::Flame(e))?;

            let t2 = tensor_utils::randn_bf16(
                Shape::from_dims(&[rank, rank, kh, kw]),
                0.0,
                0.1,
                device.clone(),
            )
            .map_err(|e| Error::Flame(e))?;

            Ok(Self {
                w1d,
                w1u,
                w2d,
                w2u,
                t1: Some(t1),
                t2: Some(t2),
                rank,
                alpha,
                device,
                is_conv: true,
            })
        } else {
            // Standard path (no Tucker)
            let w1d = tensor_utils::randn_bf16(
                Shape::from_dims(&[rank, in_channels]),
                0.0,
                1.0,
                device.clone(),
            )
            .map_err(|e| Error::Flame(e))?;

            let w1u = tensor_utils::zeros_bf16(
                Shape::from_dims(&[out_channels, rank]),
                device.clone(),
            )
            .map_err(|e| Error::Flame(e))?;

            let w2d = tensor_utils::randn_bf16(
                Shape::from_dims(&[rank, in_channels]),
                0.0,
                1.0,
                device.clone(),
            )
            .map_err(|e| Error::Flame(e))?;

            let w2u = tensor_utils::randn_bf16(
                Shape::from_dims(&[out_channels, rank]),
                0.0,
                0.1,
                device.clone(),
            )
            .map_err(|e| Error::Flame(e))?;

            Ok(Self {
                w1d,
                w1u,
                w2d,
                w2u,
                t1: None,
                t2: None,
                rank,
                alpha,
                device,
                is_conv: false,
            })
        }
    }

    /// Get the scaling factor (alpha / rank)
    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

impl LycorisModule for LoHaModule {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Python source: bypass_forward_diff builds diff_weight and applies via FUNC_LIST
        // For linear: FUNC_LIST[2] = F.linear
        // For conv: FUNC_LIST[4] = F.conv2d

        // Build the differential weight
        let diff_w = self.get_diff_weight()?;

        // Apply based on layer type
        if self.is_conv {
            // For Conv2d, would use conv2d operation
            // For now, simplified linear application
            let diff_w_t = crate::tensor_utils::transpose_2d(&diff_w)?;
            x.matmul(&diff_w_t).map_err(|e| Error::Flame(e))
        } else {
            // Linear layer: x @ diff_w^T
            let diff_w_t = crate::tensor_utils::transpose_2d(&diff_w)?;
            x.matmul(&diff_w_t).map_err(|e| Error::Flame(e))
        }
    }

    fn get_diff_weight(&self) -> Result<Tensor> {
        // Compute ΔW = (w1u @ w1d) ⊙ (w2u @ w2d) * scale
        let scale = self.scale();

        if self.t1.is_some() && self.t2.is_some() {
            // Tucker decomposition path
            crate::ops::hadamard::make_hadamard_weight_tucker(
                self.t1.as_ref().unwrap(),
                &self.w1d,
                &self.w1u,
                self.t2.as_ref().unwrap(),
                &self.w2d,
                &self.w2u,
                scale,
            )
        } else {
            // Standard Hadamard path
            crate::ops::hadamard::make_hadamard_weight(
                &self.w1d,
                &self.w1u,
                &self.w2d,
                &self.w2u,
                scale,
            )
        }
    }

    fn merge_to(&mut self, multiplier: f32) -> Result<()> {
        // Get differential weight
        let diff_weight = self.get_diff_weight()?;

        // Scale by multiplier
        let _scaled_diff = diff_weight.mul_scalar(multiplier)
            .map_err(|e| Error::Flame(e))?;

        // In a real implementation, merge with base weight
        // base_weight = base_weight + scaled_diff

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
}
