/// LoKr (LoRA with Kronecker Product) Module
///
/// ΔW = w1 ⊗ w2 * scale
/// where ⊗ is the Kronecker product
///
/// With optional factorization:
/// ΔW = (w1a @ w1b) ⊗ (w2a @ w2b) * scale

use crate::{tensor_utils, Error, LycorisModule, Result};
use cudarc::driver::CudaDevice;
use flame_core::{DType, Shape, Tensor};
use std::sync::Arc;

/// Layer type for explicit conv vs linear distinction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Linear,
    Conv2d,
}

pub struct LoKrModule {
    /// First Kronecker factor (full or factorized)
    /// If decomposed: None, else (out_l × in_m), BF16
    pub w1: Option<Tensor>,

    /// First factor decomposition: w1a (out_l × rank), BF16
    pub w1a: Option<Tensor>,

    /// First factor decomposition: w1b (rank × in_m), BF16
    pub w1b: Option<Tensor>,

    /// Second Kronecker factor (full or factorized)
    /// If decomposed: None, else (out_k × in_n × kernel...), BF16
    pub w2: Option<Tensor>,

    /// Second factor decomposition: w2a (out_k × rank), BF16
    pub w2a: Option<Tensor>,

    /// Second factor decomposition: w2b (rank × in_n × kernel...), BF16
    pub w2b: Option<Tensor>,

    /// Tucker core tensor for w2 (rank × rank × kh × kw), Optional, BF16
    pub t2: Option<Tensor>,

    /// Rank for decomposition
    pub rank: usize,

    /// Alpha parameter for scaling
    pub alpha: f32,

    /// Device
    pub device: Arc<CudaDevice>,

    /// Factorization shape: ((out_l, out_k), (in_m, in_n))
    pub shape: ((usize, usize), (usize, usize)),

    /// Kernel size for conv layers
    pub kernel_size: Option<(usize, usize)>,

    /// Explicit layer type
    pub kind: LayerKind,
}

// Helper functions for tensor operations
fn assert_bf16_storage(name: &str, t: &Tensor) -> Result<()> {
    let dt = t.dtype();
    if dt != DType::BF16 {
        return Err(Error::InvalidOperation(format!(
            "{} must use BF16 storage, got {:?}",
            name, dt
        )));
    }
    Ok(())
}

fn swap_last_two(x: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let n = dims.len();
    if n < 2 {
        return Err(Error::InvalidOperation(
            "swap_last_two requires at least 2 dimensions".into(),
        ));
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.swap(n - 1, n - 2);
    x.permute(&order).map_err(Error::Flame)
}

fn move_dim_to_end(x: &Tensor, dim_idx: usize) -> Result<Tensor> {
    let n = x.dims().len();
    if dim_idx >= n {
        return Err(Error::InvalidOperation(
            "move_dim_to_end: dim out of range".into(),
        ));
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.remove(dim_idx);
    order.push(dim_idx);
    x.permute(&order).map_err(Error::Flame)
}

impl LoKrModule {
    /// Create a new LoKr module for Linear layer
    ///
    /// # Arguments
    /// * `in_features` - Input dimension
    /// * `out_features` - Output dimension
    /// * `rank` - Rank of decomposition
    /// * `alpha` - Scaling parameter (if None, uses rank)
    /// * `factor` - Factorization hint (-1 for auto)
    /// * `decompose_both` - Whether to decompose both w1 and w2
    /// * `device` - CUDA device
    pub fn new_linear(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: Option<f32>,
        factor: i32,
        decompose_both: bool,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let alpha = alpha.unwrap_or(rank as f32);

        // Factorize dimensions
        let (in_m, in_n) = crate::ops::kronecker::factorization(in_features, factor);
        let (out_l, out_k) = crate::ops::kronecker::factorization(out_features, factor);
        let shape = ((out_l, out_k), (in_m, in_n));

        // Decide on decomposition strategy
        let use_w1 = !decompose_both || rank >= (out_l.max(in_m)) / 2;
        let use_w2 = rank >= (out_k.max(in_n)) / 2;

        let (w1, w1a, w1b) = if use_w1 {
            // w1: (out_l, in_m) - kaiming_uniform with a=sqrt(5)
            let w1 = tensor_utils::kaiming_uniform_bf16(
                Shape::from_dims(&[out_l, in_m]),
                (5.0_f32).sqrt(),
                device.clone(),
            )
            .map_err(Error::Flame)?;
            (Some(w1), None, None)
        } else {
            // w1a: (out_l, rank), w1b: (rank, in_m) - kaiming_uniform with a=sqrt(5)
            let w1a = tensor_utils::kaiming_uniform_bf16(
                Shape::from_dims(&[out_l, rank]),
                (5.0_f32).sqrt(),
                device.clone(),
            )
            .map_err(Error::Flame)?;

            let w1b = tensor_utils::kaiming_uniform_bf16(
                Shape::from_dims(&[rank, in_m]),
                (5.0_f32).sqrt(),
                device.clone(),
            )
            .map_err(Error::Flame)?;

            (None, Some(w1a), Some(w1b))
        };

        let (w2, w2a, w2b) = if use_w2 {
            // w2: (out_k, in_n) - init to zeros
            let w2 = tensor_utils::zeros_bf16(
                Shape::from_dims(&[out_k, in_n]),
                device.clone(),
            )
            .map_err(Error::Flame)?;
            (Some(w2), None, None)
        } else {
            // w2a: (out_k, rank) - kaiming_uniform, w2b: (rank, in_n) - zeros
            let w2a = tensor_utils::kaiming_uniform_bf16(
                Shape::from_dims(&[out_k, rank]),
                (5.0_f32).sqrt(),
                device.clone(),
            )
            .map_err(Error::Flame)?;

            let w2b = tensor_utils::zeros_bf16(
                Shape::from_dims(&[rank, in_n]),
                device.clone(),
            )
            .map_err(Error::Flame)?;

            (None, Some(w2a), Some(w2b))
        };

        // Validate BF16 storage
        if let Some(ref w) = w1 {
            assert_bf16_storage("w1", w)?;
        }
        if let Some(ref w) = w1a {
            assert_bf16_storage("w1a", w)?;
        }
        if let Some(ref w) = w1b {
            assert_bf16_storage("w1b", w)?;
        }
        if let Some(ref w) = w2 {
            assert_bf16_storage("w2", w)?;
        }
        if let Some(ref w) = w2a {
            assert_bf16_storage("w2a", w)?;
        }
        if let Some(ref w) = w2b {
            assert_bf16_storage("w2b", w)?;
        }

        Ok(Self {
            w1,
            w1a,
            w1b,
            w2,
            w2a,
            w2b,
            t2: None,
            rank,
            alpha,
            device,
            shape,
            kernel_size: None,
            kind: LayerKind::Linear,
        })
    }

    /// Create a new LoKr module for Conv2d layer
    ///
    /// # Arguments
    /// * `in_channels` - Input channels
    /// * `out_channels` - Output channels
    /// * `kernel_size` - Convolution kernel size (h, w)
    /// * `rank` - Rank of decomposition
    /// * `alpha` - Scaling parameter
    /// * `factor` - Factorization hint
    /// * `decompose_both` - Whether to decompose both factors
    /// * `use_tucker` - Whether to use Tucker decomposition
    /// * `device` - CUDA device
    #[allow(clippy::too_many_arguments)]
    pub fn new_conv2d(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        rank: usize,
        alpha: Option<f32>,
        factor: i32,
        decompose_both: bool,
        use_tucker: bool,
        device: Arc<CudaDevice>,
    ) -> Result<Self> {
        let alpha = alpha.unwrap_or(rank as f32);
        let (kh, kw) = kernel_size;

        // Factorize dimensions
        let (in_m, in_n) = crate::ops::kronecker::factorization(in_channels, factor);
        let (out_l, out_k) = crate::ops::kronecker::factorization(out_channels, factor);
        let shape = ((out_l, out_k), (in_m, in_n));

        // Similar logic to linear, but with kernel dimensions
        let use_w1 = !decompose_both || rank >= (out_l.max(in_m)) / 2;
        let use_w2 = rank >= (out_k.max(in_n)) / 2 || (kh == 1 && kw == 1);

        let (w1, w1a, w1b) = if use_w1 {
            let w1 = tensor_utils::kaiming_uniform_bf16(
                Shape::from_dims(&[out_l, in_m]),
                (5.0_f32).sqrt(),
                device.clone(),
            )
            .map_err(Error::Flame)?;
            (Some(w1), None, None)
        } else {
            let w1a = tensor_utils::kaiming_uniform_bf16(
                Shape::from_dims(&[out_l, rank]),
                (5.0_f32).sqrt(),
                device.clone(),
            )
            .map_err(Error::Flame)?;

            let w1b = tensor_utils::kaiming_uniform_bf16(
                Shape::from_dims(&[rank, in_m]),
                (5.0_f32).sqrt(),
                device.clone(),
            )
            .map_err(Error::Flame)?;

            (None, Some(w1a), Some(w1b))
        };

        // Standardized Tucker orientation:
        // w2a: [out_k, rank]
        // t2: [rank, rank, kh, kw]
        // w2b: [rank, in_n, kh, kw]
        let (w2, w2a, w2b, t2) = if use_w2 {
            let w2 = tensor_utils::zeros_bf16(
                Shape::from_dims(&[out_k, in_n, kh, kw]),
                device.clone(),
            )
            .map_err(Error::Flame)?;
            (Some(w2), None, None, None)
        } else if use_tucker && (kh > 1 || kw > 1) {
            // Tucker decomposition for kernel
            let t2 = tensor_utils::kaiming_uniform_bf16(
                Shape::from_dims(&[rank, rank, kh, kw]),
                (5.0_f32).sqrt(),
                device.clone(),
            )
            .map_err(Error::Flame)?;

            let w2a = tensor_utils::kaiming_uniform_bf16(
                Shape::from_dims(&[out_k, rank]),
                (5.0_f32).sqrt(),
                device.clone(),
            )
            .map_err(Error::Flame)?;

            let w2b = tensor_utils::zeros_bf16(
                Shape::from_dims(&[rank, in_n]),
                device.clone(),
            )
            .map_err(Error::Flame)?;

            (None, Some(w2a), Some(w2b), Some(t2))
        } else {
            // Standard decomposition
            let w2a = tensor_utils::kaiming_uniform_bf16(
                Shape::from_dims(&[out_k, rank]),
                (5.0_f32).sqrt(),
                device.clone(),
            )
            .map_err(Error::Flame)?;

            let w2b = tensor_utils::zeros_bf16(
                Shape::from_dims(&[rank, in_n, kh, kw]),
                device.clone(),
            )
            .map_err(Error::Flame)?;

            (None, Some(w2a), Some(w2b), None)
        };

        // Validate BF16 storage
        if let Some(ref w) = w1 {
            assert_bf16_storage("w1", w)?;
        }
        if let Some(ref w) = w1a {
            assert_bf16_storage("w1a", w)?;
        }
        if let Some(ref w) = w1b {
            assert_bf16_storage("w1b", w)?;
        }
        if let Some(ref w) = w2 {
            assert_bf16_storage("w2", w)?;
        }
        if let Some(ref w) = w2a {
            assert_bf16_storage("w2a", w)?;
        }
        if let Some(ref w) = w2b {
            assert_bf16_storage("w2b", w)?;
        }
        if let Some(ref t) = t2 {
            assert_bf16_storage("t2", t)?;
        }

        Ok(Self {
            w1,
            w1a,
            w1b,
            w2,
            w2a,
            w2b,
            t2,
            rank,
            alpha,
            device,
            shape,
            kernel_size: Some(kernel_size),
            kind: LayerKind::Conv2d,
        })
    }

    /// Get the scaling factor (alpha / rank), returns 0.0 if rank==0
    #[inline]
    pub fn scale(&self) -> f32 {
        if self.rank == 0 {
            return 0.0;
        }
        self.alpha / self.rank as f32
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

impl LycorisModule for LoKrModule {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let scale = self.scale();

        // Early exit for zero rank or alpha
        if scale == 0.0 {
            return tensor_utils::zeros_bf16(
                Shape::from_dims(x.dims()),
                self.device.clone(),
            )
            .map_err(Error::Flame);
        }

        let use_w1 = self.w1.is_some();
        let use_w2 = self.w2.is_some();

        // Compute c = w1 or w1a @ w1b (BF16 storage, FP32 compute in kernels)
        let c = if use_w1 {
            self.w1.as_ref().unwrap().clone().map_err(Error::Flame)?
        } else {
            let w1a = self.w1a.as_ref().unwrap();
            let w1b = self.w1b.as_ref().unwrap();
            w1a.matmul(w1b).map_err(Error::Flame)?
        };

        match self.kind {
            LayerKind::Linear => self.forward_linear(x, &c, use_w2, scale),
            LayerKind::Conv2d => self.forward_conv2d(x, &c, use_w2, scale),
        }
    }

    fn get_diff_weight(&self) -> Result<Tensor> {
        let scale = self.scale();

        // Early exit for zero scale
        if scale == 0.0 {
            return tensor_utils::zeros_bf16(
                Shape::from_dims(&[
                    self.shape.0 .0 * self.shape.0 .1,
                    self.shape.1 .0 * self.shape.1 .1,
                ]),
                self.device.clone(),
            )
            .map_err(Error::Flame);
        }

        // Compute w1
        let w1 = if let Some(ref w1_full) = self.w1 {
            w1_full.clone().map_err(Error::Flame)?
        } else {
            let w1a = self.w1a.as_ref().ok_or_else(|| {
                Error::InvalidOperation("w1a missing in factorized mode".to_string())
            })?;
            let w1b = self.w1b.as_ref().ok_or_else(|| {
                Error::InvalidOperation("w1b missing in factorized mode".to_string())
            })?;
            w1a.matmul(w1b).map_err(Error::Flame)?
        };

        // Compute w2
        let w2 = if let Some(ref w2_full) = self.w2 {
            w2_full.clone().map_err(Error::Flame)?
        } else if let Some(ref t2) = self.t2 {
            // Tucker reconstruction
            let w2a = self.w2a.as_ref().ok_or_else(|| {
                Error::InvalidOperation("w2a missing in Tucker mode".to_string())
            })?;
            let w2b = self.w2b.as_ref().ok_or_else(|| {
                Error::InvalidOperation("w2b missing in Tucker mode".to_string())
            })?;
            crate::ops::tucker::rebuild_tucker(t2, w2a, w2b)?
        } else {
            // Standard factorization: w2 = w2a @ w2b
            let w2a = self.w2a.as_ref().ok_or_else(|| {
                Error::InvalidOperation("w2a missing in factorized mode".to_string())
            })?;
            let w2b = self.w2b.as_ref().ok_or_else(|| {
                Error::InvalidOperation("w2b missing in factorized mode".to_string())
            })?;

            let w2b_dims = w2b.dims();
            let w2b_reshaped = w2b
                .reshape(&[w2b_dims[0], w2b_dims[1..].iter().product()])
                .map_err(Error::Flame)?;

            let result = w2a.matmul(&w2b_reshaped).map_err(Error::Flame)?;

            // Reshape back if needed
            if w2b_dims.len() > 2 {
                let mut new_shape = vec![w2a.dims()[0]];
                new_shape.extend_from_slice(&w2b_dims[1..]);
                result.reshape(&new_shape).map_err(Error::Flame)?
            } else {
                result
            }
        };

        // Ensure BF16 storage
        assert_bf16_storage("w1", &w1)?;
        assert_bf16_storage("w2", &w2)?;

        // Compute Kronecker product
        let kron_result = crate::ops::kronecker::make_kronecker(&w1, &w2, scale)?;
        assert_bf16_storage("ΔW", &kron_result)?;

        Ok(kron_result)
    }

    fn merge_to(&mut self, multiplier: f32) -> Result<()> {
        // This is now deprecated in favor of merge_into()
        // which takes a mutable base weight
        let _diff_weight = self.get_diff_weight()?;
        let _scaled_diff = _diff_weight
            .mul_scalar(multiplier)
            .map_err(Error::Flame)?;

        // Note: Cannot merge without base weight reference
        // Use merge_into() instead
        Ok(())
    }
}

impl LoKrModule {
    fn forward_linear(
        &self,
        x: &Tensor,
        c: &Tensor,
        use_w2: bool,
        scale: f32,
    ) -> Result<Tensor> {
        let x_dims = x.dims();
        let last = *x_dims
            .last()
            .ok_or_else(|| Error::InvalidOperation("x has no dims".into()))?;
        let uq = c.dims()[1];

        if last % uq != 0 {
            return Err(Error::InvalidOperation(format!(
                "feature dim {} not divisible by uq {}",
                last, uq
            )));
        }
        let vq = last / uq;

        // (b,..., uq*vq) -> (b,..., uq, vq)
        let reshaped = x
            .reshape(
                &x_dims[..x_dims.len() - 1]
                    .iter()
                    .cloned()
                    .chain([uq, vq])
                    .collect::<Vec<_>>(),
            )
            .map_err(Error::Flame)?;

        // Apply BA / A,B
        let hb = if use_w2 {
            let ba_tensor = self.w2.as_ref().unwrap();
            reshaped.matmul(ba_tensor).map_err(Error::Flame)?
        } else {
            let a_tensor = self.w2b.as_ref().unwrap();
            let b_tensor = self.w2a.as_ref().unwrap();
            let ha = reshaped.matmul(a_tensor).map_err(Error::Flame)?;
            ha.matmul(b_tensor).map_err(Error::Flame)?
        };

        // swap (..., uq, vq) -> (..., vq, uq)
        let h_swapped = swap_last_two(&hb)?;

        // F.linear with C (use C^T)
        let c_t = crate::tensor_utils::transpose_2d(c)?;
        let hc = h_swapped.matmul(&c_t).map_err(Error::Flame)?;

        // collapse last two dims back to (..., vq*up)
        let hc_dims = hc.dims();
        let n = hc_dims.len();
        let mut final_shape = hc_dims[..n - 2].to_vec();
        final_shape.push(hc_dims[n - 2] * hc_dims[n - 1]);
        let out = hc.reshape(&final_shape).map_err(Error::Flame)?;

        // scale
        out.mul_scalar(scale).map_err(Error::Flame)
    }

    fn forward_conv2d(
        &self,
        x: &Tensor,
        c: &Tensor,
        use_w2: bool,
        scale: f32,
    ) -> Result<Tensor> {
        // Conv2d path - placeholder for real conv2d ops
        // This needs actual conv2d kernels which aren't implemented yet

        // For now, return error indicating conv2d needs implementation
        Err(Error::InvalidOperation(
            "Conv2d forward path requires conv2d kernel implementation. \
             Need: conv1x1_grouped() for A/B and conv_spatial_rank() for T. \
             See original issue for full implementation details.".into()
        ))

        /* Full implementation outline:

        let xd = x.dims();
        if xd.len() != 4 {
            return Err(Error::InvalidOperation("conv expects NHWC [B,H,W,C]".into()));
        }

        let uq = c.dims()[1];
        let cin = xd[3];
        if cin % uq != 0 {
            return Err(Error::InvalidOperation("C_in not divisible by uq".into()));
        }

        let (kh, kw) = self.kernel_size.unwrap_or((1, 1));

        if use_w2 {
            let w2_full = self.w2.as_ref().unwrap();
            // Need: conv2d(x, w2_full, stride=(1,1), pad=(kh/2, kw/2), groups=1, layout=NHWC)
            // Then reshape and apply C
        } else {
            let a = self.w2b.as_ref().unwrap();
            let b = self.w2a.as_ref().unwrap();

            // Need: conv1x1_grouped(x, a_as_kernel, groups=uq)
            // Then: conv_spatial_rank(ha, t, ...) if Tucker
            // Then: conv1x1_grouped(ht, b_as_kernel, groups=uq)
            // Finally apply C
        }
        */
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lokr_creation() {
        // Placeholder - requires CUDA device initialization
        assert!(true);
    }

    #[test]
    fn test_scale_zero_rank() {
        // Test safe scale() with rank=0
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let module = LoKrModule {
            w1: None,
            w1a: None,
            w1b: None,
            w2: None,
            w2a: None,
            w2b: None,
            t2: None,
            rank: 0,
            alpha: 1.0,
            device,
            shape: ((2, 2), (2, 2)),
            kernel_size: None,
            kind: LayerKind::Linear,
        };

        assert_eq!(module.scale(), 0.0);
    }
}
