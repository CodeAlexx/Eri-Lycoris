/// Tensor utility functions for LyCORIS
///
/// Helper functions to work with Flame tensors using BF16 storage

use crate::{Error, Result};
use cudarc::driver::CudaDevice;
use flame_core::{DType, Shape, Tensor};
use std::sync::Arc;

/// Create a random tensor with BF16 dtype
pub fn randn_bf16(
    shape: Shape,
    mean: f32,
    std: f32,
    device: Arc<CudaDevice>,
) -> Result<Tensor> {
    // Create FP32 random tensor first
    let tensor_f32 = Tensor::randn(shape.clone(), mean, std, device.clone())
        .map_err(|e| Error::Flame(e))?;

    // Convert to BF16
    tensor_f32.to_dtype(DType::BF16).map_err(|e| Error::Flame(e))
}

/// Create zeros tensor with BF16 dtype
pub fn zeros_bf16(shape: Shape, device: Arc<CudaDevice>) -> Result<Tensor> {
    Tensor::zeros_dtype(shape, DType::BF16, device).map_err(|e| Error::Flame(e))
}

/// Create BF16 tensor with Kaiming uniform initialization
///
/// Python: torch.nn.init.kaiming_uniform_(tensor, a=sqrt(5))
/// Formula: U(-bound, bound) where bound = gain * sqrt(3 / fan_in)
/// gain = sqrt(2 / (1 + a²))
///
/// # Arguments
/// * `shape` - Tensor shape
/// * `a` - Negative slope parameter (use sqrt(5) for LoKr)
/// * `device` - CUDA device
pub fn kaiming_uniform_bf16(
    shape: Shape,
    a: f32,
    device: Arc<CudaDevice>,
) -> Result<Tensor> {
    let dims = shape.dims();
    let fan_in = if dims.len() >= 2 {
        dims[1]
    } else {
        dims[0]
    };

    let gain = (2.0 / (1.0 + a * a)).sqrt();
    let std = gain * (3.0 / fan_in as f32).sqrt();

    // Note: PyTorch uses uniform distribution U(-bound, bound)
    // We approximate with normal distribution N(0, std) for simplicity
    // Exact uniform would require custom kernel or Flame API extension
    randn_bf16(shape, 0.0, std, device)
}

/// Create tensor from vec with BF16 dtype
pub fn from_vec_bf16(data: Vec<f32>, shape: Shape, device: Arc<CudaDevice>) -> Result<Tensor> {
    let tensor_f32 = Tensor::from_vec(data, shape, device).map_err(|e| Error::Flame(e))?;
    tensor_f32.to_dtype(DType::BF16).map_err(|e| Error::Flame(e))
}

/// Transpose 2D tensor (handles Flame's transpose API)
pub fn transpose_2d(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.dims();
    if dims.len() != 2 {
        return Err(Error::InvalidOperation(format!(
            "transpose_2d requires 2D tensor, got {}D",
            dims.len()
        )));
    }

    // Flame's transpose() swaps last two dimensions for 2D
    tensor.transpose().map_err(|e| Error::Flame(e))
}

/// Kronecker product implementation
///
/// Computes the Kronecker product of two tensors
pub fn kronecker_product(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_dims = a.dims();
    let b_dims = b.dims();

    if a_dims.len() != 2 || b_dims.len() != 2 {
        return Err(Error::InvalidOperation(
            "Kronecker product requires 2D tensors".to_string()
        ));
    }

    let (m, n) = (a_dims[0], a_dims[1]);
    let (p, q) = (b_dims[0], b_dims[1]);

    // Get data from tensors
    let a_data = a.to_vec().map_err(|e| Error::Flame(e))?;
    let b_data = b.to_vec().map_err(|e| Error::Flame(e))?;

    // Compute Kronecker product on CPU
    let mut result = vec![0.0f32; m * p * n * q];

    for i in 0..m {
        for j in 0..n {
            let a_val = a_data[i * n + j];
            for k in 0..p {
                for l in 0..q {
                    let b_val = b_data[k * q + l];
                    result[(i * p + k) * (n * q) + (j * q + l)] = a_val * b_val;
                }
            }
        }
    }

    // Create output tensor
    let output_shape = Shape::from_dims(&[m * p, n * q]);
    let output = Tensor::from_vec(result, output_shape, a.device().clone())
        .map_err(|e| Error::Flame(e))?;

    // Convert to BF16 if input was BF16
    if a.dtype() == DType::BF16 {
        output.to_dtype(DType::BF16).map_err(|e| Error::Flame(e))
    } else {
        Ok(output)
    }
}

/// Helper to get CudaDevice from Tensor
pub fn get_cuda_device(tensor: &Tensor) -> Arc<CudaDevice> {
    tensor.device().clone()
}
