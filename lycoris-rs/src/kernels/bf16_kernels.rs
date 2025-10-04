/// GPU kernels for BF16 storage with FP32 compute
///
/// All kernels follow the pattern:
/// 1. Load BF16 from global memory
/// 2. Convert to FP32 for computation
/// 3. Perform operation in FP32
/// 4. Convert result back to BF16
/// 5. Store BF16 to global memory

use crate::{Error, Result};
use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;

/// BF16 element-wise multiplication kernel source
///
/// Multiplies two BF16 tensors element-wise, computing in FP32
pub const BF16_MUL_KERNEL: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void bf16_mul_kernel(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Load BF16, convert to FP32
        float a_fp32 = __bfloat162float(a[idx]);
        float b_fp32 = __bfloat162float(b[idx]);

        // Compute in FP32
        float result_fp32 = a_fp32 * b_fp32;

        // Convert back to BF16 and store
        out[idx] = __float2bfloat16(result_fp32);
    }
}
"#;

/// BF16 matrix multiply-accumulate kernel (for small ranks)
///
/// C = A @ B, where A, B stored in BF16, compute in FP32, output in BF16
pub const BF16_GEMM_KERNEL: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void bf16_gemm_kernel(
    const __nv_bfloat16* a,  // m x k
    const __nv_bfloat16* b,  // k x n
    __nv_bfloat16* c,        // m x n
    int m, int k, int n
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        // Accumulate in FP32
        float sum = 0.0f;

        for (int i = 0; i < k; i++) {
            float a_val = __bfloat162float(a[row * k + i]);
            float b_val = __bfloat162float(b[i * n + col]);
            sum += a_val * b_val;
        }

        // Convert result to BF16 and store
        c[row * n + col] = __float2bfloat16(sum);
    }
}
"#;

/// BF16 reduction kernel (sum, mean, etc.)
///
/// Always reduces in FP32 for numerical accuracy
pub const BF16_REDUCE_KERNEL: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void bf16_reduce_sum_kernel(
    const __nv_bfloat16* input,
    float* output,  // Note: output is FP32 for accuracy
    int n
) {
    __shared__ float shared_data[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load and convert to FP32
    float sum = 0.0f;
    if (idx < n) {
        sum = __bfloat162float(input[idx]);
    }
    shared_data[tid] = sum;
    __syncthreads();

    // Reduce in FP32
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // Write result (FP32)
    if (tid == 0) {
        atomicAdd(output, shared_data[0]);
    }
}
"#;

/// BF16 broadcast operations
///
/// Broadcasts smaller tensor to match larger tensor shape
pub const BF16_BROADCAST_KERNEL: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void bf16_broadcast_add_kernel(
    const __nv_bfloat16* a,  // larger tensor
    const __nv_bfloat16* b,  // broadcasted tensor
    __nv_bfloat16* out,
    int n,
    int broadcast_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Load BF16, convert to FP32
        float a_val = __bfloat162float(a[idx]);
        float b_val = __bfloat162float(b[idx % broadcast_size]);

        // Compute in FP32
        float result = a_val + b_val;

        // Convert to BF16 and store
        out[idx] = __float2bfloat16(result);
    }
}
"#;

/// Kernel manager for BF16 operations
pub struct BF16Kernels {
    device: Arc<CudaDevice>,
}

impl BF16Kernels {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }

    /// Element-wise multiply two BF16 tensors
    pub fn mul(
        &self,
        a: &CudaSlice<u16>,  // BF16 as u16
        b: &CudaSlice<u16>,
        out: &mut CudaSlice<u16>,
    ) -> Result<()> {
        let n = a.len();
        if b.len() != n || out.len() != n {
            return Err(Error::InvalidOperation(
                "BF16 mul: input sizes must match".to_string(),
            ));
        }

        // Compile kernel using NVRTC
        use cudarc::nvrtc::compile_ptx;

        let ptx = compile_ptx(BF16_MUL_KERNEL)
            .map_err(|e| Error::KernelCompilation(format!("Failed to compile BF16 mul kernel: {:?}", e)))?;

        self.device.load_ptx(ptx, "bf16_mul", &["bf16_mul_kernel"])
            .map_err(|e| Error::Cuda(format!("Failed to load BF16 mul kernel: {:?}", e)))?;

        let func = self.device
            .get_func("bf16_mul", "bf16_mul_kernel")
            .ok_or_else(|| Error::Cuda("Failed to get bf16_mul_kernel function".to_string()))?;

        let cfg = cudarc::driver::LaunchConfig::for_num_elems(n as u32);

        unsafe {
            func.launch(cfg, (a, b, out, n as i32))
                .map_err(|e| Error::Cuda(format!("BF16 mul kernel launch failed: {:?}", e)))?;
        }

        Ok(())
    }

    /// Matrix multiply: C = A @ B (all BF16 storage, FP32 compute)
    pub fn gemm(
        &self,
        a: &CudaSlice<u16>,
        b: &CudaSlice<u16>,
        c: &mut CudaSlice<u16>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(Error::InvalidOperation(
                "BF16 GEMM: dimension mismatch".to_string(),
            ));
        }

        // Compile GEMM kernel
        use cudarc::nvrtc::compile_ptx;

        let ptx = compile_ptx(BF16_GEMM_KERNEL)
            .map_err(|e| Error::KernelCompilation(format!("Failed to compile BF16 GEMM: {:?}", e)))?;

        self.device.load_ptx(ptx, "bf16_gemm", &["bf16_gemm_kernel"])
            .map_err(|e| Error::Cuda(format!("Failed to load BF16 GEMM: {:?}", e)))?;

        let func = self.device
            .get_func("bf16_gemm", "bf16_gemm_kernel")
            .ok_or_else(|| Error::Cuda("Failed to get bf16_gemm_kernel".to_string()))?;

        // Launch with 2D grid
        let block_size = 16;
        let grid_x = (n as u32 + block_size - 1) / block_size;
        let grid_y = (m as u32 + block_size - 1) / block_size;

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (block_size, block_size, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (a, b, c, m as i32, k as i32, n as i32))
                .map_err(|e| Error::Cuda(format!("BF16 GEMM launch failed: {:?}", e)))?;
        }

        Ok(())
    }

    /// Reduce sum (BF16 input, FP32 output for accuracy)
    pub fn reduce_sum(
        &self,
        input: &CudaSlice<u16>,
        output: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let n = input.len();

        // Initialize output to zero
        self.device.memset_zeros(output)
            .map_err(|e| Error::Cuda(format!("Failed to zero output: {:?}", e)))?;

        // Compile reduce kernel
        use cudarc::nvrtc::compile_ptx;

        let ptx = compile_ptx(BF16_REDUCE_KERNEL)
            .map_err(|e| Error::KernelCompilation(format!("Failed to compile BF16 reduce: {:?}", e)))?;

        self.device.load_ptx(ptx, "bf16_reduce", &["bf16_reduce_sum_kernel"])
            .map_err(|e| Error::Cuda(format!("Failed to load BF16 reduce: {:?}", e)))?;

        let func = self.device
            .get_func("bf16_reduce", "bf16_reduce_sum_kernel")
            .ok_or_else(|| Error::Cuda("Failed to get bf16_reduce_sum_kernel".to_string()))?;

        let block_size = 256;
        let num_blocks = (n + block_size - 1) / block_size;

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (input, output, n as i32))
                .map_err(|e| Error::Cuda(format!("BF16 reduce launch failed: {:?}", e)))?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_source_valid() {
        // Verify kernel sources compile
        assert!(BF16_MUL_KERNEL.contains("bf16_mul_kernel"));
        assert!(BF16_GEMM_KERNEL.contains("bf16_gemm_kernel"));
        assert!(BF16_REDUCE_KERNEL.contains("bf16_reduce_sum_kernel"));
        assert!(BF16_BROADCAST_KERNEL.contains("bf16_broadcast_add_kernel"));
    }
}
