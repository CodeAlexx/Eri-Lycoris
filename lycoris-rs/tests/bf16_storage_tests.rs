/// BF16 storage preservation tests
///
/// Verifies that tensors maintain BF16 storage through operations

#[cfg(test)]
mod tests {
    use flame_core::{Device, DType, Shape, Tensor};
    use lycoris_rs::dtype::DTypeExt;

    fn get_test_device() -> std::sync::Arc<Device> {
        Device::cuda_if_available(0).expect("CUDA device required for tests")
    }

    #[test]
    fn test_bf16_tensor_creation_preserves_dtype() {
        let device = get_test_device();

        // Create BF16 tensor
        let shape = Shape::from_dims(&[4, 4]);
        let tensor = Tensor::zeros_dtype(shape, DType::BF16, device.clone())
            .expect("Failed to create BF16 tensor");

        // Verify dtype is preserved
        assert!(
            tensor.dtype().is_bf16(),
            "Expected BF16 dtype, got {:?}",
            tensor.dtype()
        );
    }

    #[test]
    fn test_bf16_preserved_after_matmul() {
        let device = get_test_device();

        // Create two BF16 tensors
        let a = Tensor::randn_dtype(
            Shape::from_dims(&[4, 8]),
            0.0,
            1.0,
            DType::BF16,
            device.clone(),
        )
        .expect("Failed to create tensor A");

        let b = Tensor::randn_dtype(
            Shape::from_dims(&[8, 4]),
            0.0,
            1.0,
            DType::BF16,
            device.clone(),
        )
        .expect("Failed to create tensor B");

        // Perform matmul (should use FP32 internally but preserve BF16 storage)
        let c = a.matmul(&b).expect("Matmul failed");

        // Verify output is still BF16
        assert!(
            c.dtype().is_bf16(),
            "Expected BF16 output from matmul, got {:?}",
            c.dtype()
        );
    }

    #[test]
    fn test_bf16_preserved_through_elementwise_ops() {
        let device = get_test_device();

        let a = Tensor::randn_dtype(
            Shape::from_dims(&[4, 4]),
            0.0,
            1.0,
            DType::BF16,
            device.clone(),
        )
        .expect("Failed to create tensor");

        let b = Tensor::randn_dtype(
            Shape::from_dims(&[4, 4]),
            0.0,
            1.0,
            DType::BF16,
            device.clone(),
        )
        .expect("Failed to create tensor");

        // Element-wise multiply (Hadamard product)
        let c = a.mul(&b).expect("Multiply failed");

        assert!(
            c.dtype().is_bf16(),
            "Expected BF16 after element-wise multiply, got {:?}",
            c.dtype()
        );
    }

    #[test]
    fn test_bf16_preserved_after_reshape() {
        let device = get_test_device();

        let tensor = Tensor::randn_dtype(
            Shape::from_dims(&[4, 4]),
            0.0,
            1.0,
            DType::BF16,
            device.clone(),
        )
        .expect("Failed to create tensor");

        // Reshape
        let reshaped = tensor
            .reshape(&[2, 8])
            .expect("Reshape failed");

        assert!(
            reshaped.dtype().is_bf16(),
            "Expected BF16 after reshape, got {:?}",
            reshaped.dtype()
        );
    }

    #[test]
    fn test_bf16_preserved_after_transpose() {
        let device = get_test_device();

        let tensor = Tensor::randn_dtype(
            Shape::from_dims(&[4, 8]),
            0.0,
            1.0,
            DType::BF16,
            device.clone(),
        )
        .expect("Failed to create tensor");

        // Transpose
        let transposed = tensor
            .transpose(0, 1)
            .expect("Transpose failed");

        assert!(
            transposed.dtype().is_bf16(),
            "Expected BF16 after transpose, got {:?}",
            transposed.dtype()
        );
    }

    #[test]
    fn test_bf16_conv2d_preservation() {
        let device = get_test_device();

        // Input: (batch=1, channels=3, height=8, width=8)
        let input = Tensor::randn_dtype(
            Shape::from_dims(&[1, 3, 8, 8]),
            0.0,
            1.0,
            DType::BF16,
            device.clone(),
        )
        .expect("Failed to create input");

        // Weight: (out_channels=6, in_channels=3, kh=3, kw=3)
        let weight = Tensor::randn_dtype(
            Shape::from_dims(&[6, 3, 3, 3]),
            0.0,
            1.0,
            DType::BF16,
            device.clone(),
        )
        .expect("Failed to create weight");

        // Conv2d
        let output = input
            .conv2d(&weight, None, 1, 1, 1, 1)
            .expect("Conv2d failed");

        assert!(
            output.dtype().is_bf16(),
            "Expected BF16 after conv2d, got {:?}",
            output.dtype()
        );
    }
}
