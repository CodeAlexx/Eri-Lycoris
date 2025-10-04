/// FP32 reduce/accumulate accuracy tests
///
/// Verifies that reductions and accumulations use FP32 internally for numerical stability

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use flame_core::{Device, DType, Shape, Tensor};

    fn get_test_device() -> std::sync::Arc<Device> {
        Device::cuda_if_available(0).expect("CUDA device required for tests")
    }

    #[test]
    fn test_bf16_matmul_accuracy() {
        let device = get_test_device();

        // Create small test matrices with known values
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![5.0f32, 6.0, 7.0, 8.0];

        let a = Tensor::from_vec_dtype(
            a_data.clone(),
            Shape::from_dims(&[2, 2]),
            device.clone(),
            DType::BF16,
        )
        .expect("Failed to create tensor A");

        let b = Tensor::from_vec_dtype(
            b_data.clone(),
            Shape::from_dims(&[2, 2]),
            device.clone(),
            DType::BF16,
        )
        .expect("Failed to create tensor B");

        // Compute matmul
        let c = a.matmul(&b).expect("Matmul failed");

        // Expected result (computed in FP32):
        // [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]]
        // = [[19, 22], [43, 50]]
        let c_data = c.to_vec().expect("Failed to read result");

        // Allow some tolerance due to BF16 precision
        assert_abs_diff_eq!(c_data[0], 19.0, epsilon = 0.1);
        assert_abs_diff_eq!(c_data[1], 22.0, epsilon = 0.1);
        assert_abs_diff_eq!(c_data[2], 43.0, epsilon = 0.1);
        assert_abs_diff_eq!(c_data[3], 50.0, epsilon = 0.1);
    }

    #[test]
    fn test_bf16_sum_uses_fp32_accumulator() {
        let device = get_test_device();

        // Create tensor with many small values
        // This tests that accumulation doesn't lose precision
        let n = 1000;
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();

        let tensor = Tensor::from_vec_dtype(
            data.clone(),
            Shape::from_dims(&[n]),
            device.clone(),
            DType::BF16,
        )
        .expect("Failed to create tensor");

        // Sum all elements
        let sum = tensor.sum(None).expect("Sum failed");

        // Expected: sum of arithmetic series
        let expected = (0..n).map(|i| (i as f32) * 0.001).sum::<f32>();

        let result = sum.item().expect("Failed to get item");

        // Should be accurate due to FP32 accumulation
        assert_abs_diff_eq!(result, expected, epsilon = 0.01);
    }

    #[test]
    fn test_bf16_mean_accuracy() {
        let device = get_test_device();

        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let tensor = Tensor::from_vec_dtype(
            data.clone(),
            Shape::from_dims(&[8]),
            device.clone(),
            DType::BF16,
        )
        .expect("Failed to create tensor");

        // Compute mean
        let mean = tensor.mean(None).expect("Mean failed");

        let result = mean.item().expect("Failed to get item");
        let expected = 4.5f32;

        assert_abs_diff_eq!(result, expected, epsilon = 0.01);
    }

    #[test]
    fn test_bf16_variance_accuracy() {
        let device = get_test_device();

        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        let tensor = Tensor::from_vec_dtype(
            data.clone(),
            Shape::from_dims(&[5]),
            device.clone(),
            DType::BF16,
        )
        .expect("Failed to create tensor");

        // Variance should use FP32 for intermediate calculations
        let mean = tensor.mean(None).expect("Mean failed");
        let centered = tensor.sub(&mean).expect("Subtract failed");
        let squared = centered.mul(&centered).expect("Multiply failed");
        let variance = squared.mean(None).expect("Mean failed");

        let result = variance.item().expect("Failed to get item");

        // Expected variance of [1,2,3,4,5] is 2.0
        let expected = 2.0f32;

        assert_abs_diff_eq!(result, expected, epsilon = 0.1);
    }

    #[test]
    fn test_bf16_hadamard_product_accuracy() {
        let device = get_test_device();

        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![5.0f32, 6.0, 7.0, 8.0];

        let a = Tensor::from_vec_dtype(
            a_data.clone(),
            Shape::from_dims(&[4]),
            device.clone(),
            DType::BF16,
        )
        .expect("Failed to create tensor A");

        let b = Tensor::from_vec_dtype(
            b_data.clone(),
            Shape::from_dims(&[4]),
            device.clone(),
            DType::BF16,
        )
        .expect("Failed to create tensor B");

        // Element-wise multiply
        let c = a.mul(&b).expect("Multiply failed");

        let c_data = c.to_vec().expect("Failed to read result");

        // Expected: [5, 12, 21, 32]
        assert_abs_diff_eq!(c_data[0], 5.0, epsilon = 0.1);
        assert_abs_diff_eq!(c_data[1], 12.0, epsilon = 0.1);
        assert_abs_diff_eq!(c_data[2], 21.0, epsilon = 0.1);
        assert_abs_diff_eq!(c_data[3], 32.0, epsilon = 0.1);
    }
}
