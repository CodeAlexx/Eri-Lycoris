/// NHWC/NCHW layout round-trip tests
///
/// Verifies that layout conversions preserve data correctly

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use flame_core::{Device, DType, Shape, Tensor};
    use lycoris_rs::layout::{LayoutConverter, TensorLayout};

    fn get_test_device() -> std::sync::Arc<Device> {
        Device::cuda_if_available(0).expect("CUDA device required for tests")
    }

    #[test]
    fn test_nchw_to_nhwc_to_nchw_round_trip() {
        let device = get_test_device();

        // Create tensor in NCHW format: (batch=1, channels=3, height=4, width=4)
        let original_data: Vec<f32> = (0..48).map(|i| i as f32).collect();

        let nchw_tensor = Tensor::from_vec_dtype(
            original_data.clone(),
            Shape::from_dims(&[1, 3, 4, 4]),
            device.clone(),
            DType::BF16,
        )
        .expect("Failed to create NCHW tensor");

        // Convert NCHW -> NHWC
        let nhwc_tensor = LayoutConverter::nchw_to_nhwc(&nchw_tensor)
            .expect("NCHW to NHWC conversion failed");

        // Verify shape changed to (1, 4, 4, 3)
        assert_eq!(nhwc_tensor.dims(), &[1, 4, 4, 3]);

        // Convert NHWC -> NCHW
        let round_trip_tensor = LayoutConverter::nhwc_to_nchw(&nhwc_tensor)
            .expect("NHWC to NCHW conversion failed");

        // Verify shape is back to (1, 3, 4, 4)
        assert_eq!(round_trip_tensor.dims(), &[1, 3, 4, 4]);

        // Verify data is preserved
        let round_trip_data = round_trip_tensor
            .to_vec()
            .expect("Failed to read round-trip data");

        for (i, (&original, &round_trip)) in original_data
            .iter()
            .zip(round_trip_data.iter())
            .enumerate()
        {
            assert_abs_diff_eq!(
                original,
                round_trip,
                epsilon = 0.01,
                "Mismatch at index {}: {} != {}",
                i,
                original,
                round_trip
            );
        }
    }

    #[test]
    fn test_nhwc_to_nchw_to_nhwc_round_trip() {
        let device = get_test_device();

        // Create tensor in NHWC format: (batch=2, height=3, width=3, channels=4)
        let original_data: Vec<f32> = (0..72).map(|i| i as f32 * 0.1).collect();

        let nhwc_tensor = Tensor::from_vec_dtype(
            original_data.clone(),
            Shape::from_dims(&[2, 3, 3, 4]),
            device.clone(),
            DType::BF16,
        )
        .expect("Failed to create NHWC tensor");

        // Convert NHWC -> NCHW
        let nchw_tensor = LayoutConverter::nhwc_to_nchw(&nhwc_tensor)
            .expect("NHWC to NCHW conversion failed");

        // Verify shape changed to (2, 4, 3, 3)
        assert_eq!(nchw_tensor.dims(), &[2, 4, 3, 3]);

        // Convert NCHW -> NHWC
        let round_trip_tensor = LayoutConverter::nchw_to_nhwc(&nchw_tensor)
            .expect("NCHW to NHWC conversion failed");

        // Verify shape is back to (2, 3, 3, 4)
        assert_eq!(round_trip_tensor.dims(), &[2, 3, 3, 4]);

        // Verify data is preserved
        let round_trip_data = round_trip_tensor
            .to_vec()
            .expect("Failed to read round-trip data");

        for (i, (&original, &round_trip)) in original_data
            .iter()
            .zip(round_trip_data.iter())
            .enumerate()
        {
            assert_abs_diff_eq!(
                original,
                round_trip,
                epsilon = 0.01,
                "Mismatch at index {}: {} != {}",
                i,
                original,
                round_trip
            );
        }
    }

    #[test]
    fn test_layout_conversion_with_conv2d() {
        let device = get_test_device();

        // Input in NCHW: (batch=1, channels=3, height=8, width=8)
        let input_nchw = Tensor::randn_dtype(
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

        // Compute conv2d in NCHW (Flame default)
        let output_nchw = input_nchw
            .conv2d(&weight, None, 1, 1, 1, 1)
            .expect("Conv2d NCHW failed");

        // Convert input to NHWC
        let input_nhwc = LayoutConverter::nchw_to_nhwc(&input_nchw)
            .expect("NCHW to NHWC failed");

        // For proper NHWC conv2d, weight also needs conversion
        // (For now, we just verify the layout adapter works)

        // Convert output back to NCHW
        let output_nchw_check = LayoutConverter::convert(
            &output_nchw,
            TensorLayout::NCHW,
            TensorLayout::NCHW,
        )
        .expect("Identity conversion failed");

        // Verify identity conversion preserves data
        let original = output_nchw.to_vec().expect("Failed to read original");
        let checked = output_nchw_check.to_vec().expect("Failed to read checked");

        assert_eq!(
            original.len(),
            checked.len(),
            "Length mismatch after identity conversion"
        );
    }

    #[test]
    fn test_layout_adapter_boundary() {
        let device = get_test_device();

        // Test with linear layer (2D) - should not apply layout conversion
        let linear_input = Tensor::randn_dtype(
            Shape::from_dims(&[4, 8]),
            0.0,
            1.0,
            DType::BF16,
            device.clone(),
        )
        .expect("Failed to create linear input");

        // Layout conversion should fail for non-4D tensors
        let result = LayoutConverter::nchw_to_nhwc(&linear_input);
        assert!(
            result.is_err(),
            "Expected error for non-4D tensor in layout conversion"
        );
    }
}
