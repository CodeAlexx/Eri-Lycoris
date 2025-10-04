/// Basic usage example for LyCORIS-RS
///
/// Demonstrates creating and using LoRA modules

use flame_core::Device;
use lycoris_rs::algorithms::{LoConModule, LoHaModule, LoKrModule};
use lycoris_rs::{LycorisModule, Result};

fn main() -> Result<()> {
    println!("LyCORIS-RS Basic Usage Example\n");

    // Initialize CUDA device
    let device = Device::cuda_if_available(0).expect("CUDA device required");
    println!("Using device: CUDA:{}", device.ordinal());

    // Example 1: LoRA (LoCon) for Linear Layer
    println!("\n1. Creating LoRA (LoCon) module for linear layer...");
    let locon = LoConModule::new_linear(
        512,           // in_features
        512,           // out_features
        8,             // rank
        Some(8.0),     // alpha
        device.clone(),
    )?;
    println!("   Created LoConModule: rank={}, alpha={}", locon.rank, locon.alpha);

    // Get differential weight
    let delta_w_locon = locon.get_diff_weight()?;
    println!("   ΔW shape: {:?}", delta_w_locon.dims());

    // Example 2: LoHa (Hadamard) for Linear Layer
    println!("\n2. Creating LoHa module for linear layer...");
    let loha = LoHaModule::new_linear(
        512,
        512,
        8,
        Some(8.0),
        device.clone(),
    )?;
    println!("   Created LoHaModule: rank={}, alpha={}", loha.rank, loha.alpha);

    let delta_w_loha = loha.get_diff_weight()?;
    println!("   ΔW shape: {:?}", delta_w_loha.dims());

    // Example 3: LoKr (Kronecker) for Linear Layer
    println!("\n3. Creating LoKr module for linear layer...");
    let lokr = LoKrModule::new_linear(
        512,
        512,
        8,
        Some(8.0),
        -1,            // auto-factorization
        false,         // decompose_both
        device.clone(),
    )?;
    println!("   Created LoKrModule: rank={}, alpha={}", lokr.rank, lokr.alpha);
    println!("   Factorization shape: {:?}", lokr.shape);

    let delta_w_lokr = lokr.get_diff_weight()?;
    println!("   ΔW shape: {:?}", delta_w_lokr.dims());

    // Example 4: LoRA for Conv2d Layer
    println!("\n4. Creating LoRA module for conv2d layer...");
    let locon_conv = LoConModule::new_conv2d(
        3,                // in_channels
        64,               // out_channels
        (3, 3),           // kernel_size
        8,                // rank
        Some(8.0),        // alpha
        true,             // use_tucker
        device.clone(),
    )?;
    println!("   Created Conv2d LoConModule: rank={}, alpha={}", locon_conv.rank, locon_conv.alpha);
    println!("   Has Tucker core: {}", locon_conv.mid.is_some());

    let delta_w_conv = locon_conv.get_diff_weight()?;
    println!("   ΔW shape: {:?}", delta_w_conv.dims());

    // Example 5: Memory usage comparison
    println!("\n5. Memory usage (BF16 vs FP32):");
    let params_locon = locon.rank * (512 + 512); // down + up
    let params_loha = 2 * locon.rank * (512 + 512); // 2x for w1 and w2
    let bf16_bytes = params_locon * 2; // 2 bytes per BF16
    let fp32_bytes = params_locon * 4; // 4 bytes per FP32

    println!("   LoRA parameters: {}", params_locon);
    println!("   LoHa parameters: {}", params_loha);
    println!("   BF16 storage: {} KB", bf16_bytes / 1024);
    println!("   FP32 storage: {} KB", fp32_bytes / 1024);
    println!("   Savings: {}%", ((1.0 - (bf16_bytes as f32 / fp32_bytes as f32)) * 100.0) as i32);

    println!("\n✓ All examples completed successfully!");

    Ok(())
}
