# LyCORIS-RS: Rust Port of LyCORIS

High-performance Rust implementation of LoRA algorithms (LoRA/LoCon, LoHa, LoKr) using the Flame tensor library with BF16 storage and FP32 compute.

## Features

- **BF16 Storage + FP32 Compute**: Efficient memory usage with numerical stability
- **GPU-Accelerated**: CUDA kernels for all operations
- **Stream-Aware**: Asynchronous execution support
- **Layout Adapters**: NHWC ↔ NCHW conversion for optimal performance
- **Comprehensive Testing**: Full test suite for correctness and precision

## Algorithms

### LoRA (LoCon)
Standard low-rank adaptation: `ΔW = up @ down * scale`

```rust
use lycoris_rs::algorithms::LoConModule;

let module = LoConModule::new_linear(
    in_features: 512,
    out_features: 512,
    rank: 8,
    alpha: Some(8.0),
    device,
)?;
```

### LoHa (Hadamard Product)
Element-wise product decomposition: `ΔW = (w1u @ w1d) ⊙ (w2u @ w2d) * scale`

```rust
use lycoris_rs::algorithms::LoHaModule;

let module = LoHaModule::new_linear(
    in_features: 512,
    out_features: 512,
    rank: 8,
    alpha: Some(8.0),
    device,
)?;
```

### LoKr (Kronecker Product)
Kronecker decomposition: `ΔW = w1 ⊗ w2 * scale`

```rust
use lycoris_rs::algorithms::LoKrModule;

let module = LoKrModule::new_linear(
    in_features: 512,
    out_features: 512,
    rank: 8,
    alpha: Some(8.0),
    factor: -1,  // auto-factorization
    decompose_both: false,
    device,
)?;
```

## Architecture

```
lycoris-rs/
├── src/
│   ├── algorithms/       # LoRA algorithm implementations
│   │   ├── locon.rs     # Standard LoRA/LoCon
│   │   ├── loha.rs      # Hadamard product LoRA
│   │   └── lokr.rs      # Kronecker product LoRA
│   ├── ops/             # Core tensor operations
│   │   ├── hadamard.rs  # Hadamard product ops
│   │   ├── kronecker.rs # Kronecker product ops
│   │   └── tucker.rs    # Tucker decomposition
│   ├── kernels/         # GPU kernels
│   │   ├── bf16_kernels.rs  # BF16 storage ops
│   │   └── stream_ops.rs    # Stream-aware execution
│   ├── layout.rs        # NHWC ↔ NCHW conversion
│   ├── dtype.rs         # Data type utilities
│   └── error.rs         # Error types
└── tests/               # Integration tests
    ├── bf16_storage_tests.rs  # BF16 preservation
    ├── fp32_accuracy_tests.rs # FP32 compute accuracy
    └── layout_tests.rs        # Layout conversion

```

## Requirements

- CUDA 12.0+
- Flame tensor library with BF16 support
- Rust 1.70+

## Building

```bash
cd lycoris-rs
cargo build --release --features bf16
```

## Testing

```bash
# Run all tests
cargo test --features bf16

# Run specific test suite
cargo test --test bf16_storage_tests
cargo test --test fp32_accuracy_tests
cargo test --test layout_tests
```

## Design Principles

### BF16 Storage with FP32 Compute

All operations follow this pattern:
1. Load BF16 from GPU memory
2. Convert to FP32 in registers
3. Compute in FP32
4. Convert back to BF16
5. Store BF16 to GPU memory

This provides:
- 50% memory savings vs FP32
- Full FP32 numerical accuracy
- No loss in model quality

### Stream-Aware Operations

Operations support CUDA streams for:
- Concurrent kernel execution
- Asynchronous memory transfers
- Pipelined computation

### Layout Adapters

Automatic NHWC ↔ NCHW conversion at boundaries:
- Flame uses NCHW (PyTorch-compatible)
- cuDNN optimizes for NHWC
- Converters handle transformations transparently

## Safety Guarantees

- **No unsafe without wrappers**: All unsafe code wrapped in safe interfaces
- **No CPU execution paths**: GPU-only for consistency
- **No implicit dtype widening**: Explicit conversions required
- **No host copies beyond pinned staging**: Zero-copy where possible

## License

MIT

## Attribution

Based on the Python [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) library by KohakuBlueleaf.
