# Eri-Lycoris

A Rust implementation of LyCORIS (Lora beYond Conventional methods, Other Rank adaptation Implementations for Stable diffusion) algorithms, ported from the original [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) Python library by the brilliant [KohakuBlueleaf](https://github.com/KohakuBlueleaf).

## Overview

Eri-Lycoris brings the power of advanced LoRA algorithms to Rust, designed to work seamlessly with the [Flame](https://github.com/tracel-ai/burn) deep learning framework. This library provides efficient, memory-optimized implementations of state-of-the-art parameter-efficient fine-tuning methods.

## Key Strengths

- **High Performance**: Rust's zero-cost abstractions and memory safety provide exceptional performance without sacrificing reliability
- **Advanced Algorithms**: Implements cutting-edge LoRA variants including LoHa (Hadamard Product), LoKr (Kronecker Product), and LoCon (Convolution-aware LoRA)
- **Memory Efficient**: Optimized tensor operations with support for bfloat16 storage to minimize memory footprint
- **Flame Integration**: Built specifically to integrate with the Flame deep learning framework for seamless neural network training
- **Type Safety**: Leverages Rust's type system to catch errors at compile time, ensuring robust implementations

## Implemented Algorithms

- **LoHa**: Hadamard product-based decomposition for efficient parameter adaptation
- **LoKr**: Kronecker product decomposition for structured weight updates
- **LoCon**: Convolution-aware LoRA with specialized handling for convolutional layers

## Features

- BFloat16 and FP32 precision support
- Efficient tensor operations (Hadamard, Kronecker, Tucker decomposition)
- Flexible layout system for optimal memory access patterns
- Comprehensive test suite ensuring accuracy and reliability

## Credits

Original LyCORIS library: [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS)

This Rust port maintains the algorithmic innovations of the original while leveraging Rust's performance and safety guarantees.

## License

See LICENSE file for details.
