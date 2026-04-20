# Eri-Lycoris

Rust port of [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) (advanced LoRA-family algorithms for diffusion models) by [KohakuBlueleaf](https://github.com/KohakuBlueleaf), targeting [flame-core](https://github.com/CodeAlexx/Flame) for inference-time weight merging.

The actual crate is at [`lycoris-rs/`](lycoris-rs/) — see its [README](lycoris-rs/README.md) for full status, API, parity results, and usage.

## Status (2026-04-20)

- 4 algorithms shipped: **LoCon, LoHa, LoKr, Full** — Linear + Conv2d + Tucker variants
- Inference-only weight-merge mode (load → materialize ΔW → add into base → run inference)
- Wired into [inference-flame](https://github.com/CodeAlexx/inference-flame) as a path dep with per-model name mappers (FLUX, Z-Image, Chroma, Klein, Qwen-Image, SDXL, SD 1.5)
- `fuse_split_qkv` helper for fused-QKV models — validated against a real Z-Image LoKr (240 split-QKV adapters → 30 fused triples → 180 adapters)
- 31 tests in the crate (lib + smoke + loader + parity vs upstream Python) plus 32 in the inference-flame integration

## Credits

Original LyCORIS: [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS).
This Rust port keeps the math; ships the loader + per-model integration around it.

## License

MIT.
