//! Autograd smoke test for the four LyCORIS adapter variants.
//!
//! P0 milestone for LyCORIS in EDv2: prove that the LoKr / LoCon / LoHa /
//! Full math primitives can record autograd correctly. This is the gate —
//! every downstream LyCORIS trainer work depends on this.
//!
//! Strategy: construct each adapter with leaf weights flipped to
//! `requires_grad_(true)`, run forward (or `get_diff_weight` for Full),
//! sum to a scalar loss, run backward, and assert that the "up"-side
//! tensor's gradient is non-zero with a finite max-abs > 1e-9.
//!
//! Rationale (init choices per algo):
//!  - LoCon: down=randn, up=zeros. d_up = scale * (x @ down)^T @ d_out;
//!           non-zero because x and down are non-zero.
//!  - LoHa : branch-1 fully random, branch-2 has w2a=randn / w2b=zeros.
//!           d_w2b = w2a^T @ (d_diff ⊙ w1) is non-zero because w1 ≠ 0.
//!  - LoKr : same idea — w1a, w1b, w2a are randn; w2b is zeros.
//!           d_w2 = sum over A axes of (d_kron · w1); w1 ≠ 0, so d_w2 ≠ 0,
//!           which propagates back to d_w2b.
//!  - Full : trivial — out = strength * diff, d_diff = strength * d_out.
//!
//! Run after qwen finishes:
//!   LD_LIBRARY_PATH=/opt/libtorch-cu121/libtorch/lib:/home/alex/.local/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH \
//!     cargo test --release -p lycoris-rs --test autograd_smoke -- --nocapture

use std::sync::Arc;

use cudarc::driver::CudaDevice;
use flame_core::{DType, Shape, Tensor};

use lycoris_rs::algorithms::{
    full::FullAdapter,
    locon::LoConModule,
    loha::LoHaModule,
    lokr::LoKrModule,
};
use lycoris_rs::LycorisModule;
use lycoris_rs::tensor_utils;

const PASS_THRESHOLD: f32 = 1.0e-9;

fn try_device() -> Option<Arc<CudaDevice>> {
    CudaDevice::new(0).ok()
}

/// Cast an existing Tensor to BF16 storage and flip `requires_grad=true`.
fn bf16_grad(t: Tensor) -> Tensor {
    let bf = t.to_dtype(DType::BF16).expect("to_dtype BF16");
    bf.requires_grad_(true)
}

/// Make a randn BF16 leaf with `requires_grad=true`.
fn leaf_randn_bf16(
    shape: &[usize],
    std: f32,
    device: Arc<CudaDevice>,
) -> Tensor {
    let f32 = Tensor::randn(Shape::from_dims(shape), 0.0, std, device).expect("randn");
    bf16_grad(f32)
}

/// Make a zeros BF16 leaf with `requires_grad=true`.
fn leaf_zeros_bf16(shape: &[usize], device: Arc<CudaDevice>) -> Tensor {
    let z = tensor_utils::zeros_bf16(Shape::from_dims(shape), device).expect("zeros_bf16");
    z.requires_grad_(true)
}

/// Compute max(|grad|) on a 1-D / N-D tensor by reading it back to host F32.
fn max_abs_host(t: &Tensor) -> f32 {
    let host = t.to_dtype(DType::F32).expect("to F32 for host read")
        .to_vec().expect("to_vec");
    host.into_iter().fold(0.0f32, |acc, v| acc.max(v.abs()))
}

#[test]
fn locon_linear_autograd_records_lora_b_grad() {
    let Some(dev) = try_device() else { eprintln!("[locon_linear] no CUDA — skipped"); return; };

    const IN: usize = 64;
    const OUT: usize = 64;
    const RANK: usize = 4;
    const ALPHA: f32 = 4.0;

    // Build the module with the standard constructor (no requires_grad), then
    // overwrite the leaves with grad-enabled tensors. The LycorisModule trait's
    // forward path only reads &self.down / &self.up, so this is safe.
    let mut m = LoConModule::new_linear(IN, OUT, RANK, Some(ALPHA), dev.clone())
        .expect("LoConModule::new_linear");
    m.down = leaf_randn_bf16(&[IN, RANK], 0.1, dev.clone());
    m.up   = leaf_zeros_bf16(&[RANK, OUT], dev.clone());

    // Input
    let x = leaf_randn_bf16(&[2, IN], 1.0, dev.clone());

    // Forward + scalar loss
    let out = m.forward(&x).expect("LoCon forward");
    let loss = out.to_dtype(DType::F32).expect("to F32").sum().expect("sum");
    let grads = flame_core::autograd::backward(&loss, false).expect("backward");

    let g_up   = grads.get(m.up.id())  .expect("missing grad for up");
    let g_down = grads.get(m.down.id()).expect("missing grad for down");
    let mu = max_abs_host(g_up);
    let md = max_abs_host(g_down);
    println!("[LoCon-linear] grad_up max_abs={:e}, grad_down max_abs={:e}", mu, md);
    assert!(mu.is_finite() && mu > PASS_THRESHOLD,
        "LoCon-linear: lora_B (up) grad is zero / non-finite — autograd not recording");
    // down grad with up=0 propagates only through h ⇒ d_h = scale * d_out @ up.T
    // which is zero. So we DO NOT assert on d_down for LoCon — it is structurally
    // zero at the canonical LoRA init, and that's correct PyTorch behavior too.
    println!("[LoCon-linear] PASS (down grad expected zero with up=zeros init)");
}

#[test]
fn loha_linear_autograd_records_w2b_grad() {
    let Some(dev) = try_device() else { eprintln!("[loha_linear] no CUDA — skipped"); return; };

    const IN: usize = 64;
    const OUT: usize = 64;
    const RANK: usize = 4;
    const ALPHA: f32 = 4.0;

    let mut m = LoHaModule::new_linear(IN, OUT, RANK, Some(ALPHA), dev.clone())
        .expect("LoHaModule::new_linear");
    // Branch 1: fully nonzero so w1 = w1a @ w1b ≠ 0.
    m.w1a = leaf_randn_bf16(&[IN, RANK],  1.0, dev.clone());
    m.w1b = leaf_randn_bf16(&[RANK, OUT], 0.1, dev.clone());
    // Branch 2: w2a random, w2b zeros — checking that grad flows to w2b.
    m.w2a = leaf_randn_bf16(&[IN, RANK],  1.0, dev.clone());
    m.w2b = leaf_zeros_bf16(&[RANK, OUT], dev.clone());

    let x = leaf_randn_bf16(&[2, IN], 1.0, dev.clone());

    let out = m.forward(&x).expect("LoHa forward");
    let loss = out.to_dtype(DType::F32).expect("to F32").sum().expect("sum");
    let grads = flame_core::autograd::backward(&loss, false).expect("backward");

    let g_w2b = grads.get(m.w2b.id()).expect("missing grad for w2b");
    let g_w1b = grads.get(m.w1b.id()).expect("missing grad for w1b");
    let mw2b = max_abs_host(g_w2b);
    let mw1b = max_abs_host(g_w1b);
    println!("[LoHa-linear] grad_w2b max_abs={:e}, grad_w1b max_abs={:e}", mw2b, mw1b);
    assert!(mw2b.is_finite() && mw2b > PASS_THRESHOLD,
        "LoHa-linear: w2b grad is zero / non-finite — autograd not recording");
    assert!(mw1b.is_finite() && mw1b > PASS_THRESHOLD,
        "LoHa-linear: w1b grad is zero / non-finite — autograd not recording");
    println!("[LoHa-linear] PASS");
}

#[test]
fn lokr_linear_autograd_records_w2b_grad() {
    let Some(dev) = try_device() else { eprintln!("[lokr_linear] no CUDA — skipped"); return; };

    // Linear shape (IN, OUT) = (OL*OK, IM*IN) = (8, 16).
    // ΔW = kron(W1:[OL,IM], W2:[OK,IN]) → [OL*OK, IM*IN]
    const OL: usize = 4;
    const IM: usize = 4;
    const OK: usize = 2;
    const INN: usize = 4;
    const RANK: usize = 2;
    const ALPHA: f32 = 2.0;

    let in_total  = IM * INN;       // 16
    let out_total = OL * OK;        //  8
    // NOTE: LoKr ΔW shape after make_kronecker is [OL*OK, IM*IN] = [out_total, in_total],
    // and forward is `x.matmul(ΔW)`. So x must have last dim = OL*OK = 8.

    // Factorized W1: w1a [OL, R], w1b [R, IM]  → W1 [OL, IM]
    let w1a = leaf_randn_bf16(&[OL, RANK], 1.0, dev.clone());
    let w1b = leaf_randn_bf16(&[RANK, IM], 0.5, dev.clone());
    // Factorized W2: w2a [OK, R], w2b [R, IN]  → W2 [OK, IN]
    let w2a = leaf_randn_bf16(&[OK, RANK], 1.0, dev.clone());
    let w2b = leaf_zeros_bf16(&[RANK, INN], dev.clone());

    let m = LoKrModule {
        w1: None,
        w1a: Some(w1a),
        w1b: Some(w1b),
        w2: None,
        w2a: Some(w2a),
        w2b: Some(w2b),
        t2: None,
        rank: RANK,
        alpha: ALPHA,
        device: dev.clone(),
        shape: ((OL, OK), (IM, INN)),
        is_conv: false,
    };

    // x: [batch, OL*OK] so x.matmul(ΔW[OL*OK, IM*IN]) = [batch, IM*IN].
    let x = leaf_randn_bf16(&[2, out_total], 1.0, dev.clone());

    let out = m.forward(&x).expect("LoKr forward");
    let loss = out.to_dtype(DType::F32).expect("to F32").sum().expect("sum");
    let grads = flame_core::autograd::backward(&loss, false).expect("backward");

    // We want the grad on the W2 "up" side (w2b — all-zero leaf).
    let w2b_id = m.w2b.as_ref().unwrap().id();
    let w1b_id = m.w1b.as_ref().unwrap().id();
    let g_w2b = grads.get(w2b_id).expect("missing grad for w2b");
    let g_w1b = grads.get(w1b_id).expect("missing grad for w1b");
    let mw2b = max_abs_host(g_w2b);
    let mw1b = max_abs_host(g_w1b);
    println!("[LoKr-linear] grad_w2b max_abs={:e}, grad_w1b max_abs={:e}", mw2b, mw1b);
    assert!(mw2b.is_finite() && mw2b > PASS_THRESHOLD,
        "LoKr-linear: w2b grad is zero / non-finite — autograd not recording");
    assert!(mw1b.is_finite() && mw1b > PASS_THRESHOLD,
        "LoKr-linear: w1b grad is zero / non-finite — autograd not recording");
    println!("[LoKr-linear] PASS");
}

#[test]
fn full_autograd_records_diff_grad() {
    let Some(dev) = try_device() else { eprintln!("[full] no CUDA — skipped"); return; };

    // Full adapter is the trivial case: out = base + strength * diff.
    // Construct directly with a grad-enabled diff leaf.
    const IN: usize = 64;
    const OUT: usize = 64;
    const STRENGTH: f32 = 1.5;

    let diff = leaf_randn_bf16(&[IN, OUT], 0.1, dev.clone());
    let f = FullAdapter { diff: diff.clone(), diff_b: None };

    // delta_weight returns strength * diff; sum it as a scalar loss.
    let delta = f.delta_weight(STRENGTH).expect("delta_weight");
    let loss = delta.to_dtype(DType::F32).expect("to F32").sum().expect("sum");
    let grads = flame_core::autograd::backward(&loss, false).expect("backward");

    let g_diff = grads.get(diff.id()).expect("missing grad for diff");
    let mg = max_abs_host(g_diff);
    println!("[Full] grad_diff max_abs={:e}", mg);
    assert!(mg.is_finite() && mg > PASS_THRESHOLD,
        "Full: diff grad is zero / non-finite — autograd not recording");
    println!("[Full] PASS");
}
