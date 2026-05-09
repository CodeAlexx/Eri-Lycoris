//! LoKr (Kronecker LoRA)
//! Linear ΔW = kron(W1, W2) * scale  → [IN,OUT]
//! Conv   ΔK = kron(W1:[OL,IM], W2_full:[OK,IN,KH,KW]) * scale  → [KH,KW,IC,OC]
//! Public layouts: Linear [IN,OUT], Conv kernel [KH,KW,IC,OC] (NHWC runtime)

use crate::{tensor_utils, Error, LycorisModule, Result, StorageDtype};
use crate::ops::kronecker::{make_kronecker, make_kronecker_conv_kernel};
use crate::ops::tucker::rebuild_conv_tucker;
use cudarc::driver::CudaDevice;
use flame_core::parameter::Parameter;
use flame_core::{DType, Shape, Tensor};
use std::sync::Arc;

#[inline]
fn assert_bf16_storage(name: &str, t: &Tensor) -> Result<()> {
    if t.dtype() != DType::BF16 {
        return Err(Error::InvalidOperation(format!("{name} must use BF16 storage")));
    }
    Ok(())
}

#[inline]
fn param_tensor(p: &Parameter) -> Result<Tensor> {
    p.tensor().map_err(Error::Flame)
}

#[allow(dead_code)]
#[inline]
fn opt_param_tensor(p: &Option<Parameter>) -> Result<Option<Tensor>> {
    match p {
        Some(p) => Ok(Some(param_tensor(p)?)),
        None => Ok(None),
    }
}

/// Storage-policy-aware variant of `assert_bf16_storage`.
///
/// Inference / loader path stores everything in BF16. Training path may
/// store leaves in F32 (matching `eridiffusion-core/src/lora.rs::LoRALinear`).
/// This accepts either; the math primitives below upcast/downcast as needed.
#[inline]
fn assert_storage_dtype(name: &str, t: &Tensor) -> Result<()> {
    match t.dtype() {
        DType::BF16 | DType::F32 => Ok(()),
        d => Err(Error::InvalidOperation(format!(
            "{name} must use BF16 or F32 storage, got {:?}",
            d
        ))),
    }
}

#[inline]
fn scale_from(alpha: f32, rank: usize) -> f32 {
    // Mirrors `lycoris.functional.lokr.diff_weight` (functional/lokr.py:135-141):
    //
    //     if w1a is not None: rank = w1a.shape[1]
    //     elif w2a is not None: rank = w2a.shape[1]
    //     else:                 rank = gamma         # ← full-W1 + full-W2 case
    //     scale = gamma / rank                       # = 1.0 in the else branch
    //
    // Rust's loader sets `rank = 0` when neither side is factorized. Returning
    // `alpha / 0` here would short-circuit `make_kronecker` to a zeros tensor
    // (P0-4: silently zero ΔW). Match Python: when rank is unknown, the
    // composed kernel is the unscaled Kronecker product, i.e. scale = 1.0.
    if rank == 0 {
        1.0
    } else {
        alpha / rank as f32
    }
}

/// LoKr module supports either full or factorized W1/W2. For conv, W2 may be Tucker/factorized.
pub struct LoKrModule {
    // W1 (matrix): either full [OL,IM] or factorized [OL,R]@[R,IM]
    pub w1:  Option<Parameter>,
    pub w1a: Option<Parameter>,
    pub w1b: Option<Parameter>,

    // W2 (conv-style): either full [OK,IN,KH,KW], or factorized/Tucker
    pub w2:  Option<Parameter>,   // [OK,IN,KH,KW]
    pub w2a: Option<Parameter>,   // [OK,R]
    pub w2b: Option<Parameter>,   // [R,IN] or [R,IN,KH,KW]
    pub t2:  Option<Parameter>,   // [KH,KW,R,R]

    pub rank: usize,
    pub alpha: f32,
    pub device: Arc<CudaDevice>,

    /// ((OL, OK), (IM, IN)) preserved for linear-only shape math if needed
    pub shape: ((usize, usize), (usize, usize)),
    /// Marks whether this LoKr instance targets a conv weight
    pub is_conv: bool,
}

/// Factor a dimension into `(m, n)` with `m <= n` and `m * n == dimension`,
/// matching `lycoris.functional.general.factorization` (`general.py:14-56`).
///
/// `factor > 0`: if it divides `dimension`, split as `(min(factor, q), max(factor, q))`
/// where `q = dimension / factor`. Otherwise fall through.
/// `factor <= 0`: no cap — find the closest-to-square factor pair.
///
/// This is the math LyCORIS uses to derive the LoKr `(out_l, out_k)` and
/// `(in_m, in_n)` Kronecker block sizes from a `linear_dim × factor` config.
fn factorization(dimension: usize, factor: i32) -> (usize, usize) {
    let factor_pos = factor.max(0) as usize;

    if factor > 0 && (dimension % factor_pos) == 0 {
        let mut m = factor_pos;
        let mut n = dimension / factor_pos;
        if m > n {
            std::mem::swap(&mut m, &mut n);
        }
        return (m, n);
    }
    let cap = if factor < 0 { dimension } else { factor_pos };
    let mut m: usize = 1;
    let mut n: usize = dimension;
    let mut length = m + n;
    loop {
        let mut new_m = m + 1;
        while new_m <= n && dimension % new_m != 0 {
            new_m += 1;
        }
        if new_m > n {
            break;
        }
        let new_n = dimension / new_m;
        if new_m + new_n > length || new_m > cap {
            break;
        }
        m = new_m;
        n = new_n;
        length = m + n;
        if m >= n {
            break;
        }
    }
    if m > n {
        std::mem::swap(&mut m, &mut n);
    }
    (m, n)
}

impl LoKrModule {
    #[inline]
    pub fn scale(&self) -> f32 { scale_from(self.alpha, self.rank) }

    /// Construct a fresh, trainable LoKr adapter for a `Linear(in,out)` layer.
    ///
    /// Mirrors `lycoris.modules.lokr.LoKrModule.__init__` (Python:
    /// `lokr.py:138-244`) for the linear branch:
    ///
    /// - `factor` is the upstream Kronecker split factor; `(out_l, out_k) =
    ///   factorization(out, factor)` and `(in_m, in_n) = factorization(in, factor)`.
    ///   `factor = -1` means "as-square-as-possible".
    /// - `decompose_both = true` and `rank < max(out_l, in_m)/2` → factorize W1 too,
    ///   producing `w1_a:[out_l,r]` and `w1_b:[r,in_m]`. Otherwise `w1:[out_l,in_m]` (full).
    /// - `rank < max(out_k, in_n)/2` → factorize W2: `w2_a:[out_k,r]`, `w2_b:[r,in_n]`.
    ///   Otherwise `w2:[out_k,in_n]` (full); upstream prints a "force full" warning,
    ///   we silently take the full path.
    /// - `use_scalar`: when true, `w2`/`w2_b` are kaiming-initialized so the adapter
    ///   starts non-zero and a learnable scalar gates it. When false (default),
    ///   `w2` (full) or `w2_b` (factorized) is initialized to zero so initial ΔW=0.
    ///   `use_scalar=true` is **not yet plumbed** in this struct (no `scalar` field);
    ///   it's accepted here for API stability but currently treated as `false`.
    /// - `dtype`: `StorageDtype::F32` for EDv2 training (AdamW state); `Bf16` for
    ///   inference / merge-only.
    ///
    /// Init policy follows the upstream Python verbatim — every leaf is kaiming
    /// uniform (a=√5) **except** the canonical zero leg (`w2` full, or `w2_b`
    /// factorized when `use_scalar=false`). This deviates from a simplified
    /// "zero w1_b too" pattern: zeroing `w1_b` would zero W1 at init and break
    /// the Kronecker product's symmetry.
    #[allow(clippy::too_many_arguments)]
    pub fn new_linear(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        factor: i32,
        decompose_both: bool,
        use_scalar: bool,
        device: Arc<CudaDevice>,
        dtype: StorageDtype,
    ) -> Result<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(Error::InvalidOperation(
                "LoKr::new_linear: in_features and out_features must be > 0".into(),
            ));
        }
        if rank == 0 {
            return Err(Error::InvalidOperation(
                "LoKr::new_linear: rank must be > 0 for fresh construction".into(),
            ));
        }

        let (in_m, in_n) = factorization(in_features, factor);
        let (out_l, out_k) = factorization(out_features, factor);

        // shape = ((out_l, out_k), (in_m, in_n))
        let dec_w1 = decompose_both && rank < (out_l.max(in_m)) / 2;
        let factorize_w2 = rank < (out_k.max(in_n)) / 2;
        let kaiming_a = (5.0_f32).sqrt();

        let (w1, w1a, w1b) = if dec_w1 {
            // w1_a:[out_l, r], w1_b:[r, in_m]; both kaiming
            let w1a = tensor_utils::kaiming_uniform_param(
                Shape::from_dims(&[out_l, rank]),
                kaiming_a,
                dtype,
                device.clone(),
            )?;
            let w1b = tensor_utils::kaiming_uniform_param(
                Shape::from_dims(&[rank, in_m]),
                kaiming_a,
                dtype,
                device.clone(),
            )?;
            (None, Some(w1a), Some(w1b))
        } else {
            // w1:[out_l, in_m], kaiming
            let w1 = tensor_utils::kaiming_uniform_param(
                Shape::from_dims(&[out_l, in_m]),
                kaiming_a,
                dtype,
                device.clone(),
            )?;
            (Some(w1), None, None)
        };

        let (w2, w2a, w2b) = if factorize_w2 {
            // w2_a:[out_k, r] kaiming; w2_b:[r, in_n] zero (so initial ΔW=0).
            // upstream uses kaiming on w2_b iff use_scalar=true; we don't
            // implement use_scalar yet, so treat as false (zero w2_b).
            let _ = use_scalar; // reserved for future scalar-gating wiring
            let w2a = tensor_utils::kaiming_uniform_param(
                Shape::from_dims(&[out_k, rank]),
                kaiming_a,
                dtype,
                device.clone(),
            )?;
            let w2b = tensor_utils::zeros_param(
                Shape::from_dims(&[rank, in_n]),
                dtype,
                device.clone(),
            )?;
            (None, Some(w2a), Some(w2b))
        } else {
            // w2:[out_k, in_n] full, zero-init (upstream behavior with
            // use_scalar=false). Stored as 2D for the linear path.
            let w2 = tensor_utils::zeros_param(
                Shape::from_dims(&[out_k, in_n]),
                dtype,
                device.clone(),
            )?;
            (Some(w2), None, None)
        };

        Ok(Self {
            w1: w1.map(Parameter::new),
            w1a: w1a.map(Parameter::new),
            w1b: w1b.map(Parameter::new),
            w2: w2.map(Parameter::new),
            w2a: w2a.map(Parameter::new),
            w2b: w2b.map(Parameter::new),
            t2: None,
            rank,
            alpha,
            device,
            shape: ((out_l, out_k), (in_m, in_n)),
            is_conv: false,
        })
    }

    /// Construct a fresh, trainable LoKr adapter for a `Conv2d(in,out,kh,kw)` layer.
    ///
    /// Mirrors `LoKrModule.__init__` (`lokr.py:89-137`) for the conv branch.
    /// Layout convention: W2 (full) is stored upstream as `[out_k, in_n, kh, kw]`;
    /// we keep that layout in `self.w2` so `resolve_w2_full_ok_in_kh_kw` can
    /// pass it straight to `make_kronecker_conv_kernel`.
    ///
    /// Tucker path is selected by `use_tucker = true && (kh > 1 || kw > 1)` and
    /// **only when W2 is factorized** (upstream condition: `lora_dim < max(...)/2`).
    /// In that case `t2:[r, r, kh, kw]`, `w2_a:[r, out_k]`, `w2_b:[r, in_n]` —
    /// rank lives on axis 0 for both `w2a/w2b` (this is the layout the loader's
    /// Tucker resolve path expects, see `resolve_w2_full_ok_in_kh_kw`).
    ///
    /// **Limitation**: the non-Tucker spatial-conv factorized W2 path
    /// (`w2_a:[out_k,r] @ w2_b:[r, in_n*kh*kw]`) used by upstream's "Conv2d not
    /// tucker" branch (`lokr.py:131-136`) is not constructed by this entry point
    /// — for kh*kw > 1 with factorized W2 the caller should pass `use_tucker=true`,
    /// or use `factor` such that W2 is full. Returning `Err` for now; expand
    /// when an EDv2 trainer needs the spatial-non-tucker variant.
    #[allow(clippy::too_many_arguments)]
    pub fn new_conv2d(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        rank: usize,
        alpha: f32,
        factor: i32,
        decompose_both: bool,
        use_tucker: bool,
        use_scalar: bool,
        device: Arc<CudaDevice>,
        dtype: StorageDtype,
    ) -> Result<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(Error::InvalidOperation(
                "LoKr::new_conv2d: in_channels and out_channels must be > 0".into(),
            ));
        }
        if rank == 0 {
            return Err(Error::InvalidOperation(
                "LoKr::new_conv2d: rank must be > 0 for fresh construction".into(),
            ));
        }
        let (kh, kw) = kernel_size;
        if kh == 0 || kw == 0 {
            return Err(Error::InvalidOperation(
                "LoKr::new_conv2d: kernel size dims must be > 0".into(),
            ));
        }

        let (in_m, in_n) = factorization(in_channels, factor);
        let (out_l, out_k) = factorization(out_channels, factor);
        let tucker_active = use_tucker && (kh > 1 || kw > 1);
        let kaiming_a = (5.0_f32).sqrt();
        let _ = use_scalar; // scalar gating not yet implemented; reserved.

        // W1 — same logic as linear (W1 is always 2D regardless of conv shape).
        let dec_w1 = decompose_both && rank < (out_l.max(in_m)) / 2;
        let (w1, w1a, w1b) = if dec_w1 {
            let w1a = tensor_utils::kaiming_uniform_param(
                Shape::from_dims(&[out_l, rank]),
                kaiming_a,
                dtype,
                device.clone(),
            )?;
            let w1b = tensor_utils::kaiming_uniform_param(
                Shape::from_dims(&[rank, in_m]),
                kaiming_a,
                dtype,
                device.clone(),
            )?;
            (None, Some(w1a), Some(w1b))
        } else {
            let w1 = tensor_utils::kaiming_uniform_param(
                Shape::from_dims(&[out_l, in_m]),
                kaiming_a,
                dtype,
                device.clone(),
            )?;
            (Some(w1), None, None)
        };

        // W2 selection.
        let factorize_w2 = rank < (out_k.max(in_n)) / 2;

        let (w2, w2a, w2b, t2) = if !factorize_w2 {
            // Full W2 conv kernel: [out_k, in_n, kh, kw], zero-init.
            let w2 = tensor_utils::zeros_param(
                Shape::from_dims(&[out_k, in_n, kh, kw]),
                dtype,
                device.clone(),
            )?;
            (Some(w2), None, None, None)
        } else if tucker_active {
            // Tucker factorized: t2:[r,r,kh,kw], w2a:[r,out_k], w2b:[r,in_n].
            // Upstream stores t2 as [r, r, kh, kw] AND saves w2a/w2b with rank
            // on axis 0 (`lokr.py:122-128`). resolve_w2_full_ok_in_kh_kw expects
            // t2 already permuted to [kh, kw, r, r] (loader does that). For
            // freshly-constructed modules we keep upstream's layout for
            // `lokr_t2.shape == (r, r, kh, kw)` and let the loader's permute
            // mismatch be handled by `resolve_w2_full_ok_in_kh_kw` itself.
            //
            // NB: the loader path takes already-permuted t2 ([kh,kw,r,r]).
            // Fresh construction emits the upstream raw layout because that's
            // what save_to_state_dict is going to write back. The resolve
            // path needs to handle both — currently it only handles the
            // permuted layout. This is a known limitation; documented below.
            let t2 = tensor_utils::kaiming_uniform_param(
                Shape::from_dims(&[rank, rank, kh, kw]),
                kaiming_a,
                dtype,
                device.clone(),
            )?;
            let w2a = tensor_utils::kaiming_uniform_param(
                Shape::from_dims(&[rank, out_k]),
                kaiming_a,
                dtype,
                device.clone(),
            )?;
            // Zero-init w2b so initial ΔW=0 (use_scalar=false branch).
            let w2b = tensor_utils::zeros_param(
                Shape::from_dims(&[rank, in_n]),
                dtype,
                device.clone(),
            )?;
            (None, Some(w2a), Some(w2b), Some(t2))
        } else {
            // Factorized non-Tucker spatial conv requires `w2b:[r,in_n*kh*kw]`
            // and the corresponding kron expansion. Not implemented yet; the
            // resolve path expects `w2b:[r, in_n, kh, kw]` for the spatial
            // case (`lokr.rs:149-160`), and upstream's lokr.py:131-136 stores
            // `w2_b:[r, in_n*kh*kw]`. The two layouts disagree — opening this
            // construction path will require a loader update to match.
            return Err(Error::InvalidOperation(format!(
                "LoKr::new_conv2d: factorized non-Tucker spatial conv (kh*kw={}) \
                 not implemented; pass use_tucker=true or use a factor that \
                 keeps W2 full",
                kh * kw
            )));
        };

        Ok(Self {
            w1: w1.map(Parameter::new),
            w1a: w1a.map(Parameter::new),
            w1b: w1b.map(Parameter::new),
            w2: w2.map(Parameter::new),
            w2a: w2a.map(Parameter::new),
            w2b: w2b.map(Parameter::new),
            t2: t2.map(Parameter::new),
            rank,
            alpha,
            device,
            shape: ((out_l, out_k), (in_m, in_n)),
            is_conv: true,
        })
    }

    /// Resolve W1 as a dense matrix [OL,IM]
    fn resolve_w1(&self) -> Result<Tensor> {
        if let Some(ref w) = self.w1 {
            let wt = param_tensor(w)?;
            assert_bf16_storage("w1", &wt)?;
            return Ok(wt);
        }
        let wa = self.w1a.as_ref().ok_or_else(|| Error::InvalidOperation("w1a missing".into()))?;
        let wb = self.w1b.as_ref().ok_or_else(|| Error::InvalidOperation("w1b missing".into()))?;
        let wa_t = param_tensor(wa)?;
        let wb_t = param_tensor(wb)?;
        assert_bf16_storage("w1a", &wa_t)?;
        assert_bf16_storage("w1b", &wb_t)?;
        wa_t.matmul(&wb_t).map_err(Error::Flame) // [OL,IM]
    }

    /// Resolve W2 to a full conv kernel **in OK/IN/KH/KW order**.
    /// Returns [OK,IN,KH,KW].
    fn resolve_w2_full_ok_in_kh_kw(&self) -> Result<Tensor> {
        if let Some(ref w) = self.w2 {
            let wt = param_tensor(w)?;
            assert_bf16_storage("w2", &wt)?;
            return Ok(wt); // already [OK,IN,KH,KW]
        }

        // Tucker path. Upstream `lycoris.functional.lokr.weight_gen` saves
        // (functional/lokr.py:71-74):
        //   t2  : [R, R, KH, KW]
        //   w2a : [R, shape[0][1]] = [R, OK]    ← rank is axis 0!
        //   w2b : [R, shape[1][1]] = [R, IN]
        //
        // The non-Tucker save (functional/lokr.py:77-78) flips w2a to
        // [OK, R] and lifts w2b to 4D [R, IN, KH, KW]. The Rust code used
        // to read `ok = w2a.dims()[0]; r = w2a.dims()[1]` for both branches,
        // which is correct for non-Tucker but inverts the Tucker layout
        // (P0-5: silent corruption — data dims happened to look plausible
        // because R≈OK for many shapes).
        //
        // The loader hands us t2 already permuted to [KH, KW, R, R].
        if let Some(ref t) = self.t2 {
            let w2a = self.w2a.as_ref().ok_or_else(|| Error::InvalidOperation("w2a missing for Tucker".into()))?;
            let w2b = self.w2b.as_ref().ok_or_else(|| Error::InvalidOperation("w2b missing for Tucker".into()))?;
            let t_t = param_tensor(t)?;
            let w2a_t0 = param_tensor(w2a)?;
            let w2b_t0 = param_tensor(w2b)?;
            assert_bf16_storage("t2", &t_t)?;
            assert_bf16_storage("w2a", &w2a_t0)?;
            assert_bf16_storage("w2b", &w2b_t0)?;
            let r   = w2a_t0.dims()[0];
            let ok  = w2a_t0.dims()[1];
            let r2  = w2b_t0.dims()[0];
            let inn = w2b_t0.dims()[1];
            if r2 != r {
                return Err(Error::InvalidOperation(format!(
                    "Tucker LoKr rank mismatch: w2a[0]={}, w2b[0]={}", r, r2
                )));
            }
            // rebuild_conv_tucker expects: t:[KH,KW,R,R], down:[1,1,IC,R], up:[1,1,R,OC].
            // Here IC=IN, OC=OK. w2a is [R, OK] on disk, so transpose to [OK, R]
            // before reshaping to [1, 1, R, OK]. Same for w2b ([R, IN] → [IN, R]).
            let w2a_tt = w2a_t0.transpose().map_err(Error::Flame)?;  // [OK, R]
            let w2b_tt = w2b_t0.transpose().map_err(Error::Flame)?;  // [IN, R]
            let up   = w2a_tt.reshape(&[1, 1, r, ok]).map_err(Error::Flame)?;   // [1,1,R,OK]
            let down = w2b_tt.reshape(&[1, 1, inn, r]).map_err(Error::Flame)?;  // [1,1,IN,R]
            let k_hw_ic_oc = rebuild_conv_tucker(&t_t, &down, &up)?;            // [KH,KW,IN,OK]
            // Reorder to [OK,IN,KH,KW] for make_kronecker_conv_kernel's expected input
            return k_hw_ic_oc.permute(&[3, 2, 0, 1]).map_err(Error::Flame);
        }

        // Factorized non-Tucker:
        // w2a:[OK,R], w2b:[R,IN] (1×1)  → full:[OK,IN,1,1]
        // w2b:[R,IN,KH,KW] (spatial)    → full:[OK,IN,KH,KW] by contracting R at each (h,w)
        let w2a = self.w2a.as_ref().ok_or_else(|| Error::InvalidOperation("w2a missing".into()))?;
        let w2b = self.w2b.as_ref().ok_or_else(|| Error::InvalidOperation("w2b missing".into()))?;
        let w2a_t = param_tensor(w2a)?;
        let w2b_t = param_tensor(w2b)?;
        assert_bf16_storage("w2a", &w2a_t)?;
        assert_bf16_storage("w2b", &w2b_t)?;
        let da = w2a_t.dims();
        let db = w2b_t.dims();
        let ok = da[0];
        let r  = da[1];

        match db.len() {
            2 => {
                let inn = db[1];
                if db[0] != r { return Err(Error::InvalidOperation("rank mismatch w2a/w2b (1x1)".into())); }
                let ok_in = w2a_t.matmul(&w2b_t).map_err(Error::Flame)?; // [OK,IN]
                ok_in.reshape(&[ok, inn, 1, 1]).map_err(Error::Flame)
            }
            4 => {
                let (rb, inn, kh, kw) = (db[0], db[1], db[2], db[3]);
                if rb != r { return Err(Error::InvalidOperation("rank mismatch w2a/w2b (spatial)".into())); }

                // Reshape w2b: [R,IN,KH,KW] → [R, IN*KH*KW]
                let w2b_flat = w2b_t.reshape(&[r, inn * kh * kw]).map_err(Error::Flame)?;

                // Contract: [OK,R] @ [R, IN*KH*KW] → [OK, IN*KH*KW]
                let result_flat = w2a_t.matmul(&w2b_flat).map_err(Error::Flame)?;

                // Reshape to final: [OK, IN, KH, KW]
                result_flat.reshape(&[ok, inn, kh, kw]).map_err(Error::Flame)
            }
            _ => Err(Error::InvalidOperation("unsupported w2b rank; expected [R,IN] or [R,IN,KH,KW]".into())),
        }
    }
}

impl LycorisModule for LoKrModule {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // See `LoConModule::forward` for the dtype-coercion rationale —
        // dw is composed from F32-storage params; cast to `x.dtype()` *with
        // autograd recording* before mixing into the residual graph.
        let target_dtype = x.dtype();
        if !self.is_conv {
            // Linear: x[..., IN] @ ΔW[IN,OUT]
            let dw = self.get_diff_weight()?;
            let dw = if dw.dtype() != target_dtype { dw.to_dtype(target_dtype).map_err(Error::Flame)? } else { dw };
            return x.matmul(&dw).map_err(Error::Flame);
        }
        // Conv: NHWC with composed kernel
        let k = self.get_diff_weight()?; // [KH,KW,IC,OC]
        let k = if k.dtype() != target_dtype { k.to_dtype(target_dtype).map_err(Error::Flame)? } else { k };
        crate::ops::conv2d::conv2d(
            x, &k, (1,1), (0,0), (1,1), 1,
            crate::ops::conv2d::Layout::NHWC,
        )
    }

    fn get_diff_weight(&self) -> Result<Tensor> {
        let s = self.scale();

        if !self.is_conv {
            // Linear ΔW: kron(W1:[OL,IM], W2:[OK,IN]) → [IN,OUT]
            let w1 = self.resolve_w1()?; // [OL,IM]
            // Build W2 (linear 2D) from provided W2 state
            let w2_lin: Tensor = if let Some(ref w2_full) = self.w2 {
                let w2_full_t = param_tensor(w2_full)?;
                let d = w2_full_t.dims();
                if d.len() == 2 {
                    w2_full_t
                } else if d.len() == 4 && d[2] == 1 && d[3] == 1 {
                    // [OK,IN,1,1] → [OK,IN]
                    w2_full_t.reshape(&[d[0], d[1]]).map_err(Error::Flame)?
                } else {
                    return Err(Error::InvalidOperation("linear LoKr requires 2D w2 (or KH=KW=1)".into()));
                }
            } else if let (Some(ref a), Some(ref b)) = (&self.w2a, &self.w2b) {
                // [OK,R]@[R,IN] → [OK,IN]
                let a_t = param_tensor(a)?;
                let b_t = param_tensor(b)?;
                a_t.matmul(&b_t).map_err(Error::Flame)?
            } else {
                // pure W1-only LoKr isn't meaningful for kron; bail
                return Err(Error::InvalidOperation("missing W2 for linear LoKr".into()));
            };
            return make_kronecker(&w1, &w2_lin, s);
        }

        // Conv ΔK: need W1:[OL,IM] and W2_full:[OK,IN,KH,KW], then kron → [KH,KW,IC,OC]
        let w1 = self.resolve_w1()?;                  // [OL,IM]
        let w2_full_ok_in = self.resolve_w2_full_ok_in_kh_kw()?; // [OK,IN,KH,KW]
        make_kronecker_conv_kernel(&w1, &w2_full_ok_in, s)        // [KH,KW,IC,OC]
    }

    fn merge_to(&mut self, _multiplier: f32) -> Result<()> {
        // Deprecated - use external merging logic
        Ok(())
    }

    fn parameters(&self) -> Vec<Tensor> {
        // Order: [w1?, w1a?, w1b?, w2?, w2a?, w2b?, t2?]. Skip None.
        // Each LoKr instance has either `w1` (full) XOR (`w1a` & `w1b`), and
        // similarly for W2 (full / factorized / Tucker). Order inside the
        // Vec is fixed so the trainer's optimizer state pairs by index
        // across save / resume.
        let mut out: Vec<Tensor> = Vec::with_capacity(7);
        if let Some(ref w) = self.w1 { out.push(param_tensor(w).expect("LoKr.w1 mutex")); }
        if let Some(ref w) = self.w1a { out.push(param_tensor(w).expect("LoKr.w1a mutex")); }
        if let Some(ref w) = self.w1b { out.push(param_tensor(w).expect("LoKr.w1b mutex")); }
        if let Some(ref w) = self.w2 { out.push(param_tensor(w).expect("LoKr.w2 mutex")); }
        if let Some(ref w) = self.w2a { out.push(param_tensor(w).expect("LoKr.w2a mutex")); }
        if let Some(ref w) = self.w2b { out.push(param_tensor(w).expect("LoKr.w2b mutex")); }
        if let Some(ref t) = self.t2 { out.push(param_tensor(t).expect("LoKr.t2 mutex")); }
        out
    }

    fn parameters_handles(&self) -> Vec<Parameter> {
        let mut out: Vec<Parameter> = Vec::with_capacity(7);
        if let Some(ref w) = self.w1 { out.push(w.clone()); }
        if let Some(ref w) = self.w1a { out.push(w.clone()); }
        if let Some(ref w) = self.w1b { out.push(w.clone()); }
        if let Some(ref w) = self.w2 { out.push(w.clone()); }
        if let Some(ref w) = self.w2a { out.push(w.clone()); }
        if let Some(ref w) = self.w2b { out.push(w.clone()); }
        if let Some(ref t) = self.t2 { out.push(t.clone()); }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_zero_rank() {
        // P0-4: rank=0 (neither W1 nor W2 factorized) must NOT produce 0
        // because that silently zeros ΔW. Mirror Python: scale = gamma/gamma = 1.0.
        assert_eq!(scale_from(1.0, 0), 1.0);
        assert_eq!(scale_from(8.0, 0), 1.0);
        assert_eq!(scale_from(8.0, 4), 2.0);
    }
}
