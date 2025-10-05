//! LoKr (Kronecker LoRA)
//! Linear ΔW = kron(W1, W2) * scale  → [IN,OUT]
//! Conv   ΔK = kron(W1:[OL,IM], W2_full:[OK,IN,KH,KW]) * scale  → [KH,KW,IC,OC]
//! Public layouts: Linear [IN,OUT], Conv kernel [KH,KW,IC,OC] (NHWC runtime)

use crate::{Error, LycorisModule, Result};
use crate::ops::kronecker::{make_kronecker, make_kronecker_conv_kernel};
use crate::ops::tucker::rebuild_conv_tucker;
use cudarc::driver::CudaDevice;
use flame_core::{DType, Tensor};
use std::sync::Arc;

#[inline]
fn assert_bf16_storage(name: &str, t: &Tensor) -> Result<()> {
    if t.dtype() != DType::BF16 {
        return Err(Error::InvalidOperation(format!("{name} must use BF16 storage")));
    }
    Ok(())
}

#[inline]
fn scale_from(alpha: f32, rank: usize) -> f32 {
    if rank == 0 { 0.0 } else { alpha / rank as f32 }
}

/// LoKr module supports either full or factorized W1/W2. For conv, W2 may be Tucker/factorized.
pub struct LoKrModule {
    // W1 (matrix): either full [OL,IM] or factorized [OL,R]@[R,IM]
    pub w1:  Option<Tensor>,
    pub w1a: Option<Tensor>,
    pub w1b: Option<Tensor>,

    // W2 (conv-style): either full [OK,IN,KH,KW], or factorized/Tucker
    pub w2:  Option<Tensor>,   // [OK,IN,KH,KW]
    pub w2a: Option<Tensor>,   // [OK,R]
    pub w2b: Option<Tensor>,   // [R,IN] or [R,IN,KH,KW]
    pub t2:  Option<Tensor>,   // [KH,KW,R,R]

    pub rank: usize,
    pub alpha: f32,
    pub device: Arc<CudaDevice>,

    /// ((OL, OK), (IM, IN)) preserved for linear-only shape math if needed
    pub shape: ((usize, usize), (usize, usize)),
    /// Marks whether this LoKr instance targets a conv weight
    pub is_conv: bool,
}

impl LoKrModule {
    #[inline]
    pub fn scale(&self) -> f32 { scale_from(self.alpha, self.rank) }

    /// Resolve W1 as a dense matrix [OL,IM]
    fn resolve_w1(&self) -> Result<Tensor> {
        if let Some(ref w) = self.w1 {
            assert_bf16_storage("w1", w)?;
            return Ok(w.clone());
        }
        let wa = self.w1a.as_ref().ok_or_else(|| Error::InvalidOperation("w1a missing".into()))?;
        let wb = self.w1b.as_ref().ok_or_else(|| Error::InvalidOperation("w1b missing".into()))?;
        assert_bf16_storage("w1a", wa)?;
        assert_bf16_storage("w1b", wb)?;
        wa.matmul(wb).map_err(Error::Flame) // [OL,IM]
    }

    /// Resolve W2 to a full conv kernel **in OK/IN/KH/KW order**.
    /// Returns [OK,IN,KH,KW].
    fn resolve_w2_full_ok_in_kh_kw(&self) -> Result<Tensor> {
        if let Some(ref w) = self.w2 {
            assert_bf16_storage("w2", w)?;
            return Ok(w.clone()); // already [OK,IN,KH,KW]
        }

        // Tucker path: t2:[KH,KW,R,R], w2a:[OK,R], w2b:[R,IN]
        if let Some(ref t) = self.t2 {
            let w2a = self.w2a.as_ref().ok_or_else(|| Error::InvalidOperation("w2a missing for Tucker".into()))?;
            let w2b = self.w2b.as_ref().ok_or_else(|| Error::InvalidOperation("w2b missing for Tucker".into()))?;
            assert_bf16_storage("t2", t)?;
            assert_bf16_storage("w2a", w2a)?;
            assert_bf16_storage("w2b", w2b)?;
            let ok  = w2a.dims()[0];
            let r   = w2a.dims()[1];
            let inn = w2b.dims()[1];
            if w2b.dims()[0] != r {
                return Err(Error::InvalidOperation("rank mismatch w2a/w2b".into()));
            }
            // rebuild_conv_tucker expects: t:[KH,KW,R,R], down:[1,1,IN,R], up:[1,1,R,OK]
            let up   = w2a.reshape(&[1,1, r, ok]).map_err(Error::Flame)?;   // [1,1,R,OK]
            let down = w2b.reshape(&[1,1, inn, r]).map_err(Error::Flame)?;  // [1,1,IN,R]
            let k_hw_ic_oc = rebuild_conv_tucker(t, &down, &up)?;           // [KH,KW,IN,OK]
            // Reorder to [OK,IN,KH,KW] for make_kronecker_conv_kernel's expected input
            return k_hw_ic_oc.permute(&[3,2,0,1]).map_err(Error::Flame);
        }

        // Factorized non-Tucker:
        // w2a:[OK,R], w2b:[R,IN] (1×1)  → full:[OK,IN,1,1]
        // w2b:[R,IN,KH,KW] (spatial)    → full:[OK,IN,KH,KW] by contracting R at each (h,w)
        let w2a = self.w2a.as_ref().ok_or_else(|| Error::InvalidOperation("w2a missing".into()))?;
        let w2b = self.w2b.as_ref().ok_or_else(|| Error::InvalidOperation("w2b missing".into()))?;
        assert_bf16_storage("w2a", w2a)?;
        assert_bf16_storage("w2b", w2b)?;
        let da = w2a.dims();
        let db = w2b.dims();
        let ok = da[0];
        let r  = da[1];

        match db.len() {
            2 => {
                let inn = db[1];
                if db[0] != r { return Err(Error::InvalidOperation("rank mismatch w2a/w2b (1x1)".into())); }
                let ok_in = w2a.matmul(w2b).map_err(Error::Flame)?; // [OK,IN]
                ok_in.reshape(&[ok, inn, 1, 1]).map_err(Error::Flame)
            }
            4 => {
                let (rb, inn, kh, kw) = (db[0], db[1], db[2], db[3]);
                if rb != r { return Err(Error::InvalidOperation("rank mismatch w2a/w2b (spatial)".into())); }

                // Reshape w2b: [R,IN,KH,KW] → [R, IN*KH*KW]
                let w2b_flat = w2b.reshape(&[r, inn * kh * kw]).map_err(Error::Flame)?;

                // Contract: [OK,R] @ [R, IN*KH*KW] → [OK, IN*KH*KW]
                let result_flat = w2a.matmul(&w2b_flat).map_err(Error::Flame)?;

                // Reshape to final: [OK, IN, KH, KW]
                result_flat.reshape(&[ok, inn, kh, kw]).map_err(Error::Flame)
            }
            _ => Err(Error::InvalidOperation("unsupported w2b rank; expected [R,IN] or [R,IN,KH,KW]".into())),
        }
    }
}

impl LycorisModule for LoKrModule {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if !self.is_conv {
            // Linear: x[..., IN] @ ΔW[IN,OUT]
            let dw = self.get_diff_weight()?;
            return x.matmul(&dw).map_err(Error::Flame);
        }
        // Conv: NHWC with composed kernel
        let k = self.get_diff_weight()?; // [KH,KW,IC,OC]
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
                let d = w2_full.dims();
                if d.len() == 2 {
                    w2_full.clone()
                } else if d.len() == 4 && d[2] == 1 && d[3] == 1 {
                    // [OK,IN,1,1] → [OK,IN]
                    w2_full.reshape(&[d[0], d[1]]).map_err(Error::Flame)?
                } else {
                    return Err(Error::InvalidOperation("linear LoKr requires 2D w2 (or KH=KW=1)".into()));
                }
            } else if let (Some(ref a), Some(ref b)) = (&self.w2a, &self.w2b) {
                // [OK,R]@[R,IN] → [OK,IN]
                a.matmul(b).map_err(Error::Flame)?
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_zero_rank() {
        assert_eq!(scale_from(1.0, 0), 0.0);
        assert_eq!(scale_from(8.0, 4), 2.0);
    }
}
