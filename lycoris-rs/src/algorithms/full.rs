//! Full adapter — the trivial case. The on-disk tensors `diff` and (optional)
//! `diff_b` are direct weight/bias deltas.
//!
//! Used for fine-tuned full-rank weight deltas where no decomposition was
//! applied. Inference does:
//!     base.weight ← base.weight + strength * diff
//!     base.bias   ← base.bias   + strength * diff_b   (if diff_b present)
//!
//! Upstream save format: `lycoris/modules/full.py:128-132` (`custom_state_dict`).

use crate::{Error, Result};
use flame_core::Tensor;

pub struct FullAdapter {
    /// Raw weight delta tensor, in whatever shape the base weight uses.
    pub diff: Tensor,
    /// Optional bias delta tensor (1D), shape matches the layer's bias if any.
    /// P0-7: previously `.diff_b` was silently dropped by the loader, so
    /// bias-using layers (linear projections, MLP outs, group norms) lost
    /// the bias delta entirely.
    pub diff_b: Option<Tensor>,
}

impl FullAdapter {
    /// Returns `strength * diff`. Caller adds this to the base weight.
    pub fn delta_weight(&self, strength: f32) -> Result<Tensor> {
        if strength == 1.0 {
            // Avoid an unnecessary scalar mul kernel.
            return Ok(self.diff.clone());
        }
        self.diff.mul_scalar(strength).map_err(Error::Flame)
    }

    /// Returns `Some(strength * diff_b)` when a bias delta is present.
    pub fn delta_bias(&self, strength: f32) -> Result<Option<Tensor>> {
        match &self.diff_b {
            None => Ok(None),
            Some(b) if strength == 1.0 => Ok(Some(b.clone())),
            Some(b) => Ok(Some(b.mul_scalar(strength).map_err(Error::Flame)?)),
        }
    }
}
