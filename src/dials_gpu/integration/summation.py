from __future__ import annotations

import numpy as np
import torch

from dials.array_family import flex
from dials_gpu.integration import _shoebox_pack_ext
from dials_gpu.integration.background import (
    MASK_BACKGROUND,
    MASK_BACKGROUND_USED,
    MASK_FOREGROUND,
    MASK_OVERLAPPED,
    MASK_VALID,
    _get_device,
)

MASK_BG_USED_VALID = MASK_VALID | MASK_BACKGROUND | MASK_BACKGROUND_USED  # 19
MASK_FG_VALID = MASK_VALID | MASK_FOREGROUND  # 5


def compute_summed_intensity_gpu(
    reflections: flex.reflection_table, image_volume=None, device=None
):
    """
    GPU-accelerated summation integration on a reflection table using PyTorch.
    Matches `reflections.compute_summed_intensity()` (`dials/algorithms/integration/sum/summation.h`).
    """
    if image_volume is not None:
        return reflections.compute_summed_intensity(image_volume=image_volume)

    n_refl = len(reflections)
    if n_refl == 0:
        return

    shoeboxes = reflections["shoebox"]
    if hasattr(_shoebox_pack_ext, "compute_summation_fast"):
        if "flags" not in reflections:
            reflections["flags"] = flex.size_t(n_refl, 0)
        i_val, i_var, b_val, b_var = _shoebox_pack_ext.compute_summation_fast(
            shoeboxes,
            reflections["flags"],
            MASK_FOREGROUND,
            MASK_FG_VALID,
            MASK_OVERLAPPED,
            MASK_BG_USED_VALID,
        )
        reflections["intensity.sum.value"] = i_val
        reflections["intensity.sum.variance"] = i_var
        reflections["background.sum.value"] = b_val
        reflections["background.sum.variance"] = b_var
        return

    dev = _get_device(device)

    sum_p_np = np.empty(n_refl, dtype=np.float64)
    sum_b_np = np.empty(n_refl, dtype=np.float64)
    n_sig_np = np.empty(n_refl, dtype=np.float64)
    n_bg_np = np.empty(n_refl, dtype=np.float64)
    bad_fg_np = np.empty(n_refl, dtype=np.int32)

    _shoebox_pack_ext.extract_summation_totals(
        shoeboxes,
        MASK_FOREGROUND,
        MASK_FG_VALID,
        MASK_OVERLAPPED,
        MASK_BG_USED_VALID,
        sum_p_np.ctypes.data,
        sum_b_np.ctypes.data,
        n_sig_np.ctypes.data,
        n_bg_np.ctypes.data,
        bad_fg_np.ctypes.data,
    )

    sum_p = torch.as_tensor(sum_p_np, dtype=torch.float64, device=dev)
    sum_b = torch.as_tensor(sum_b_np, dtype=torch.float64, device=dev)
    n_signal = torch.as_tensor(n_sig_np, dtype=torch.float64, device=dev)
    n_bg = torch.as_tensor(n_bg_np, dtype=torch.float64, device=dev)
    bad_fg_count = torch.as_tensor(bad_fg_np, dtype=torch.int32, device=dev)

    success_t = bad_fg_count == 0
    m_ratio = torch.where(
        n_bg > 0, n_signal / n_bg.clamp(min=1.0), torch.zeros_like(n_signal)
    )

    intensity_val = sum_p - sum_b
    bg_var = torch.abs(sum_b) * (1.0 + m_ratio)
    intensity_var = torch.abs(intensity_val) + bg_var
    bg_val = sum_b

    reflections["intensity.sum.value"] = flex.double(
        np.ascontiguousarray(intensity_val.cpu().numpy(), dtype=np.float64)
    )
    reflections["intensity.sum.variance"] = flex.double(
        np.ascontiguousarray(intensity_var.cpu().numpy(), dtype=np.float64)
    )
    reflections["background.sum.value"] = flex.double(
        np.ascontiguousarray(bg_val.cpu().numpy(), dtype=np.float64)
    )
    reflections["background.sum.variance"] = flex.double(
        np.ascontiguousarray(bg_var.cpu().numpy(), dtype=np.float64)
    )

    success_flex = flex.bool(
        np.ascontiguousarray(success_t.cpu().numpy(), dtype=bool)
    )
    fail_flex = ~success_flex
    reflections.set_flags(success_flex, reflections.flags.integrated_sum)
    reflections.set_flags(fail_flex, reflections.flags.failed_during_summation)


class SummationGPU:
    """GPU-accelerated summation integration algorithm."""

    def __init__(self, device=None):
        self.device = _get_device(device)

    def __call__(self, reflections: flex.reflection_table, image_volume=None):
        compute_summed_intensity_gpu(
            reflections, image_volume=image_volume, device=self.device
        )
