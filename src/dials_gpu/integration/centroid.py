from __future__ import annotations

import dxtbx.flumpy as flumpy
import numpy as np
import torch

from dials.array_family import flex
from dials.extensions.simple_centroid_ext import SimpleCentroidExt
from dials_gpu.integration import _shoebox_pack_ext
from dials_gpu.integration.background import (
    MASK_FOREGROUND,
    MASK_OVERLAPPED,
    MASK_VALID,
    _get_device,
)

MASK_FG_VALID = MASK_VALID | MASK_FOREGROUND  # 5


def compute_centroid_gpu(
    reflections: flex.reflection_table, experiments, image_volume=None, device=None
):
    """
    GPU-accelerated centroid calculation on a reflection table using PyTorch.
    Matches `reflections.compute_centroid(experiments)` (`shoebox.centroid_foreground_minus_background()`
    + `reflections.centroid_px_to_mm(experiments)`).
    """
    if image_volume is not None:
        return reflections.compute_centroid(experiments, image_volume=image_volume)

    n_refl = len(reflections)
    if n_refl == 0:
        return

    shoeboxes = reflections["shoebox"]
    if hasattr(_shoebox_pack_ext, "compute_centroid_fast"):
        has_single_panel_px2mm = (
            len(experiments) == 1
            and experiments[0].detector is not None
            and len(experiments[0].detector) == 1
            and experiments[0].detector[0].get_px_mm_strategy().name()
            in ("SimplePxMmStrategy", "ParallaxCorrectedPxMmStrategy")
            and tuple(experiments[0].detector[0].get_raw_image_offset()) == (0, 0)
            and (
                experiments[0].scan is None
                or not experiments[0].scan.has_property("time_of_flight")
            )
        )
        if has_single_panel_px2mm:
            panel = experiments[0].detector[0]
            scan = experiments[0].scan
            px_x, px_y = panel.get_pixel_size()
            if scan is None or not scan.has_property("oscillation"):
                osc0, osc_w, arr0 = 0.0, 0.0, 0.0
            else:
                osc0, osc_w = scan.get_oscillation(deg=False)
                arr0 = float(scan.get_array_range()[0])
            strat = panel.get_px_mm_strategy()
            if strat.name() == "ParallaxCorrectedPxMmStrategy":
                px2mm_mode = 1
                mu = float(strat.mu())
                t0 = float(strat.t0())
            else:
                px2mm_mode = 0
                mu = 0.0
                t0 = 0.0
            px_pos, px_var, mm_pos, mm_var = _shoebox_pack_ext.compute_centroid_fast(
                shoeboxes,
                MASK_FG_VALID,
                MASK_OVERLAPPED,
                True,
                px2mm_mode,
                float(px_x),
                float(px_y),
                mu,
                t0,
                panel.get_fast_axis(),
                panel.get_slow_axis(),
                panel.get_origin(),
                float(osc0),
                float(osc_w),
                float(arr0),
            )
            reflections["xyzobs.px.value"] = px_pos
            reflections["xyzobs.px.variance"] = px_var
            reflections["xyzobs.mm.value"] = mm_pos
            reflections["xyzobs.mm.variance"] = mm_var
            return
        else:
            px_pos, px_var = _shoebox_pack_ext.compute_centroid_fast(
                shoeboxes,
                MASK_FG_VALID,
                MASK_OVERLAPPED,
                False,
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0),
                0.0,
                0.0,
                0.0,
            )
            reflections["xyzobs.px.value"] = px_pos
            reflections["xyzobs.px.variance"] = px_var
            reflections.centroid_px_to_mm(experiments)
            return

    dev = _get_device(device)
    bbox_np = np.column_stack([p.as_numpy_array() for p in reflections["bbox"].parts()])

    fg_counts = np.empty(n_refl, dtype=np.int64)
    flat_flags = np.empty(n_refl, dtype=np.uint8)
    moments_np = np.empty((n_refl, 8), dtype=np.float64)

    _shoebox_pack_ext.extract_centroid_moments(
        shoeboxes,
        MASK_FG_VALID,
        MASK_OVERLAPPED,
        fg_counts.ctypes.data,
        flat_flags.ctypes.data,
        moments_np.ctypes.data,
    )

    bbox_t = torch.as_tensor(bbox_np, dtype=torch.float64, device=dev)
    flat_t = torch.as_tensor(flat_flags != 0, dtype=torch.bool, device=dev)
    counts = torch.as_tensor(fg_counts, dtype=torch.long, device=dev)
    mom_t = torch.as_tensor(moments_np, dtype=torch.float64, device=dev)

    xmid = (bbox_t[:, 1] + bbox_t[:, 0]) * 0.5
    ymid = (bbox_t[:, 3] + bbox_t[:, 2]) * 0.5
    zmid = (bbox_t[:, 5] + bbox_t[:, 4]) * 0.5

    sum_w = mom_t[:, 0]
    sum_wsq = mom_t[:, 1]
    mean_x = mom_t[:, 2]
    mean_y = mom_t[:, 3]
    mean_z = mom_t[:, 4]
    sum_wdx2 = mom_t[:, 5]
    sum_wdy2 = mom_t[:, 6]
    sum_wdz2 = mom_t[:, 7]

    has_points = (counts > 0) & (sum_w > 0)

    # Offset calculation: zoff = (bbox[5] + bbox[4]) // 2 if flat else bbox[4]
    zoff = torch.where(
        flat_t,
        torch.div(bbox_t[:, 5] + bbox_t[:, 4], 2, rounding_mode="trunc"),
        bbox_t[:, 4],
    )
    pos_x = torch.where(has_points, mean_x + bbox_t[:, 0], xmid)
    pos_y = torch.where(has_points, mean_y + bbox_t[:, 2], ymid)
    pos_z = torch.where(has_points, mean_z + zoff, zmid)

    # Variance calculation matching CentroidPoints (dials/algorithms/image/centroid/centroid_points.h):
    denom = (sum_w * sum_w) - sum_wsq
    valid_var = (counts > 1) & (denom > 0.0)
    scale = 1.0 / denom.clamp(min=1e-30)
    half_sq = 1.0 / 12.0

    var_x = torch.where(valid_var, sum_wdx2 * scale + half_sq, torch.zeros_like(sum_w))
    var_y = torch.where(valid_var, sum_wdy2 * scale + half_sq, torch.zeros_like(sum_w))
    var_z = torch.where(valid_var, sum_wdz2 * scale + half_sq, torch.zeros_like(sum_w))

    pos_3 = torch.stack([pos_x, pos_y, pos_z], dim=1)
    var_3 = torch.stack([var_x, var_y, var_z], dim=1)

    reflections["xyzobs.px.value"] = flex.vec3_double(
        np.ascontiguousarray(pos_3.cpu().numpy(), dtype=np.float64)
    )
    reflections["xyzobs.px.variance"] = flex.vec3_double(
        np.ascontiguousarray(var_3.cpu().numpy(), dtype=np.float64)
    )

    if (
        len(experiments) == 1
        and experiments[0].detector is not None
        and len(experiments[0].detector) == 1
        and experiments[0].detector[0].get_px_mm_strategy().name()
        in ("SimplePxMmStrategy", "ParallaxCorrectedPxMmStrategy")
        and tuple(experiments[0].detector[0].get_raw_image_offset()) == (0, 0)
        and (
            experiments[0].scan is None
            or not experiments[0].scan.has_property("time_of_flight")
        )
    ):
        panel = experiments[0].detector[0]
        scan = experiments[0].scan
        px_x, px_y = panel.get_pixel_size()
        if scan is None or not scan.has_property("oscillation"):
            osc0, osc_w = 0.0, 0.0
            arr0 = 0.0
            z_rad = torch.zeros_like(pos_z)
        else:
            osc0, osc_w = scan.get_oscillation(deg=False)
            arr0 = float(scan.get_array_range()[0])
            z_rad = osc0 + (pos_z - arr0) * osc_w
        xy_0 = pos_x * px_x
        xy_1 = pos_y * px_y
        strat = panel.get_px_mm_strategy()
        if strat.name() == "ParallaxCorrectedPxMmStrategy":
            mu = float(strat.mu())
            t0 = float(strat.t0())
            f0, f1, f2 = panel.get_fast_axis()
            s0, s1, s2 = panel.get_slow_axis()
            o0, o1, o2 = panel.get_origin()
            n0 = f1 * s2 - f2 * s1
            n1 = f2 * s0 - f0 * s2
            n2 = f0 * s1 - f1 * s0
            if o0 * n0 + o1 * n1 + o2 * n2 < 0.0:
                n0, n1, n2 = -n0, -n1, -n2
            s1_x = o0 + xy_0 * f0 + xy_1 * s0
            s1_y = o1 + xy_0 * f1 + xy_1 * s1
            s1_z = o2 + xy_0 * f2 + xy_1 * s2
            inv_len = 1.0 / torch.sqrt(s1_x * s1_x + s1_y * s1_y + s1_z * s1_z)
            s1_x = s1_x * inv_len
            s1_y = s1_y * inv_len
            s1_z = s1_z * inv_len
            cos_t = s1_x * n0 + s1_y * n1 + s1_z * n2
            inv_mu = 1.0 / mu
            t0_over_cos_t = t0 / cos_t
            atten_o = inv_mu - (t0_over_cos_t + inv_mu) * torch.exp(
                -mu * t0_over_cos_t
            )
            s1_dot_f = s1_x * f0 + s1_y * f1 + s1_z * f2
            s1_dot_s = s1_x * s0 + s1_y * s1 + s1_z * s2
            xy_0 = xy_0 - s1_dot_f * atten_o
            xy_1 = xy_1 - s1_dot_s * atten_o
        mm_pos = torch.stack([xy_0, xy_1, z_rad], dim=1)
        mm_var = torch.stack(
            [
                var_x * (px_x * px_x),
                var_y * (px_y * px_y),
                var_z * (osc_w * osc_w),
            ],
            dim=1,
        )
        reflections["xyzobs.mm.value"] = flex.vec3_double(
            np.ascontiguousarray(mm_pos.cpu().numpy(), dtype=np.float64)
        )
        reflections["xyzobs.mm.variance"] = flex.vec3_double(
            np.ascontiguousarray(mm_var.cpu().numpy(), dtype=np.float64)
        )
    else:
        reflections.centroid_px_to_mm(experiments)


class SimpleCentroidGPU:
    """
    GPU-accelerated SimpleCentroid extension matching `dials.extensions.simple_centroid_ext.SimpleCentroidExt`.
    """

    name = "simple"

    @staticmethod
    def phil():
        return SimpleCentroidExt.phil()

    def __init__(self, params, experiments, device=None):
        self.params = params
        self.experiments = experiments
        self.device = _get_device(device)

    def compute_centroid(
        self, reflections: flex.reflection_table, image_volume=None
    ):
        compute_centroid_gpu(
            reflections,
            self.experiments,
            image_volume=image_volume,
            device=self.device,
        )
