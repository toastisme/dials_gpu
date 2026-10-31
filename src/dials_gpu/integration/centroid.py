from __future__ import annotations

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

    dev = _get_device(device)
    shoeboxes = reflections["shoebox"]
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
        == "SimplePxMmStrategy"
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
        mm_pos = torch.stack([pos_x * px_x, pos_y * px_y, z_rad], dim=1)
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
