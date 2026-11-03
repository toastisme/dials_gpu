"""
GPU-accelerated scaling model components, Gaussian smoothers (1D/2D/3D),
RefinerCalculatorGPU, and crystal-frame / spherical harmonic utilities
using PyTorch.
"""

from __future__ import annotations

import math
from math import acos

import numpy as np
import torch
from dxtbx import flumpy
from scitbx import sparse

from dials.algorithms.scaling.basis_functions import RefinerCalculator
from dials.algorithms.scaling.model.components.analytical_component import (
    AnalyticalComponent,
)
from dials.algorithms.scaling.model.components.scale_components import (
    LinearDoseDecay,
    QuadraticDoseDecay,
    ScaleComponentBase,
    SHScaleComponent,
    SingleBScaleFactor,
    SingleScaleFactor,
)
from dials.algorithms.scaling.model.components.smooth_scale_components import (
    GaussianSmoother1D,
    GaussianSmoother2D,
    GaussianSmoother3D,
    SmoothBScaleComponent1D,
    SmoothScaleComponent1D,
    SmoothScaleComponent2D,
    SmoothScaleComponent3D,
)
from dials.array_family import flex
from dials_scaling_ext import row_multiply


def _get_device(device=None) -> torch.device:
    if device is not None:
        d = torch.device(device)
        if d.type == "cuda" and d.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return d
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


_MIN_CUDA_REFL = 20000


def _get_minimization_device(n_refl: int, device=None) -> torch.device:
    """Select optimal PyTorch device for iterative minimization based on reflection count."""
    if device is not None:
        return _get_device(device)
    if n_refl >= _MIN_CUDA_REFL and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _boost_iround(x: torch.Tensor) -> torch.Tensor:
    """Replicate boost::math::iround (round half away from zero)."""
    return torch.where(
        x >= 0.0,
        torch.floor(x + 0.5),
        torch.ceil(x - 0.5),
    ).to(torch.int64)


def _idx_range_gpu(
    z: torch.Tensor, nvalues: int, half_naverage: float, naverage: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized replica of GaussianSmoother::idx_range on GPU without host sync stalls."""
    if nvalues <= 3:
        i1 = torch.zeros_like(z, dtype=torch.int64)
        i2 = torch.full_like(z, nvalues, dtype=torch.int64)
        return i1, i2

    i1 = _boost_iround(z - half_naverage) + 1
    i2 = i1 + naverage

    neg_mask = i1 < 0
    i1 = torch.where(neg_mask, torch.zeros_like(i1), i1)
    i2 = torch.where(neg_mask, torch.clamp(i2, min=2), i2)

    over_mask = i2 > nvalues
    i2 = torch.where(over_mask, torch.full_like(i2, nvalues), i2)
    i1 = torch.where(over_mask, torch.clamp(i1, max=nvalues - 2), i1)

    return i1, i2


def _get_smoother_1d_weights_gpu(
    smoother: GaussianSmoother1D,
    x_gpu: torch.Tensor,
    first_fixed: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cached (weight, inv_sw, dv_dp) for 1D GaussianSmoother on x_gpu."""
    device = x_gpu.device
    n_refl = x_gpu.shape[0]
    nvalues = int(smoother.num_values())

    cache = getattr(smoother, "_gpu_weight_cache", None)
    key = (int(x_gpu.data_ptr()), n_refl, nvalues, first_fixed, device)
    if cache is not None and cache[0] == key and cache[1] is x_gpu:
        return cache[2], cache[3], cache[4]

    positions = getattr(smoother, "_gpu_positions", None)
    if positions is None or positions.device != device or positions.shape[0] != nvalues:
        pos_list = list(smoother.positions())
        positions = torch.tensor(pos_list, dtype=torch.float64, device=device)
        smoother._gpu_positions = positions
        smoother._gpu_spacing = float(smoother.spacing())
        smoother._gpu_sigma = float(smoother.sigma())
        smoother._gpu_naverage = int(smoother.num_average())
        smoother._gpu_half_naverage = float(smoother.num_average()) / 2.0
        smoother._gpu_x0 = float(getattr(smoother, "_x0", 0.0))

    spacing = smoother._gpu_spacing
    sigma = smoother._gpu_sigma
    naverage = smoother._gpu_naverage
    half_naverage = smoother._gpu_half_naverage
    x0 = smoother._gpu_x0

    z = (x_gpu - x0) / spacing
    i1, i2 = _idx_range_gpu(z, nvalues, half_naverage, naverage)

    idx = torch.arange(nvalues, dtype=torch.int64, device=device)[None, :]
    mask = (idx >= i1[:, None]) & (idx < i2[:, None])
    ds = (z[:, None] - positions[None, :]) / sigma
    if device.type == "cpu":
        ds_np = ds.numpy()
        exp_t = torch.from_numpy(np.exp(-(ds_np * ds_np)))
    else:
        exp_t = torch.exp(-(ds * ds))
    weight = torch.where(mask, exp_t, torch.zeros_like(ds))
    sumweight = torch.zeros_like(z)
    for k in range(nvalues):
        sumweight = sumweight + weight[:, k]
    safe_sw = torch.where(sumweight > 0.0, sumweight, torch.ones_like(sumweight))
    inv_sw = torch.where(sumweight > 0.0, 1.0 / safe_sw, torch.zeros_like(sumweight))

    if first_fixed:
        dv_dp = weight[:, 1:] * inv_sw[:, None]
    else:
        dv_dp = weight * inv_sw[:, None]

    smoother._gpu_weight_cache = (key, x_gpu, weight, safe_sw, dv_dp)
    return weight, safe_sw, dv_dp


def eval_smoother_1d_gpu(
    smoother: GaussianSmoother1D,
    x_gpu: torch.Tensor,
    values_gpu: torch.Tensor,
    first_fixed: bool = False,
    compute_derivatives: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate 1D GaussianSmoother on GPU, returning (value, dv_dp)."""
    device = x_gpu.device
    n_refl = x_gpu.shape[0]
    nvalues = int(smoother.num_values())
    if n_refl == 0:
        n_cols = (nvalues - 1) if first_fixed else nvalues
        return (
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty((0, n_cols), dtype=torch.float64, device=device)
            if compute_derivatives
            else None,
        )

    weight, safe_sw, dv_dp = _get_smoother_1d_weights_gpu(smoother, x_gpu, first_fixed)
    sumwv = torch.zeros_like(safe_sw)
    for k in range(nvalues):
        sumwv = sumwv + weight[:, k] * values_gpu[k]
    value = torch.where(safe_sw > 0.0, sumwv / safe_sw, torch.zeros_like(safe_sw))

    if not compute_derivatives:
        return value, None
    return value, dv_dp


def _get_smoother_2d_weights_gpu(
    smoother: GaussianSmoother2D,
    x_gpu: torch.Tensor,
    y_gpu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cached (weight, inv_sw, dv_dp) for 2D GaussianSmoother on (x_gpu, y_gpu)."""
    device = x_gpu.device
    n_refl = x_gpu.shape[0]
    nxvalues = int(smoother.num_x_values())
    nyvalues = int(smoother.num_y_values())
    n_total = nxvalues * nyvalues

    cache = getattr(smoother, "_gpu_weight_cache", None)
    key = (
        int(x_gpu.data_ptr()),
        int(y_gpu.data_ptr()),
        n_refl,
        nxvalues,
        nyvalues,
        device,
    )
    if (
        cache is not None
        and cache[0] == key
        and cache[1] is x_gpu
        and cache[2] is y_gpu
    ):
        return cache[3], cache[4], cache[5]

    x_pos = getattr(smoother, "_gpu_x_positions", None)
    if (
        x_pos is None
        or x_pos.device != device
        or x_pos.shape[0] != nxvalues
        or smoother._gpu_y_positions.shape[0] != nyvalues
    ):
        x_pos = torch.tensor(
            list(smoother.x_positions()), dtype=torch.float64, device=device
        )
        y_pos = torch.tensor(
            list(smoother.y_positions()), dtype=torch.float64, device=device
        )
        smoother._gpu_x_positions = x_pos
        smoother._gpu_y_positions = y_pos
        smoother._gpu_x_spacing = float(smoother.x_spacing())
        smoother._gpu_y_spacing = float(smoother.y_spacing())
        smoother._gpu_sigma = float(smoother.sigma())
        smoother._gpu_nx_avg = int(smoother.num_x_average())
        smoother._gpu_ny_avg = int(smoother.num_y_average())
        smoother._gpu_half_nx_avg = float(smoother._gpu_nx_avg) / 2.0
        smoother._gpu_half_ny_avg = float(smoother._gpu_ny_avg) / 2.0
        smoother._gpu_x0 = float(getattr(smoother, "_x0", 0.0))
        smoother._gpu_y0 = float(getattr(smoother, "_y0", 0.0))
    else:
        y_pos = smoother._gpu_y_positions

    z1 = (x_gpu - smoother._gpu_x0) / smoother._gpu_x_spacing
    z2 = (y_gpu - smoother._gpu_y0) / smoother._gpu_y_spacing

    i1, i2 = _idx_range_gpu(
        z1, nxvalues, smoother._gpu_half_nx_avg, smoother._gpu_nx_avg
    )
    j1, j2 = _idx_range_gpu(
        z2, nyvalues, smoother._gpu_half_ny_avg, smoother._gpu_ny_avg
    )

    ix = torch.arange(nxvalues, dtype=torch.int64, device=device)[None, :]
    iy = torch.arange(nyvalues, dtype=torch.int64, device=device)[None, :]
    mask_x = (ix >= i1[:, None]) & (ix < i2[:, None])
    mask_y = (iy >= j1[:, None]) & (iy < j2[:, None])

    dx2 = (z1[:, None] - x_pos[None, :]) ** 2
    dy2 = (z2[:, None] - y_pos[None, :]) ** 2
    inv_sigma2 = 1.0 / (smoother._gpu_sigma * smoother._gpu_sigma)

    # Index order in C++ is idx = i + j * nxvalues (j is outer, i is inner)
    mask_2d = (mask_y[:, :, None] & mask_x[:, None, :]).reshape(n_refl, n_total)
    ds2 = (dy2[:, :, None] + dx2[:, None, :]).reshape(n_refl, n_total) * inv_sigma2
    weight = torch.where(mask_2d, torch.exp(-ds2), torch.zeros_like(ds2))

    sumweight = weight.sum(dim=1)
    safe_sw = torch.where(sumweight > 0.0, sumweight, torch.ones_like(sumweight))
    inv_sw = torch.where(sumweight > 0.0, 1.0 / safe_sw, torch.zeros_like(sumweight))
    dv_dp = weight * inv_sw[:, None]

    smoother._gpu_weight_cache = (key, x_gpu, y_gpu, weight, inv_sw, dv_dp)
    return weight, inv_sw, dv_dp


def eval_smoother_2d_gpu(
    smoother: GaussianSmoother2D,
    x_gpu: torch.Tensor,
    y_gpu: torch.Tensor,
    values_gpu: torch.Tensor,
    compute_derivatives: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate 2D GaussianSmoother on GPU, returning (value, dv_dp)."""
    device = x_gpu.device
    n_refl = x_gpu.shape[0]
    nxvalues = int(smoother.num_x_values())
    nyvalues = int(smoother.num_y_values())
    n_total = nxvalues * nyvalues
    if n_refl == 0:
        return (
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty((0, n_total), dtype=torch.float64, device=device)
            if compute_derivatives
            else None,
        )

    weight, inv_sw, dv_dp = _get_smoother_2d_weights_gpu(smoother, x_gpu, y_gpu)
    sumwv = torch.mv(weight, values_gpu[:n_total])
    value = sumwv * inv_sw

    if not compute_derivatives:
        return value, None
    return value, dv_dp


def _get_smoother_3d_weights_gpu(
    smoother: GaussianSmoother3D,
    x_gpu: torch.Tensor,
    y_gpu: torch.Tensor,
    z_gpu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cached (weight, inv_sw, dv_dp) for 3D GaussianSmoother on (x_gpu, y_gpu, z_gpu)."""
    device = x_gpu.device
    n_refl = x_gpu.shape[0]
    nxvalues = int(smoother.num_x_values())
    nyvalues = int(smoother.num_y_values())
    nzvalues = int(smoother.num_z_values())
    n_total = nxvalues * nyvalues * nzvalues

    cache = getattr(smoother, "_gpu_weight_cache", None)
    key = (
        int(x_gpu.data_ptr()),
        int(y_gpu.data_ptr()),
        int(z_gpu.data_ptr()),
        n_refl,
        nxvalues,
        nyvalues,
        nzvalues,
        device,
    )
    if (
        cache is not None
        and cache[0] == key
        and cache[1] is x_gpu
        and cache[2] is y_gpu
        and cache[3] is z_gpu
    ):
        return cache[4], cache[5], cache[6]

    x_pos = getattr(smoother, "_gpu_x_positions", None)
    if (
        x_pos is None
        or x_pos.device != device
        or x_pos.shape[0] != nxvalues
        or smoother._gpu_y_positions.shape[0] != nyvalues
        or smoother._gpu_z_positions.shape[0] != nzvalues
    ):
        x_pos = torch.tensor(
            list(smoother.x_positions()), dtype=torch.float64, device=device
        )
        y_pos = torch.tensor(
            list(smoother.y_positions()), dtype=torch.float64, device=device
        )
        z_pos = torch.tensor(
            list(smoother.z_positions()), dtype=torch.float64, device=device
        )
        smoother._gpu_x_positions = x_pos
        smoother._gpu_y_positions = y_pos
        smoother._gpu_z_positions = z_pos
        smoother._gpu_x_spacing = float(smoother.x_spacing())
        smoother._gpu_y_spacing = float(smoother.y_spacing())
        smoother._gpu_z_spacing = float(smoother.z_spacing())
        smoother._gpu_sigma = float(smoother.sigma())
        smoother._gpu_nx_avg = int(smoother.num_x_average())
        smoother._gpu_ny_avg = int(smoother.num_y_average())
        smoother._gpu_nz_avg = int(smoother.num_z_average())
        smoother._gpu_half_nx_avg = float(smoother._gpu_nx_avg) / 2.0
        smoother._gpu_half_ny_avg = float(smoother._gpu_ny_avg) / 2.0
        smoother._gpu_half_nz_avg = float(smoother._gpu_nz_avg) / 2.0
        smoother._gpu_x0 = float(getattr(smoother, "_x0", 0.0))
        smoother._gpu_y0 = float(getattr(smoother, "_y0", 0.0))
        smoother._gpu_z0 = float(getattr(smoother, "_z0", 0.0))
    else:
        y_pos = smoother._gpu_y_positions
        z_pos = smoother._gpu_z_positions

    z1 = (x_gpu - smoother._gpu_x0) / smoother._gpu_x_spacing
    z2 = (y_gpu - smoother._gpu_y0) / smoother._gpu_y_spacing
    z3 = (z_gpu - smoother._gpu_z0) / smoother._gpu_z_spacing

    i1, i2 = _idx_range_gpu(
        z1, nxvalues, smoother._gpu_half_nx_avg, smoother._gpu_nx_avg
    )
    j1, j2 = _idx_range_gpu(
        z2, nyvalues, smoother._gpu_half_ny_avg, smoother._gpu_ny_avg
    )
    k1, k2 = _idx_range_gpu(
        z3, nzvalues, smoother._gpu_half_nz_avg, smoother._gpu_nz_avg
    )

    ix = torch.arange(nxvalues, dtype=torch.int64, device=device)[None, :]
    iy = torch.arange(nyvalues, dtype=torch.int64, device=device)[None, :]
    iz = torch.arange(nzvalues, dtype=torch.int64, device=device)[None, :]
    mask_x = (ix >= i1[:, None]) & (ix < i2[:, None])
    mask_y = (iy >= j1[:, None]) & (iy < j2[:, None])
    mask_z = (iz >= k1[:, None]) & (iz < k2[:, None])

    dx2 = (z1[:, None] - x_pos[None, :]) ** 2
    dy2 = (z2[:, None] - y_pos[None, :]) ** 2
    dz2 = (z3[:, None] - z_pos[None, :]) ** 2
    inv_sigma2 = 1.0 / (smoother._gpu_sigma * smoother._gpu_sigma)

    # Index order in C++ is idx = i + j * nxvalues + k * nxvalues * nyvalues
    mask_3d = (
        mask_z[:, :, None, None] & mask_y[:, None, :, None] & mask_x[:, None, None, :]
    ).reshape(n_refl, n_total)
    ds2 = (
        (dz2[:, :, None, None] + dy2[:, None, :, None] + dx2[:, None, None, :]).reshape(
            n_refl, n_total
        )
        * inv_sigma2
    )
    weight = torch.where(mask_3d, torch.exp(-ds2), torch.zeros_like(ds2))

    sumweight = weight.sum(dim=1)
    safe_sw = torch.where(sumweight > 0.0, sumweight, torch.ones_like(sumweight))
    inv_sw = torch.where(sumweight > 0.0, 1.0 / safe_sw, torch.zeros_like(sumweight))
    dv_dp = weight * inv_sw[:, None]

    smoother._gpu_weight_cache = (key, x_gpu, y_gpu, z_gpu, weight, inv_sw, dv_dp)
    return weight, inv_sw, dv_dp


def eval_smoother_3d_gpu(
    smoother: GaussianSmoother3D,
    x_gpu: torch.Tensor,
    y_gpu: torch.Tensor,
    z_gpu: torch.Tensor,
    values_gpu: torch.Tensor,
    compute_derivatives: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate 3D GaussianSmoother on GPU, returning (value, dv_dp)."""
    device = x_gpu.device
    n_refl = x_gpu.shape[0]
    nxvalues = int(smoother.num_x_values())
    nyvalues = int(smoother.num_y_values())
    nzvalues = int(smoother.num_z_values())
    n_total = nxvalues * nyvalues * nzvalues
    if n_refl == 0:
        return (
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty((0, n_total), dtype=torch.float64, device=device)
            if compute_derivatives
            else None,
        )

    weight, inv_sw, dv_dp = _get_smoother_3d_weights_gpu(
        smoother, x_gpu, y_gpu, z_gpu
    )
    sumwv = torch.mv(weight, values_gpu[:n_total])
    value = sumwv * inv_sw

    if not compute_derivatives:
        return value, None
    return value, dv_dp


class GaussianSmoother1DGPU(GaussianSmoother1D):
    """1D Gaussian smoother with GPU tensor evaluation support."""

    def __init__(self, x_range, num_intervals):
        super().__init__(x_range, num_intervals)
        self._x0 = float(x_range[0])


class GaussianSmoother2DGPU(GaussianSmoother2D):
    """2D Gaussian smoother with GPU tensor evaluation support."""

    def __init__(self, x_range, num_x_intervals, y_range, num_y_intervals):
        super().__init__(x_range, num_x_intervals, y_range, num_y_intervals)
        self._x0 = float(x_range[0])
        self._y0 = float(y_range[0])


class GaussianSmoother3DGPU(GaussianSmoother3D):
    """3D Gaussian smoother with GPU tensor evaluation support."""

    def __init__(
        self,
        x_range,
        num_x_intervals,
        y_range,
        num_y_intervals,
        z_range,
        num_z_intervals,
    ):
        super().__init__(
            x_range,
            num_x_intervals,
            y_range,
            num_y_intervals,
            z_range,
            num_z_intervals,
        )
        self._x0 = float(x_range[0])
        self._y0 = float(y_range[0])
        self._z0 = float(z_range[0])


def _get_cached_block_tensor(
    component, attr_name: str, block_id: int, flex_list, device: torch.device
) -> torch.Tensor:
    """Fetch or create a cached GPU tensor for a component's per-block reflection data."""
    cache_attr = f"_gpu_cache_{attr_name}"
    cache = getattr(component, cache_attr, None)
    flex_arr = flex_list[block_id]
    if (
        cache is None
        or len(cache) != len(flex_list)
        or cache[block_id] is None
        or cache[block_id][0] is not flex_arr
        or cache[block_id][1].device != device
    ):
        if cache is None or len(cache) != len(flex_list):
            cache = [None] * len(flex_list)
            setattr(component, cache_attr, cache)
        np_arr = np.asarray(flumpy.to_numpy(flex_arr), dtype=np.float64)
        t = torch.from_numpy(np_arr).to(device=device, dtype=torch.float64)
        cache[block_id] = (flex_arr, t)
    return cache[block_id][1]


def _get_param_tensor(component, flex_params, device: torch.device) -> torch.Tensor:
    """Return a 1D float64 tensor for `flex_params` on `device`, reusing zero-copy CPU views when possible."""
    if device.type == "cpu":
        cache = getattr(component, "_gpu_cpu_param_cache", None)
        if cache is not None and cache[0] is flex_params:
            return cache[1]
        t = torch.from_numpy(flumpy.to_numpy(flex_params))
        component._gpu_cpu_param_cache = (flex_params, t)
        return t
    return torch.from_numpy(flumpy.to_numpy(flex_params)).to(
        device=device, dtype=torch.float64
    )


def eval_sph_harm_dense_gpu(
    s0c_flex,
    s1c_flex,
    lmax: int,
    quantize: bool = False,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Evaluate real spherical harmonics (N_refl, n_params) on GPU/CPU matching
    C++ `create_sph_harm_table` (when quantize=False) or `create_sph_harm_lookup_table`
    with points_per_degree=2 (when quantize=True) to ~1e-15."""
    s0_np = flumpy.to_numpy(s0c_flex).reshape(-1, 3)
    s1_np = flumpy.to_numpy(s1c_flex).reshape(-1, 3)
    n_refl = s0_np.shape[0]
    n_params = int(lmax * (lmax + 2))
    dev = _get_minimization_device(n_refl, device)
    if n_refl == 0 or n_params == 0:
        return torch.empty((n_refl, n_params), dtype=torch.float64, device=dev)

    s0_xyz = torch.from_numpy(np.ascontiguousarray(s0_np, dtype=np.float64)).to(
        device=dev
    )
    s1_xyz = torch.from_numpy(np.ascontiguousarray(s1_np, dtype=np.float64)).to(
        device=dev
    )

    def _get_tp(xyz: torch.Tensor):
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        theta = torch.atan2(torch.sqrt(x * x + y * y), z)
        phi = torch.fmod(torch.atan2(y, x) + (2.0 * math.pi), 2.0 * math.pi)
        if quantize:
            idx = 720 * torch.floor(theta * (360.0 / math.pi)).to(
                torch.int64
            ) + torch.floor(phi * (360.0 / math.pi)).to(torch.int64)
            theta = torch.floor(idx.to(torch.float64) / 720.0) * (math.pi / 360.0)
            phi = (idx % 720).to(torch.float64) * (math.pi / 360.0)
        return theta, phi

    t0, p0 = _get_tp(s0_xyz)
    t1, p1 = _get_tp(s1_xyz)
    x0, x1 = torch.cos(t0), torch.cos(t1)
    s_0 = torch.sqrt(torch.clamp((1.0 - x0) * (1.0 + x0), min=0.0))
    s_1 = torch.sqrt(torch.clamp((1.0 - x1) * (1.0 + x1), min=0.0))

    P0 = {}
    P1 = {}
    for m in range(lmax + 1):
        if m == 0:
            pmm0 = torch.ones_like(x0)
            pmm1 = torch.ones_like(x1)
        else:
            fact = 2.0 * m - 1.0
            pmm0 = -P0[(m - 1, m - 1)] * fact * s_0
            pmm1 = -P1[(m - 1, m - 1)] * fact * s_1
        P0[(m, m)] = pmm0
        P1[(m, m)] = pmm1
        if m + 1 <= lmax:
            P0[(m + 1, m)] = x0 * (2 * m + 1) * pmm0
            P1[(m + 1, m)] = x1 * (2 * m + 1) * pmm1
        for l in range(m + 2, lmax + 1):
            P0[(l, m)] = (
                x0 * (2 * l - 1) * P0[(l - 1, m)] - (l + m - 1) * P0[(l - 2, m)]
            ) / (l - m)
            P1[(l, m)] = (
                x1 * (2 * l - 1) * P1[(l - 1, m)] - (l + m - 1) * P1[(l - 2, m)]
            ) / (l - m)

    sqrt2 = 1.414213562
    cols = []
    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            norm = math.sqrt(
                (2.0 * l + 1.0)
                / (4.0 * math.pi)
                * math.factorial(l - am)
                / math.factorial(l + am)
            )
            plm0 = norm * P0[(l, am)]
            plm1 = norm * P1[(l, am)]
            if m < 0:
                pref = sqrt2 * ((-1.0) ** m) / 2.0
                val = pref * (plm0 * torch.sin(am * p0) + plm1 * torch.sin(am * p1))
            elif m == 0:
                val = 0.5 * (plm0 + plm1)
            else:
                pref = sqrt2 * ((-1.0) ** m) / 2.0
                val = pref * (plm0 * torch.cos(am * p0) + plm1 * torch.cos(am * p1))
            cols.append(val)
    return torch.stack(cols, dim=1)


class SHScaleComponentGPU(SHScaleComponent):
    """GPU-accelerated SHScaleComponent storing a dense (N_refl, n_params) harmonic
    array/tensor directly instead of scitbx.sparse.matrix or lookup tables."""

    def update_reflection_data(self, selection=None, block_selections=None):
        if getattr(self, "_mode", None) != "gpu_dense":
            return super().update_reflection_data(selection, block_selections)

        full_t = getattr(self, "_gpu_full_sh_tensor", None)
        full_np = self._gpu_full_sh_np
        dev = full_t.device if full_t is not None else torch.device("cpu")

        if selection is not None:
            sel_np = flumpy.to_numpy(selection)
            if full_t is not None and dev.type == "cuda":
                sel_t = torch.from_numpy(np.ascontiguousarray(sel_np)).to(device=dev)
                b_t = full_t[sel_t]
                self._gpu_block_sh_tensors = [b_t]
                self._n_refl = [int(b_t.shape[0])]
            else:
                b_np = full_np[sel_np]
                self._gpu_block_sh_tensors = [torch.from_numpy(b_np)]
                self._n_refl = [int(b_np.shape[0])]
        elif block_selections is not None:
            # Check if single block selection is 0..N-1 (as in expand_scales_to_all_reflections)
            if (
                len(block_selections) == 1
                and len(block_selections[0]) == full_np.shape[0]
            ):
                sel0_np = flumpy.to_numpy(block_selections[0])
                if sel0_np.size == 0 or (
                    sel0_np[0] == 0
                    and sel0_np[-1] == full_np.shape[0] - 1
                    and np.all(sel0_np == np.arange(full_np.shape[0], dtype=sel0_np.dtype))
                ):
                    self._gpu_block_sh_tensors = [
                        full_t if full_t is not None else torch.from_numpy(full_np)
                    ]
                    self._n_refl = [int(full_np.shape[0])]
                    self._gpu_sh_cache = None
                    return

            blocks_t = []
            n_refls = []
            for sel in block_selections:
                sel_np = flumpy.to_numpy(sel)
                if full_t is not None and dev.type == "cuda":
                    sel_t = torch.from_numpy(
                        np.ascontiguousarray(sel_np, dtype=np.int64)
                    ).to(device=dev)
                    b_t = full_t[sel_t]
                else:
                    b_np = full_np[sel_np]
                    b_t = torch.from_numpy(np.ascontiguousarray(b_np, dtype=np.float64))
                blocks_t.append(b_t)
                n_refls.append(int(b_t.shape[0]))
            self._gpu_block_sh_tensors = blocks_t
            self._n_refl = n_refls
        else:
            self._gpu_block_sh_tensors = [
                full_t if full_t is not None else torch.from_numpy(full_np)
            ]
            self._n_refl = [int(full_np.shape[0])]
        self._gpu_sh_cache = None

    def calculate_scales(self, block_id=0):
        if getattr(self, "_mode", None) != "gpu_dense":
            return super().calculate_scales(block_id)
        sh_t = self._gpu_block_sh_tensors[block_id]
        params_t = _get_param_tensor(self, self._parameters, sh_t.device)
        scales_t = 1.0 + torch.mv(sh_t, params_t)
        scales_np = np.ascontiguousarray(scales_t.cpu().numpy(), dtype=np.float64)
        return flumpy.from_numpy(scales_np)

    def calculate_scales_and_derivatives(self, block_id=0):
        if getattr(self, "_mode", None) != "gpu_dense":
            return super().calculate_scales_and_derivatives(block_id)
        sh_t = self._gpu_block_sh_tensors[block_id]
        params_t = _get_param_tensor(self, self._parameters, sh_t.device)
        scales_t = 1.0 + torch.mv(sh_t, params_t)
        scales_np = np.ascontiguousarray(scales_t.cpu().numpy(), dtype=np.float64)
        sh_np = np.ascontiguousarray(sh_t.cpu().numpy(), dtype=np.float64)
        n_rows, n_cols = sh_np.shape
        d_mat = sparse.matrix(n_rows, n_cols)
        if n_rows > 0 and n_cols > 0:
            b = flumpy.from_numpy(sh_np)
            b.reshape(flex.grid(n_rows, n_cols))
            d_mat.assign_block(b, 0, 0)
        return flumpy.from_numpy(scales_np), d_mat


def configure_absorption_component_gpu(model, reflection_table) -> None:
    """Configure `model.components['absorption']` with `SHScaleComponentGPU` and
    dense GPU/CPU spherical harmonic matrix."""
    lmax = int(model.configdict["lmax"])
    comp = model.components["absorption"]
    if not isinstance(comp, SHScaleComponentGPU):
        comp.__class__ = SHScaleComponentGPU

    quantize = reflection_table.size() > 100000
    sh_t = eval_sph_harm_dense_gpu(
        reflection_table["s0c"], reflection_table["s1c"], lmax, quantize=quantize
    )
    if sh_t.shape[1] != comp.n_params:
        sh_t = sh_t[:, : comp.n_params]
    comp._mode = "gpu_dense"
    comp._gpu_full_sh_tensor = sh_t
    comp._gpu_full_sh_np = (
        sh_t.numpy()
        if sh_t.device.type == "cpu"
        else np.ascontiguousarray(sh_t.cpu().numpy(), dtype=np.float64)
    )
    comp._gpu_block_sh_tensors = [sh_t]
    comp._n_refl = [int(sh_t.shape[0])]

    surface_weight = model.configdict["abs_surface_weight"]
    parameter_restraints = flex.double([])
    for i in range(1, lmax + 1):
        parameter_restraints.extend(flex.double([1.0] * ((2 * i) + 1)))
    parameter_restraints *= surface_weight
    comp.parameter_restraints = parameter_restraints


def _get_cached_sh_matrix_gpu(
    component: SHScaleComponent, block_id: int, device: torch.device
) -> torch.Tensor:
    """Return cached (N_refl, N_params) spherical harmonic matrix on GPU for block_id."""
    if getattr(component, "_mode", None) == "gpu_dense":
        b_t = component._gpu_block_sh_tensors[block_id]
        if b_t.device != device:
            b_t = b_t.to(device=device, dtype=torch.float64)
            component._gpu_block_sh_tensors[block_id] = b_t
        return b_t

    cache = getattr(component, "_gpu_sh_cache", None)
    src = component._harmonic_values[block_id]

    n_blocks = len(component._harmonic_values)
    if (
        cache is None
        or len(cache) != n_blocks
        or cache[block_id] is None
        or cache[block_id][0] is not src
        or cache[block_id][1].device != device
    ):
        if cache is None or len(cache) != n_blocks:
            cache = [None] * n_blocks
            component._gpu_sh_cache = cache

        if component._mode == "memory":
            # component._harmonic_values[block_id] is a list of flex.double arrays (one per param)
            cols = [
                np.asarray(flumpy.to_numpy(col_arr), dtype=np.float64)
                for col_arr in src
            ]
            if cols:
                mat_np = np.column_stack(cols)
            else:
                mat_np = np.empty((0, component.n_params), dtype=np.float64)
        else:
            # speed mode: src is a scitbx.sparse.matrix of shape (n_refl, n_params)
            dense_flex = src.as_dense_matrix()
            mat_np = np.asarray(flumpy.to_numpy(dense_flex), dtype=np.float64).reshape(
                src.n_rows, src.n_cols
            )

        t = torch.from_numpy(mat_np).to(device=device, dtype=torch.float64)
        cache[block_id] = (src, t)
    return cache[block_id][1]


def _get_cached_inv_2d2_tensor(
    component, block_id: int, flex_list, device: torch.device
) -> torch.Tensor:
    """Return cached 1 / (2 * d^2) tensor for a component's per-block d_values."""
    cache = getattr(component, "_gpu_cache_inv_2d2", None)
    flex_arr = flex_list[block_id]
    if (
        cache is None
        or len(cache) != len(flex_list)
        or cache[block_id] is None
        or cache[block_id][0] is not flex_arr
        or cache[block_id][1].device != device
    ):
        if cache is None or len(cache) != len(flex_list):
            cache = [None] * len(flex_list)
            component._gpu_cache_inv_2d2 = cache
        d_gpu = _get_cached_block_tensor(component, "d_val", block_id, flex_list, device)
        inv_2d2 = 1.0 / (2.0 * (d_gpu * d_gpu))
        cache[block_id] = (flex_arr, inv_2d2)
    return cache[block_id][1]


def evaluate_component_gpu(
    component: ScaleComponentBase,
    block_id: int = 0,
    compute_derivatives: bool = True,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate a DIALS scale component on the GPU, returning (scales_gpu, derivatives_gpu).

    scales_gpu has shape (N_refl,) and derivatives_gpu has shape (N_refl, N_free_params)
    with dtype=torch.float64 on `device`.
    """
    n_refl = int(component.n_refl[block_id])
    dev = _get_minimization_device(n_refl, device)

    # Fast paths for each component type (check subclasses before base classes!)
    if isinstance(component, SmoothBScaleComponent1D):
        params_gpu = _get_param_tensor(component, component._parameters, dev)
        x_gpu = _get_cached_block_tensor(
            component, "norm_x", block_id, component._normalised_values, dev
        )
        prefac = _get_cached_inv_2d2_tensor(
            component, block_id, component._d_values, dev
        )
        b_val, db_dp = eval_smoother_1d_gpu(
            component._smoother,
            x_gpu,
            params_gpu,
            first_fixed=bool(component._fixed_initial),
            compute_derivatives=compute_derivatives,
        )
        if dev.type == "cpu":
            scales = torch.from_numpy(np.exp((b_val * prefac).numpy()))
        else:
            scales = torch.exp(b_val * prefac)
        if not compute_derivatives:
            return scales, None
        derivs = db_dp * (scales * prefac)[:, None]
        return scales, derivs

    if type(component) is SmoothScaleComponent1D or (
        isinstance(component, SmoothScaleComponent1D)
        and not isinstance(component, SmoothBScaleComponent1D)
    ):
        params_gpu = _get_param_tensor(component, component._parameters, dev)
        x_gpu = _get_cached_block_tensor(
            component, "norm_x", block_id, component._normalised_values, dev
        )
        return eval_smoother_1d_gpu(
            component._smoother,
            x_gpu,
            params_gpu,
            first_fixed=bool(component._fixed_initial),
            compute_derivatives=compute_derivatives,
        )

    if isinstance(component, SmoothScaleComponent2D):
        params_gpu = _get_param_tensor(component, component._parameters, dev)
        x_gpu = _get_cached_block_tensor(
            component, "norm_x", block_id, component._normalised_x_values, dev
        )
        y_gpu = _get_cached_block_tensor(
            component, "norm_y", block_id, component._normalised_y_values, dev
        )
        return eval_smoother_2d_gpu(
            component._smoother,
            x_gpu,
            y_gpu,
            params_gpu,
            compute_derivatives=compute_derivatives,
        )

    if isinstance(component, SmoothScaleComponent3D):
        params_gpu = _get_param_tensor(component, component._parameters, dev)
        x_gpu = _get_cached_block_tensor(
            component, "norm_x", block_id, component._normalised_x_values, dev
        )
        y_gpu = _get_cached_block_tensor(
            component, "norm_y", block_id, component._normalised_y_values, dev
        )
        z_gpu = _get_cached_block_tensor(
            component, "norm_z", block_id, component._normalised_z_values, dev
        )
        return eval_smoother_3d_gpu(
            component._smoother,
            x_gpu,
            y_gpu,
            z_gpu,
            params_gpu,
            compute_derivatives=compute_derivatives,
        )

    if isinstance(component, SHScaleComponent):
        sh_mat = _get_cached_sh_matrix_gpu(component, block_id, dev)
        params_gpu = _get_param_tensor(component, component._parameters, dev)
        scales = 1.0 + torch.mv(sh_mat, params_gpu)
        if not compute_derivatives:
            return scales, None
        return scales, sh_mat

    if type(component) is SingleScaleFactor:
        k_val = float(component.parameters[0])
        scales = torch.full((n_refl,), k_val, dtype=torch.float64, device=dev)
        if not compute_derivatives:
            return scales, None
        derivs = torch.ones((n_refl, 1), dtype=torch.float64, device=dev)
        return scales, derivs

    if type(component) is SingleBScaleFactor:
        b_val = float(component.parameters[0])
        inv_2d2 = _get_cached_inv_2d2_tensor(
            component, block_id, component._d_values, dev
        )
        scales = torch.exp(b_val * inv_2d2)
        if not compute_derivatives:
            return scales, None
        derivs = (scales * inv_2d2)[:, None]
        return scales, derivs

    if type(component) is QuadraticDoseDecay:
        c_val = float(component.parameters[0])
        d_gpu = _get_cached_block_tensor(
            component, "d_val", block_id, component._d_values, dev
        )
        x_gpu = _get_cached_block_tensor(
            component, "x_val", block_id, component._x, dev
        )
        factor = x_gpu / (d_gpu * d_gpu)
        scales = torch.exp(c_val * factor)
        if not compute_derivatives:
            return scales, None
        derivs = (scales * factor)[:, None]
        return scales, derivs

    if type(component) is LinearDoseDecay:
        c_val = float(component.parameters[0])
        d_gpu = _get_cached_block_tensor(
            component, "d_val", block_id, component._d_values, dev
        )
        x_gpu = _get_cached_block_tensor(
            component, "x_val", block_id, component._x, dev
        )
        factor = x_gpu / d_gpu
        scales = torch.exp(c_val * factor)
        if not compute_derivatives:
            return scales, None
        derivs = (scales * factor)[:, None]
        return scales, derivs

    if isinstance(component, AnalyticalComponent):
        scales = _get_cached_block_tensor(
            component, "corr", block_id, component._analytical_corrections, dev
        )
        if not compute_derivatives:
            return scales, None
        derivs = torch.empty((n_refl, 0), dtype=torch.float64, device=dev)
        return scales, derivs

    # Generic fallback for custom/test components
    if compute_derivatives:
        s_flex, d_sparse = component.calculate_scales_and_derivatives(block_id)
        s_np = np.asarray(flumpy.to_numpy(s_flex), dtype=np.float64)
        s_gpu = torch.from_numpy(s_np).to(device=dev, dtype=torch.float64)
        if d_sparse.n_rows == 0 or d_sparse.n_cols == 0:
            d_gpu = torch.empty(
                (d_sparse.n_rows, d_sparse.n_cols), dtype=torch.float64, device=dev
            )
        else:
            d_np = np.asarray(
                flumpy.to_numpy(d_sparse.as_dense_matrix()), dtype=np.float64
            ).reshape(d_sparse.n_rows, d_sparse.n_cols)
            d_gpu = torch.from_numpy(d_np).to(device=dev, dtype=torch.float64)
        return s_gpu, d_gpu
    else:
        s_flex = component.calculate_scales(block_id)
        s_np = np.asarray(flumpy.to_numpy(s_flex), dtype=np.float64)
        s_gpu = torch.from_numpy(s_np).to(device=dev, dtype=torch.float64)
        return s_gpu, None


class RefinerCalculatorGPU(RefinerCalculator):
    """GPU-accelerated RefinerCalculator that computes overall scale factors and
    chain-rule parameter derivatives on the GPU using PyTorch."""

    @staticmethod
    def calculate_scales_and_derivatives_gpu(
        apm,
        block_id: int,
        device: torch.device | str | None = None,
        compute_derivatives: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute (scales_gpu, derivatives_gpu) on the GPU for an active_parameter_manager."""
        comp_scales = []
        comp_derivs = []
        for comp in apm.components.values():
            s_gpu, d_gpu = evaluate_component_gpu(
                comp["object"],
                block_id=block_id,
                compute_derivatives=compute_derivatives,
                device=device,
            )
            comp_scales.append(s_gpu)
            if compute_derivatives:
                comp_derivs.append(d_gpu)

        if comp_scales:
            dev = comp_scales[0].device
        else:
            n_const = (
                len(apm.constant_g_values[block_id])
                if getattr(apm, "constant_g_values", None) is not None
                else 0
            )
            dev = _get_minimization_device(n_const, device)

        const_g_gpu = None
        if getattr(apm, "constant_g_values", None) is not None:
            const_g_gpu = _get_cached_block_tensor(
                apm, "const_g", block_id, apm.constant_g_values, dev
            )

        if not comp_scales:
            empty_d = (
                torch.empty((0, 0), dtype=torch.float64, device=dev)
                if compute_derivatives
                else None
            )
            if const_g_gpu is not None:
                return const_g_gpu, empty_d
            return (
                torch.empty(0, dtype=torch.float64, device=dev),
                empty_d,
            )

        n_obs = comp_scales[0].shape[0]
        if len(comp_scales) == 1:
            total_scales = comp_scales[0]
            if const_g_gpu is not None:
                total_scales = total_scales * const_g_gpu
                total_derivs = (
                    comp_derivs[0] * const_g_gpu[:, None]
                    if compute_derivatives
                    else None
                )
            else:
                total_derivs = comp_derivs[0] if compute_derivatives else None
            return total_scales, total_derivs

        # Multiply all component scales
        total_scales = comp_scales[0].clone()
        for s in comp_scales[1:]:
            total_scales = total_scales * s
        if const_g_gpu is not None:
            total_scales = total_scales * const_g_gpu

        if not compute_derivatives:
            return total_scales, None

        # Chain rule for each component's derivative block
        scaled_deriv_blocks = []
        for i, d_i in enumerate(comp_derivs):
            mult = torch.ones(n_obs, dtype=torch.float64, device=dev)
            for j, s_j in enumerate(comp_scales):
                if i != j:
                    mult = mult * s_j
            if const_g_gpu is not None:
                mult = mult * const_g_gpu
            scaled_deriv_blocks.append(d_i * mult[:, None])

        total_derivs = torch.cat(scaled_deriv_blocks, dim=1)
        return total_scales, total_derivs


def rotate_vectors_about_axis_gpu(
    rot_axis, vectors, angles, device: torch.device | str | None = None
):
    """GPU implementation of dials_scaling_ext.rotate_vectors_about_axis."""
    axis_np = np.asarray(flumpy.to_numpy(rot_axis), dtype=np.float64).reshape(-1, 3)[0]
    vec_np = np.asarray(flumpy.to_numpy(vectors), dtype=np.float64).reshape(-1, 3)
    ang_np = np.asarray(flumpy.to_numpy(angles), dtype=np.float64).reshape(-1)
    dev = _get_minimization_device(vec_np.shape[0], device)

    axis_t = torch.from_numpy(axis_np).to(device=dev, dtype=torch.float64)
    axis_t = axis_t / torch.linalg.norm(axis_t)
    ux, uy, uz = axis_t[0], axis_t[1], axis_t[2]

    vec_t = torch.from_numpy(vec_np).to(device=dev, dtype=torch.float64)
    ang_t = torch.from_numpy(ang_np).to(device=dev, dtype=torch.float64)

    cos_a = torch.cos(ang_t)
    sin_a = torch.sin(ang_t)
    one_m_cos = 1.0 - cos_a

    vx, vy, vz = vec_t[:, 0], vec_t[:, 1], vec_t[:, 2]
    rx = (
        (cos_a + ux * ux * one_m_cos) * vx
        + (ux * uy * one_m_cos - uz * sin_a) * vy
        + (uz * ux * one_m_cos + uy * sin_a) * vz
    )
    ry = (
        (ux * uy * one_m_cos + uz * sin_a) * vx
        + (cos_a + uy * uy * one_m_cos) * vy
        + (uz * uy * one_m_cos - ux * sin_a) * vz
    )
    rz = (
        (ux * uz * one_m_cos - uy * sin_a) * vx
        + (uy * uz * one_m_cos + ux * sin_a) * vy
        + (cos_a + uz * uz * one_m_cos) * vz
    )
    rot_np = np.ascontiguousarray(
        torch.stack([rx, ry, rz], dim=1).cpu().numpy(), dtype=np.float64
    )
    return flumpy.vec_from_numpy(rot_np)


def _alignment_rotation_matrix(alignment_axis) -> np.ndarray:
    """Return 3x3 rotation matrix R such that v_aligned = v @ R.T aligns `alignment_axis` with +z."""
    ux, uy, uz = (
        float(alignment_axis[0]),
        float(alignment_axis[1]),
        float(alignment_axis[2]),
    )
    if (ux, uy, uz) == (0.0, 0.0, 1.0):
        return np.eye(3, dtype=np.float64)
    ax, ay, az = uy, -ux, 0.0
    norm_a = math.hypot(ax, ay)
    if norm_a > 0.0:
        ax /= norm_a
        ay /= norm_a
    angle = acos(uz / math.sqrt(ux * ux + uy * uy + uz * uz))
    c = math.cos(angle)
    s = math.sin(angle)
    omc = 1.0 - c
    return np.array(
        [
            [c + ax * ax * omc, ax * ay * omc - az * s, az * ax * omc + ay * s],
            [ax * ay * omc + az * s, c + ay * ay * omc, az * ay * omc - ax * s],
            [ax * az * omc - ay * s, ay * az * omc + ax * s, c + az * az * omc],
        ],
        dtype=np.float64,
    )


def align_axis_along_z_gpu(
    alignment_axis, vectors, device: torch.device | str | None = None
):
    """GPU implementation of dials.algorithms.scaling.scaling_utilities.align_axis_along_z."""
    if tuple(alignment_axis) == (0.0, 0.0, 1.0):
        return vectors
    R_np = _alignment_rotation_matrix(alignment_axis)
    vec_np = np.asarray(flumpy.to_numpy(vectors), dtype=np.float64).reshape(-1, 3)
    dev = _get_minimization_device(vec_np.shape[0], device)
    vec_t = torch.from_numpy(vec_np).to(device=dev, dtype=torch.float64)
    R_t = torch.from_numpy(R_np.T.copy()).to(device=dev, dtype=torch.float64)
    rot_t = torch.mm(vec_t, R_t)
    rot_np = np.ascontiguousarray(rot_t.cpu().numpy(), dtype=np.float64)
    return flumpy.vec_from_numpy(rot_np)


def calc_theta_phi_gpu(xyz, device: torch.device | str | None = None):
    """GPU implementation of dials_scaling_ext.calc_theta_phi."""
    xyz_np = np.asarray(flumpy.to_numpy(xyz), dtype=np.float64).reshape(-1, 3)
    dev = _get_minimization_device(xyz_np.shape[0], device)
    xyz_t = torch.from_numpy(xyz_np).to(device=dev, dtype=torch.float64)
    x, y, z = xyz_t[:, 0], xyz_t[:, 1], xyz_t[:, 2]
    theta = torch.atan2(torch.sqrt(y * y + x * x), z)
    two_pi = 2.0 * math.pi
    phi = torch.fmod(torch.atan2(y, x) + two_pi, two_pi)
    tp_np = np.ascontiguousarray(
        torch.stack([theta, phi], dim=1).cpu().numpy(), dtype=np.float64
    )
    return flumpy.vec_from_numpy(tp_np)


def calc_lookup_index_gpu(
    theta_phi, points_per_degree: float, device: torch.device | str | None = None
):
    """GPU implementation of dials_scaling_ext.calc_lookup_index."""
    tp_np = np.asarray(flumpy.to_numpy(theta_phi), dtype=np.float64).reshape(-1, 2)
    dev = _get_minimization_device(tp_np.shape[0], device)
    tp_t = torch.from_numpy(tp_np).to(device=dev, dtype=torch.float64)
    rad_to_deg_ppd = float(points_per_degree) * 180.0 / math.pi
    idx_t = (
        360.0 * float(points_per_degree) * torch.floor(tp_t[:, 0] * rad_to_deg_ppd)
        + torch.floor(tp_t[:, 1] * rad_to_deg_ppd)
    ).to(torch.int64)
    idx_np = np.ascontiguousarray(idx_t.cpu().numpy(), dtype=np.uint64)
    return flumpy.from_numpy(idx_np)


def calc_crystal_frame_vectors_gpu(
    reflection_table,
    experiment,
    device: torch.device | str | None = None,
    align_axis: tuple[float, float, float] | None = None,
):
    """GPU implementation of dials.algorithms.scaling.scaling_utilities.calc_crystal_frame_vectors."""
    n_refl = len(reflection_table)
    dev = _get_minimization_device(n_refl, device)
    gonio = experiment.goniometer
    fixed_rot_np = np.array(gonio.get_fixed_rotation(), dtype=np.float64).reshape(3, 3)
    if align_axis is not None and tuple(align_axis) != (0.0, 0.0, 1.0):
        R_align = _alignment_rotation_matrix(align_axis)
        fixed_rot_np = fixed_rot_np @ R_align.T
    fixed_rot_t = torch.from_numpy(np.ascontiguousarray(fixed_rot_np)).to(
        device=dev, dtype=torch.float64
    )
    setting_rot_t = torch.tensor(
        gonio.get_setting_rotation(), dtype=torch.float64, device=dev
    ).reshape(3, 3)
    axis_t = torch.tensor(
        gonio.get_rotation_axis_datum(), dtype=torch.float64, device=dev
    )
    axis_t = axis_t / torch.linalg.norm(axis_t)

    s0_t = torch.tensor(
        experiment.beam.get_sample_to_source_direction(),
        dtype=torch.float64,
        device=dev,
    )
    s1_np = flumpy.to_numpy(reflection_table["s1"])
    xyz_np = flumpy.to_numpy(reflection_table["xyzobs.px.value"])
    z_px_np = np.ascontiguousarray(xyz_np[:, 2], dtype=np.float64)
    s1_t = torch.from_numpy(np.ascontiguousarray(s1_np, dtype=np.float64)).to(
        device=dev
    )
    z_px_t = torch.from_numpy(z_px_np).to(device=dev)
    osc = experiment.scan.get_oscillation(deg=False)
    i0 = float(experiment.scan.get_array_range()[0])
    phi_t = float(osc[0]) + (z_px_t - i0) * float(osc[1])

    lengths = torch.linalg.norm(s1_t, dim=1)
    valid = lengths > 0.0
    ux, uy, uz = axis_t[0], axis_t[1], axis_t[2]

    # Matrix-free Rodrigues rotation: R_inv @ v = F^T @ R_{-phi} @ (S^T @ v)
    w0 = torch.mv(setting_rot_t.t(), s0_t)
    u_dot_w0 = torch.dot(axis_t, w0)
    u_cross_w0 = torch.linalg.cross(axis_t, w0)
    a0 = torch.mv(fixed_rot_t.t(), w0)
    b0 = torch.mv(fixed_rot_t.t(), u_cross_w0)
    c0 = torch.mv(fixed_rot_t.t(), u_dot_w0 * axis_t)

    if bool(valid.all()):
        cos_a = torch.cos(phi_t)
        sin_a = torch.sin(phi_t)
        omc = 1.0 - cos_a
        s0c_t = (
            cos_a[:, None] * a0[None, :]
            - sin_a[:, None] * b0[None, :]
            + omc[:, None] * c0[None, :]
        )
        s1n = s1_t / lengths[:, None]
        w1 = torch.mm(s1n, setting_rot_t)
        w1x, w1y, w1z = w1[:, 0], w1[:, 1], w1[:, 2]
        u_dot_w1 = ux * w1x + uy * w1y + uz * w1z
        cx = uy * w1z - uz * w1y
        cy = uz * w1x - ux * w1z
        cz = ux * w1y - uy * w1x
        rot_s1 = (
            cos_a[:, None] * w1
            - sin_a[:, None] * torch.stack([cx, cy, cz], dim=1)
            + (omc * u_dot_w1)[:, None] * axis_t[None, :]
        )
        s1c_t = torch.mm(rot_s1, fixed_rot_t)
    else:
        s0c_t = torch.zeros((n_refl, 3), dtype=torch.float64, device=dev)
        s1c_t = torch.zeros((n_refl, 3), dtype=torch.float64, device=dev)
        if bool(valid.any()):
            ang = phi_t[valid]
            cos_a = torch.cos(ang)
            sin_a = torch.sin(ang)
            omc = 1.0 - cos_a
            s0c_t[valid] = (
                cos_a[:, None] * a0[None, :]
                - sin_a[:, None] * b0[None, :]
                + omc[:, None] * c0[None, :]
            )
            s1n = s1_t[valid] / lengths[valid, None]
            w1 = torch.mm(s1n, setting_rot_t)
            w1x, w1y, w1z = w1[:, 0], w1[:, 1], w1[:, 2]
            u_dot_w1 = ux * w1x + uy * w1y + uz * w1z
            cx = uy * w1z - uz * w1y
            cy = uz * w1x - ux * w1z
            cz = ux * w1y - uy * w1x
            rot_s1 = (
                cos_a[:, None] * w1
                - sin_a[:, None] * torch.stack([cx, cy, cz], dim=1)
                + (omc * u_dot_w1)[:, None] * axis_t[None, :]
            )
            s1c_t[valid] = torch.mm(rot_s1, fixed_rot_t)

    s0c_np = np.ascontiguousarray(s0c_t.cpu().numpy(), dtype=np.float64)
    s1c_np = np.ascontiguousarray(s1c_t.cpu().numpy(), dtype=np.float64)
    reflection_table["s0c"] = flumpy.vec_from_numpy(s0c_np)
    reflection_table["s1c"] = flumpy.vec_from_numpy(s1c_np)
    return reflection_table
