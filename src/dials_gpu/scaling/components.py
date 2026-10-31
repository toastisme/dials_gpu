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
    weight = torch.where(mask, torch.exp(-(ds * ds)), torch.zeros_like(ds))
    sumweight = weight.sum(dim=1)
    safe_sw = torch.where(sumweight > 0.0, sumweight, torch.ones_like(sumweight))
    inv_sw = torch.where(sumweight > 0.0, 1.0 / safe_sw, torch.zeros_like(sumweight))

    if first_fixed:
        dv_dp = weight[:, 1:] * inv_sw[:, None]
    else:
        dv_dp = weight * inv_sw[:, None]

    smoother._gpu_weight_cache = (key, x_gpu, weight, inv_sw, dv_dp)
    return weight, inv_sw, dv_dp


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

    weight, inv_sw, dv_dp = _get_smoother_1d_weights_gpu(smoother, x_gpu, first_fixed)
    sumwv = torch.mv(weight, values_gpu[:nvalues])
    value = sumwv * inv_sw

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


def _get_cached_sh_matrix_gpu(
    component: SHScaleComponent, block_id: int, device: torch.device
) -> torch.Tensor:
    """Return cached (N_refl, N_params) spherical harmonic matrix on GPU for block_id."""
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
        params_np = np.asarray(flumpy.to_numpy(component.value), dtype=np.float64)
        params_gpu = torch.from_numpy(params_np).to(device=dev, dtype=torch.float64)
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
        scales = torch.exp(b_val * prefac)
        if not compute_derivatives:
            return scales, None
        derivs = db_dp * (scales * prefac)[:, None]
        return scales, derivs

    if type(component) is SmoothScaleComponent1D or (
        isinstance(component, SmoothScaleComponent1D)
        and not isinstance(component, SmoothBScaleComponent1D)
    ):
        params_np = np.asarray(flumpy.to_numpy(component.value), dtype=np.float64)
        params_gpu = torch.from_numpy(params_np).to(device=dev, dtype=torch.float64)
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
        params_np = np.asarray(flumpy.to_numpy(component.value), dtype=np.float64)
        params_gpu = torch.from_numpy(params_np).to(device=dev, dtype=torch.float64)
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
        params_np = np.asarray(flumpy.to_numpy(component.value), dtype=np.float64)
        params_gpu = torch.from_numpy(params_np).to(device=dev, dtype=torch.float64)
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
        params_np = np.asarray(flumpy.to_numpy(component.parameters), dtype=np.float64)
        params_gpu = torch.from_numpy(params_np).to(device=dev, dtype=torch.float64)
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
        apm, block_id: int, device: torch.device | str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (scales_gpu, derivatives_gpu) on the GPU for an active_parameter_manager."""
        comp_scales = []
        comp_derivs = []
        for comp in apm.components.values():
            s_gpu, d_gpu = evaluate_component_gpu(
                comp["object"],
                block_id=block_id,
                compute_derivatives=True,
                device=device,
            )
            comp_scales.append(s_gpu)
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
            if const_g_gpu is not None:
                return const_g_gpu, torch.empty((0, 0), dtype=torch.float64, device=dev)
            return (
                torch.empty(0, dtype=torch.float64, device=dev),
                torch.empty((0, 0), dtype=torch.float64, device=dev),
            )

        n_obs = comp_scales[0].shape[0]
        if len(comp_scales) == 1:
            total_scales = comp_scales[0]
            if const_g_gpu is not None:
                total_scales = total_scales * const_g_gpu
                total_derivs = comp_derivs[0] * const_g_gpu[:, None]
            else:
                total_derivs = comp_derivs[0]
            return total_scales, total_derivs

        # Multiply all component scales
        total_scales = comp_scales[0].clone()
        for s in comp_scales[1:]:
            total_scales = total_scales * s
        if const_g_gpu is not None:
            total_scales = total_scales * const_g_gpu

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


def align_axis_along_z_gpu(
    alignment_axis, vectors, device: torch.device | str | None = None
):
    """GPU implementation of dials.algorithms.scaling.scaling_utilities.align_axis_along_z."""
    if tuple(alignment_axis) == (0.0, 0.0, 1.0):
        return vectors
    ux, uy, uz = alignment_axis
    cross_prod_uz = flex.vec3_double([(uy, -1.0 * ux, 0.0)])
    angle_between_u_z = +1.0 * acos(uz / ((ux**2 + uy**2 + uz**2) ** 0.5))
    phi = flex.double(vectors.size(), angle_between_u_z)
    return rotate_vectors_about_axis_gpu(cross_prod_uz, vectors, phi, device=device)


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
    reflection_table, experiment, device: torch.device | str | None = None
):
    """GPU implementation of dials.algorithms.scaling.scaling_utilities.calc_crystal_frame_vectors."""
    n_refl = len(reflection_table)
    dev = _get_minimization_device(n_refl, device)
    gonio = experiment.goniometer
    fixed_rot_t = torch.tensor(
        gonio.get_fixed_rotation(), dtype=torch.float64, device=dev
    ).reshape(3, 3)
    setting_rot_t = torch.tensor(
        gonio.get_setting_rotation(), dtype=torch.float64, device=dev
    ).reshape(3, 3)
    axis_t = torch.tensor(
        gonio.get_rotation_axis_datum(), dtype=torch.float64, device=dev
    )
    axis_t = axis_t / torch.linalg.norm(axis_t)

    s0c_t = torch.zeros((n_refl, 3), dtype=torch.float64, device=dev)
    s1c_t = torch.zeros((n_refl, 3), dtype=torch.float64, device=dev)

    s0_t = torch.tensor(
        experiment.beam.get_sample_to_source_direction(),
        dtype=torch.float64,
        device=dev,
    )
    s1_np = np.asarray(flumpy.to_numpy(reflection_table["s1"]), dtype=np.float64)
    phi_np = np.asarray(
        flumpy.to_numpy(
            experiment.scan.get_angle_from_array_index(
                reflection_table["xyzobs.px.value"].parts()[2], deg=False
            )
        ),
        dtype=np.float64,
    )
    s1_t = torch.from_numpy(s1_np).to(device=dev, dtype=torch.float64)
    phi_t = torch.from_numpy(phi_np).to(device=dev, dtype=torch.float64)

    lengths = torch.linalg.norm(s1_t, dim=1)
    valid = lengths > 0.0
    if bool(valid.any()):
        s1n = s1_t[valid] / lengths[valid, None]
        ang = phi_t[valid]
        ux, uy, uz = axis_t[0], axis_t[1], axis_t[2]
        cos_a = torch.cos(ang)
        sin_a = torch.sin(ang)
        omc = 1.0 - cos_a

        # Batched Rodrigues rotation matrices (M, 3, 3)
        rot_mat = torch.empty((ang.shape[0], 3, 3), dtype=torch.float64, device=dev)
        rot_mat[:, 0, 0] = cos_a + ux * ux * omc
        rot_mat[:, 0, 1] = ux * uy * omc - uz * sin_a
        rot_mat[:, 0, 2] = uz * ux * omc + uy * sin_a
        rot_mat[:, 1, 0] = ux * uy * omc + uz * sin_a
        rot_mat[:, 1, 1] = cos_a + uy * uy * omc
        rot_mat[:, 1, 2] = uz * uy * omc - ux * sin_a
        rot_mat[:, 2, 0] = ux * uz * omc - uy * sin_a
        rot_mat[:, 2, 1] = uy * uz * omc + ux * sin_a
        rot_mat[:, 2, 2] = cos_a + uz * uz * omc

        R = torch.matmul(
            setting_rot_t[None, :, :], torch.matmul(rot_mat, fixed_rot_t[None, :, :])
        )
        R_inv = R.transpose(1, 2)
        s0c_t[valid] = torch.matmul(R_inv, s0_t)
        s1c_t[valid] = torch.einsum("ijk,ik->ij", R_inv, s1n)

    s0c_np = np.ascontiguousarray(s0c_t.cpu().numpy(), dtype=np.float64)
    s1c_np = np.ascontiguousarray(s1c_t.cpu().numpy(), dtype=np.float64)
    reflection_table["s0c"] = flumpy.vec_from_numpy(s0c_np)
    reflection_table["s1c"] = flumpy.vec_from_numpy(s1c_np)
    return reflection_table
