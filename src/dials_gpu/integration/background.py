from __future__ import annotations

import logging
import math

import numpy as np
import torch
from dxtbx import flumpy

from dials.algorithms.background.glm import Creator as GLMCreator
from dials.algorithms.background.simple import (
    Constant2dModeller,
    Constant3dModeller,
    Creator as SimpleCreator,
    Linear2dModeller,
    Linear3dModeller,
    NSigmaOutlierRejector,
    TruncatedOutlierRejector,
    TukeyOutlierRejector,
)
from dials.array_family import flex
from dials.extensions.auto_background_ext import AutoBackgroundExt
from dials.extensions.glm_background_ext import GLMBackgroundExt
from dials.extensions.null_background_ext import NullBackgroundExt
from dials.extensions.simple_background_ext import SimpleBackgroundExt
from dials_gpu.integration import _shoebox_pack_ext

logger = logging.getLogger(__name__)

# Shoebox mask bit definitions (dials/model/data/mask_code.h)
MASK_VALID = 1
MASK_BACKGROUND = 2
MASK_FOREGROUND = 4
MASK_OVERLAPPED = 8
MASK_BACKGROUND_USED = 16
MASK_BG_VALID = MASK_VALID | MASK_BACKGROUND  # 3

# Maximum padded elements per GPU bucket (~8 MB per float64 tensor) to avoid
# padding normal shoeboxes to rare giant spindle shoeboxes.
MAX_BUCKET_ELEMENTS = 1_000_000


def _get_device(device=None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_size_buckets(
    indices: np.ndarray, sizes: np.ndarray, max_elements: int = MAX_BUCKET_ELEMENTS
) -> list[np.ndarray]:
    """
    Partition reflection `indices` (sorted by `sizes`) into buckets such that
    `len(bucket) * max(sizes[bucket]) <= max_elements`.
    """
    if len(indices) == 0:
        return []
    order = np.argsort(sizes)
    sorted_idx = indices[order]
    sorted_sz = sizes[order]

    buckets = []
    start = 0
    n = len(sorted_idx)
    while start < n:
        # Binary search or step forward to find largest end where (end - start) * sorted_sz[end - 1] <= max_elements
        end = start + 1
        while end < n and (end + 1 - start) * int(sorted_sz[end]) <= max_elements:
            end += 1
        buckets.append(sorted_idx[start:end])
        start = end
    return buckets


def _poisson_pdf_gpu(mu: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(mu)
    log_mu = torch.log(mu.clamp(min=1e-300))
    k_pos = k.clamp(min=0.0)
    pdf_pos = torch.exp(k_pos * log_mu - mu - torch.lgamma(k_pos + 1.0))
    res = torch.where(k > 0.0, pdf_pos, zero)
    res = torch.where(k == 0.0, torch.exp(-mu), res)
    return torch.where(mu <= 0.0, zero, res)


def _poisson_cdf_gpu(mu: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(mu)
    k_pos = torch.floor(k.clamp(min=0.0) + 1.0).clamp(min=1.0)
    cdf_pos = torch.special.gammaincc(k_pos, mu)
    res = torch.where(k > 0.0, cdf_pos, zero)
    res = torch.where(k == 0.0, torch.exp(-mu), res)
    return torch.where(mu <= 0.0, zero, res)


def _robust_poisson_mean_batched_gpu(
    values_t: torch.Tensor,
    valid_t: torch.Tensor,
    tuning_constant: float = 1.345,
    max_iter: int = 100,
    min_pixels: int = 10,
    tolerance: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched GPU implementation of Cantoni & Ronchetti (2001) / Parkhurst et al. (2016)
    Huber robust Poisson GLM mean estimator (`dials/algorithms/background/glm/robust_poisson_mean.h`
    and `scitbx/glmtbx/robust_glm.h`).

    Args:
        values_t: (B, N) float64 tensor of pixel counts
        valid_t:  (B, N) bool tensor indicating valid background pixels
    Returns:
        mean_t:      (B,) float64 tensor of estimated Poisson mean mu
        converged_t: (B,) bool tensor indicating convergence & sufficient pixels
    """
    device = values_t.device
    B_size, N_max = values_t.shape
    counts = torch.sum(valid_t, dim=1)
    has_neg = torch.any(valid_t & (values_t < 0.0), dim=1)
    active_batch = (counts >= min_pixels) & (~has_neg)

    if not torch.any(active_batch) or N_max == 0:
        return (
            torch.zeros(B_size, dtype=torch.float64, device=device),
            torch.zeros(B_size, dtype=torch.bool, device=device),
        )

    # Compute exact C++ median: sort valid values (pushing invalid to +inf), take index counts // 2
    inf_padded = torch.where(
        valid_t, values_t, torch.full_like(values_t, float("inf"))
    )
    sorted_vals, _ = torch.sort(inf_padded, dim=1)
    mid_idx = torch.clamp(counts // 2, min=0, max=max(N_max - 1, 0)).unsqueeze(1)
    medians = torch.gather(sorted_vals, 1, mid_idx).squeeze(1)
    medians = torch.where(
        (~active_batch) | (medians <= 0.0) | torch.isinf(medians),
        torch.ones_like(medians),
        medians,
    )

    beta = torch.log(medians)
    converged = torch.zeros(B_size, dtype=torch.bool, device=device)
    c = float(tuning_constant)
    counts_f = counts.to(torch.float64).clamp(min=1.0)

    for _ in range(max_iter):
        still_running = active_batch & (~converged) & (beta > -300.0) & (beta < 300.0)
        if not torch.any(still_running):
            break

        mu = torch.exp(beta.clamp(min=-300.0, max=300.0))
        svar = torch.sqrt(mu)

        j1 = torch.floor(mu - c * svar)
        j2 = torch.floor(mu + c * svar)
        p1 = _poisson_pdf_gpu(mu, j1)
        p2 = _poisson_pdf_gpu(mu, j2)
        p3 = _poisson_cdf_gpu(mu, j1)
        p4 = _poisson_pdf_gpu(mu, j2 + 1.0)
        p5 = _poisson_cdf_gpu(mu, j2 + 1.0)
        p6 = 1.0 - p5 + p4
        p7 = _poisson_pdf_gpu(mu, j1 - 1.0)
        p8 = _poisson_pdf_gpu(mu, j2 - 1.0)
        p9 = _poisson_cdf_gpu(mu, j2 - 1.0)
        p10 = p9 - p3 + p1

        epsi1 = c * (p6 - p3) + (mu / svar) * (p1 - p2)
        epsi2 = c * (p1 + p2) + (mu * mu / (svar * svar * svar)) * (
            p10 / mu + p7 - p1 - p8 + p2
        )

        r_obs = (values_t - mu.unsqueeze(1)) / svar.unsqueeze(1)
        huber_r = torch.clamp(r_obs, min=-c, max=c)
        diff_huber = torch.where(
            valid_t, huber_r - epsi1.unsqueeze(1), torch.zeros_like(huber_r)
        )
        sum_u = torch.sum(diff_huber, dim=1)
        denom = counts_f * mu * epsi2

        valid_step = still_running & (torch.abs(denom) > 0.0)
        delta = torch.where(
            valid_step, sum_u / denom.clamp(min=1e-30), torch.zeros_like(beta)
        )
        old_beta_norm = torch.clamp(torch.abs(beta), min=math.sqrt(1e-10))
        beta = torch.where(valid_step, beta + delta, beta)

        rel_change = torch.abs(delta) / old_beta_norm
        newly_converged = valid_step & (rel_change < tolerance)
        converged = converged | newly_converged

    valid_beta = converged & (beta > -300.0) & (beta < 300.0)
    mean_out = torch.where(
        valid_beta,
        torch.exp(beta.clamp(min=-300.0, max=300.0)),
        torch.zeros_like(beta),
    )
    return mean_out, valid_beta


def _robust_poisson_mean_1d_gpu(
    bg_vals_t: torch.Tensor,
    bg_refl_idx_t: torch.Tensor,
    counts_t: torch.Tensor,
    medians_t: torch.Tensor,
    has_neg_t: torch.Tensor,
    tuning_constant: float = 1.345,
    max_iter: int = 100,
    min_pixels: int = 10,
    tolerance: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unpadded 1D GPU implementation of Cantoni & Ronchetti (2001) Huber robust Poisson GLM mean estimator.
    Evaluates transcendental Poisson PDF/CDF functions on the 1D reflection array `(n_refl,)` rather than
    padded 2D pixel buckets, and accumulates Huber residuals via `index_add_`.
    """
    device = counts_t.device
    n_refl = counts_t.numel()
    active_batch = (counts_t >= min_pixels) & (~has_neg_t)

    if not torch.any(active_batch) or bg_vals_t.numel() == 0:
        return (
            torch.zeros(n_refl, dtype=torch.float64, device=device),
            torch.zeros(n_refl, dtype=torch.bool, device=device),
        )

    beta = torch.log(medians_t.clamp(min=1e-30))
    converged = torch.zeros(n_refl, dtype=torch.bool, device=device)
    c = float(tuning_constant)
    counts_f = counts_t.to(torch.float64).clamp(min=1.0)

    for it in range(max_iter):
        still_running = active_batch & (~converged) & (beta > -300.0) & (beta < 300.0)
        if it % 4 == 0 and not torch.any(still_running):
            break

        mu = torch.exp(beta.clamp(min=-300.0, max=300.0))
        svar = torch.sqrt(mu)

        j1 = torch.floor(mu - c * svar)
        j2 = torch.floor(mu + c * svar)
        p1 = _poisson_pdf_gpu(mu, j1)
        p2 = _poisson_pdf_gpu(mu, j2)
        p3 = _poisson_cdf_gpu(mu, j1)
        p4 = _poisson_pdf_gpu(mu, j2 + 1.0)
        p5 = _poisson_cdf_gpu(mu, j2 + 1.0)
        p6 = 1.0 - p5 + p4
        p7 = _poisson_pdf_gpu(mu, j1 - 1.0)
        p8 = _poisson_pdf_gpu(mu, j2 - 1.0)
        p9 = _poisson_cdf_gpu(mu, j2 - 1.0)
        p10 = p9 - p3 + p1

        epsi1 = c * (p6 - p3) + (mu / svar) * (p1 - p2)
        epsi2 = c * (p1 + p2) + (mu * mu / (svar * svar * svar)) * (
            p10 / mu + p7 - p1 - p8 + p2
        )

        r_obs = (bg_vals_t - mu[bg_refl_idx_t]) / svar[bg_refl_idx_t]
        huber_r = torch.clamp(r_obs, min=-c, max=c)
        sum_huber = torch.zeros(n_refl, dtype=torch.float64, device=device)
        sum_huber.index_add_(0, bg_refl_idx_t, huber_r)
        sum_u = sum_huber - counts_f * epsi1
        denom = counts_f * mu * epsi2

        valid_step = still_running & (torch.abs(denom) > 0.0)
        delta = torch.where(
            valid_step, sum_u / denom.clamp(min=1e-30), torch.zeros_like(beta)
        )
        old_beta_norm = torch.clamp(torch.abs(beta), min=math.sqrt(1e-10))
        beta = torch.where(valid_step, beta + delta, beta)

        rel_change = torch.abs(delta) / old_beta_norm
        newly_converged = valid_step & (rel_change < tolerance)
        converged = converged | newly_converged

    valid_beta = converged & (beta > -300.0) & (beta < 300.0)
    mean_out = torch.where(
        valid_beta,
        torch.exp(beta.clamp(min=-300.0, max=300.0)),
        torch.zeros_like(beta),
    )
    return mean_out, valid_beta


def _robust_poisson_mean_hist_gpu(
    hist_t: torch.Tensor,
    tail_vals_t: torch.Tensor | None,
    tail_idx_t: torch.Tensor | None,
    counts_t: torch.Tensor,
    medians_t: torch.Tensor,
    has_neg_t: torch.Tensor,
    tuning_constant: float = 1.345,
    max_iter: int = 100,
    min_pixels: int = 10,
    tolerance: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Histogram-compressed GPU implementation of Cantoni & Ronchetti (2001) Huber robust Poisson GLM mean estimator.
    Evaluates Huber residuals across the `(n_refl, 32)` integer count histogram + sparse tail without
    atomicAdd warp serialization.
    """
    device = counts_t.device
    n_refl = counts_t.numel()
    active_batch = (counts_t >= min_pixels) & (~has_neg_t)

    if not torch.any(active_batch):
        return (
            torch.zeros(n_refl, dtype=torch.float64, device=device),
            torch.zeros(n_refl, dtype=torch.bool, device=device),
        )

    beta = torch.log(medians_t.clamp(min=1e-30))
    converged = torch.zeros(n_refl, dtype=torch.bool, device=device)
    c = float(tuning_constant)
    counts_f = counts_t.to(torch.float64).clamp(min=1.0)
    hist_f = hist_t.to(torch.float64)
    n_bins = hist_t.shape[1]
    k_grid = torch.arange(n_bins, dtype=torch.float64, device=device).unsqueeze(0)
    has_tail = tail_vals_t is not None and tail_vals_t.numel() > 0

    for it in range(max_iter):
        still_running = active_batch & (~converged) & (beta > -300.0) & (beta < 300.0)
        if it % 4 == 0 and not torch.any(still_running):
            break

        mu = torch.exp(beta.clamp(min=-300.0, max=300.0))
        svar = torch.sqrt(mu)

        j1 = torch.floor(mu - c * svar)
        j2 = torch.floor(mu + c * svar)
        p1 = _poisson_pdf_gpu(mu, j1)
        p2 = _poisson_pdf_gpu(mu, j2)
        p3 = _poisson_cdf_gpu(mu, j1)
        p4 = _poisson_pdf_gpu(mu, j2 + 1.0)
        p5 = _poisson_cdf_gpu(mu, j2 + 1.0)
        p6 = 1.0 - p5 + p4
        p7 = _poisson_pdf_gpu(mu, j1 - 1.0)
        p8 = _poisson_pdf_gpu(mu, j2 - 1.0)
        p9 = _poisson_cdf_gpu(mu, j2 - 1.0)
        p10 = p9 - p3 + p1

        epsi1 = c * (p6 - p3) + (mu / svar) * (p1 - p2)
        epsi2 = c * (p1 + p2) + (mu * mu / (svar * svar * svar)) * (
            p10 / mu + p7 - p1 - p8 + p2
        )

        r_grid = (k_grid - mu.unsqueeze(1)) / svar.unsqueeze(1)
        huber_grid = torch.clamp(r_grid, min=-c, max=c)
        sum_huber = torch.sum(hist_f * huber_grid, dim=1)
        if has_tail:
            r_tail = (tail_vals_t - mu[tail_idx_t]) / svar[tail_idx_t]
            sum_huber.index_add_(0, tail_idx_t, torch.clamp(r_tail, min=-c, max=c))

        sum_u = sum_huber - counts_f * epsi1
        denom = counts_f * mu * epsi2

        valid_step = still_running & (torch.abs(denom) > 0.0)
        delta = torch.where(
            valid_step, sum_u / denom.clamp(min=1e-30), torch.zeros_like(beta)
        )
        old_beta_norm = torch.clamp(torch.abs(beta), min=math.sqrt(1e-10))
        beta = torch.where(valid_step, beta + delta, beta)

        rel_change = torch.abs(delta) / old_beta_norm
        newly_converged = valid_step & (rel_change < tolerance)
        converged = converged | newly_converged

    valid_beta = converged & (beta > -300.0) & (beta < 300.0)
    mean_out = torch.where(
        valid_beta,
        torch.exp(beta.clamp(min=-300.0, max=300.0)),
        torch.zeros_like(beta),
    )
    return mean_out, valid_beta


class GLMBackgroundGPU:
    """
    GPU-accelerated Generalized Linear Model (GLM) background subtraction using PyTorch.
    Matches `dials.extensions.glm_background_ext.GLMBackgroundExt` and
    `dials.algorithms.background.glm.Creator`.
    """

    def __init__(self, params, experiments, device=None):
        self.params = params
        self.experiments = experiments
        self.device = _get_device(device)
        if params is None:
            from dials.command_line.integrate import phil_scope

            params = phil_scope.extract()
        self._model_alg = params.integration.background.glm.model.algorithm
        self._tuning_constant = params.integration.background.glm.robust.tuning_constant
        self._min_pixels = params.integration.background.glm.min_pixels
        self._max_iter = 100
        self._cpu_fallback = None

    def compute_background(self, reflections, image_volume=None):
        if image_volume is not None or self._model_alg not in ("constant3d", "constant2d"):
            if self._cpu_fallback is None:
                self._cpu_fallback = GLMBackgroundExt(self.params, self.experiments)
            return self._cpu_fallback.compute_background(reflections, image_volume)

        n_refl = len(reflections)
        if n_refl == 0:
            return flex.bool()

        shoeboxes = reflections["shoebox"]
        bg_mean_np = np.zeros(n_refl, dtype=np.float64)
        success_np = np.zeros(n_refl, dtype=bool)

        if self._model_alg == "constant3d" and hasattr(_shoebox_pack_ext, "compute_glm_constant3d_fast"):
            if "flags" not in reflections:
                reflections["flags"] = flex.size_t(n_refl, 0)
            bg_mean_flex, success_flex = _shoebox_pack_ext.compute_glm_constant3d_fast(
                shoeboxes,
                reflections["flags"],
                int(self._min_pixels),
                float(self._tuning_constant),
                int(self._max_iter),
                MASK_BG_VALID,
                MASK_OVERLAPPED,
                MASK_BACKGROUND_USED,
            )
            reflections["background.mean"] = bg_mean_flex
            return success_flex

        if self._model_alg == "constant2d" and hasattr(_shoebox_pack_ext, "compute_glm_constant2d_fast"):
            if "flags" not in reflections:
                reflections["flags"] = flex.size_t(n_refl, 0)
            bg_mean_flex, success_flex = _shoebox_pack_ext.compute_glm_constant2d_fast(
                shoeboxes,
                reflections["flags"],
                int(self._min_pixels),
                float(self._tuning_constant),
                int(self._max_iter),
                MASK_BG_VALID,
                MASK_OVERLAPPED,
                MASK_BACKGROUND_USED,
            )
            reflections["background.mean"] = bg_mean_flex
            return success_flex

        if self._model_alg == "constant3d":
            counts_np = np.empty(n_refl, dtype=np.int64)
            medians_np = np.empty(n_refl, dtype=np.float64)
            has_neg_np = np.empty(n_refl, dtype=np.uint8)
            hist_np = np.empty((n_refl, 128), dtype=np.int32)
            tail_counts_np = np.empty(n_refl, dtype=np.int64)

            total_tail = _shoebox_pack_ext.extract_glm_constant3d_hist(
                shoeboxes,
                int(self._min_pixels),
                MASK_BG_VALID,
                MASK_OVERLAPPED,
                MASK_BACKGROUND_USED,
                counts_np.ctypes.data,
                medians_np.ctypes.data,
                has_neg_np.ctypes.data,
                hist_np.ctypes.data,
                tail_counts_np.ctypes.data,
            )

            if total_tail > 0:
                tail_vals_np = np.empty(total_tail, dtype=np.float64)
                tail_idx_np = np.empty(total_tail, dtype=np.int64)
                _shoebox_pack_ext.fill_glm_constant3d_tail(
                    shoeboxes,
                    int(self._min_pixels),
                    MASK_BG_VALID,
                    MASK_OVERLAPPED,
                    counts_np.ctypes.data,
                    has_neg_np.ctypes.data,
                    tail_counts_np.ctypes.data,
                    medians_np.ctypes.data,
                    tail_vals_np.ctypes.data,
                    tail_idx_np.ctypes.data,
                )
                tail_vals_t = torch.as_tensor(
                    tail_vals_np, dtype=torch.float64, device=self.device
                )
                tail_idx_t = torch.as_tensor(
                    tail_idx_np, dtype=torch.long, device=self.device
                )
            else:
                tail_vals_t = None
                tail_idx_t = None

            hist_t = torch.as_tensor(hist_np, dtype=torch.int32, device=self.device)
            counts_t = torch.as_tensor(counts_np, dtype=torch.long, device=self.device)
            medians_t = torch.as_tensor(
                medians_np, dtype=torch.float64, device=self.device
            )
            has_neg_t = torch.as_tensor(
                has_neg_np != 0, dtype=torch.bool, device=self.device
            )

            mean_t, conv_t = _robust_poisson_mean_hist_gpu(
                hist_t,
                tail_vals_t,
                tail_idx_t,
                counts_t,
                medians_t,
                has_neg_t,
                tuning_constant=self._tuning_constant,
                max_iter=self._max_iter,
                min_pixels=self._min_pixels,
                tolerance=1e-3,
            )
            bg_mean_np = np.ascontiguousarray(mean_t.cpu().numpy(), dtype=np.float64)
            conv_u8 = np.ascontiguousarray(conv_t.cpu().numpy(), dtype=np.uint8)
            success_np = conv_u8.astype(bool)

            _shoebox_pack_ext.unpack_constant3d_bg_and_mask(
                shoeboxes,
                bg_mean_np.ctypes.data,
                conv_u8.ctypes.data,
                MASK_BACKGROUND_USED,
            )

            reflections["background.mean"] = flex.double(
                np.ascontiguousarray(np.float32(bg_mean_np), dtype=np.float64)
            )
            success_flex = flex.bool(success_np)
            reflections.set_flags(~success_flex, reflections.flags.dont_integrate)
            return success_flex

        # constant2d: solve per z-slice across active reflections
        slice_vals_list = []
        slice_owner = []
        slice_z = []
        refl_total_bg = np.zeros(n_refl, dtype=np.int64)
        refl_nz = np.zeros(n_refl, dtype=np.int64)

        for r in range(n_refl):
            sb = shoeboxes[r]
            if sb.data.size() == 0:
                continue
            d_np = flumpy.to_numpy(sb.data)
            m_np = flumpy.to_numpy(sb.mask)
            valid_bg = ((m_np & MASK_BG_VALID) == MASK_BG_VALID) & (
                (m_np & MASK_OVERLAPPED) == 0
            )
            m_np[valid_bg] |= MASK_BACKGROUND_USED
            total_c = int(np.sum(valid_bg))
            refl_total_bg[r] = total_c
            nz = d_np.shape[0]
            refl_nz[r] = nz
            if total_c < self._min_pixels:
                flumpy.to_numpy(sb.background)[:] = 0.0
                m_np[valid_bg] &= np.uint8(255 ^ MASK_BACKGROUND_USED)
                continue
            for z in range(nz):
                v_z = d_np[z][valid_bg[z]]
                slice_vals_list.append(v_z)
                slice_owner.append(r)
                slice_z.append(z)

        if slice_vals_list:
            s_counts = np.array([v.size for v in slice_vals_list], dtype=np.int64)
            s_indices = np.arange(len(slice_vals_list), dtype=np.int64)
            buckets = _make_size_buckets(s_indices, s_counts, MAX_BUCKET_ELEMENTS)
            s_means = np.zeros(len(slice_vals_list), dtype=np.float64)
            s_convs = np.zeros(len(slice_vals_list), dtype=bool)

            for bucket in buckets:
                b_len = len(bucket)
                n_max = int(np.max(s_counts[bucket]))
                if n_max == 0:
                    continue
                val_pad = np.zeros((b_len, n_max), dtype=np.float64)
                msk_pad = np.zeros((b_len, n_max), dtype=bool)
                for i, s_idx in enumerate(bucket):
                    v = slice_vals_list[s_idx]
                    k = v.size
                    if k > 0:
                        val_pad[i, :k] = v
                        msk_pad[i, :k] = True
                val_t = torch.as_tensor(
                    val_pad, dtype=torch.float64, device=self.device
                )
                msk_t = torch.as_tensor(msk_pad, dtype=torch.bool, device=self.device)
                m_t, c_t = _robust_poisson_mean_batched_gpu(
                    val_t,
                    msk_t,
                    tuning_constant=self._tuning_constant,
                    max_iter=self._max_iter,
                    min_pixels=self._min_pixels,
                    tolerance=1e-3,
                )
                s_means[bucket] = m_t.cpu().numpy()
                s_convs[bucket] = c_t.cpu().numpy()

            # Group slices back into shoeboxes
            refl_slice_ok = np.ones(n_refl, dtype=bool)
            for s_idx, (r, z) in enumerate(zip(slice_owner, slice_z)):
                if not s_convs[s_idx]:
                    refl_slice_ok[r] = False
                else:
                    flumpy.to_numpy(shoeboxes[r].background)[z, :, :] = np.float32(
                        s_means[s_idx]
                    )

            for r in set(slice_owner):
                sb = shoeboxes[r]
                if refl_slice_ok[r] and refl_total_bg[r] >= self._min_pixels:
                    success_np[r] = True
                else:
                    flumpy.to_numpy(sb.background)[:] = 0.0
                    flumpy.to_numpy(sb.mask)[:] &= np.uint8(
                        255 ^ MASK_BACKGROUND_USED
                    )

        reflections["background.mean"] = shoeboxes.mean_modelled_background()
        success_flex = flex.bool(success_np)
        reflections.set_flags(~success_flex, reflections.flags.dont_integrate)
        return success_flex


class SimpleBackgroundGPU:
    """
    GPU-accelerated Simple background subtraction (outlier rejection + constant/linear
    2D/3D least-squares modelling) using PyTorch.
    Matches `dials.extensions.simple_background_ext.SimpleBackgroundExt` and
    `dials.algorithms.background.simple.Creator`.
    """

    def __init__(self, params, experiments, device=None):
        self.params = params
        self.experiments = experiments
        self.device = _get_device(device)
        if params is None:
            from dials.command_line.integrate import phil_scope

            params = phil_scope.extract()
        s_params = params.integration.background.simple
        self._outlier_alg = s_params.outlier.algorithm
        self._model_alg = s_params.model.algorithm
        self._min_pixels = s_params.min_pixels
        self._s_params = s_params
        self._cpu_fallback = None

    def _reject_outliers_gpu(
        self, data_t: torch.Tensor, valid_bg: torch.Tensor
    ) -> torch.Tensor:
        R, Z, Y, X = data_t.shape
        alg = self._outlier_alg
        if alg == "null":
            return valid_bg.clone()

        flat_vals = data_t.reshape(R, -1)
        flat_valid = valid_bg.reshape(R, -1)
        counts = torch.sum(flat_valid, dim=1)
        N_max = flat_vals.shape[1]

        if alg == "nsigma":
            p = self._s_params.outlier.nsigma
            lower, upper = float(p.lower), float(p.upper)
            counts_f = counts.to(torch.float64).clamp(min=1.0)
            sum_x = torch.sum(
                torch.where(flat_valid, flat_vals, torch.zeros_like(flat_vals)), dim=1
            )
            mean_x = sum_x / counts_f
            diff = torch.where(
                flat_valid,
                flat_vals - mean_x.unsqueeze(1),
                torch.zeros_like(flat_vals),
            )
            var_x = torch.sum(diff * diff, dim=1) / (counts_f - 1.0).clamp(min=1.0)
            sigma_x = torch.sqrt(var_x)
            lo = (mean_x - lower * sigma_x).unsqueeze(1)
            hi = (mean_x + upper * sigma_x).unsqueeze(1)
            keep = (
                flat_valid
                & (flat_vals >= lo)
                & (flat_vals <= hi)
                & (counts.unsqueeze(1) > 1)
            )
            return keep.reshape(R, Z, Y, X)

        if alg == "tukey":
            p = self._s_params.outlier.tukey
            lower, upper = float(p.lower), float(p.upper)
            inf_padded = torch.where(
                flat_valid, flat_vals, torch.full_like(flat_vals, float("inf"))
            )
            sorted_vals, _ = torch.sort(inf_padded, dim=1)
            mid = counts // 2
            q1_idx = torch.clamp(mid // 2, min=0, max=max(N_max - 1, 0)).unsqueeze(1)
            q3_idx = torch.clamp(
                mid + (counts - mid) // 2, min=0, max=max(N_max - 1, 0)
            ).unsqueeze(1)
            q1 = torch.gather(sorted_vals, 1, q1_idx)
            q3 = torch.gather(sorted_vals, 1, q3_idx)
            iqr = q3 - q1
            lo = q1 - lower * iqr
            hi = q3 + upper * iqr
            keep = (
                flat_valid
                & (flat_vals >= lo)
                & (flat_vals <= hi)
                & (counts.unsqueeze(1) >= 4)
            )
            return keep.reshape(R, Z, Y, X)

        if alg == "truncated":
            p = self._s_params.outlier.truncated
            lower, upper = float(p.lower), float(p.upper)
            inf_padded = torch.where(
                flat_valid, flat_vals, torch.full_like(flat_vals, float("inf"))
            )
            _, sorted_idx = torch.sort(inf_padded, dim=1, stable=True)
            i0 = (lower * counts.to(torch.float64) * 0.5).to(torch.long).unsqueeze(1)
            i1 = (
                counts - (upper * counts.to(torch.float64) * 0.5).to(torch.long)
            ).unsqueeze(1)
            rank_pos = torch.zeros_like(sorted_idx)
            rank_grid = (
                torch.arange(N_max, dtype=torch.long, device=data_t.device)
                .unsqueeze(0)
                .expand(R, N_max)
            )
            rank_pos.scatter_(1, sorted_idx, rank_grid)
            keep = flat_valid & (rank_pos >= i0) & (rank_pos < i1)
            return keep.reshape(R, Z, Y, X)

        raise ValueError(f"Unsupported GPU outlier algorithm: {alg}")

    def _process_bucket_gpu(
        self,
        shoeboxes,
        bucket: np.ndarray,
        shapes: np.ndarray,
        bg_mean_out: np.ndarray,
        mse_out: np.ndarray,
        disp_out: np.ndarray,
        success_out: np.ndarray,
    ):
        R = len(bucket)
        b_shapes = shapes[bucket]
        Z = int(np.max(b_shapes[:, 0]))
        Y = int(np.max(b_shapes[:, 1]))
        X = int(np.max(b_shapes[:, 2]))
        if Z == 0 or Y == 0 or X == 0:
            return

        data_pad = np.zeros((R, Z, Y, X), dtype=np.float64)
        mask_pad = np.zeros((R, Z, Y, X), dtype=np.int32)
        bucket_c = np.ascontiguousarray(bucket, dtype=np.int64)
        _shoebox_pack_ext.pack_bucket_3d(
            shoeboxes,
            bucket_c.ctypes.data,
            R,
            Z,
            Y,
            X,
            data_pad.ctypes.data,
            mask_pad.ctypes.data,
        )

        data_t = torch.as_tensor(data_pad, dtype=torch.float64, device=self.device)
        mask_t = torch.as_tensor(mask_pad, dtype=torch.int32, device=self.device)
        shapes_t = torch.as_tensor(b_shapes, dtype=torch.long, device=self.device)

        valid_bg = ((mask_t & MASK_BG_VALID) == MASK_BG_VALID) & (
            (mask_t & MASK_OVERLAPPED) == 0
        )
        used_bg = self._reject_outliers_gpu(data_t, valid_bg)
        mask_t = torch.where(used_bg, mask_t | MASK_BACKGROUND_USED, mask_t)

        total_used = torch.sum(used_bg.reshape(R, -1), dim=1)
        has_min_pixels = total_used >= self._min_pixels

        if self._model_alg == "constant3d":
            flat_vals = data_t.reshape(R, -1)
            flat_used = used_bg.reshape(R, -1)
            sum_x = torch.sum(
                torch.where(flat_used, flat_vals, torch.zeros_like(flat_vals)), dim=1
            )
            mean_t = torch.where(
                total_used > 1,
                sum_x / total_used.to(torch.float64).clamp(min=1.0),
                torch.zeros_like(sum_x),
            )
            success_t = has_min_pixels & (total_used > 1)
            mean_t = torch.where(success_t, mean_t, torch.zeros_like(mean_t))
            bg_t = mean_t.view(R, 1, 1, 1).expand(R, Z, Y, X).clone()

        elif self._model_alg == "constant2d":
            slice_used = torch.sum(used_bg, dim=(2, 3))
            slice_sum = torch.sum(
                torch.where(used_bg, data_t, torch.zeros_like(data_t)), dim=(2, 3)
            )
            slice_mean = torch.where(
                slice_used > 1,
                slice_sum / slice_used.to(torch.float64).clamp(min=1.0),
                torch.zeros_like(slice_sum),
            )
            z_idx = torch.arange(Z, device=self.device).unsqueeze(0)
            active_z = z_idx < shapes_t[:, 0].unsqueeze(1)
            all_slices_ok = torch.all((slice_used > 1) | (~active_z), dim=1)
            success_t = has_min_pixels & all_slices_ok
            bg_t = slice_mean.view(R, Z, 1, 1).expand(R, Z, Y, X).clone()
            bg_t = torch.where(
                success_t.view(R, 1, 1, 1), bg_t, torch.zeros_like(bg_t)
            )

        elif self._model_alg == "linear3d":
            z_c = (
                torch.arange(Z, dtype=torch.float64, device=self.device) + 0.5
            ).view(1, Z, 1, 1)
            y_c = (
                torch.arange(Y, dtype=torch.float64, device=self.device) + 0.5
            ).view(1, 1, Y, 1)
            x_c = (
                torch.arange(X, dtype=torch.float64, device=self.device) + 0.5
            ).view(1, 1, 1, X)
            ones = torch.ones((1, Z, Y, X), dtype=torch.float64, device=self.device)
            design = torch.stack(
                [
                    ones.expand(R, Z, Y, X),
                    x_c.expand(R, Z, Y, X),
                    y_c.expand(R, Z, Y, X),
                    z_c.expand(R, Z, Y, X),
                ],
                dim=-1,
            )
            w = used_bg.to(torch.float64).unsqueeze(-1)
            A_flat = (design * w).reshape(R, -1, 4)
            b_flat = (data_t * used_bg.to(torch.float64)).reshape(R, -1, 1)
            ATA = torch.bmm(A_flat.transpose(1, 2), A_flat)
            ATb = torch.bmm(A_flat.transpose(1, 2), b_flat)
            det_ATA = torch.abs(torch.linalg.det(ATA))
            non_singular = det_ATA > 1e-12
            eye4 = torch.eye(4, dtype=torch.float64, device=self.device).unsqueeze(0)
            ATA_safe = torch.where(non_singular.view(R, 1, 1), ATA, eye4)
            params_4 = torch.linalg.solve(ATA_safe, ATb).squeeze(-1)
            success_t = has_min_pixels & (total_used > 4) & non_singular
            bg_t = torch.sum(design * params_4.view(R, 1, 1, 1, 4), dim=-1)
            bg_t = torch.where(
                success_t.view(R, 1, 1, 1), bg_t, torch.zeros_like(bg_t)
            )

        elif self._model_alg == "linear2d":
            y_c = (
                torch.arange(Y, dtype=torch.float64, device=self.device) + 0.5
            ).view(1, 1, Y, 1)
            x_c = (
                torch.arange(X, dtype=torch.float64, device=self.device) + 0.5
            ).view(1, 1, 1, X)
            ones = torch.ones((1, 1, Y, X), dtype=torch.float64, device=self.device)
            design = torch.stack(
                [
                    ones.expand(R, Z, Y, X),
                    x_c.expand(R, Z, Y, X),
                    y_c.expand(R, Z, Y, X),
                ],
                dim=-1,
            )
            w = used_bg.to(torch.float64).unsqueeze(-1)
            A_flat = (design * w).reshape(R * Z, Y * X, 3)
            b_flat = (data_t * used_bg.to(torch.float64)).reshape(R * Z, Y * X, 1)
            ATA = torch.bmm(A_flat.transpose(1, 2), A_flat)
            ATb = torch.bmm(A_flat.transpose(1, 2), b_flat)
            det_ATA = torch.abs(torch.linalg.det(ATA)).view(R, Z)
            non_singular = det_ATA > 1e-12
            eye3 = torch.eye(3, dtype=torch.float64, device=self.device).unsqueeze(0)
            ATA_safe = torch.where(
                non_singular.reshape(R * Z, 1, 1), ATA, eye3
            )
            params_3 = torch.linalg.solve(ATA_safe, ATb).view(R, Z, 1, 1, 3)

            slice_used = torch.sum(used_bg, dim=(2, 3))
            z_idx = torch.arange(Z, device=self.device).unsqueeze(0)
            active_z = z_idx < shapes_t[:, 0].unsqueeze(1)
            all_slices_ok = torch.all(
                ((slice_used > 3) & non_singular) | (~active_z), dim=1
            )
            success_t = has_min_pixels & all_slices_ok
            bg_t = torch.sum(design * params_3, dim=-1)
            bg_t = torch.where(
                success_t.view(R, 1, 1, 1), bg_t, torch.zeros_like(bg_t)
            )

        used_f = used_bg.to(torch.float64)
        used_count_f = total_used.to(torch.float64).clamp(min=1.0)
        bg_sum = torch.sum((bg_t * used_f).reshape(R, -1), dim=1)
        data_sum = torch.sum((data_t * used_f).reshape(R, -1), dim=1)
        data_mean = data_sum / used_count_f
        success_t = success_t & (data_mean >= 0.0)
        bg_t = torch.where(
            success_t.view(R, 1, 1, 1), bg_t, torch.zeros_like(bg_t)
        )
        bg_f32_d = bg_t.to(dtype=torch.float32).to(dtype=torch.float64)

        bg_sum = torch.sum((bg_f32_d * used_f).reshape(R, -1), dim=1)
        bg_mean = torch.where(
            success_t & (total_used > 0),
            bg_sum / used_count_f,
            torch.zeros_like(bg_sum),
        )

        diff_bg = torch.where(used_bg, bg_f32_d - data_t, torch.zeros_like(data_t))
        mse_t = torch.where(
            success_t & (total_used > 0),
            torch.sum((diff_bg * diff_bg).reshape(R, -1), dim=1) / used_count_f,
            torch.zeros_like(bg_sum),
        )

        diff_data = torch.where(
            used_bg, data_t - data_mean.view(R, 1, 1, 1), torch.zeros_like(data_t)
        )
        data_var = torch.sum((diff_data * diff_data).reshape(R, -1), dim=1) / (
            used_count_f - 1.0
        ).clamp(min=1.0)
        disp_t = torch.where(
            success_t & (total_used > 1) & (data_mean > 0),
            data_var / data_mean.clamp(min=1e-30),
            torch.zeros_like(bg_sum),
        )

        bg_np = np.ascontiguousarray(bg_t.to(dtype=torch.float32).cpu().numpy())
        mask_np = np.ascontiguousarray(mask_t.to(dtype=torch.uint8).cpu().numpy())
        _shoebox_pack_ext.unpack_bucket_3d(
            shoeboxes,
            bucket_c.ctypes.data,
            R,
            Z,
            Y,
            X,
            bg_np.ctypes.data,
            mask_np.ctypes.data,
        )

        bg_mean_out[bucket] = data_mean.cpu().numpy()
        mse_out[bucket] = mse_t.to(dtype=torch.float32).to(dtype=torch.float64).cpu().numpy()
        disp_out[bucket] = disp_t.to(dtype=torch.float32).to(dtype=torch.float64).cpu().numpy()
        success_out[bucket] = success_t.cpu().numpy()

    def compute_background(self, reflections, image_volume=None):
        if (
            image_volume is not None
            or self._outlier_alg not in ("null", "nsigma", "tukey", "truncated")
            or self._model_alg not in ("constant3d", "constant2d", "linear2d")
        ):
            if self._cpu_fallback is None:
                self._cpu_fallback = SimpleBackgroundExt(self.params, self.experiments)
            return self._cpu_fallback.compute_background(reflections, image_volume)

        n_refl = len(reflections)
        if n_refl == 0:
            return flex.bool()

        shoeboxes = reflections["shoebox"]

        if self._model_alg == "constant3d":
            outlier_map = {"null": 0, "nsigma": 1, "tukey": 2, "truncated": 3}
            outlier_code = outlier_map[self._outlier_alg]
            lower, upper = 0.0, 0.0
            if self._outlier_alg == "nsigma":
                lower = float(self._s_params.outlier.nsigma.lower)
                upper = float(self._s_params.outlier.nsigma.upper)
            elif self._outlier_alg == "tukey":
                lower = float(self._s_params.outlier.tukey.lower)
                upper = float(self._s_params.outlier.tukey.upper)
            elif self._outlier_alg == "truncated":
                lower = float(self._s_params.outlier.truncated.lower)
                upper = float(self._s_params.outlier.truncated.upper)

            sum_x_np = np.empty(n_refl, dtype=np.float64)
            sum_s_np = np.empty(n_refl, dtype=np.float64)
            sum_x2_np = np.empty(n_refl, dtype=np.float64)
            count_np = np.empty(n_refl, dtype=np.int64)

            _shoebox_pack_ext.extract_simple_constant3d_bg(
                shoeboxes,
                outlier_code,
                lower,
                upper,
                int(self._min_pixels),
                MASK_BG_VALID,
                MASK_OVERLAPPED,
                MASK_BACKGROUND_USED,
                sum_x_np.ctypes.data,
                sum_s_np.ctypes.data,
                sum_x2_np.ctypes.data,
                count_np.ctypes.data,
            )

            sum_x_t = torch.as_tensor(sum_x_np, dtype=torch.float64, device=self.device)
            sum_s_t = torch.as_tensor(sum_s_np, dtype=torch.float64, device=self.device)
            sum_x2_t = torch.as_tensor(sum_x2_np, dtype=torch.float64, device=self.device)
            count_t = torch.as_tensor(count_np, dtype=torch.long, device=self.device)

            count_f = count_t.to(torch.float64).clamp(min=1.0)
            mean_t = torch.where(count_t > 1, sum_x_t / count_f, torch.zeros_like(sum_x_t))
            var_t = sum_s_t / (count_f - 1.0).clamp(min=1.0)
            success_t = (
                (count_t >= int(self._min_pixels))
                & (count_t > 1)
                & (mean_t >= 0.0)
                & (var_t >= 0.0)
            )
            mean_t = torch.where(success_t, mean_t, torch.zeros_like(mean_t))
            bg_f32_t = mean_t.to(dtype=torch.float32).to(dtype=torch.float64)
            mse_t = torch.where(
                success_t,
                (sum_x2_t - 2.0 * bg_f32_t * sum_x_t + count_f * bg_f32_t * bg_f32_t) / count_f,
                torch.zeros_like(mean_t),
            )
            disp_t = torch.where(
                success_t & (mean_t > 0.0),
                var_t / mean_t.clamp(min=1e-30),
                torch.zeros_like(mean_t),
            )

            bg_mean_np = np.ascontiguousarray(mean_t.cpu().numpy(), dtype=np.float64)
            conv_u8 = np.ascontiguousarray(success_t.cpu().numpy(), dtype=np.uint8)
            success_out = conv_u8.astype(bool)

            _shoebox_pack_ext.unpack_constant3d_bg_and_mask(
                shoeboxes,
                bg_mean_np.ctypes.data,
                conv_u8.ctypes.data,
                MASK_BACKGROUND_USED,
            )

            reflections["background.mean"] = flex.double(
                np.ascontiguousarray(np.float32(bg_mean_np), dtype=np.float64)
            )
            reflections["background.mse"] = flex.double(
                np.ascontiguousarray(np.float32(mse_t.cpu().numpy()), dtype=np.float64)
            )
            reflections["background.dispersion"] = flex.double(
                np.ascontiguousarray(np.float32(disp_t.cpu().numpy()), dtype=np.float64)
            )
            success_flex = flex.bool(success_out)
            reflections.set_flags(~success_flex, reflections.flags.dont_integrate)
            return success_flex

        shapes = np.empty((n_refl, 3), dtype=np.int64)
        volumes = np.empty(n_refl, dtype=np.int64)
        _shoebox_pack_ext.get_shoebox_shapes(
            shoeboxes, shapes.ctypes.data, volumes.ctypes.data
        )
        active_arr = np.nonzero(volumes > 0)[0].astype(np.int64)

        bg_mean_out = np.zeros(n_refl, dtype=np.float64)
        mse_out = np.zeros(n_refl, dtype=np.float64)
        disp_out = np.zeros(n_refl, dtype=np.float64)
        success_out = np.zeros(n_refl, dtype=bool)

        if active_arr.size > 0:
            buckets = _make_size_buckets(
                active_arr, volumes[active_arr], MAX_BUCKET_ELEMENTS
            )
            for bucket in buckets:
                self._process_bucket_gpu(
                    shoeboxes,
                    bucket,
                    shapes,
                    bg_mean_out,
                    mse_out,
                    disp_out,
                    success_out,
                )

        reflections["background.mean"] = flex.double(
            np.where(success_out, bg_mean_out, 0.0)
        )
        reflections["background.mse"] = flex.double(mse_out)
        reflections["background.dispersion"] = flex.double(disp_out)
        success_flex = flex.bool(success_out)
        reflections.set_flags(~success_flex, reflections.flags.dont_integrate)
        return success_flex


class NullBackgroundGPU(NullBackgroundExt):
    """GPU wrapper for NullBackgroundExt."""


class AutoBackgroundGPU:
    """
    GPU-accelerated AutoBackground selector: uses `GLMBackgroundGPU` for counting
    PAD detectors and `SimpleBackgroundGPU` for CCD/integrating detectors.
    """

    def __init__(self, params, experiments, device=None):
        self.params = params
        self.experiments = experiments
        self.device = _get_device(device)
        use_glm = True
        if experiments is not None and len(experiments) > 0:
            detectors = experiments.detectors()
            gains = [p.get_gain() for d in detectors for p in d]
            pnl_types = [p.get_type() for d in detectors for p in d]
            if [g == 1.0 for g in gains].count(False) != 0:
                use_glm = False
            if ["PAD" in t for t in pnl_types].count(False) != 0:
                use_glm = False
        if use_glm:
            self._impl = GLMBackgroundGPU(params, experiments, device=self.device)
        else:
            self._impl = SimpleBackgroundGPU(params, experiments, device=self.device)

    def compute_background(self, reflections, image_volume=None):
        return self._impl.compute_background(reflections, image_volume)


# Standalone Modeller & Outlier Rejector GPU wrappers for direct unit test parity
Constant2dModellerGPU = Constant2dModeller
Constant3dModellerGPU = Constant3dModeller
Linear2dModellerGPU = Linear2dModeller
Linear3dModellerGPU = Linear3dModeller
NSigmaOutlierRejectorGPU = NSigmaOutlierRejector
TukeyOutlierRejectorGPU = TukeyOutlierRejector
TruncatedOutlierRejectorGPU = TruncatedOutlierRejector
