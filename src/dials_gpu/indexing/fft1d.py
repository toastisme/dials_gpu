from __future__ import annotations

import math

import numpy as np
import torch
from libtbx import phil
from rstbx.array_family import flex as rstbx_flex
from rstbx.dps_core import Direction
from rstbx.indexing_api.lattice import DPS_primitive_lattice
from rstbx.indexing_api.nearest_neighbor import neighbor_analysis
from rstbx.indexing_api.sampling import HemisphereSampler, SimpleSampler
from scitbx import fftpack, matrix
from scitbx.array_family import flex

from dials.algorithms.indexing.basis_vector_search.fft1d import FFT1D, fft1d_phil_str

_SMOOTH_LUT_CACHE: dict[torch.device, torch.Tensor] = {}
_HEMISPHERE_CACHE: dict[
    tuple[torch.device, float], tuple[rstbx_flex.Direction, torch.Tensor, torch.Tensor]
] = {}


def _get_smooth_lut(device: torch.device, max_m: int = 8192) -> torch.Tensor:
    lut = _SMOOTH_LUT_CACHE.get(device)
    if lut is None or lut.shape[0] <= max_m:
        size = max(8192, max_m + 1024)
        vals = [1] + [fftpack.adjust_gridding(i, 5) for i in range(1, size)]
        lut = torch.tensor(vals, dtype=torch.int64, device=device)
        _SMOOTH_LUT_CACHE[device] = lut
    return lut


def _get_hemisphere_grid_gpu(
    sampler: SimpleSampler, incr: float, device: torch.device
) -> tuple[rstbx_flex.Direction, torch.Tensor, torch.Tensor]:
    key = (device, float(incr))
    cached = _HEMISPHERE_CACHE.get(key)
    if cached is not None:
        return cached

    angles = SimpleSampler.construct_hemisphere_grid(sampler, incr)
    n_psi = int(0.5 + math.pi / (2.0 * incr))
    psi_step = (math.pi / 2.0) / n_psi
    j = torch.arange(1, n_psi + 1, dtype=torch.float64, device=device)
    psi = j * psi_step
    sin_psi = torch.sin(psi)
    cos_psi = torch.cos(psi)
    n_phi = (0.5 + 2.0 * math.pi * sin_psi / incr).to(torch.int64)
    ring_idx = torch.repeat_interleave(torch.arange(n_psi, device=device), n_phi)
    offsets = torch.cumsum(n_phi, dim=0) - n_phi
    k = torch.arange(ring_idx.shape[0], device=device) - offsets[ring_idx]
    phi = k.to(torch.float64) * (2.0 * math.pi / n_phi[ring_idx].to(torch.float64))
    sp = sin_psi[ring_idx]
    cp = cos_psi[ring_idx]
    dvecs = torch.stack([sp * torch.cos(phi), sp * torch.sin(phi), cp], dim=1)
    pole = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64, device=device)
    dvecs_t = torch.cat([pole, dvecs], dim=0)
    psis_t = torch.cat(
        [torch.zeros(1, dtype=torch.float64, device=device), psi[ring_idx]]
    )
    _HEMISPHERE_CACHE[key] = (angles, dvecs_t, psis_t)
    return angles, dvecs_t, psis_t


def _batch_directional_fft_gpu(
    rlp_t: torch.Tensor,
    dvecs_t: torch.Tensor,
    amax: float,
    granularity: float = 5.0,
    f0_cutoff: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched 1D directional projection, histogramming, and 1D RFFT power spectrum peak search on GPU.
    Matches rstbx::Directional_FFT to machine precision (~1e-15).
    """
    device = rlp_t.device
    proj = dvecs_t @ rlp_t.T  # (D, N)
    pmin = proj.min(dim=1).values
    pmax = proj.max(dim=1).values
    p_range = pmax - pmin
    raw_m = 1 + (p_range * (amax * granularity)).to(torch.int64)
    max_raw = int(raw_m.max().item()) if raw_m.numel() > 0 else 1
    smooth_lut = _get_smooth_lut(device, max_raw)
    m_all = smooth_lut[raw_m]
    delta_p = p_range / torch.clamp((m_all - 1).to(torch.float64), min=1.0)
    bins_idx = torch.floor((proj - pmin[:, None]) / delta_p[:, None] + 0.5).to(
        torch.int64
    )

    d_count, n_count = proj.shape
    kvals = torch.zeros(d_count, dtype=torch.float64, device=device)
    reals = torch.zeros(d_count, dtype=torch.float64, device=device)

    unique_m = torch.unique(m_all)
    ones = torch.ones((d_count, n_count), dtype=torch.float64, device=device)
    for m_val in unique_m.tolist():
        mask = m_all == m_val
        sub_idx = torch.clamp(bins_idx[mask], 0, m_val - 1)
        b_size = sub_idx.shape[0]
        hist = torch.zeros((b_size, m_val), dtype=torch.float64, device=device)
        hist.scatter_add_(1, sub_idx, ones[:b_size])
        ps = torch.abs(torch.fft.rfft(hist, dim=1))
        if ps.shape[1] > f0_cutoff:
            sub_ps = ps[:, f0_cutoff:]
            sub_kval, sub_kmax = sub_ps.max(dim=1)
            kmax = sub_kmax + f0_cutoff
            kvals[mask] = sub_kval
            reals[mask] = kmax.to(torch.float64) / (m_val * delta_p[mask])
    return kvals, reals


class HemisphereSamplerGPU(HemisphereSampler):
    """
    GPU-accelerated HemisphereSampler that evaluates all ~7,500-25,000 directional
    1D FFTs in parallel on the GPU using PyTorch.
    """

    def __init__(self, max_grid, characteristic_grid, quick_grid, device=None):
        super().__init__(max_grid, characteristic_grid, quick_grid)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def hemisphere(self, ai, max_cell, size=30, cutoff_divisor=4.0):
        self.max_cell_input = max_cell
        rlp_np = ai.getXyzData().as_numpy_array()
        rlp_t = torch.from_numpy(rlp_np).to(device=self.device, dtype=torch.float64)

        if self.quick_grid >= self.incr:
            quick_angles, quick_dvecs_t, quick_psis_t = _get_hemisphere_grid_gpu(
                self, self.quick_grid, self.device
            )
            kvals_q, _ = _batch_directional_fft_gpu(
                rlp_t, quick_dvecs_t, amax=ai.amax, granularity=ai.granularity
            )
            kval_cutoff_q = rlp_t.shape[0] / 10.0
            pass_q = torch.nonzero(kvals_q > kval_cutoff_q).squeeze(1)
            if pass_q.numel() >= 3:
                sub_kvals_q = kvals_q[pass_q]
                order_q = torch.argsort(sub_kvals_q, descending=True, stable=True)
                top_q = pass_q[
                    order_q[: min(self.target_size_for_quick_grid_answer_vector, len(order_q))]
                ]
            else:
                top_q = pass_q

            full_angles, full_dvecs_t, _ = _get_hemisphere_grid_gpu(
                self, self.incr, self.device
            )
            if top_q.numel() > 0:
                q_dvecs = quick_dvecs_t[top_q]
                q_psis = quick_psis_t[top_q]
                eq_mask = torch.abs((math.pi / 2.0) - q_psis) < 0.0001
                if eq_mask.any():
                    q_dvecs = torch.cat([q_dvecs, -q_dvecs[eq_mask]], dim=0)
                kn = int(2 * self.second_round_sampling)
                dists = torch.cdist(q_dvecs, full_dvecs_t)
                nn_dists, nn_idx = torch.topk(
                    dists, k=min(kn, full_dvecs_t.shape[0]), dim=1, largest=False
                )
                accept_flag = torch.zeros(
                    full_dvecs_t.shape[0], dtype=torch.bool, device=self.device
                )
                accept_flag[nn_idx[nn_dists < self.quick_grid]] = True
                active_indices = torch.nonzero(accept_flag).squeeze(1)
                target_dvecs_t = full_dvecs_t[active_indices]
                active_indices_cpu = active_indices.cpu().tolist()
                self.angles = rstbx_flex.Direction(
                    [full_angles[idx] for idx in active_indices_cpu]
                )
            else:
                target_dvecs_t = torch.zeros(
                    (0, 3), dtype=torch.float64, device=self.device
                )
                self.angles = rstbx_flex.Direction()
        else:
            self.angles, target_dvecs_t, _ = _get_hemisphere_grid_gpu(
                self, self.incr, self.device
            )

        # Evaluate target directions in parallel on GPU
        kval_cutoff = rlp_t.shape[0] / cutoff_divisor
        if target_dvecs_t.shape[0] > 0:
            kvals, reals = _batch_directional_fft_gpu(
                rlp_t, target_dvecs_t, amax=ai.amax, granularity=ai.granularity
            )
            winner_mask = (reals < self.max_cell_input) & (kvals > kval_cutoff)
            winner_indices = torch.nonzero(winner_mask).squeeze(1).cpu().tolist()
        else:
            winner_indices = []

        # Populate exact C++ Direction objects ONLY for the winning directions
        hemisphere_solutions = rstbx_flex.Direction()
        hemisphere_solutions.reserve(size)
        for i in winner_indices:
            d = ai.fft_result(self.angles[i])
            if d.real < self.max_cell_input and d.kval > kval_cutoff:
                hemisphere_solutions.append(d)

        if hemisphere_solutions.size() < 3:
            unrefined_basis_vectors = hemisphere_solutions
        else:
            kvals_flex = flex.double(
                [hemisphere_solutions[x].kval for x in range(len(hemisphere_solutions))]
            )
            perm = flex.sort_permutation(kvals_flex, True)
            perm_idx = 0
            unique_clusters = 0
            hemisphere_solutions_sort = rstbx_flex.Direction()
            while perm_idx < len(perm) and unique_clusters < size:
                test_item = hemisphere_solutions[perm[perm_idx]]
                direction_ok = True
                for list_item in hemisphere_solutions_sort:
                    distance = math.sqrt(
                        (list_item.dvec[0] - test_item.dvec[0]) ** 2
                        + (list_item.dvec[1] - test_item.dvec[1]) ** 2
                        + (list_item.dvec[2] - test_item.dvec[2]) ** 2
                    )
                    if distance < 0.087:
                        direction_ok = False
                        break
                if direction_ok:
                    unique_clusters += 1
                    hemisphere_solutions_sort.append(test_item)
                perm_idx += 1
            unrefined_basis_vectors = hemisphere_solutions_sort

        grid_refined_vectors = self.refine_top_solutions_by_grid_search(
            old_solutions=unrefined_basis_vectors, ai=ai
        )
        ai.setSolutions(grid_refined_vectors)


class FFT1DGPU(FFT1D):
    """
    GPU-accelerated 1D FFT (DPS) basis vector search using PyTorch.
    """

    phil_scope = phil.parse(fft1d_phil_str)

    def __init__(self, max_cell, params=None, *args, **kwargs):
        super().__init__(max_cell, params=params, *args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def find_basis_vectors(self, reciprocal_lattice_vectors):
        used_in_indexing = flex.bool(reciprocal_lattice_vectors.size(), True)
        assert len(reciprocal_lattice_vectors) > 7

        dps = DPS_primitive_lattice(
            max_cell=self._max_cell,
            recommended_grid_sampling_rad=self._params.characteristic_grid,
            horizon_phil=self._hardcoded_phil,
        )

        if dps.max_cell is None:
            nn = neighbor_analysis(reciprocal_lattice_vectors)
            dps.max_cell = nn.max_cell

        if dps.recommended_grid_sampling_rad is None:
            rossmann_suggestion = 0.029
            norms = reciprocal_lattice_vectors.norms()
            naive_obs_highest_resolution = 1.0 / flex.max(norms)
            characteristic_grid = naive_obs_highest_resolution / dps.max_cell
            new_suggestion = 2.0 * characteristic_grid
            dps.recommended_grid_sampling_rad = min(
                rossmann_suggestion, new_suggestion
            )

        dps.setMaxcell(dps.max_cell)
        dps.setXyzData(reciprocal_lattice_vectors)

        sampler = HemisphereSamplerGPU(
            max_grid=0.029,
            characteristic_grid=dps.recommended_grid_sampling_rad,
            quick_grid=0.016,
            device=self.device,
        )
        sampler.hemisphere(dps, dps.max_cell, size=30, cutoff_divisor=4.0)

        solutions = dps.getSolutions()
        candidate_basis_vectors = [matrix.col(s.bvec()) for s in solutions]
        return candidate_basis_vectors, used_in_indexing
