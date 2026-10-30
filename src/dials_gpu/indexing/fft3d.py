from __future__ import annotations

import logging
import math

import numpy as np
import torch
from cctbx import crystal, uctbx, xray
from libtbx import libtbx, phil
from scitbx import fftpack, matrix
from scitbx.array_family import flex
from scitbx.math import five_number_summary

from dials.algorithms import indexing
from dials.algorithms.indexing.basis_vector_search.fft3d import FFT3D, fft3d_phil_str
from dials.algorithms.indexing.basis_vector_search.utils import (
    group_vectors,
    is_approximate_integer_multiple,
)

logger = logging.getLogger(__name__)

# Reusable GPU buffers keyed by (device, n_points) to avoid repeated allocation
_FFT3D_GPU_CACHE: dict[tuple[torch.device, int], dict[str, torch.Tensor]] = {}


def _get_fft3d_buffers(device: torch.device, n_points: int) -> dict[str, torch.Tensor]:
    key = (device, n_points)
    bufs = _FFT3D_GPU_CACHE.get(key)
    if bufs is None:
        bufs = {
            "idx_grid": torch.full(
                (n_points, n_points, n_points), -1, dtype=torch.int32, device=device
            ),
            "offsets": torch.tensor(
                [
                    [1, 0, 0],
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, 1 * -1, 0],
                    [0, 0, 1],
                    [0, 0, -1],
                ],
                dtype=torch.int64,
                device=device,
            ),
            "grid_1d": torch.zeros(n_points**3, dtype=torch.float64, device=device),
        }
        _FFT3D_GPU_CACHE[key] = bufs
    return bufs


def map_centroids_to_reciprocal_space_grid_gpu(
    reciprocal_space_vectors: flex.vec3_double | np.ndarray | torch.Tensor,
    n_points: int,
    d_min: float,
    b_iso: float = 0.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU implementation of dials_algorithms_indexing_ext.map_centroids_to_reciprocal_space_grid.
    Returns (grid_t [n_points, n_points, n_points] float64, used_in_indexing_t [N] bool).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(reciprocal_space_vectors, torch.Tensor):
        rlp_t = reciprocal_space_vectors.to(device=device, dtype=torch.float64)
    elif isinstance(reciprocal_space_vectors, np.ndarray):
        rlp_t = torch.from_numpy(reciprocal_space_vectors).to(
            device=device, dtype=torch.float64
        )
    else:
        rlp_t = torch.from_numpy(reciprocal_space_vectors.as_numpy_array()).to(
            device=device, dtype=torch.float64
        )

    bufs = _get_fft3d_buffers(device, n_points)
    grid_1d = bufs["grid_1d"]
    grid_1d.zero_()

    if rlp_t.shape[0] == 0:
        return grid_1d.view(n_points, n_points, n_points), torch.zeros(
            0, dtype=torch.bool, device=device
        )

    v_len = torch.linalg.norm(rlp_t, dim=1)
    # Handle zero-length vectors safely
    valid_len = v_len > 0
    d_spacings = torch.where(valid_len, 1.0 / v_len, torch.zeros_like(v_len))
    valid_d = valid_len & (d_spacings >= d_min)

    rlgrid = 2.0 / (d_min * n_points)
    one_over_rlgrid = 1.0 / rlgrid
    half_n = n_points // 2

    scaled = rlp_t * one_over_rlgrid
    # Match scitbx::math::iround (half away from zero)
    coords = (
        torch.sign(scaled) * torch.floor(torch.abs(scaled) + 0.5)
    ).to(torch.int64) + half_n

    in_bounds = (coords >= 0).all(dim=1) & (coords < n_points).all(dim=1)
    used_gpu = valid_d & in_bounds

    valid_coords = coords[used_gpu]
    if valid_coords.shape[0] > 0:
        lin_idx = (
            valid_coords[:, 0] * n_points + valid_coords[:, 1]
        ) * n_points + valid_coords[:, 2]
        if b_iso != 0:
            weights = torch.exp(-b_iso * (v_len[used_gpu] ** 2) / 4.0)
            # In C++, sequential loop overwrites grid(coord) = T so last index wins
            m_valid = lin_idx.shape[0]
            order = torch.arange(m_valid, dtype=torch.int64, device=device)
            unique_lin, inv = torch.unique(lin_idx, return_inverse=True)
            last_pos = torch.zeros(
                unique_lin.shape[0], dtype=torch.int64, device=device
            )
            last_pos.scatter_reduce_(0, inv, order, reduce="amax", include_self=False)
            grid_1d[unique_lin] = weights[last_pos]
        else:
            grid_1d[lin_idx] = 1.0

    return grid_1d.view(n_points, n_points, n_points), used_gpu


def flood_fill_periodic_3d_gpu(
    active_mask: torch.Tensor,
    n_points: int,
    idx_grid: torch.Tensor,
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU implementation of cctbx.masks.flood_fill on a 3D periodic binary grid.
    Returns (centres_of_mass_frac [n_voids, 3] float64, grid_points_per_void [n_voids] int32).
    """
    device = active_mask.device
    active_coords = torch.nonzero(active_mask)
    k_active = active_coords.shape[0]
    if k_active == 0:
        return (
            torch.zeros((0, 3), dtype=torch.float64, device=device),
            torch.zeros(0, dtype=torch.int32, device=device),
        )

    idx_grid[active_coords[:, 0], active_coords[:, 1], active_coords[:, 2]] = (
        torch.arange(k_active, dtype=torch.int32, device=device)
    )
    nbr_coords = (active_coords[:, None, :] + offsets[None, :, :]) % n_points
    nbr_idx = idx_grid[nbr_coords[:, :, 0], nbr_coords[:, :, 1], nbr_coords[:, :, 2]]
    self_idx = torch.arange(k_active, dtype=torch.int64, device=device)
    nbr_idx = torch.where(nbr_idx >= 0, nbr_idx.to(torch.int64), self_idx[:, None])

    # Shiloach-Vishkin connected components on the sparse active graph
    labels = self_idx.clone()
    for _ in range(8):
        min_nbr = labels[nbr_idx].min(dim=1).values
        labels.scatter_reduce_(
            0, labels.clone(), min_nbr, reduce="amin", include_self=True
        )
        labels = labels[labels]
        labels = labels[labels]

    # Reset sparse index lookup table
    idx_grid[active_coords[:, 0], active_coords[:, 1], active_coords[:, 2]] = -1

    unique_roots, inverse_idx, counts = torch.unique(
        labels, return_inverse=True, return_counts=True
    )
    half_n = n_points // 2
    seed_coords = active_coords[labels]
    diff = active_coords - seed_coords
    unwrapped = torch.where(
        diff > half_n,
        active_coords - n_points,
        torch.where(diff < -half_n, active_coords + n_points, active_coords),
    ).to(torch.int64)

    n_voids = unique_roots.shape[0]
    sum_coords = torch.zeros((n_voids, 3), dtype=torch.int64, device=device)
    sum_coords.index_add_(0, inverse_idx, unwrapped)
    com = sum_coords.to(torch.float64) / counts[:, None].to(torch.float64)

    # Wrap into [-n_points, n_points] as in flood_fill.h::centres_of_mass()
    com = torch.fmod(com, float(n_points))
    com_frac = com / float(n_points)
    return com_frac, counts.to(torch.int32)


class FFT3DGPU(FFT3D):
    """
    GPU-accelerated 3D FFT basis vector search using PyTorch (torch.fft.fftn)
    and sparse 3D periodic Shiloach-Vishkin flood fill.
    """

    phil_scope = phil.parse(fft3d_phil_str)

    def __init__(self, max_cell, min_cell=3, params=None, *args, **kwargs):
        super().__init__(max_cell, min_cell=min_cell, params=params, *args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def find_basis_vectors(self, reciprocal_lattice_vectors):
        if self._params.reciprocal_space_grid.d_min is libtbx.Auto:
            max_cell = self._max_cell
            d_min = 5 * max_cell / self._n_points
            rlp_np = reciprocal_lattice_vectors.as_numpy_array()
            norms = np.linalg.norm(rlp_np, axis=1)
            valid_norms = norms[norms > 0]
            if len(valid_norms) == 0:
                raise indexing.DialsIndexError(
                    "Indexing failed: no non-zero reciprocal lattice vectors."
                )
            d_min = max(d_min, float(1.0 / np.max(valid_norms)))
            logger.info("Setting d_min: %.2f", d_min)
        else:
            d_min = self._params.reciprocal_space_grid.d_min

        if self._params.b_iso is libtbx.Auto:
            self._params.b_iso = -4 * d_min**2 * math.log(0.05)
            logger.debug("Setting b_iso = %.1f", self._params.b_iso)

        logger.info("FFT gridding: (%i,%i,%i)" % self._gridding)
        grid_t, used_gpu = map_centroids_to_reciprocal_space_grid_gpu(
            reciprocal_lattice_vectors,
            self._n_points,
            d_min,
            b_iso=self._params.b_iso,
            device=self.device,
        )
        n_centroids = int((grid_t > 0).sum().item())
        logger.info("Number of centroids used: %i", n_centroids)
        used_in_indexing = flex.bool(np.ascontiguousarray(used_gpu.cpu().numpy()))

        # Perform 3D C2C forward FFT on GPU
        fft_c = torch.fft.fftn(grid_t)
        grid_real_t = fft_c.real.square()
        del fft_c

        self.sites, self.volumes = self._find_peaks_gpu(grid_real_t, d_min)

        self.crystal_symmetry = crystal.symmetry(
            unit_cell=self._fft_cell, space_group_symbol="P1"
        )
        # In P1 with cubic cell (L, L, L, 90, 90, 90), sites_mod_short wraps fractional
        # coordinates into [-0.5, 0.5] and orthogonalizes by multiplying by L.
        xs = xray.structure(crystal_symmetry=self.crystal_symmetry)
        for i, site in enumerate(self.sites):
            xs.add_scatterer(xray.scatterer("C%i" % i, site=site))

        xs = xs.sites_mod_short()
        sites_cart = xs.sites_cart()
        lengths = flex.double([matrix.col(sc).length() for sc in sites_cart])
        perm = flex.sort_permutation(lengths)
        xs = xs.select(perm)
        volumes = self.volumes.select(perm)

        vectors = xs.sites_cart()
        norms = vectors.norms()
        sel = (norms > self._min_cell) & (norms < (2 * self._max_cell))
        vectors = vectors.select(sel)
        vectors = [matrix.col(v) for v in vectors]
        volumes = volumes.select(sel)

        vector_groups = group_vectors(vectors, volumes)
        vectors = [g.mean for g in vector_groups]
        volumes = flex.double(max(g.weights) for g in vector_groups)

        perm = flex.sort_permutation(volumes, reverse=True)
        volumes = volumes.select(perm)
        vectors = [vectors[i] for i in perm]

        for i, (v, volume) in enumerate(zip(vectors, volumes)):
            logger.debug(f"{i} {v.length()} {volume}")

        lengths = flex.double(v.length() for v in vectors)
        perm = flex.sort_permutation(lengths)

        unique_vectors = []
        unique_volumes = flex.double()
        for p in perm:
            v = vectors[p]
            is_unique = True
            for i, v_u in enumerate(unique_vectors):
                if (unique_volumes[i] > volumes[p]) and is_approximate_integer_multiple(
                    v_u, v
                ):
                    logger.debug(
                        "rejecting %s: integer multiple of %s", v.length(), v_u.length()
                    )
                    is_unique = False
                    break
            if is_unique:
                unique_vectors.append(v)
                unique_volumes.append(volumes[p])

        perm = flex.sort_permutation(unique_volumes, reverse=True)
        self.candidate_basis_vectors = [unique_vectors[i] for i in perm]
        return self.candidate_basis_vectors, used_in_indexing

    def _find_peaks_gpu(self, grid_real_t: torch.Tensor, d_min: float):
        mean_val = grid_real_t.mean()
        rmsd_val = torch.sqrt(torch.mean((grid_real_t - mean_val).square()))
        cutoff = self._params.rmsd_cutoff * rmsd_val
        # Note: in CPU FFT3D, values < cutoff are set to 0, >0 set to 1
        active_mask = (grid_real_t >= cutoff) & (grid_real_t > 0)

        cell_lengths = [self._n_points * d_min / 2 for _ in range(3)]
        self._fft_cell = uctbx.unit_cell(cell_lengths + [90] * 3)

        bufs = _get_fft3d_buffers(self.device, self._n_points)
        com_frac_t, counts_t = flood_fill_periodic_3d_gpu(
            active_mask,
            self._n_points,
            bufs["idx_grid"],
            bufs["offsets"],
        )
        n_voids = counts_t.shape[0]
        if n_voids < 4:
            raise indexing.DialsIndexError(
                "Indexing failed: fft3d peak search failed to find sufficient number of peaks."
            )

        counts_np = np.ascontiguousarray(counts_t.cpu().numpy(), dtype=np.int32)
        com_np = np.ascontiguousarray(com_frac_t.cpu().numpy(), dtype=np.float64)

        grid_points_per_void = flex.int(counts_np)
        outliers = flex.bool(n_voids, False)
        min_x, q1_x, med_x, q3_x, max_x = five_number_summary(grid_points_per_void)
        iqr_multiplier = 5
        iqr_x = q3_x - q1_x
        cut_x = iqr_multiplier * iqr_x
        outliers.set_selected(grid_points_per_void.as_double() > (q3_x + cut_x), True)
        isel = (
            grid_points_per_void
            > int(
                self._params.peak_volume_cutoff
                * flex.max(grid_points_per_void.select(~outliers))
            )
        ).iselection()

        all_sites = flex.vec3_double(com_np)
        sites = all_sites.select(isel)
        volumes = grid_points_per_void.select(isel)
        return sites, volumes
