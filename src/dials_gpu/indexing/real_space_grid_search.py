from __future__ import annotations

import logging
import math

import numpy as np
import torch
from libtbx import phil
from rstbx.array_family import flex
from rstbx.dps_core import SimpleSamplerTool
from scitbx import matrix

from dials.algorithms.indexing import DialsIndexError
from dials.algorithms.indexing.basis_vector_search.real_space_grid_search import (
    RealSpaceGridSearch,
    real_space_grid_search_phil_str,
)
from dials.algorithms.indexing.basis_vector_search.strategy import Strategy
from dials.algorithms.indexing.basis_vector_search.utils import group_vectors

logger = logging.getLogger(__name__)

_RSGS_DIRECTION_CACHE: dict[tuple[str, float], tuple[np.ndarray, torch.Tensor]] = {}


def _get_cached_hemisphere_directions(
    device: torch.device, characteristic_grid: float
) -> tuple[np.ndarray, torch.Tensor]:
    key = (str(device), float(characteristic_grid))
    if key not in _RSGS_DIRECTION_CACHE:
        sst = SimpleSamplerTool(characteristic_grid)
        sst.construct_hemisphere_grid(sst.incr)
        directions_np = np.array([d.dvec for d in sst.angles], dtype=np.float64)
        directions_t = torch.from_numpy(directions_np).to(
            device=device, dtype=torch.float64
        )
        _RSGS_DIRECTION_CACHE[key] = (directions_np, directions_t)
    return _RSGS_DIRECTION_CACHE[key]


class RealSpaceGridSearchGPU(RealSpaceGridSearch):
    """
    GPU-accelerated RealSpaceGridSearch using PyTorch matrix multiplication and cosine sum.
    Caches hemisphere sampling directions on GPU to avoid per-call SimpleSamplerTool overhead.
    """

    phil_scope = phil.parse(real_space_grid_search_phil_str)

    def __init__(self, max_cell, target_unit_cell, params=None, *args, **kwargs):
        Strategy.__init__(self, max_cell, params=params, *args, **kwargs)
        if target_unit_cell is None:
            raise DialsIndexError(
                "Target unit cell must be provided for real_space_grid_search"
            )
        self._target_unit_cell = target_unit_cell
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        directions_np, directions_t = _get_cached_hemisphere_directions(
            self.device, self._params.characteristic_grid
        )
        unique_dims = sorted(set(target_unit_cell.parameters()[:3]))
        self._directions_np = directions_np
        self._unique_dims = unique_dims
        self._sv_tensor = torch.cat([directions_t * l for l in unique_dims], dim=0)
        self._sv_array_cached = None
        self._sv_flex_cached = None

    @property
    def _sv_array(self):
        if self._sv_array_cached is None:
            self._sv_array_cached = np.vstack(
                [self._directions_np * l for l in self._unique_dims]
            )
        return self._sv_array_cached

    @property
    def _sv_flex(self):
        if self._sv_flex_cached is None:
            self._sv_flex_cached = flex.vec3_double(
                np.ascontiguousarray(self._sv_array, dtype=np.float64)
            )
        return self._sv_flex_cached

    def _compute_scores_tensor(self, rlv_t: torch.Tensor) -> torch.Tensor:
        n_sv = self._sv_tensor.shape[0]
        n_rlv = rlv_t.shape[0]
        two_pi_rlv_t = (2.0 * math.pi) * rlv_t.T
        if n_sv * n_rlv <= 8_000_000:
            return torch.cos(self._sv_tensor @ two_pi_rlv_t).sum(dim=1)
        chunk_size = max(1024, 8_000_000 // max(1, n_rlv))
        scores_t = torch.empty(n_sv, dtype=torch.float64, device=self.device)
        for start in range(0, n_sv, chunk_size):
            end = min(start + chunk_size, n_sv)
            scores_t[start:end] = torch.cos(
                self._sv_tensor[start:end] @ two_pi_rlv_t
            ).sum(dim=1)
        return scores_t

    def score_vectors(self, reciprocal_lattice_vectors):
        if isinstance(reciprocal_lattice_vectors, torch.Tensor):
            rlv_t = reciprocal_lattice_vectors.to(
                device=self.device, dtype=torch.float64
            )
        else:
            rlv_t = torch.from_numpy(
                reciprocal_lattice_vectors.as_numpy_array()
            ).to(device=self.device, dtype=torch.float64)

        scores_t = self._compute_scores_tensor(rlv_t)
        scores_np = np.ascontiguousarray(scores_t.cpu().numpy(), dtype=np.float64)
        return self._sv_flex, flex.double(scores_np)

    def find_basis_vectors(self, reciprocal_lattice_vectors):
        used_in_indexing = flex.bool(reciprocal_lattice_vectors.size(), True)
        logger.info("Indexing from %i reflections", used_in_indexing.count(True))

        if isinstance(reciprocal_lattice_vectors, torch.Tensor):
            rlv_t = reciprocal_lattice_vectors.to(
                device=self.device, dtype=torch.float64
            )
        else:
            rlv_t = torch.from_numpy(
                reciprocal_lattice_vectors.as_numpy_array()
            ).to(device=self.device, dtype=torch.float64)

        scores_t = self._compute_scores_tensor(rlv_t)
        order_t = torch.argsort(scores_t, descending=True, stable=True)

        # Transfer only the top slice first (group_vectors stops at max_groups=30,
        # which typically requires inspecting < 500 vectors). Fall back to full array if needed.
        top_k = min(order_t.shape[0], max(2048, self._params.max_vectors * 64))
        top_order = order_t[:top_k]
        sorted_sv_np = np.ascontiguousarray(
            self._sv_tensor[top_order].cpu().numpy(), dtype=np.float64
        )
        sorted_weights_np = np.ascontiguousarray(
            scores_t[top_order].cpu().numpy(), dtype=np.float64
        )

        vectors = [matrix.col(tuple(v)) for v in sorted_sv_np]
        weights = sorted_weights_np.tolist()
        groups = group_vectors(vectors, weights, max_groups=self._params.max_vectors)

        if len(groups) < self._params.max_vectors and top_k < order_t.shape[0]:
            full_sv_np = np.ascontiguousarray(
                self._sv_tensor[order_t].cpu().numpy(), dtype=np.float64
            )
            full_weights_np = np.ascontiguousarray(
                scores_t[order_t].cpu().numpy(), dtype=np.float64
            )
            vectors = flex.vec3_double(full_sv_np)
            weights = flex.double(full_weights_np)
            groups = group_vectors(
                vectors, weights, max_groups=self._params.max_vectors
            )

        unique_vectors = []
        unique_weights = []
        for g in groups:
            idx = flex.max_index(flex.double(g.weights))
            unique_vectors.append(g.vectors[idx])
            unique_weights.append(g.weights[idx])

        logger.info("Number of unique vectors: %i", len(unique_vectors))

        for v, w in zip(unique_vectors, unique_weights):
            logger.debug("%s %s %s", w, v.length(), str(v.elems))

        return unique_vectors, used_in_indexing
