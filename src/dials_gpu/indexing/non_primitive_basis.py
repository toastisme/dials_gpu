from __future__ import annotations

import logging

import numpy as np
import scitbx.matrix
import torch
from cctbx.uctbx.reduction_base import iteration_limit_exceeded
from rstbx.dps_core.cell_assessment import unit_cell_too_small
from rstbx.indexing_api import tools
from scitbx.array_family import flex

from dials.algorithms.indexing import DialsIndexError

logger = logging.getLogger(__name__)

# Precompute (3, 28) test vectors and (28,) moduli from rstbx.indexing_api.tools.R
_R_VECS_NP = np.array([t["vec"] for t in tools.R], dtype=np.int64).T  # (3, 28)
_R_MODS_NP = np.array([t["mod"] for t in tools.R], dtype=np.int64)  # (28,)
_R_CACHE: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _get_r_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    cached = _R_CACHE.get(device)
    if cached is None:
        cached = (
            torch.from_numpy(_R_VECS_NP).to(device=device),
            torch.from_numpy(_R_MODS_NP).to(device=device),
        )
        _R_CACHE[device] = cached
    return cached


def detect_gpu(miller_indices, threshold=0.9, device=None):
    """
    GPU/vectorized replacement for dials.algorithms.indexing.non_primitive_basis.detect.
    Evaluates all 28 systematic absence tests from rstbx.indexing_api.tools.R in one matrix multiply.
    """
    if isinstance(miller_indices, torch.Tensor):
        n_hkl = miller_indices.shape[0]
        if n_hkl == 0:
            return None
        dev = miller_indices.device
        vecs_t, mods_t = _get_r_tensors(dev)
        dots = miller_indices.to(torch.int64) @ vecs_t  # (N, 28)
        rem_zero = (torch.remainder(dots, mods_t[None, :]) == 0).sum(dim=0)
        fracs = rem_zero.to(torch.float64) / float(n_hkl)
        hits = torch.nonzero(fracs > threshold).squeeze(1)
        if hits.numel() > 0:
            idx = int(hits[0].item())
            test = tools.R[idx]
            logger.debug(
                "Detected exclusive presence of %dH %dK %dL = %dn, remainder 0",
                test["vec"][0],
                test["vec"][1],
                test["vec"][2],
                test["mod"],
            )
            return test["trans"]
        return None

    n_hkl = len(miller_indices)
    if n_hkl == 0:
        return None
    hkl_np = miller_indices.as_vec3_double().as_numpy_array().astype(np.int64)
    dots = hkl_np @ _R_VECS_NP  # (N, 28)
    rem_zero = (np.mod(dots, _R_MODS_NP[None, :]) == 0).sum(axis=0)
    fracs = rem_zero / float(n_hkl)
    hits = np.flatnonzero(fracs > threshold)
    if hits.size > 0:
        idx = int(hits[0])
        test = tools.R[idx]
        logger.debug(
            "Detected exclusive presence of %dH %dK %dL = %dn, remainder 0",
            test["vec"][0],
            test["vec"][1],
            test["vec"][2],
            test["mod"],
        )
        return test["trans"]
    return None


def correct_gpu(experiments, reflections, assign_indices, threshold=0.9):
    """
    GPU-accelerated replacement for dials.algorithms.indexing.non_primitive_basis.correct.
    """
    assert len(experiments.crystals()) == 1
    while True:
        sel = reflections["miller_index"] != (0, 0, 0)
        if sel.count(True) == 0:
            break
        t_mat = detect_gpu(reflections["miller_index"].select(sel), threshold=threshold)
        if t_mat is None:
            break

        crystal_model = experiments.crystals()[0]
        direct_matrix = scitbx.matrix.sqr(crystal_model.get_A()).inverse()
        m_mat = t_mat.inverse().transpose()
        new_direct_matrix = m_mat * direct_matrix

        crystal_model.set_A(new_direct_matrix.inverse())

        unit_cell_too_small(crystal_model.get_unit_cell())
        try:
            cb_op = crystal_model.get_unit_cell().change_of_basis_op_to_niggli_cell()
        except iteration_limit_exceeded as e:
            raise DialsIndexError(e)
        crystal_model.update(crystal_model.change_basis(cb_op))

        reflections["id"] = flex.int(len(reflections), -1)
        reflections.unset_flags(
            flex.bool(len(reflections), True), reflections.flags.indexed
        )
        assign_indices(reflections, experiments)
