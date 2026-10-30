from __future__ import annotations

import logging
import math
from itertools import combinations as iter_combinations

import numpy as np
import torch
from cctbx import sgtbx
from cctbx.uctbx.reduction_base import iteration_limit_exceeded
from dxtbx.model import Crystal

from dials.algorithms.indexing import DialsIndexError
from dials.algorithms.indexing.basis_vector_search.combinations import (
    filter_known_symmetry,
    filter_similar_orientations,
)

logger = logging.getLogger(__name__)


def candidate_orientation_matrices_gpu(
    basis_vectors, max_combinations=None, device=None
):
    """
    GPU-accelerated candidate orientation matrix triplet generation and angular filtering.
    Matches dials.algorithms.indexing.basis_vector_search.combinations.candidate_orientation_matrices.
    """
    n = len(basis_vectors)
    if n < 3:
        return []
    n = min(n, 100)
    basis_vectors = basis_vectors[:n]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idxs_np = np.array(list(iter_combinations(range(n), 3)), dtype=np.int64)
    norms_sq = (idxs_np**2).sum(axis=1)
    idxs_np = idxs_np[np.argsort(norms_sq, kind="stable")]

    if max_combinations is not None and max_combinations < len(idxs_np):
        idxs_np = idxs_np[:max_combinations]

    idxs = torch.from_numpy(idxs_np).to(device=device)

    half_pi = 0.5 * math.pi
    min_angle = 20.0 / 180.0 * math.pi

    bv_np = np.array([tuple(v) for v in basis_vectors], dtype=np.float64)
    bv = torch.from_numpy(bv_np).to(device=device, dtype=torch.float64)

    a_arr = bv[idxs[:, 0]]
    b_arr = bv[idxs[:, 1]]
    c_arr = bv[idxs[:, 2]]

    a_ns = (a_arr * a_arr).sum(dim=1)
    b_ns = (b_arr * b_arr).sum(dim=1)
    c_ns = (c_arr * c_arr).sum(dim=1)

    ab_dot = (a_arr * b_arr).sum(dim=1)
    angle_ab = torch.acos(
        torch.clamp(ab_dot / torch.sqrt(a_ns * b_ns), -1.0, 1.0)
    )
    m1 = (angle_ab >= min_angle) & ((math.pi - angle_ab) >= min_angle)
    a_arr, b_arr, c_arr = a_arr[m1], b_arr[m1], c_arr[m1]
    b_ns, c_ns, angle_ab = b_ns[m1], c_ns[m1], angle_ab[m1]

    flip_b = (angle_ab < half_pi)[:, None]
    b_arr = torch.where(flip_b, -b_arr, b_arr)

    a_cross_b = torch.linalg.cross(a_arr, b_arr, dim=1)
    acb_ns = (a_cross_b * a_cross_b).sum(dim=1)

    acb_c_dot = (a_cross_b * c_arr).sum(dim=1)
    angle_acb_c = torch.acos(
        torch.clamp(acb_c_dot / torch.sqrt(acb_ns * c_ns), -1.0, 1.0)
    )
    m2 = torch.abs(half_pi - angle_acb_c) >= min_angle
    a_arr, b_arr, c_arr = a_arr[m2], b_arr[m2], c_arr[m2]
    a_cross_b, b_ns, c_ns = a_cross_b[m2], b_ns[m2], c_ns[m2]

    bc_dot = (b_arr * c_arr).sum(dim=1)
    alpha = torch.acos(
        torch.clamp(bc_dot / torch.sqrt(b_ns * c_ns), -1.0, 1.0)
    )
    flip_c = (alpha < half_pi)[:, None]
    c_arr = torch.where(flip_c, -c_arr, c_arr)

    acb_dot_c = (a_cross_b * c_arr).sum(dim=1)
    flip_all = (acb_dot_c < 0)[:, None]
    a_arr = torch.where(flip_all, -a_arr, a_arr)
    b_arr = torch.where(flip_all, -b_arr, b_arr)
    c_arr = torch.where(flip_all, -c_arr, c_arr)

    a_list = a_arr.cpu().tolist()
    b_list = b_arr.cpu().tolist()
    c_list = c_arr.cpu().tolist()

    sg_p1 = sgtbx.space_group()
    for a_row, b_row, c_row in zip(a_list, b_list, c_list):
        model = Crystal(a_row, b_row, c_row, space_group=sg_p1)
        uc = model.get_unit_cell()
        try:
            cb_op_to_niggli = uc.change_of_basis_op_to_niggli_cell()
        except iteration_limit_exceeded as e:
            raise DialsIndexError(e)
        model = model.change_basis(cb_op_to_niggli)

        uc = model.get_unit_cell()
        params = uc.parameters()
        if uc.volume() > (params[0] * params[1] * params[2] / 100):
            yield model


__all__ = [
    "candidate_orientation_matrices_gpu",
    "filter_known_symmetry",
    "filter_similar_orientations",
]
