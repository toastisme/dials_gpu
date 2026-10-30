from __future__ import annotations

import math

import numpy as np
import torch
from cctbx.array_family import flex

import dials_algorithms_indexing_ext as ext
from dials.algorithms.indexing import DialsIndexError
from dials.algorithms.indexing.assign_indices import (
    AssignIndicesGlobal,
    AssignIndicesLocal,
    AssignIndicesStrategy,
)


def assign_indices_global_gpu(
    rlps_np: np.ndarray,
    phi_np: np.ndarray,
    ub_matrices_np: np.ndarray,
    tolerance: float = 0.3,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    GPU implementation of dials::algorithms::AssignIndices (index.h).
    Args:
        rlps_np: (N, 3) float64 array of reciprocal lattice points
        phi_np: (N,) float64 array of rotation angles (radians)
        ub_matrices_np: (L, 3, 3) float64 array of crystal A = UB matrices
        tolerance: HKL fractional distance cutoff (default 0.3)
    Returns:
        (miller_indices [N, 3] int32, crystal_ids [N] int32)
    """
    n_refs = rlps_np.shape[0]
    n_lattices = ub_matrices_np.shape[0]
    if n_refs == 0 or n_lattices == 0:
        return (
            np.zeros((n_refs, 3), dtype=np.int32),
            np.full(n_refs, -1, dtype=np.int32),
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rlp_t = torch.from_numpy(rlps_np).to(device=device, dtype=torch.float64)
    phi_t = torch.from_numpy(phi_np).to(device=device, dtype=torch.float64)
    ub_t = torch.from_numpy(ub_matrices_np).to(device=device, dtype=torch.float64)
    a_inv_t = torch.linalg.inv(ub_t)  # (L, 3, 3)

    # hkl_f[l, i, :] = rlp[i, :] @ A_inv[l].T -> (L, N, 3)
    hkl_f = torch.einsum("nj,lkj->lnk", rlp_t, a_inv_t)
    # Match scitbx::math::iround (half away from zero)
    hkl_i = (torch.sign(hkl_f) * torch.floor(torch.abs(hkl_f) + 0.5)).to(torch.int32)
    diff = hkl_f - hkl_i.to(torch.float64)
    lengths_sq = (diff * diff).sum(dim=2)  # (L, N)

    # Choose best lattice per reflection
    best_d2, best_lattice = lengths_sq.min(dim=0)  # (N,)
    ref_arange = torch.arange(n_refs, device=device)
    best_hkl = hkl_i[best_lattice, ref_arange]  # (N, 3)

    tol_sq = tolerance * tolerance
    non_zero_hkl = (best_hkl != 0).any(dim=1)
    valid_mask = (best_d2 <= tol_sq) & non_zero_hkl

    miller_out = torch.where(
        valid_mask[:, None], best_hkl, torch.zeros_like(best_hkl)
    )
    cryst_out = torch.where(
        valid_mask,
        best_lattice.to(torch.int32),
        torch.full((n_refs,), -1, dtype=torch.int32, device=device),
    )

    # Vectorized GPU duplicate HKL disambiguation within |phi_i - phi_j| <= pi/4 on same crystal
    valid_idx = torch.nonzero(valid_mask).squeeze(1)
    if valid_idx.numel() > 1:
        v_hkl = miller_out[valid_idx].to(torch.int64)
        keys = (
            ((v_hkl[:, 0] + 32768) << 32)
            | ((v_hkl[:, 1] + 32768) << 16)
            | (v_hkl[:, 2] + 32768)
        )
        order = torch.argsort(keys, stable=True)
        s_keys = keys[order]
        s_refs = valid_idx[order]

        dup_mask = torch.zeros_like(s_keys, dtype=torch.bool)
        dup_mask[:-1] |= s_keys[:-1] == s_keys[1:]
        dup_mask[1:] |= s_keys[1:] == s_keys[:-1]

        if dup_mask.any():
            d_keys = s_keys[dup_mask]
            d_refs = s_refs[dup_mask]
            u_keys, inv, counts = torch.unique_consecutive(
                d_keys, return_inverse=True, return_counts=True
            )
            max_c = int(counts.max().item())
            num_g = u_keys.shape[0]
            g_starts = torch.cumsum(counts, dim=0) - counts
            col_idx = torch.arange(d_keys.shape[0], device=device) - g_starts[inv]

            g_refs = torch.full(
                (num_g, max_c), -1, dtype=torch.int64, device=device
            )
            g_refs[inv, col_idx] = d_refs
            clamped_refs = g_refs.clamp(min=0)
            g_cryst = torch.where(
                g_refs >= 0,
                cryst_out[clamped_refs],
                torch.full_like(g_refs, -1, dtype=torch.int32),
            )
            g_phi = torch.where(
                g_refs >= 0,
                phi_t[clamped_refs],
                torch.zeros((num_g, max_c), dtype=torch.float64, device=device),
            )
            g_d2 = torch.where(
                g_refs >= 0,
                best_d2[clamped_refs],
                torch.zeros((num_g, max_c), dtype=torch.float64, device=device),
            )

            pi_4 = math.pi / 4.0
            for i in range(max_c):
                for j in range(i + 1, max_c):
                    ci = g_cryst[:, i]
                    cj = g_cryst[:, j]
                    active = (
                        (ci >= 0)
                        & (ci == cj)
                        & (torch.abs(g_phi[:, i] - g_phi[:, j]) <= pi_4)
                    )
                    if active.any():
                        j_better = active & (g_d2[:, j] < g_d2[:, i])
                        i_better = active & ~j_better
                        g_cryst[:, i] = torch.where(j_better, -1, g_cryst[:, i])
                        g_cryst[:, j] = torch.where(i_better, -1, g_cryst[:, j])

            updated_cryst = g_cryst[inv, col_idx]
            cryst_out[d_refs] = updated_cryst
            invalidated = d_refs[updated_cryst < 0]
            miller_out[invalidated] = 0

    return (
        np.ascontiguousarray(miller_out.cpu().numpy(), dtype=np.int32),
        np.ascontiguousarray(cryst_out.cpu().numpy(), dtype=np.int32),
    )


class AssignIndicesGlobalGPU(AssignIndicesGlobal):
    """
    GPU-accelerated replacement for AssignIndicesGlobal.
    """

    def __init__(self, tolerance=0.3, device=None, gpu_threshold=4096):
        super().__init__(tolerance=tolerance)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.gpu_threshold = gpu_threshold

    def __call__(self, reflections, experiments, d_min=None):
        reciprocal_lattice_points = reflections["rlp"]
        reflections["miller_index"] = flex.miller_index(len(reflections), (0, 0, 0))
        if d_min is not None:
            d_spacings = 1 / reciprocal_lattice_points.norms()
            inside_resolution_limit = d_spacings > d_min
        else:
            inside_resolution_limit = flex.bool(reciprocal_lattice_points.size(), True)
        sel = inside_resolution_limit & (reflections["id"] == -1)
        isel = sel.iselection()
        rlps = reciprocal_lattice_points.select(isel)
        refs = reflections.select(isel)
        phi = refs["xyzobs.mm.value"].parts()[2]

        crystals = experiments.crystals()
        n_cryst = len(crystals)
        imgset_ids = reflections["imageset_id"].select(sel)

        for i_imgset, imgset in enumerate(experiments.imagesets()):
            sel_imgset = imgset_ids == i_imgset
            sub_rlps = rlps.select(sel_imgset)
            sub_phi = phi.select(sel_imgset)

            if len(sub_rlps) * n_cryst >= self.gpu_threshold:
                ub_np = np.array(
                    [cm.get_A() for cm in crystals], dtype=np.float64
                ).reshape(-1, 3, 3)
                miller_np, cryst_np = assign_indices_global_gpu(
                    sub_rlps.as_numpy_array(),
                    sub_phi.as_numpy_array(),
                    ub_np,
                    tolerance=self._tolerance,
                    device=self.device,
                )
                miller_indices = flex.miller_index(miller_np)
                crystal_ids = flex.int(cryst_np)
            else:
                ub_matrices = flex.mat3_double([cm.get_A() for cm in crystals])
                result = ext.AssignIndices(
                    sub_rlps,
                    sub_phi,
                    ub_matrices,
                    tolerance=self._tolerance,
                )
                miller_indices = result.miller_indices()
                crystal_ids = result.crystal_ids()

            expt_ids = flex.int(crystal_ids.size(), -1)
            for i_cryst, cryst in enumerate(crystals):
                sel_cryst = crystal_ids == i_cryst
                for i_expt in experiments.where(crystal=cryst, imageset=imgset):
                    expt_ids.set_selected(sel_cryst, i_expt)
                    if experiments[i_expt].identifier:
                        reflections.experiment_identifiers()[i_expt] = experiments[
                            i_expt
                        ].identifier

            sub_isel = isel.select(sel_imgset)
            reflections["miller_index"].set_selected(sub_isel, miller_indices)
            reflections["id"].set_selected(sub_isel, expt_ids)
            reflections.set_flags(
                reflections["miller_index"] != (0, 0, 0), reflections.flags.indexed
            )
            reflections["id"].set_selected(reflections["miller_index"] == (0, 0, 0), -1)


class AssignIndicesLocalGPU(AssignIndicesLocal):
    """
    AssignIndicesLocal wrapper maintaining full compatibility with DIALS local index assignment.
    """

    pass


__all__ = [
    "AssignIndicesStrategy",
    "AssignIndicesGlobal",
    "AssignIndicesGlobalGPU",
    "AssignIndicesLocal",
    "AssignIndicesLocalGPU",
    "assign_indices_global_gpu",
]
