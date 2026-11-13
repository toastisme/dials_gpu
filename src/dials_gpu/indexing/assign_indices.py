from __future__ import annotations

import math

import numpy as np
import torch
import triton
import triton.language as tl
from cctbx.array_family import flex
from dxtbx import flumpy

import dials_algorithms_indexing_ext as ext
from dials.algorithms.indexing import DialsIndexError
from dials.algorithms.indexing.assign_indices import (
    AssignIndicesGlobal,
    AssignIndicesLocal,
    AssignIndicesStrategy,
)


@triton.jit
def _disambiguate_dup_hkl_kernel(
    d_refs_ptr,
    g_starts_ptr,
    counts_ptr,
    phi_ptr,
    best_d2_ptr,
    cryst_out_ptr,
    pi_4,
):
    g = tl.program_id(0)
    start = tl.load(g_starts_ptr + g)
    cnt = tl.load(counts_ptr + g)
    for i in range(cnt):
        i_ref = tl.load(d_refs_ptr + start + i)
        ci = tl.load(cryst_out_ptr + i_ref, volatile=True)
        for j in range(i + 1, cnt):
            if ci >= 0:
                j_ref = tl.load(d_refs_ptr + start + j)
                cj = tl.load(cryst_out_ptr + j_ref, volatile=True)
                if ci == cj:
                    phi_i = tl.load(phi_ptr + i_ref)
                    phi_j = tl.load(phi_ptr + j_ref)
                    if tl.abs(phi_i - phi_j) <= pi_4:
                        d2_i = tl.load(best_d2_ptr + i_ref)
                        d2_j = tl.load(best_d2_ptr + j_ref)
                        if d2_j < d2_i:
                            tl.store(cryst_out_ptr + i_ref, -1)
                            ci = -1
                        else:
                            tl.store(cryst_out_ptr + j_ref, -1)


_LAST_RLP_PHI_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]] = {}


def assign_indices_global_gpu(
    rlps_np: np.ndarray,
    phi_np: np.ndarray,
    ub_matrices_np: np.ndarray,
    tolerance: float = 0.3,
    device: torch.device | None = None,
    _cached_tensors: tuple[torch.Tensor, torch.Tensor] | None = None,
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

    if _cached_tensors is not None:
        rlp_t, phi_t = _cached_tensors
    else:
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

        d_keys = s_keys[dup_mask]
        d_refs = s_refs[dup_mask]
        if d_refs.numel() > 1:
            u_keys, inv, counts = torch.unique_consecutive(
                d_keys, return_inverse=True, return_counts=True
            )
            num_g = int(u_keys.shape[0])
            if num_g > 0 and device.type == "cuda":
                counts_i32 = counts.to(torch.int32)
                g_starts_i32 = (torch.cumsum(counts, dim=0) - counts).to(torch.int32)
                _disambiguate_dup_hkl_kernel[(num_g,)](
                    d_refs,
                    g_starts_i32,
                    counts_i32,
                    phi_t,
                    best_d2,
                    cryst_out,
                    math.pi / 4.0,
                    num_warps=1,
                )
                invalidated = d_refs[cryst_out[d_refs] < 0]
                miller_out[invalidated] = 0
            elif num_g > 0:
                max_c = int(counts.max().item())
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
        self._last_cache = None

    def __call__(self, reflections, experiments, d_min=None):
        reciprocal_lattice_points = reflections["rlp"]
        n_total = len(reflections)
        reflections["miller_index"] = flex.miller_index(n_total, (0, 0, 0))
        id_col = reflections["id"]
        all_unindexed = d_min is None and (id_col == -1).count(False) == 0

        if all_unindexed:
            isel = None
            rlps = reciprocal_lattice_points
            phi = reflections["xyzobs.mm.value"].parts()[2]
            imgset_ids = reflections["imageset_id"]
        else:
            if d_min is not None:
                d_spacings = 1 / reciprocal_lattice_points.norms()
                sel = (d_spacings > d_min) & (id_col == -1)
            else:
                sel = id_col == -1
            isel = sel.iselection()
            rlps = reciprocal_lattice_points.select(isel)
            phi = reflections["xyzobs.mm.value"].select(isel).parts()[2]
            imgset_ids = reflections["imageset_id"].select(sel)

        crystals = experiments.crystals()
        n_cryst = len(crystals)
        imgsets = experiments.imagesets()
        n_imgsets = len(imgsets)

        for i_imgset, imgset in enumerate(imgsets):
            if n_imgsets == 1 and (imgset_ids == i_imgset).count(False) == 0:
                sel_imgset = None
                sub_rlps = rlps
                sub_phi = phi
            else:
                sel_imgset = imgset_ids == i_imgset
                sub_rlps = rlps.select(sel_imgset)
                sub_phi = phi.select(sel_imgset)

            if len(sub_rlps) * n_cryst >= self.gpu_threshold:
                ub_np = np.array(
                    [cm.get_A() for cm in crystals], dtype=np.float64
                ).reshape(-1, 3, 3)
                rlps_np = sub_rlps.as_numpy_array()
                phi_np = sub_phi.as_numpy_array()
                cached_tensors = None
                n_sub = len(rlps_np)
                if n_sub >= 16:
                    sig = (
                        n_sub,
                        float(rlps_np[0, 0]),
                        float(rlps_np[n_sub // 2, 1]),
                        float(rlps_np[-1, 2]),
                        float(phi_np[0]),
                        float(phi_np[-1]),
                    )
                    if (
                        self._last_cache is not None
                        and self._last_cache[0] == sig
                        and np.array_equal(self._last_cache[1], rlps_np)
                        and np.array_equal(self._last_cache[2], phi_np)
                    ):
                        cached_tensors = self._last_cache[3]
                    else:
                        rlp_t = torch.from_numpy(rlps_np).to(
                            device=self.device, dtype=torch.float64
                        )
                        phi_t = torch.from_numpy(phi_np).to(
                            device=self.device, dtype=torch.float64
                        )
                        cached_tensors = (rlp_t, phi_t)
                        self._last_cache = (
                            sig,
                            rlps_np.copy(),
                            phi_np.copy(),
                            cached_tensors,
                        )
                miller_np, cryst_np = assign_indices_global_gpu(
                    rlps_np,
                    phi_np,
                    ub_np,
                    tolerance=self._tolerance,
                    device=self.device,
                    _cached_tensors=cached_tensors,
                )
                miller_indices = flex.miller_index(
                    flumpy.from_numpy(np.ascontiguousarray(miller_np[:, 0])),
                    flumpy.from_numpy(np.ascontiguousarray(miller_np[:, 1])),
                    flumpy.from_numpy(np.ascontiguousarray(miller_np[:, 2])),
                )
                crystal_ids = flumpy.from_numpy(np.ascontiguousarray(cryst_np))
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

            if isel is None and sel_imgset is None:
                reflections["miller_index"] = miller_indices
                reflections["id"] = expt_ids
            else:
                if isel is None:
                    sub_isel = sel_imgset.iselection()
                elif sel_imgset is None:
                    sub_isel = isel
                else:
                    sub_isel = isel.select(sel_imgset)
                reflections["miller_index"].set_selected(sub_isel, miller_indices)
                reflections["id"].set_selected(sub_isel, expt_ids)
            nz_mask = reflections["miller_index"] != (0, 0, 0)
            reflections.set_flags(nz_mask, reflections.flags.indexed)
            reflections["id"].set_selected(~nz_mask, -1)


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
