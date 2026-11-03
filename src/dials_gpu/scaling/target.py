"""
GPU-accelerated scaling target functions (ScalingTargetGPU, ScalingTargetFixedIHGPU)
computing residuals, functional, adjoint gradients, and Jacobians on the GPU.
"""

from __future__ import annotations

import numpy as np
import torch
from dxtbx import flumpy
from scitbx import sparse

from dials.algorithms.scaling.scaling_restraints import ScalingRestraintsCalculator
from dials.algorithms.scaling.target_function import ScalingTarget, ScalingTargetFixedIH
from dials.array_family import flex
from dials_scaling_ext import calc_jacobian, row_multiply

from .components import _get_device, _get_minimization_device
from .ih_table import (
    _np_to_tensor,
    calc_Ih_gpu,
    get_block_Static_tensors_gpu,
    sum_in_groups_gpu,
)


def _get_block_device(block, device=None) -> torch.device:
    if device is not None:
        return _get_device(device)
    g_gpu = getattr(block, "_gpu_scales", None)
    if g_gpu is not None and g_gpu.shape[0] == block.size:
        return g_gpu.device
    return _get_minimization_device(block.size, device)


def _extract_derivs_blocks_gpu(block, device: torch.device):
    """Return list of (r0, r1, c0, c1, D_gpu) blocks and total n_params."""
    derivs_blocks = getattr(block, "_gpu_derivs_blocks", None)
    if derivs_blocks is not None:
        n_params = int(getattr(block, "_gpu_n_params", 0))
        if derivs_blocks and derivs_blocks[0][4].device != device:
            derivs_blocks = [
                (r0, r1, c0, c1, d.to(device=device))
                for r0, r1, c0, c1, d in derivs_blocks
            ]
        return derivs_blocks, n_params

    # Fallback when called from unit tests where block.derivatives was set directly
    d_sparse = block.derivatives
    if d_sparse is None:
        raise ValueError("block.derivatives is None and _gpu_derivs_blocks is not set")
    n_rows = d_sparse.n_rows
    n_cols = d_sparse.n_cols
    if n_rows == 0 or n_cols == 0:
        d_gpu = torch.empty((n_rows, n_cols), dtype=torch.float64, device=device)
    else:
        d_np = np.asarray(
            flumpy.to_numpy(d_sparse.as_dense_matrix()), dtype=np.float64
        ).reshape(n_rows, n_cols)
        d_gpu = torch.from_numpy(d_np).to(device=device, dtype=torch.float64)
    return [(0, n_rows, 0, n_cols, d_gpu)], n_cols


def _get_block_state_gpu(block, device: torch.device, fixed_Ih: bool = False):
    """Fetch or compute (group_ids, I_gpu, w_gpu, g_gpu, Ih_gpu, sumgsq_gpu) on GPU."""
    group_ids_gpu, I_gpu, w_gpu = get_block_Static_tensors_gpu(block, device)
    g_gpu = getattr(block, "_gpu_scales", None)
    if g_gpu is None or g_gpu.shape[0] != block.size or g_gpu.device != device:
        g_gpu = _np_to_tensor(block.inverse_scale_factors, device, dtype=torch.float64)

    Ih_gpu = getattr(block, "_gpu_Ih_values", None)
    if Ih_gpu is None or Ih_gpu.shape[0] != block.size or Ih_gpu.device != device:
        Ih_gpu = _np_to_tensor(block.Ih_values, device, dtype=torch.float64)

    if fixed_Ih:
        sumgsq_gpu = None
    else:
        sumgsq_gpu = getattr(block, "_gpu_sumgsq", None)
        if (
            sumgsq_gpu is None
            or sumgsq_gpu.shape[0] != block.n_groups
            or sumgsq_gpu.device != device
        ):
            sumgsq_gpu = sum_in_groups_gpu(
                g_gpu * g_gpu * w_gpu, block.n_groups, group_ids_gpu, "per_group"
            )
    return group_ids_gpu, I_gpu, w_gpu, g_gpu, Ih_gpu, sumgsq_gpu


class ScalingTargetGPU(ScalingTarget):
    """GPU-accelerated ScalingTarget using PyTorch."""

    def rmsds(self, Ih_table, apm):
        R = 0.0
        n = 0
        for block in Ih_table.blocked_data_list:
            dev = _get_block_device(block)
            _, I_gpu, w_gpu, g_gpu, Ih_gpu, _ = _get_block_state_gpu(
                block, dev, fixed_Ih=True
            )
            res = I_gpu - g_gpu * Ih_gpu
            R += float(torch.sum(res * res * w_gpu).item())
            n += block.size
        if self.param_restraints:
            restraints = ScalingRestraintsCalculator.calculate_restraints(apm)
            if restraints:
                R += float(np.sum(restraints[0]))
            else:
                self.param_restraints = False
        self._rmsds = [(R / n) ** 0.5]
        return self._rmsds

    @staticmethod
    def calculate_residuals(Ih_table):
        dev = _get_block_device(Ih_table)
        _, I_gpu, _, g_gpu, Ih_gpu, _ = _get_block_state_gpu(
            Ih_table, dev, fixed_Ih=True
        )
        res_gpu = I_gpu - g_gpu * Ih_gpu
        return res_gpu.cpu().numpy()

    @staticmethod
    def calculate_gradients_gpu(
        Ih_table, device: torch.device | str | None = None
    ) -> tuple[float, torch.Tensor]:
        """Compute (functional, gradients_gpu) directly on GPU using adjoint reduction."""
        dev = _get_block_device(Ih_table, device)
        group_ids_gpu, I_gpu, w_gpu, g_gpu, Ih_gpu, sumgsq_gpu = _get_block_state_gpu(
            Ih_table, dev, fixed_Ih=False
        )
        derivs_blocks, n_params = _extract_derivs_blocks_gpu(Ih_table, dev)

        resids = I_gpu - g_gpu * Ih_gpu
        functional = float(torch.sum(resids * resids * w_gpu).item())

        prefactor = -2.0 * w_gpu * resids
        dIh = (I_gpu - 2.0 * g_gpu * Ih_gpu) * w_gpu

        # Adjoint group factor u_h = sum_{j in h}(prefactor_j * g_j) / sumgsq_h
        sum_pg = sum_in_groups_gpu(
            prefactor * g_gpu, Ih_table.n_groups, group_ids_gpu, "per_group"
        )
        u_refl = (sum_pg / sumgsq_gpu)[group_ids_gpu]

        # Combined per-reflection sensitivity w.r.t. g_i:
        adj = prefactor * Ih_gpu + u_refl * dIh

        grad_parts = []
        for r0, r1, c0, c1, D_j in derivs_blocks:
            if D_j.shape[1] == 0:
                continue
            grad_parts.append(torch.mv(D_j.t(), adj[r0:r1]))

        if grad_parts:
            grad_gpu = torch.cat(grad_parts, dim=0)
        else:
            grad_gpu = torch.zeros(n_params, dtype=torch.float64, device=dev)

        return functional, grad_gpu

    @staticmethod
    def calculate_gradients(Ih_table):
        _, grad_gpu = ScalingTargetGPU.calculate_gradients_gpu(Ih_table)
        grad_np = np.ascontiguousarray(grad_gpu.cpu().numpy(), dtype=np.float64)
        return flumpy.from_numpy(grad_np)

    @staticmethod
    def calculate_dense_jacobian_gpu(
        Ih_table, device: torch.device | str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute (residuals_gpu, jacobian_gpu, weights_gpu) on the GPU."""
        dev = _get_block_device(Ih_table, device)
        group_ids_gpu, I_gpu, w_gpu, g_gpu, Ih_gpu, sumgsq_gpu = _get_block_state_gpu(
            Ih_table, dev, fixed_Ih=False
        )
        derivs_blocks, n_params = _extract_derivs_blocks_gpu(Ih_table, dev)

        resids_gpu = I_gpu - g_gpu * Ih_gpu
        dIh = (I_gpu - 2.0 * g_gpu * Ih_gpu) * w_gpu
        n_groups = Ih_table.n_groups

        # dIh_by_dpi has shape (n_groups, n_params)
        if (
            len(derivs_blocks) == 1
            and derivs_blocks[0][0] == 0
            and derivs_blocks[0][1] == Ih_table.size
            and derivs_blocks[0][2] == 0
            and derivs_blocks[0][3] == n_params
            and Ih_table.size >= 8000
        ):
            D_0 = derivs_blocks[0][4]
            dIh_by_dpi = (
                sum_in_groups_gpu(
                    dIh[:, None] * D_0, n_groups, group_ids_gpu, "per_group"
                )
                / sumgsq_gpu[:, None]
            )
            J_gpu = (
                -D_0 * Ih_gpu[:, None] - g_gpu[:, None] * dIh_by_dpi[group_ids_gpu, :]
            )
            return resids_gpu, J_gpu, w_gpu

        dIh_by_dpi = torch.zeros((n_groups, n_params), dtype=torch.float64, device=dev)
        sumgsq_refl = sumgsq_gpu[group_ids_gpu]
        if Ih_table.size >= 8000:
            dIh_scaled = dIh / sumgsq_refl
            for r0, r1, c0, c1, D_j in derivs_blocks:
                if c1 > c0:
                    term = dIh_scaled[r0:r1, None] * D_j
                    dIh_by_dpi[:, c0:c1] = sum_in_groups_gpu(
                        term, n_groups, group_ids_gpu[r0:r1], "per_group"
                    )
        else:
            for r0, r1, c0, c1, D_j in derivs_blocks:
                if c1 > c0:
                    term = (dIh[r0:r1, None] * D_j) / sumgsq_refl[r0:r1, None]
                    dIh_by_dpi[:, c0:c1] = sum_in_groups_gpu(
                        term, n_groups, group_ids_gpu[r0:r1], "per_group"
                    )

        # Jacobian J_{i, k} = -D_{i, k} * Ih_i - g_i * dIh_by_dpi[h(i), k]
        J_gpu = torch.zeros(
            (Ih_table.size, n_params), dtype=torch.float64, device=dev
        )
        for r0, r1, c0, c1, D_j in derivs_blocks:
            if c1 > c0:
                J_gpu[r0:r1, c0:c1] -= D_j * Ih_gpu[r0:r1, None]
        J_gpu -= g_gpu[:, None] * dIh_by_dpi[group_ids_gpu, :]

        return resids_gpu, J_gpu, w_gpu

    @classmethod
    def compute_functional_gradients(cls, Ih_table):
        functional, grad_gpu = cls.calculate_gradients_gpu(Ih_table)
        Ih_table.derivatives = None
        Ih_table._gpu_derivs_blocks = None
        grad_np = np.ascontiguousarray(grad_gpu.cpu().numpy(), dtype=np.float64)
        return functional, flumpy.from_numpy(grad_np)

    @classmethod
    def compute_residuals(cls, Ih_table):
        dev = _get_block_device(Ih_table)
        _, I_gpu, w_gpu, g_gpu, Ih_gpu, _ = _get_block_state_gpu(
            Ih_table, dev, fixed_Ih=True
        )
        res_gpu = I_gpu - g_gpu * Ih_gpu
        res_np = np.ascontiguousarray(res_gpu.cpu().numpy(), dtype=np.float64)
        w_np = np.ascontiguousarray(w_gpu.cpu().numpy(), dtype=np.float64)
        return flumpy.from_numpy(res_np), flumpy.from_numpy(w_np)

    @classmethod
    def compute_residuals_and_dense_jacobian_gpu(cls, Ih_table):
        res_gpu, J_gpu, w_gpu = cls.calculate_dense_jacobian_gpu(Ih_table)
        Ih_table.derivatives = None
        Ih_table._gpu_derivs_blocks = None
        return res_gpu, J_gpu, w_gpu


class ScalingTargetFixedIHGPU(ScalingTargetGPU, ScalingTargetFixedIH):
    """GPU-accelerated ScalingTargetFixedIH for targeted scaling."""

    @staticmethod
    def calculate_gradients_gpu(
        Ih_table, device: torch.device | str | None = None
    ) -> tuple[float, torch.Tensor]:
        dev = _get_block_device(Ih_table, device)
        _, I_gpu, w_gpu, g_gpu, Ih_gpu, _ = _get_block_state_gpu(
            Ih_table, dev, fixed_Ih=True
        )
        derivs_blocks, n_params = _extract_derivs_blocks_gpu(Ih_table, dev)

        resids = I_gpu - g_gpu * Ih_gpu
        functional = float(torch.sum(resids * resids * w_gpu).item())
        adj = -2.0 * resids * w_gpu * Ih_gpu

        grad_parts = []
        for r0, r1, c0, c1, D_j in derivs_blocks:
            if D_j.shape[1] == 0:
                continue
            grad_parts.append(torch.mv(D_j.t(), adj[r0:r1]))

        if grad_parts:
            grad_gpu = torch.cat(grad_parts, dim=0)
        else:
            grad_gpu = torch.zeros(n_params, dtype=torch.float64, device=dev)
        return functional, grad_gpu

    @staticmethod
    def calculate_gradients(Ih_table):
        _, grad_gpu = ScalingTargetFixedIHGPU.calculate_gradients_gpu(Ih_table)
        grad_np = np.ascontiguousarray(grad_gpu.cpu().numpy(), dtype=np.float64)
        return flumpy.from_numpy(grad_np)

    @staticmethod
    def calculate_dense_jacobian_gpu(
        Ih_table, device: torch.device | str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = _get_block_device(Ih_table, device)
        _, I_gpu, w_gpu, g_gpu, Ih_gpu, _ = _get_block_state_gpu(
            Ih_table, dev, fixed_Ih=True
        )
        derivs_blocks, n_params = _extract_derivs_blocks_gpu(Ih_table, dev)

        resids_gpu = I_gpu - g_gpu * Ih_gpu
        J_gpu = torch.zeros((Ih_table.size, n_params), dtype=torch.float64, device=dev)
        for r0, r1, c0, c1, D_j in derivs_blocks:
            if c1 > c0:
                J_gpu[r0:r1, c0:c1] = -Ih_gpu[r0:r1, None] * D_j
        return resids_gpu, J_gpu, w_gpu

    @staticmethod
    def calculate_jacobian(Ih_table):
        if Ih_table.derivatives is not None:
            return row_multiply(
                Ih_table.derivatives, flumpy.from_numpy(-1.0 * Ih_table.Ih_values)
            )
        _, J_gpu, _ = ScalingTargetFixedIHGPU.calculate_dense_jacobian_gpu(Ih_table)
        J_np = np.ascontiguousarray(J_gpu.cpu().numpy(), dtype=np.float64)
        j_sparse = sparse.matrix(J_np.shape[0], J_np.shape[1])
        j_sparse.assign_block(flumpy.from_numpy(J_np), 0, 0)
        return j_sparse
