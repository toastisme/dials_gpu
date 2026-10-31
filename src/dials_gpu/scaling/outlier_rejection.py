"""
GPU-accelerated outlier rejection algorithms for scaling using PyTorch.
"""

from __future__ import annotations

import copy
import logging

import numpy as np
import torch
from dxtbx import flumpy
from scitbx.array_family import flex

from dials.algorithms.scaling.Ih_table import IhTable
from dials.algorithms.scaling.outlier_rejection import (
     NormDevOutlierRejection,
    OutlierRejectionBase,
    SimpleNormDevOutlierRejection,
    TargetedOutlierRejection,
    determine_Esq_outlier_index_arrays,
)
from dials_scaling_ext import limit_outlier_weights

from .components import _get_device, _get_minimization_device
from .ih_table import (
    IhTableGPU,
    _np_to_tensor,
    calc_Ih_gpu,
    get_block_group_ids_gpu,
    sum_in_groups_gpu,
)

logger = logging.getLogger("dials")


class TargetedOutlierRejectionGPU(TargetedOutlierRejection):
    """GPU-accelerated TargetedOutlierRejection."""

    def __init__(self, Ih_table, zmax, target):
        assert (
            target.n_work_blocks == 1
        ), "Targeted outlier rejection requires a target Ih_table with nblocks = 1"
        self._target_Ih_table_block = target.blocked_data_list[0]
        calc_Ih_gpu(self._target_Ih_table_block, sync_to_cpu=True)
        OutlierRejectionBase.__init__(self, Ih_table, zmax)

    def _do_outlier_rejection(self):
        Ih_table = self._Ih_table_block
        dev = _get_minimization_device(Ih_table.size)
        target = self._target_Ih_table_block
        target_asu_Ih_dict = dict(
            zip(target.asu_miller_index, zip(target.Ih_values, target.variances))
        )
        target_Ih_value = np.zeros(Ih_table.size, dtype=np.float64)
        target_Ih_sigmasq = np.zeros(Ih_table.size, dtype=np.float64)
        for j, miller_idx in enumerate(Ih_table.asu_miller_index):
            vals = target_asu_Ih_dict.get(miller_idx)
            if vals is not None:
                target_Ih_value[j] = vals[0]
                target_Ih_sigmasq[j] = vals[1]

        nz_sel = target_Ih_value != 0.0
        if not np.any(nz_sel):
            return

        t_Ih_gpu = torch.from_numpy(target_Ih_value[nz_sel]).to(
            device=dev, dtype=torch.float64
        )
        t_var_gpu = torch.from_numpy(target_Ih_sigmasq[nz_sel]).to(
            device=dev, dtype=torch.float64
        )
        I_gpu = torch.from_numpy(
            np.ascontiguousarray(Ih_table.intensities[nz_sel], dtype=np.float64)
        ).to(device=dev, dtype=torch.float64)
        g_gpu = torch.from_numpy(
            np.ascontiguousarray(
                Ih_table.inverse_scale_factors[nz_sel], dtype=np.float64
            )
        ).to(device=dev, dtype=torch.float64)
        var_gpu = torch.from_numpy(
            np.ascontiguousarray(Ih_table.variances[nz_sel], dtype=np.float64)
        ).to(device=dev, dtype=torch.float64)

        norm_dev = (I_gpu - g_gpu * t_Ih_gpu) / torch.sqrt(
            var_gpu + (g_gpu * g_gpu) * t_var_gpu
        )
        outliers_sel = (torch.abs(norm_dev) > float(self._zmax)).cpu().numpy()
        outliers_isel = np.nonzero(nz_sel)[0][outliers_sel]

        outliers = np.full(self._Ih_table_block.size, False, dtype=bool)
        outliers[outliers_isel] = True

        self._outlier_indices = np.concatenate(
            [
                self._outlier_indices,
                self._Ih_table_block.Ih_table["loc_indices"].iloc[outliers].to_numpy(),
            ]
        )
        self._datasets = np.concatenate(
            [
                self._datasets,
                self._Ih_table_block.Ih_table["dataset_id"].iloc[outliers].to_numpy(),
            ]
        )


class SimpleNormDevOutlierRejectionGPU(SimpleNormDevOutlierRejection):
    """GPU-accelerated SimpleNormDevOutlierRejection."""

    def _do_outlier_rejection(self):
        block = self._Ih_table_block
        if block.size == 0:
            return
        dev = _get_minimization_device(block.size)
        group_ids_gpu = get_block_group_ids_gpu(block, dev)
        n_groups = block.n_groups

        I_gpu = _np_to_tensor(block.intensities, dev, dtype=torch.float64)
        g_gpu = _np_to_tensor(block.inverse_scale_factors, dev, dtype=torch.float64)
        w_gpu = _np_to_tensor(self.weights, dev, dtype=torch.float64)

        assert bool(torch.all(w_gpu > 0.0).item())

        wg = w_gpu * g_gpu
        wgIsum = sum_in_groups_gpu(wg * I_gpu, n_groups, group_ids_gpu, "per_refl")
        wg2sum = sum_in_groups_gpu(wg * g_gpu, n_groups, group_ids_gpu, "per_refl")

        zero_sel = wg2sum == 0.0
        safe_wg2sum = torch.where(zero_sel, torch.ones_like(wg2sum), wg2sum)

        norm_dev = (I_gpu - (g_gpu * wgIsum / safe_wg2sum)) / torch.sqrt(
            (1.0 / w_gpu) + torch.square(g_gpu / safe_wg2sum)
        )
        norm_dev = torch.where(
            zero_sel, torch.full_like(norm_dev, 1000.0), norm_dev
        )
        outliers = (torch.abs(norm_dev) > float(self._zmax)).cpu().numpy()

        self._outlier_indices = np.concatenate(
            [
                self._outlier_indices,
                block.Ih_table["loc_indices"].iloc[outliers].to_numpy(),
            ]
        )
        self._datasets = np.concatenate(
            [
                self._datasets,
                block.Ih_table["dataset_id"].iloc[outliers].to_numpy(),
            ]
        )


class NormDevOutlierRejectionGPU(NormDevOutlierRejection):
    """GPU-accelerated iterative leave-one-out NormDevOutlierRejection.

    Runs all leave-one-out normal deviation rounds on GPU tensors without
    repeatedly slicing scitbx.sparse.matrix or pandas DataFrames.
    """

    def _do_outlier_rejection(self):
        block = self._Ih_table_block
        if block.size == 0:
            return
        dev = _get_minimization_device(block.size)

        group_ids_gpu = get_block_group_ids_gpu(block, dev)
        n_groups = block.n_groups
        I_gpu = _np_to_tensor(block.intensities, dev, dtype=torch.float64)
        g_gpu = _np_to_tensor(block.inverse_scale_factors, dev, dtype=torch.float64)
        w_gpu = _np_to_tensor(self.weights, dev, dtype=torch.float64)
        loc_indices_np = block.Ih_table["loc_indices"].to_numpy()
        dataset_ids_np = block.Ih_table["dataset_id"].to_numpy()

        # Track indices into the initial block arrays
        active_idx_gpu = torch.arange(block.size, dtype=torch.int64, device=dev)
        zmax = float(self._zmax)

        outlier_block_indices = []

        while active_idx_gpu.shape[0] > 0:
            m = active_idx_gpu.shape[0]
            ones = torch.ones(m, dtype=torch.float64, device=dev)
            nh_refl = sum_in_groups_gpu(ones, n_groups, group_ids_gpu, "per_refl")
            gt2_mask = nh_refl > 2.0
            if not bool(gt2_mask.any().item()):
                break

            wg = w_gpu * g_gpu
            wgI = wg * I_gpu
            wg2 = wg * g_gpu
            wgIsum = sum_in_groups_gpu(wgI, n_groups, group_ids_gpu, "per_refl")
            wg2sum = sum_in_groups_gpu(wg2, n_groups, group_ids_gpu, "per_refl")

            wgIsum_others = wgIsum - wgI
            wg2sum_others = wg2sum - wg2

            # Guard against zero division on reflections with nh > 2
            zero_sel = (wg2sum_others == 0.0) & gt2_mask
            safe_wg2 = torch.where(
                wg2sum_others == 0.0, torch.ones_like(wg2sum_others), wg2sum_others
            )

            assert bool(torch.all(w_gpu[gt2_mask] > 0.0).item())
            norm_dev = (I_gpu - (g_gpu * wgIsum_others / safe_wg2)) / torch.sqrt(
                (1.0 / w_gpu) + (g_gpu * g_gpu) / safe_wg2
            )
            z_score = torch.where(
                gt2_mask, torch.abs(norm_dev), torch.zeros_like(norm_dev)
            )
            z_score = torch.where(
                zero_sel, torch.full_like(z_score, 1000.0), z_score
            )

            # Find max z_score per group
            max_z_group = torch.full((n_groups,), zmax, dtype=torch.float64, device=dev)
            max_z_group.scatter_reduce_(
                0, group_ids_gpu, z_score, reduce="amax", include_self=True
            )
            group_has_outlier = max_z_group > zmax
            if not bool(group_has_outlier.any().item()):
                break

            # Break ties in favor of the first reflection in each group
            is_max = (z_score == max_z_group[group_ids_gpu]) & (z_score > zmax)
            local_arange = torch.arange(m, dtype=torch.int64, device=dev)
            cand_idx = torch.where(
                is_max, local_arange, torch.full_like(local_arange, m)
            )
            winner_local = torch.full((n_groups,), m, dtype=torch.int64, device=dev)
            winner_local.scatter_reduce_(
                0, group_ids_gpu, cand_idx, reduce="amin", include_self=True
            )

            valid_winners = winner_local[winner_local < m]
            outlier_block_indices.append(
                active_idx_gpu[valid_winners].cpu().numpy()
            )

            # Keep other reflections from groups that had an outlier for the next round
            keep_mask = group_has_outlier[group_ids_gpu] & (
                local_arange != winner_local[group_ids_gpu]
            )
            if not bool(keep_mask.any().item()):
                break

            active_idx_gpu = active_idx_gpu[keep_mask]
            group_ids_gpu = group_ids_gpu[keep_mask]
            I_gpu = I_gpu[keep_mask]
            g_gpu = g_gpu[keep_mask]
            w_gpu = w_gpu[keep_mask]

        if outlier_block_indices:
            all_rej = np.concatenate(outlier_block_indices)
            self._outlier_indices = np.concatenate(
                [self._outlier_indices, loc_indices_np[all_rej]]
            )
            self._datasets = np.concatenate(
                [self._datasets, dataset_ids_np[all_rej]]
            )


def determine_outlier_index_arrays_gpu(
    Ih_table, method="standard", zmax=6.0, target=None
):
    """GPU-accelerated drop-in replacement for determine_outlier_index_arrays."""
    outlier_rej = None
    if method == "standard":
        outlier_rej = NormDevOutlierRejectionGPU(Ih_table, zmax)
    elif method == "simple":
        outlier_rej = SimpleNormDevOutlierRejectionGPU(Ih_table, zmax)
    elif method == "target":
        assert target is not None
        outlier_rej = TargetedOutlierRejectionGPU(Ih_table, zmax, target)
    elif method is not None:
        raise ValueError(f"Invalid choice of outlier rejection method: {method}")
    if not outlier_rej:
        return [flex.size_t([]) for _ in range(Ih_table.n_datasets)]
    outlier_rej.run()
    outlier_index_arrays = outlier_rej.final_outlier_arrays
    if Ih_table.n_datasets > 1:
        msg = "Combined outlier rejection has been performed across multiple datasets, \n"
    else:
        msg = "A round of outlier rejection has been performed, \n"
    n_outliers = sum(len(i) for i in outlier_index_arrays)
    msg += f"{n_outliers} outliers have been identified. \n"
    logger.info(msg)
    return outlier_index_arrays


def reject_outliers_gpu(reflection_table, experiment, method="standard", zmax=6.0):
    """GPU-accelerated drop-in replacement for reject_outliers."""
    assert "intensity" in reflection_table, "reflection table has no 'intensity' column"
    assert "variance" in reflection_table, "reflection table has no 'variance' column"

    if "inverse_scale_factor" not in reflection_table:
        reflection_table["inverse_scale_factor"] = flex.double(
            reflection_table.size(), 1.0
        )

    Ih_table = IhTableGPU(
        [reflection_table], experiment.crystal.get_space_group(), nblocks=1
    )
    outlier_indices = determine_outlier_index_arrays_gpu(
        Ih_table, method=method, zmax=zmax
    )[0]

    reflection_table.unset_flags(
        reflection_table.get_flags(reflection_table.flags.outlier_in_scaling),
        reflection_table.flags.outlier_in_scaling,
    )
    reflection_table.set_flags(
        flumpy.from_numpy(outlier_indices), reflection_table.flags.outlier_in_scaling
    )
    return reflection_table
