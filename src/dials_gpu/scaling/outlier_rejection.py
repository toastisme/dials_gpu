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


def limit_outlier_weights_gpu(block, device=None) -> np.ndarray:
    """Vectorized group-median weight clamping matching C++ `limit_outlier_weights`
    without constructing or iterating over `scitbx.sparse.matrix` `h_index_matrix`."""
    if block.size == 0 or block.n_groups == 0:
        w_np = block.weights.copy()
        block._gpu_clamped_weights = None
        return w_np
    dev = _get_minimization_device(block.size, device)
    group_ids_gpu = get_block_group_ids_gpu(block, dev)
    w_gpu = _np_to_tensor(block.weights, dev, dtype=torch.float64)
    n_groups = block.n_groups

    p1 = torch.argsort(w_gpu)
    p2 = torch.argsort(group_ids_gpu[p1], stable=True)
    sorted_w = w_gpu[p1[p2]]
    counts = torch.bincount(group_ids_gpu, minlength=n_groups)
    ends = torch.cumsum(counts, dim=0)
    starts = ends - counts
    safe_counts = torch.clamp(counts, min=1)
    max_idx = w_gpu.numel() - 1
    mid1 = torch.clamp(starts + (safe_counts - 1) // 2, max=max_idx)
    mid2 = torch.clamp(starts + safe_counts // 2, max=max_idx)
    med_g = 0.5 * (sorted_w[mid1] + sorted_w[mid2])
    w_clamped = torch.minimum(w_gpu, 10.0 * med_g[group_ids_gpu])
    block._gpu_clamped_weights = w_clamped
    return w_clamped.cpu().numpy()


def determine_Esq_outlier_index_arrays_gpu(Ih_table, experiment, emax=10.0):
    """Fast cached version of `determine_Esq_outlier_index_arrays` that caches
    the resolution binner, `1 / epsilons`, and bin assignments across outlier
    rejection rounds on the same `IhTableBlock`."""
    from cctbx import crystal, miller, uctbx

    block = Ih_table.Ih_table_blocks[0]
    if block.size == 0:
        return [np.array([], dtype=np.uint64)] * Ih_table.n_datasets

    uc = experiment.crystal.get_unit_cell()
    uc_key = tuple(round(x, 8) for x in uc.parameters())
    sg_key = Ih_table.space_group.type().hall_symbol()
    cache = getattr(block, "_gpu_quasi_norm_cache", None)

    if cache is None or cache[0] != (block.size, uc_key, sg_key):
        miller_set = miller.set(
            crystal_symmetry=crystal.symmetry(
                unit_cell=uc,
                space_group=Ih_table.space_group,
                assert_is_compatible_unit_cell=False,
            ),
            indices=block.asu_miller_index,
            anomalous_flag=False,
        )
        n = block.size
        if n > 20000:
            n_refl_shells = 20
        elif n > 15000:
            n_refl_shells = 15
        else:
            n_refl_shells = 10
        d_star_sq = miller_set.d_star_sq().data()
        d_star_sq_max = flex.max(d_star_sq)
        d_star_sq_min = flex.min(d_star_sq)
        span = d_star_sq_max - d_star_sq_min
        if span <= 0.0:
            return determine_Esq_outlier_index_arrays(Ih_table, experiment, emax=emax)
        d_star_sq_max += span * 1e-6
        d_star_sq_min -= span * 1e-6
        if d_star_sq_min <= 0.0:
            return determine_Esq_outlier_index_arrays(Ih_table, experiment, emax=emax)
        d_star_sq_step = (d_star_sq_max - d_star_sq_min) / n_refl_shells
        binner = miller_set.setup_binner_d_star_sq_step(
            d_min=uctbx.d_star_sq_as_d(d_star_sq_min),
            d_max=uctbx.d_star_sq_as_d(d_star_sq_max),
            d_star_sq_step=d_star_sq_step,
            auto_binning=False,
        )
        if binner.n_bin_d_too_large_or_small() != 0:
            return determine_Esq_outlier_index_arrays(Ih_table, experiment, emax=emax)
        inv_eps_np = 1.0 / flumpy.to_numpy(miller_set.epsilons().data().as_double())
        used_bins = list(binner.range_used())
        bin_idx_np = np.zeros(n, dtype=np.int64)
        bin_counts = np.zeros(len(used_bins), dtype=np.int64)
        for k, i_bin in enumerate(used_bins):
            sel_k = flumpy.to_numpy(binner.selection(i_bin))
            bin_idx_np[sel_k] = k
            bin_counts[k] = int(np.count_nonzero(sel_k))
        cache = ((block.size, uc_key, sg_key), binner, inv_eps_np, bin_idx_np, bin_counts)
        block._gpu_quasi_norm_cache = cache

    _, binner, inv_eps_np, bin_idx_np, bin_counts = cache
    raw_iobs = block.intensities / block.inverse_scale_factors
    pos_iobs = np.maximum(raw_iobs, 0.0)
    f_sq_over_eps = pos_iobs * inv_eps_np
    sums = np.bincount(bin_idx_np, weights=f_sq_over_eps, minlength=len(bin_counts))
    means = np.where(bin_counts > 0, sums / np.maximum(bin_counts, 1), 0.0)
    mean_flex = flumpy.from_numpy(np.ascontiguousarray(means, dtype=np.float64))
    norm_interp = flumpy.to_numpy(binner.interpolate(mean_flex, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        sel_np = (raw_iobs / norm_interp) > (emax**2)

    n_e2_outliers = int(np.count_nonzero(sel_np))
    if n_e2_outliers:
        logger.info(
            f"{n_e2_outliers} outliers identified from normalised intensity analysis (E\xb2 > {(emax**2)})"
        )
    outlier_indices = block.Ih_table["loc_indices"].to_numpy()[sel_np]
    if Ih_table.n_datasets == 1:
        return [outlier_indices]
    datasets = block.Ih_table["dataset_id"].to_numpy()[sel_np]
    return [outlier_indices[datasets == i] for i in range(Ih_table.n_datasets)]


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

    def __init__(self, Ih_table, zmax):
        OutlierRejectionBase.__init__(self, Ih_table, zmax)
        self.weights = limit_outlier_weights_gpu(self._Ih_table_block)

    def _do_outlier_rejection(self):
        block = self._Ih_table_block
        if block.size == 0:
            return
        dev = _get_minimization_device(block.size)
        group_ids_gpu = get_block_group_ids_gpu(block, dev)
        n_groups = block.n_groups

        I_gpu = _np_to_tensor(block.intensities, dev, dtype=torch.float64)
        g_gpu = _np_to_tensor(block.inverse_scale_factors, dev, dtype=torch.float64)
        w_cached = getattr(block, "_gpu_clamped_weights", None)
        if w_cached is not None and w_cached.device == dev and w_cached.shape[0] == block.size:
            w_gpu = w_cached
        else:
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

    def __init__(self, Ih_table, zmax):
        OutlierRejectionBase.__init__(self, Ih_table, zmax)
        self.weights = limit_outlier_weights_gpu(self._Ih_table_block)

    def _do_outlier_rejection(self):
        block = self._Ih_table_block
        if block.size == 0:
            return
        dev = _get_minimization_device(block.size)

        group_ids_gpu = get_block_group_ids_gpu(block, dev)
        n_groups = block.n_groups
        I_gpu = _np_to_tensor(block.intensities, dev, dtype=torch.float64)
        g_gpu = _np_to_tensor(block.inverse_scale_factors, dev, dtype=torch.float64)
        w_cached = getattr(block, "_gpu_clamped_weights", None)
        if w_cached is not None and w_cached.device == dev and w_cached.shape[0] == block.size:
            w_gpu = w_cached
        else:
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
