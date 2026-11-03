"""
GPU-accelerated profile/summation intensity combination for scaling using PyTorch.
"""

from __future__ import annotations

import logging

import boost_adaptbx.boost.python
import numpy as np
import torch
from cctbx import miller
from dxtbx import flumpy

from dials.algorithms.scaling.combine_intensities import (
    MultiDatasetIntensityCombiner,
    SingleDatasetIntensityCombiner,
    _determine_Imids,
    _determine_inverse_partiality,
    _get_filter_selection,
    _make_reflection_table_from_scaler,
    combine_intensities,
    fast_merging_stats,
    map_indices_to_asu,
)
from dials.algorithms.scaling.scaling_utilities import DialsMergingStatisticsError
from dials.array_family import flex
from dials.util import tabulate

from .components import _get_device
from .ih_table import _pack_hkl

miller_ext = boost_adaptbx.boost.python.import_ext("cctbx_miller_ext")
logger = logging.getLogger("dials")


def _fast_merging_stats_presorted(array):
    """Calculate merging stats on a miller.array already sorted by packed_indices."""
    assert array.sigmas() is not None
    positive_sel = array.sigmas() > 0
    i_over_sigma_sel = (array.data() / array.sigmas()) > 1.0
    array = array.select(positive_sel & i_over_sigma_sel)
    if not array.size():
        return -1.0, -1.0
    merge_ext = miller_ext.merge_equivalents_obs(
        array.indices(), array.data(), array.sigmas(), use_internal_variance=True
    )
    r_meas = merge_ext.r_meas
    cc_one_half = miller.compute_cc_one_half(unmerged=array, return_n_refl=False)
    return r_meas, cc_one_half


def _calculate_combined_raw_intensities_gpu(
    Iprf_t: torch.Tensor,
    Isum_t: torch.Tensor,
    Vprf_t: torch.Tensor,
    Vsum_t: torch.Tensor,
    Imid: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU evaluation of profile/summation intensity combination."""
    ratio = Isum_t / float(Imid)
    w = torch.where(
        Isum_t <= 0.0,
        torch.ones_like(Isum_t),
        1.0 / (1.0 + ratio * ratio * ratio),
    )
    one_m_w = 1.0 - w
    Icomb = w * Iprf_t + one_m_w * Isum_t
    Vcomb = w * Vprf_t + one_m_w * Vsum_t
    return Icomb, Vcomb


def combine_intensities_gpu(
    reflections, Imid: float, device: torch.device | str | None = None
):
    """GPU-accelerated drop-in replacement for combine_intensities."""
    dev = _get_device(device)
    has_prf = "intensity.prf.value" in reflections
    assert "intensity.sum.value" in reflections
    assert "prescaling_correction" in reflections

    n_refl = reflections.size()
    if n_refl == 0:
        return flex.double([]), flex.double([])

    conv_np = np.asarray(
        flumpy.to_numpy(reflections["prescaling_correction"]), dtype=np.float64
    )
    Isum_np = np.asarray(
        flumpy.to_numpy(reflections["intensity.sum.value"]), dtype=np.float64
    )
    Vsum_np = np.asarray(
        flumpy.to_numpy(reflections["intensity.sum.variance"]), dtype=np.float64
    )
    conv_t = torch.from_numpy(conv_np).to(device=dev, dtype=torch.float64)
    Isum_t = torch.from_numpy(Isum_np).to(device=dev, dtype=torch.float64)
    Vsum_t = torch.from_numpy(Vsum_np).to(device=dev, dtype=torch.float64)

    if "partiality" in reflections:
        p_np = np.asarray(flumpy.to_numpy(reflections["partiality"]), dtype=np.float64)
        p_t = torch.from_numpy(p_np).to(device=dev, dtype=torch.float64)
        inv_p_t = torch.where(p_t > 0.0, 1.0 / torch.where(p_t > 0.0, p_t, torch.ones_like(p_t)), torch.ones_like(p_t))
        sum_conv_t = conv_t * inv_p_t
    else:
        inv_p_t = None
        sum_conv_t = conv_t

    if has_prf:
        Ipr_np = np.asarray(
            flumpy.to_numpy(reflections["intensity.prf.value"]), dtype=np.float64
        )
        Vpr_np = np.asarray(
            flumpy.to_numpy(reflections["intensity.prf.variance"]), dtype=np.float64
        )
        Ipr_t = torch.from_numpy(Ipr_np).to(device=dev, dtype=torch.float64)
        Vpr_t = torch.from_numpy(Vpr_np).to(device=dev, dtype=torch.float64)

    flags_np = np.asarray(flumpy.to_numpy(reflections["flags"]), dtype=np.int64)
    flags_t = torch.from_numpy(flags_np).to(device=dev, dtype=torch.int64)
    prf_flag = int(reflections.flags.integrated_prf)
    sum_flag = int(reflections.flags.integrated_sum)
    both_flag = int(reflections.flags.integrated)
    not_prf_t = (flags_t & prf_flag) == 0
    not_sum_t = (flags_t & sum_flag) == 0
    both_t = (flags_t & both_flag) == both_flag

    if Imid == 1:
        intensity_t = Isum_t * sum_conv_t
        variance_t = Vsum_t * sum_conv_t * sum_conv_t
        if has_prf:
            intensity_t = torch.where(not_sum_t, Ipr_t * conv_t, intensity_t)
            variance_t = torch.where(
                not_sum_t, Vpr_t * conv_t * conv_t, variance_t
            )
    else:
        intensity_t = Ipr_t * conv_t
        variance_t = Vpr_t * conv_t * conv_t
        intensity_t = torch.where(not_prf_t, Isum_t * sum_conv_t, intensity_t)
        variance_t = torch.where(
            not_prf_t, Vsum_t * sum_conv_t * sum_conv_t, variance_t
        )
        if Imid != 0:
            if inv_p_t is not None:
                Int_c, Var_c = _calculate_combined_raw_intensities_gpu(
                    Ipr_t,
                    Isum_t * inv_p_t,
                    Vpr_t,
                    Vsum_t * inv_p_t * inv_p_t,
                    Imid,
                )
            else:
                Int_c, Var_c = _calculate_combined_raw_intensities_gpu(
                    Ipr_t, Isum_t, Vpr_t, Vsum_t, Imid
                )
            intensity_t = torch.where(both_t, Int_c * conv_t, intensity_t)
            variance_t = torch.where(
                both_t, Var_c * conv_t * conv_t, variance_t
            )

    int_np = np.ascontiguousarray(intensity_t.cpu().numpy(), dtype=np.float64)
    var_np = np.ascontiguousarray(variance_t.cpu().numpy(), dtype=np.float64)
    return flumpy.from_numpy(int_np), flumpy.from_numpy(var_np)


def _eval_dataset_Imid_gpu(dataset, Imid: float, dev: torch.device):
    """Evaluate scaled (I_obs, sigma_obs) on GPU for a candidate Imid."""
    cache = getattr(dataset, "_gpu_comb_cache", None)
    if cache is None or cache["dev"] != dev:
        Ipr_t = torch.from_numpy(
            np.asarray(
                flumpy.to_numpy(dataset["intensity.prf.value"]), dtype=np.float64
            )
        ).to(device=dev, dtype=torch.float64)
        Vpr_t = torch.from_numpy(
            np.asarray(
                flumpy.to_numpy(dataset["intensity.prf.variance"]), dtype=np.float64
            )
        ).to(device=dev, dtype=torch.float64)
        Isum_t = torch.from_numpy(
            np.asarray(
                flumpy.to_numpy(dataset["intensity.sum.value"]), dtype=np.float64
            )
        ).to(device=dev, dtype=torch.float64)
        Vsum_t = torch.from_numpy(
            np.asarray(
                flumpy.to_numpy(dataset["intensity.sum.variance"]), dtype=np.float64
            )
        ).to(device=dev, dtype=torch.float64)
        if "partiality" in dataset:
            p_t = torch.from_numpy(
                np.asarray(flumpy.to_numpy(dataset["partiality"]), dtype=np.float64)
            ).to(device=dev, dtype=torch.float64)
            Isum_t = Isum_t / p_t
            Vsum_t = Vsum_t / (p_t * p_t)
        conv_t = torch.from_numpy(
            np.asarray(
                flumpy.to_numpy(dataset["prescaling_correction"]), dtype=np.float64
            )
        ).to(device=dev, dtype=torch.float64)
        g_t = torch.from_numpy(
            np.asarray(
                flumpy.to_numpy(dataset["inverse_scale_factor"]), dtype=np.float64
            )
        ).to(device=dev, dtype=torch.float64)
        scale_mult_t = conv_t / g_t
        cache = {
            "dev": dev,
            "Ipr": Ipr_t,
            "Vpr": Vpr_t,
            "Isum": Isum_t,
            "Vsum": Vsum_t,
            "conv": conv_t,
            "g": g_t,
            "scale_mult": scale_mult_t,
        }
        dataset._gpu_comb_cache = cache

    if Imid == 0:
        Int_t = cache["Ipr"]
        Var_t = cache["Vpr"]
    elif Imid == 1:
        Int_t = cache["Isum"]
        Var_t = cache["Vsum"]
    else:
        Int_t, Var_t = _calculate_combined_raw_intensities_gpu(
            cache["Ipr"], cache["Isum"], cache["Vpr"], cache["Vsum"], Imid
        )

    i_obs_t = Int_t * cache["scale_mult"]
    sig_obs_t = torch.sqrt(Var_t) * cache["scale_mult"]
    return i_obs_t, sig_obs_t


class SingleDatasetIntensityCombinerGPU(SingleDatasetIntensityCombiner):
    """GPU-accelerated SingleDatasetIntensityCombiner."""

    def calculate_suitable_combined_intensities(self):
        reflections = self.scaler.reflection_table.select(
            self.scaler.suitable_refl_for_scaling_sel
        )
        return combine_intensities_gpu(reflections, self.max_key)

    def _test_Imid_combinations(self):
        dev = _get_device()
        rows = []
        results = {}
        if self.dataset.size() > 0:
            hkl_np = flumpy.to_numpy(self.dataset["miller_index"])
            perm_np = np.ascontiguousarray(
                np.argsort(_pack_hkl(hkl_np), kind="stable"), dtype=np.uint64
            )
            self.dataset = self.dataset.select(flumpy.from_numpy(perm_np))
        miller_set = miller.set(
            crystal_symmetry=self.experiment.crystal.get_crystal_symmetry(
                assert_is_compatible_unit_cell=False
            ),
            indices=self.dataset["miller_index"],
            anomalous_flag=False,
        )
        for Imid in self.Imids:
            i_obs_t, sig_obs_t = _eval_dataset_Imid_gpu(self.dataset, Imid, dev)
            i_np = np.ascontiguousarray(i_obs_t.cpu().numpy(), dtype=np.float64)
            s_np = np.ascontiguousarray(sig_obs_t.cpu().numpy(), dtype=np.float64)
            i_obs = miller.array(miller_set, data=flumpy.from_numpy(i_np))
            i_obs.set_observation_type_xray_intensity()
            i_obs.set_sigmas(flumpy.from_numpy(s_np))
            try:
                rmeas, cchalf = _fast_merging_stats_presorted(array=i_obs)
                logger.debug("Imid: %s, Rmeas %s, cchalf %s", Imid, rmeas, cchalf)
            except RuntimeError:
                raise DialsMergingStatisticsError(
                    "Unable to merge for intensity combination"
                )
            results[Imid] = rmeas
            res_str = {0: "prf only", 1: "sum only"}
            if Imid not in res_str:
                res_str[Imid] = "Imid = " + str(round(Imid, 2))
            rows.append([res_str[Imid], str(round(cchalf, 5)), str(round(rmeas, 5))])
        return rows, results


class MultiDatasetIntensityCombinerGPU(MultiDatasetIntensityCombiner):
    """GPU-accelerated MultiDatasetIntensityCombiner."""

    def calculate_suitable_combined_intensities(self, dataset):
        scaler = self.active_scalers[dataset]
        reflections = scaler.reflection_table.select(
            scaler.suitable_refl_for_scaling_sel
        )
        Imid = 1 if dataset not in self.good_datasets else self.max_key
        return combine_intensities_gpu(reflections, Imid)

    def _test_Imid_combinations(self):
        dev = _get_device()
        rows = []
        results = {}
        combined_indices = flex.miller_index([])
        for dataset in self.datasets:
            combined_indices.extend(dataset["miller_index"])
        perm_gpu = None
        if combined_indices.size() > 0:
            hkl_np = flumpy.to_numpy(combined_indices)
            perm_np = np.ascontiguousarray(
                np.argsort(_pack_hkl(hkl_np), kind="stable"), dtype=np.int64
            )
            combined_indices = combined_indices.select(
                flumpy.from_numpy(perm_np.view(np.uint64))
            )
            perm_gpu = torch.from_numpy(perm_np).to(device=dev, dtype=torch.int64)
        miller_set = miller.set(
            crystal_symmetry=self.active_scalers[
                0
            ].experiment.crystal.get_crystal_symmetry(),
            indices=combined_indices,
            anomalous_flag=False,
        )
        for Imid in self.Imids:
            i_parts = []
            s_parts = []
            for dataset in self.datasets:
                i_obs_t, sig_obs_t = _eval_dataset_Imid_gpu(dataset, Imid, dev)
                i_parts.append(i_obs_t)
                s_parts.append(sig_obs_t)
            all_i_t = torch.cat(i_parts, dim=0)
            all_s_t = torch.cat(s_parts, dim=0)
            if perm_gpu is not None:
                all_i_t = all_i_t[perm_gpu]
                all_s_t = all_s_t[perm_gpu]
            i_np = np.ascontiguousarray(all_i_t.cpu().numpy(), dtype=np.float64)
            s_np = np.ascontiguousarray(all_s_t.cpu().numpy(), dtype=np.float64)
            i_obs = miller.array(miller_set, data=flumpy.from_numpy(i_np))
            i_obs.set_observation_type_xray_intensity()
            i_obs.set_sigmas(flumpy.from_numpy(s_np))
            try:
                rmeas, cchalf = _fast_merging_stats_presorted(array=i_obs)
                logger.debug("Imid: %s, Rmeas %s, cchalf %s", Imid, rmeas, cchalf)
            except RuntimeError:
                raise DialsMergingStatisticsError(
                    "Unable to merge for intensity combination"
                )
            results[Imid] = rmeas
            res_str = {0: "prf only", 1: "sum only"}
            if Imid not in res_str:
                res_str[Imid] = "Imid = " + str(round(Imid, 2))
            rows.append([res_str[Imid], str(round(cchalf, 5)), str(round(rmeas, 5))])
        return rows, results
