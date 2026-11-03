"""
GPU-accelerated error model classes and refinement targets for scaling using PyTorch.
"""

from __future__ import annotations

import logging
import types
from math import ceil, exp, log

import numpy as np
import torch
from dxtbx import flumpy
from scitbx import sparse
from scipy.stats import norm

from dials.algorithms.scaling.error_model.engine import (
    ErrorModelRefinery,
    ErrorModelRegressionRefiner,
    error_model_refinery,
    run_error_model_refinement,
)
from dials.algorithms.scaling.error_model.error_model import (
    AComponent,
    BasicErrorModel,
    BComponent,
    ErrorModelA_APM,
    ErrorModelB_APM,
    ErrorModelBinner,
    ErrorModelRegressionAPM,
    calc_deltahl,
    calc_sigmaprime,
    extract_error_model_groups,
    filter_unsuitable_reflections,
    filter_unsuitable_reflections_stills,
    phil_scope,
)
from dials.algorithms.scaling.error_model.error_model_target import (
    ErrorModelTarget,
    ErrorModelTargetA,
    ErrorModelTargetB,
    ErrorModelTargetRegression,
    calculate_regression_x_y,
)
from dials.array_family import flex

from .components import _get_device, _get_minimization_device
from .ih_table import _np_to_tensor, get_block_group_ids_gpu, sum_in_groups_gpu

logger = logging.getLogger("dials")

_NORM_PPF_CACHE: dict[int, tuple[int, int, flex.double]] = {}
_V1_CDF = float(norm.cdf(-1.5))
_V2_CDF = float(norm.cdf(1.5))


class ErrorModelBinnerGPU(ErrorModelBinner):
    """GPU-accelerated ErrorModelBinner that builds dense bin summation matrices
    without a per-reflection Python loop over scitbx.sparse.matrix."""

    def __init__(self, Ih_table, min_reflections_required=250, n_bins=10):
        self.binning_info = {
            "initial_variances": [],
            "bin_boundaries": [],
            "mean_intensities": np.array([], dtype=float).reshape((0,)),
            "bin_variances": [],
            "refl_per_bin": [],
            "n_reflections": None,
        }
        self.n_bins = n_bins
        self.Ih_table = Ih_table
        self.min_reflections_required = min_reflections_required
        self.n_h = self.Ih_table.calc_nh()
        self._smat_dense_np = self._create_summation_matrix_dense()
        self._summation_matrix_lazy = None
        self.weights = np.array(self.binning_info["mean_intensities"])
        self._sigmaprime_np = None
        self._delta_hl_np = None
        self._last_gpu_params = None
        update_binner_gpu(self, [1.0, 0.0])
        self.binning_info["initial_variances"] = self.binning_info["bin_variances"]

    @property
    def summation_matrix(self):
        if self._summation_matrix_lazy is None:
            n_rows, n_cols = self._smat_dense_np.shape
            smat = sparse.matrix(n_rows, n_cols)
            if n_rows > 0 and n_cols > 0:
                b = flumpy.from_numpy(
                    np.ascontiguousarray(self._smat_dense_np, dtype=np.float64)
                )
                b.reshape(flex.grid(n_rows, n_cols))
                smat.assign_block(b, 0, 0)
            self._summation_matrix_lazy = smat
        return self._summation_matrix_lazy

    @summation_matrix.setter
    def summation_matrix(self, value):
        self._summation_matrix_lazy = value
        if value is not None:
            self._smat_dense_np = np.asarray(
                flumpy.to_numpy(value.as_dense_matrix()), dtype=np.float64
            ).reshape(value.n_rows, value.n_cols)

    @property
    def sigmaprime(self):
        if self._sigmaprime_np is None:
            cache = getattr(self, "_gpu_cache", None)
            if cache is not None and "sigmaprime" in cache:
                self._sigmaprime_np = cache["sigmaprime"].cpu().numpy()
            else:
                self._sigmaprime_np = calc_sigmaprime([1.0, 0.0], self.Ih_table)
        return self._sigmaprime_np

    @sigmaprime.setter
    def sigmaprime(self, value):
        self._sigmaprime_np = value

    @property
    def delta_hl(self):
        if self._delta_hl_np is None:
            cache = getattr(self, "_gpu_cache", None)
            if cache is not None and "delta_hl" in cache:
                self._delta_hl_np = cache["delta_hl"].cpu().numpy()
            else:
                self._delta_hl_np = calc_deltahl(
                    self.Ih_table, self.n_h, self.sigmaprime
                )
        return self._delta_hl_np

    @delta_hl.setter
    def delta_hl(self, value):
        self._delta_hl_np = value

    def update(self, parameters):
        update_binner_gpu(self, parameters)

    def _create_summation_matrix_dense(self) -> np.ndarray:
        n = self.Ih_table.size
        self.binning_info["n_reflections"] = n
        Ih = self.Ih_table.Ih_values * self.Ih_table.inverse_scale_factors
        if "partiality" in self.Ih_table.Ih_table:
            Ih = Ih * self.Ih_table.Ih_table["partiality"].to_numpy()
        size_order = np.argsort(-Ih, kind="stable")
        Imax = float(Ih.max())
        min_Ih = float(Ih.min())
        Imin = max(1.0, min_Ih)
        spacing = (log(Imax) - log(Imin)) / float(self.n_bins)
        boundaries = [Imax] + [
            exp(log(Imax) - (i * spacing)) for i in range(1, self.n_bins + 1)
        ]
        boundaries[-1] = min_Ih - 0.01
        self.binning_info["bin_boundaries"] = np.array(boundaries)

        n_cumul = 0
        if Ih.size > 100 * self.min_reflections_required:
            self.min_reflections_required = int(Ih.size / 100.0)
        min_per_bin = min(self.min_reflections_required, int(n / (3.0 * self.n_bins)))

        bin_Boundaries = self.binning_info["bin_boundaries"]
        sels = []
        refl_per_bin = []
        for i in range(len(bin_Boundaries) - 1):
            maximum = bin_Boundaries[i]
            minimum = bin_Boundaries[i + 1]
            sel1 = Ih <= maximum
            sel2 = Ih > minimum
            sel = sel1 & sel2
            n_in_bin = int(np.count_nonzero(sel))
            if n_in_bin < min_per_bin:
                m = n_cumul + min_per_bin
                if m < n:
                    idx = size_order[m]
                    intensity = Ih[idx]
                    bin_Boundaries[i + 1] = intensity
                    minimum = intensity
                    sel = sel1 & (Ih > minimum)
                    n_in_bin = int(np.count_nonzero(sel))
            refl_per_bin.append(float(n_in_bin))
            sels.append(sel)
            n_cumul += n_in_bin

        keep_cols = [
            i for i, cnt in enumerate(refl_per_bin) if cnt >= (min_per_bin - 5)
        ]
        if len(keep_cols) == self.n_bins:
            self.binning_info["refl_per_bin"] = np.array(refl_per_bin, dtype=float)
            mean_ints = [float(np.mean(Ih[s])) for s in sels]
            self.binning_info["mean_intensities"] = np.array(mean_ints, dtype=float)
            smat_dense = np.zeros((n, self.n_bins), dtype=np.float64)
            for i, s in enumerate(sels):
                smat_dense[s, i] = 1.0
            return smat_dense

        new_refl_per_bin = [refl_per_bin[i] for i in keep_cols]
        new_bounds = [bin_Boundaries[i] for i in keep_cols] + [bin_Boundaries[-1]]
        self.binning_info["refl_per_bin"] = np.array(new_refl_per_bin, dtype=float)
        self.binning_info["bin_boundaries"] = np.array(new_bounds, dtype=float)
        smat_dense = np.zeros((n, len(keep_cols)), dtype=np.float64)
        mean_ints = []
        for next_col, i in enumerate(keep_cols):
            smat_dense[sels[i], next_col] = 1.0
            maximum = new_bounds[next_col]
            minimum = new_bounds[next_col + 1]
            sel = (Ih <= maximum) & (Ih > minimum)
            mean_ints.append(float(np.mean(Ih[sel])))
        self.binning_info["mean_intensities"] = np.array(mean_ints, dtype=float)
        return smat_dense


def _get_binner_gpu_cache(binner: ErrorModelBinner, dev: torch.device):
    """Fetch or create cached GPU tensors for an ErrorModelBinner."""
    cache = getattr(binner, "_gpu_cache", None)
    if cache is None or cache["dev"] != dev or cache["n"] != binner.Ih_table.size:
        block = binner.Ih_table
        var_t = _np_to_tensor(block.variances, dev, dtype=torch.float64)
        I_t = _np_to_tensor(block.intensities, dev, dtype=torch.float64)
        g_t = _np_to_tensor(block.inverse_scale_factors, dev, dtype=torch.float64)
        Ih_t = _np_to_tensor(block.Ih_values, dev, dtype=torch.float64)
        nh_t = _np_to_tensor(binner.n_h, dev, dtype=torch.float64)
        prefac_t = torch.sqrt((nh_t - 1.0) / nh_t)
        diff_t = prefac_t * ((I_t / g_t) - Ih_t)

        smat_dense_np = getattr(binner, "_smat_dense_np", None)
        if smat_dense_np is None:
            smat = binner.summation_matrix
            smat_dense_np = np.asarray(
                flumpy.to_numpy(smat.as_dense_matrix()), dtype=np.float64
            ).reshape(smat.n_rows, smat.n_cols)
        smat_t = _np_to_tensor(smat_dense_np, dev, dtype=torch.float64)

        counts_t = _np_to_tensor(
            binner.binning_info["refl_per_bin"], dev, dtype=torch.float64
        )
        weights_t = _np_to_tensor(binner.weights, dev, dtype=torch.float64)

        cache = {
            "dev": dev,
            "n": block.size,
            "var": var_t,
            "I": I_t,
            "I_sq": I_t * I_t,
            "g": g_t,
            "g_sq": g_t * g_t,
            "Ih": Ih_t,
            "nh": nh_t,
            "diff": diff_t,
            "smat": smat_t,
            "counts": counts_t,
            "counts_sq": counts_t * counts_t,
            "weights": weights_t,
            "sum_weights": torch.sum(weights_t),
        }
        binner._gpu_cache = cache
    return cache


def update_binner_gpu(
    binner: ErrorModelBinner, parameters, device: torch.device | str | None = None
):
    """Update ErrorModelBinner sigmaprime, delta_hl, and bin_variances on the GPU."""
    dev = _get_minimization_device(binner.Ih_table.size, device)
    c = _get_binner_gpu_cache(binner, dev)
    a = float(parameters[0])
    b = float(parameters[1])
    param_key = (a, b, dev)
    if getattr(binner, "_last_gpu_params", None) == param_key and "bin_vars" in c:
        return

    sigmaprime_t = (a * torch.sqrt(c["var"] + (b * b) * c["I_sq"])) / c["g"]
    delta_hl_t = c["diff"] / sigmaprime_t

    sum_deltasq_t = torch.mv(c["smat"].t(), delta_hl_t * delta_hl_t)
    sum_delta_t = torch.mv(c["smat"].t(), delta_hl_t)
    bin_vars_t = (sum_deltasq_t / c["counts"]) - (
        (sum_delta_t * sum_delta_t) / c["counts_sq"]
    )

    c["sigmaprime"] = sigmaprime_t
    c["delta_hl"] = delta_hl_t
    c["bin_vars"] = bin_vars_t
    binner._last_gpu_params = param_key

    if isinstance(binner, ErrorModelBinnerGPU):
        binner._sigmaprime_np = None
        binner._delta_hl_np = None
    else:
        binner.sigmaprime = sigmaprime_t.cpu().numpy()
        binner.delta_hl = delta_hl_t.cpu().numpy()
    bin_vars_np = bin_vars_t.cpu().numpy()
    binner.bin_variances = bin_vars_np
    binner.binning_info["bin_variances"] = bin_vars_np


class ErrorModelTargetAGPU(ErrorModelTargetA):
    """GPU-accelerated ErrorModelTargetA."""

    def _get_xy_gpu(self, dev: torch.device):
        em = self.error_model
        sx_gpu = getattr(em, "_gpu_sortedx", None)
        sy_gpu = getattr(em, "_gpu_sortedy", None)
        if (
            sx_gpu is not None
            and sy_gpu is not None
            and getattr(em, "_gpu_sortedx_ref", None) is em.sortedx
            and getattr(em, "_gpu_sortedy_ref", None) is em.sortedy
            and sx_gpu.device == dev
        ):
            return sx_gpu, sy_gpu
        cache = getattr(self, "_gpu_sorted_xy_cache", None)
        if (
            cache is None
            or cache[0] is not em.sortedx
            or cache[1] is not em.sortedy
            or cache[2].device != dev
        ):
            sx_t = torch.from_numpy(
                np.ascontiguousarray(flumpy.to_numpy(em.sortedx), dtype=np.float64)
            ).to(device=dev)
            sy_t = torch.from_numpy(
                np.ascontiguousarray(flumpy.to_numpy(em.sortedy), dtype=np.float64)
            ).to(device=dev)
            cache = (em.sortedx, em.sortedy, sx_t, sy_t)
            self._gpu_sorted_xy_cache = cache
        return cache[2], cache[3]

    def calculate_residuals(self, apm):
        dev = _get_minimization_device(len(self.error_model.sortedx))
        sx_t, sy_t = self._get_xy_gpu(dev)
        x0 = float(apm.x[0])
        x1 = float(apm.x[1])
        diff = sy_t - (x1 * sx_t) - x0
        R_np = np.ascontiguousarray((diff * diff).cpu().numpy(), dtype=np.float64)
        return flumpy.from_numpy(R_np)

    def calculate_gradients(self, apm):
        dev = _get_minimization_device(len(self.error_model.sortedx))
        sx_t, sy_t = self._get_xy_gpu(dev)
        x0 = float(apm.x[0])
        x1 = float(apm.x[1])
        R = sy_t - (x1 * sx_t) - x0
        g0 = float((-2.0 * torch.sum(R)).item())
        g1 = float((-2.0 * torch.sum(R * sx_t)).item())
        return flex.double([g0, g1])


class ErrorModelTargetBGPU(ErrorModelTargetB):
    """GPU-accelerated ErrorModelTargetB."""

    def predict(self, _):
        update_binner_gpu(self.error_model.binner, self.error_model.parameters)

    def calculate_residuals(self, _):
        dev = _get_minimization_device(self.error_model.binner.Ih_table.size)
        c = _get_binner_gpu_cache(self.error_model.binner, dev)
        if "bin_vars" not in c:
            update_binner_gpu(self.error_model.binner, self.error_model.parameters, dev)
        bv = c["bin_vars"]
        R_t = (
            ((0.5 - bv) ** 2 + (1.0 / bv) - 1.25)
            * c["weights"]
            / c["sum_weights"]
        )
        R_np = np.ascontiguousarray(R_t.cpu().numpy(), dtype=np.float64)
        return flumpy.from_numpy(R_np)

    def calculate_gradients(self, apm):
        dev = _get_minimization_device(self.error_model.binner.Ih_table.size)
        c = _get_binner_gpu_cache(self.error_model.binner, dev)
        if "bin_vars" not in c:
            update_binner_gpu(self.error_model.binner, self.error_model.parameters, dev)
        a = float(self.error_model.components["a"].parameters[0])
        b = float(apm.x[0])
        sigmaprime_t = c["sigmaprime"]
        delta_hl_t = c["delta_hl"]
        bv = c["bin_vars"]

        dsig_dc = b * c["I_sq"] * (a * a) / (sigmaprime_t * c["g_sq"])
        ddelta_dsigma = -1.0 * delta_hl_t / sigmaprime_t
        deriv = ddelta_dsigma * dsig_dc

        dphi_by_dvar = -2.0 * (0.5 - bv + (1.0 / (2.0 * bv * bv)))
        term1 = torch.mv(c["smat"].t(), 2.0 * delta_hl_t * deriv)
        term2a = torch.mv(c["smat"].t(), delta_hl_t)
        term2b = torch.mv(c["smat"].t(), deriv)
        grad_t = dphi_by_dvar * (
            (term1 / c["counts"]) - (2.0 * term2a * term2b / c["counts_sq"])
        )
        g_val = float((torch.sum(grad_t * c["weights"]) / c["sum_weights"]).item())
        return flex.double([g_val])


class ErrorModelTargetRegressionGPU(ErrorModelTargetRegression):
    """GPU-accelerated ErrorModelTargetRegression."""

    def __init__(self, error_model):
        ErrorModelTarget.__init__(self, error_model)
        block = self.error_model.filtered_Ih_table
        dev = _get_minimization_device(block.size)
        group_ids_gpu = get_block_group_ids_gpu(block, dev)
        n_groups = block.n_groups
        I_t = _np_to_tensor(block.intensities, dev, dtype=torch.float64)
        g_t = _np_to_tensor(block.inverse_scale_factors, dev, dtype=torch.float64)
        Ih_t = _np_to_tensor(block.Ih_values, dev, dtype=torch.float64)
        var_t = _np_to_tensor(block.variances, dev, dtype=torch.float64)

        ones = torch.ones_like(I_t)
        n_t = sum_in_groups_gpu(ones, n_groups, group_ids_gpu, "per_refl")
        var_over_nm1 = ((I_t - g_t * Ih_t) ** 2) / (n_t - 1.0)
        sigmasq_obs = sum_in_groups_gpu(
            var_over_nm1, n_groups, group_ids_gpu, "per_refl"
        )
        isq = I_t * I_t
        self._y_gpu = sigmasq_obs / isq
        self._x_gpu = var_t / isq
        self.x = self._x_gpu.cpu().numpy()
        self.y = self._y_gpu.cpu().numpy()
        self.n_refl = self.y.size

    def calculate_residuals(self, apm):
        params = apm.x
        if apm.active_parameters == ["a"]:
            b = float(self.error_model.parameters[1])
            p0 = float(params[0])
            R_t = (self._y_gpu - (p0 * self._x_gpu) - p0 * (b * b)) ** 2
        elif apm.active_parameters == ["b"]:
            a = float(self.error_model.parameters[0])
            a2 = a * a
            p0 = float(params[0])
            R_t = (self._y_gpu - (a2 * self._x_gpu) - (a2 * p0)) ** 2
        else:
            p0 = float(params[0])
            p1 = float(params[1])
            R_t = (self._y_gpu - (p0 * self._x_gpu) - p1) ** 2
        R_np = np.ascontiguousarray(R_t.cpu().numpy(), dtype=np.float64)
        return flumpy.from_numpy(R_np)

    def calculate_gradients(self, apm):
        params = apm.x
        if apm.active_parameters == ["a"]:
            b = float(self.error_model.parameters[1])
            b2 = b * b
            p0 = float(params[0])
            R = self._y_gpu - (p0 * self._x_gpu) - p0 * b2
            g0 = float((-2.0 * torch.sum(R * (self._x_gpu + b2))).item())
            return flex.double([g0])
        elif apm.active_parameters == ["b"]:
            a = float(self.error_model.parameters[0])
            a2 = a * a
            p0 = float(params[0])
            R = self._y_gpu - (a2 * self._x_gpu) - (a2 * p0)
            g0 = float((-2.0 * torch.sum(R * a2)).item())
            return flex.double([g0])
        else:
            p0 = float(params[0])
            p1 = float(params[1])
            R = self._y_gpu - (p0 * self._x_gpu) - p1
            g0 = float((-2.0 * torch.sum(R * self._x_gpu)).item())
            g1 = float((-2.0 * torch.sum(R)).item())
            return flex.double([g0, g1])


class ErrorModelRefineryGPU(ErrorModelRefinery):
    """GPU-accelerated ErrorModelRefinery."""

    def _refine_a(self):
        parameterisation = ErrorModelA_APM(self.model)
        self._refine_component(
            self.model, ErrorModelTargetAGPU(self.model), parameterisation
        )

    def _refine_b(self):
        parameterisation = ErrorModelB_APM(self.model)
        self._refine_component(
            self.model, ErrorModelTargetBGPU(self.model), parameterisation
        )


def error_model_refinery_gpu(
    model, active_parameters, error_model_scope, max_iterations
):
    """Return GPU-accelerated error model refinery engine."""
    if model.id_ == "basic":
        if error_model_scope.minimisation == "individual":
            return ErrorModelRefineryGPU(
                model=model,
                max_iterations=max_iterations,
                parameters_to_refine=active_parameters,
            )
        if error_model_scope.minimisation == "regression":
            parameterisation = ErrorModelRegressionAPM(model, active_parameters)
            target = ErrorModelTargetRegressionGPU(model)
            return ErrorModelRegressionRefiner(
                model=model,
                target=target,
                prediction_parameterisation=parameterisation,
                max_iterations=max_iterations,
            )
    return None


def _calculate_sorted_deviations_gpu(self, parameters):
    """GPU-accelerated replacement for BasicErrorModel.calculate_sorted_deviations."""
    block = self.filtered_Ih_table
    n = block.size
    dev = _get_minimization_device(n)
    if self.binner is not None and self.binner.Ih_table is block:
        update_binner_gpu(self.binner, parameters, dev)
        delta_hl_t = self.binner._gpu_cache["delta_hl"]
    else:
        a = float(parameters[0])
        b = float(parameters[1])
        var_t = _np_to_tensor(block.variances, dev, dtype=torch.float64)
        I_t = _np_to_tensor(block.intensities, dev, dtype=torch.float64)
        g_t = _np_to_tensor(block.inverse_scale_factors, dev, dtype=torch.float64)
        Ih_t = _np_to_tensor(block.Ih_values, dev, dtype=torch.float64)
        nh_t = _np_to_tensor(block.calc_nh(), dev, dtype=torch.float64)
        sigmaprime_t = (a * torch.sqrt(var_t + (b * b) * (I_t * I_t))) / g_t
        delta_hl_t = torch.sqrt((nh_t - 1.0) / nh_t) * ((I_t / g_t) - Ih_t) / sigmaprime_t

    ppf_entry = _NORM_PPF_CACHE.get(n)
    if ppf_entry is None:
        idx_cutoff_min = ceil((_V1_CDF * n) - 0.5)
        idx_cutoff_max = ceil((_V2_CDF * n) - 0.5)
        central_n = idx_cutoff_max - idx_cutoff_min
        v_np = np.linspace(
            start=(idx_cutoff_min + 0.5) / n,
            stop=(idx_cutoff_max + 0.5) / n,
            endpoint=False,
            num=central_n,
        )
        v_t = torch.from_numpy(v_np).to(device=dev, dtype=torch.float64)
        ppf_t = torch.special.ndtri(v_t)
        sortedx_flex = flumpy.from_numpy(
            np.ascontiguousarray(ppf_t.cpu().numpy(), dtype=np.float64)
        )
        ppf_entry = (idx_cutoff_min, idx_cutoff_max, sortedx_flex, ppf_t)
        _NORM_PPF_CACHE[n] = ppf_entry
    else:
        idx_cutoff_min, idx_cutoff_max, sortedx_flex, ppf_t = ppf_entry
        if ppf_t.device != dev:
            ppf_t = ppf_t.to(device=dev)

    sortedy_t = torch.sort(delta_hl_t)[0][idx_cutoff_min:idx_cutoff_max]
    sortedy_np = np.ascontiguousarray(sortedy_t.cpu().numpy(), dtype=np.float64)
    sortedy_flex = flumpy.from_numpy(sortedy_np)

    self.sortedx = sortedx_flex
    self.sortedy = sortedy_flex
    self._gpu_sortedx = ppf_t
    self._gpu_sortedy = sortedy_t
    self._gpu_sortedx_ref = sortedx_flex
    self._gpu_sortedy_ref = sortedy_flex


def _configure_error_model_for_refinement_gpu(
    model: BasicErrorModel, Ih_table_block, min_partiality=0.4, use_stills_filtering=False
):
    """Configure BasicErrorModel using ErrorModelBinnerGPU and GPU sorted deviations."""
    model.filtered_Ih_table = model.filter_unsuitable_reflections(
        Ih_table_block,
        model.params,
        min_partiality,
        use_stills_filtering,
    )
    model.binner = ErrorModelBinnerGPU(
        model.filtered_Ih_table, model.min_reflections_required, model.params.n_bins
    )
    model.calculate_sorted_deviations = types.MethodType(
        _calculate_sorted_deviations_gpu, model
    )
    model.binner.update(model.parameters)
    model.calculate_sorted_deviations(model.parameters)


def run_error_model_refinement_gpu(
    model: BasicErrorModel, Ih_table, min_partiality=0.4, use_stills_filtering=False
) -> BasicErrorModel:
    """GPU-accelerated drop-in replacement for run_error_model_refinement."""
    assert Ih_table.n_work_blocks == 1
    if isinstance(model, BasicErrorModel):
        _configure_error_model_for_refinement_gpu(
            model, Ih_table.blocked_data_list[0], min_partiality, use_stills_filtering
        )
    else:
        model.configure_for_refinement(
            Ih_table.blocked_data_list[0], min_partiality, use_stills_filtering
        )
    if not model.active_parameters:
        logger.info("All error model parameters fixed, skipping refinement")
    else:
        logger.info("Performing a round of error model refinement.")
        refinery = error_model_refinery_gpu(
            model=model,
            active_parameters=model.active_parameters,
            error_model_scope=model.params,
            max_iterations=100,
        )
        if refinery:
            refinery.run()
            refinery.print_step_table()
        logger.info(model)
    model.finalise()
    return model
