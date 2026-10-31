"""
GPU-accelerated error model classes and refinement targets for scaling using PyTorch.
"""

from __future__ import annotations

import logging
from math import ceil

import numpy as np
import torch
from dxtbx import flumpy
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

        smat = binner.summation_matrix
        smat_np = np.asarray(
            flumpy.to_numpy(smat.as_dense_matrix()), dtype=np.float64
        ).reshape(smat.n_rows, smat.n_cols)
        smat_t = _np_to_tensor(smat_np, dev, dtype=torch.float64)

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

    binner.sigmaprime = sigmaprime_t.cpu().numpy()
    binner.delta_hl = delta_hl_t.cpu().numpy()
    bin_vars_np = bin_vars_t.cpu().numpy()
    binner.bin_variances = bin_vars_np
    binner.binning_info["bin_variances"] = bin_vars_np


class ErrorModelTargetAGPU(ErrorModelTargetA):
    """GPU-accelerated ErrorModelTargetA."""

    def _get_xy_gpu(self, dev: torch.device):
        em = self.error_model
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


def run_error_model_refinement_gpu(
    model: BasicErrorModel, Ih_table, min_partiality=0.4, use_stills_filtering=False
) -> BasicErrorModel:
    """GPU-accelerated drop-in replacement for run_error_model_refinement."""
    assert Ih_table.n_work_blocks == 1
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
