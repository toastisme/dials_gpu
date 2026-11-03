"""
GPU-accelerated scaling refinement engines (SimpleLBFGS, GaussNewton, LevMar).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from dxtbx import flumpy
from scitbx import sparse

from dials.algorithms.refinement.engine import (
    GaussNewtonIterations,
    LevenbergMarquardtIterations,
    SimpleLBFGS,
)
from dials.algorithms.scaling.scaling_refiner import (
    DOF_TOO_LOW,
    MAX_ITERATIONS,
    MAX_TRIAL_ITERATIONS,
    OBJECTIVE_INCREASE,
    RMSD_CONVERGED,
    STEP_TOO_SMALL,
    TARGET_ACHIEVED,
    ScalingGaussNewtonIterations,
    ScalingLevenbergMarquardtIterations,
    ScalingLstbxBuildUpMixin,
    ScalingRefinery,
    ScalingSimpleLBFGS,
    print_step_table,
    scaling_refinery_phil_scope,
    scaling_refinery_phil_str,
)
from dials.algorithms.scaling.scaling_utilities import log_memory_usage
from dials.array_family import flex

from .components import _get_device

logger = logging.getLogger("dials")


def _get_reducing_matrix_gpu(apm, device: torch.device) -> torch.Tensor | None:
    """If `apm` is a shared_active_parameter_manager, return its reducing_matrix on GPU."""
    red_mat = getattr(apm, "reducing_matrix", None)
    if red_mat is None:
        return None
    cache = getattr(apm, "_gpu_reducing_matrix", None)
    if cache is None or cache.device != device:
        dense_flex = red_mat.as_dense_matrix()
        mat_np = np.asarray(flumpy.to_numpy(dense_flex), dtype=np.float64).reshape(
            red_mat.n_rows, red_mat.n_cols
        )
        cache = torch.from_numpy(mat_np).to(device=device, dtype=torch.float64)
        apm._gpu_reducing_matrix = cache
    return cache


class ScalingSimpleLBFGSGPU(ScalingSimpleLBFGS):
    """GPU-accelerated L-BFGS minimizer for scaling."""

    def run(self):
        work_blocks = self._scaler.get_blocks_for_minimisation()
        for b in work_blocks:
            b._gpu_in_minimisation = True
        try:
            return super().run()
        finally:
            for b in work_blocks:
                b._gpu_in_minimisation = False
                b._gpu_fast_static = None

    def compute_functional_gradients_and_curvatures(self):
        self.prepare_for_step()
        work_blocks = self._scaler.get_blocks_for_minimisation()

        f = []
        gi = []
        has_gpu_update = hasattr(self._scaler, "update_for_minimisation_gpu")
        for block_id, block in enumerate(work_blocks):
            if has_gpu_update:
                self._scaler.update_for_minimisation_gpu(
                    self._parameters, block_id, calc_derivs=True
                )
            else:
                self._scaler.update_for_minimisation(self._parameters, block_id)
            fb, gb = self._parameters.compute_functional_gradients(block)
            f.append(fb)
            gi.append(gb)

        f = sum(f)
        g = gi[0]
        for i in range(1, len(gi)):
            g += gi[i]

        restraints = self._parameters.compute_restraints_functional_gradients(
            self._parameters
        )
        if restraints:
            f += restraints[0]
            g += restraints[1]

        logger.debug("Functional : %s", f)
        logger.debug("Gradients : %s", list(g))
        log_memory_usage()
        logger.debug("\n")
        return f, g, None


class ScalingLstbxBuildUpMixinGPU(ScalingLstbxBuildUpMixin):
    """GPU-accelerated lstbx build_up mixin that computes dense Jacobians on GPU
    without constructing intermediate scitbx.sparse.matrix objects."""

    def run(self):
        work_blocks = self._scaler.get_blocks_for_minimisation()
        for b in work_blocks:
            b._gpu_in_minimisation = True
        try:
            return super().run()
        finally:
            for b in work_blocks:
                b._gpu_in_minimisation = False
                b._gpu_fast_static = None

    def build_up(self, objective_only=False):
        self.prepare_for_step()
        self.reset()

        work_blocks = self._scaler.get_blocks_for_minimisation()
        has_gpu_update = hasattr(self._scaler, "update_for_minimisation_gpu")
        need_explicit_jacobian = (
            "jacobian_structure" in self.history
            or "parameter_correlation" in self.history
            or "condition_number" in self.history
        )
        use_fast_dense = (
            hasattr(self._target, "compute_residuals_and_dense_jacobian_gpu")
            and "jacobian_structure" not in self.history
        )

        if objective_only:
            for block_id, block in enumerate(work_blocks):
                if has_gpu_update:
                    self._scaler.update_for_minimisation_gpu(
                        self._parameters, block_id, calc_derivs=False
                    )
                else:
                    self._scaler.update_for_minimisation(self._parameters, block_id)
                residuals, weights = self._parameters.compute_residuals(block)
                self.add_residuals(residuals, weights)
        else:
            self._jacobian = None

            for block_id, block in enumerate(work_blocks):
                if has_gpu_update:
                    self._scaler.update_for_minimisation_gpu(
                        self._parameters, block_id, calc_derivs=True
                    )
                else:
                    self._scaler.update_for_minimisation(self._parameters, block_id)

                if use_fast_dense:
                    res_gpu, J_gpu, w_gpu = (
                        self._target.compute_residuals_and_dense_jacobian_gpu(block)
                    )
                    red_mat_gpu = _get_reducing_matrix_gpu(
                        self._parameters, J_gpu.device
                    )
                    if red_mat_gpu is not None:
                        J_gpu = torch.mm(J_gpu, red_mat_gpu)
                    n_obs, n_par = int(J_gpu.shape[0]), int(J_gpu.shape[1])
                    if (
                        n_obs >= 8000
                        and n_obs > n_par > 0
                        and not need_explicit_jacobian
                    ):
                        Jw = J_gpu * w_gpu[:, None]
                        AtWA_np = torch.mm(J_gpu.t(), Jw).cpu().numpy()
                        rhs_np = (
                            torch.mm(Jw.t(), res_gpu[:, None])
                            .squeeze(1)
                            .cpu()
                            .numpy()
                        )
                        tot_wr2 = float(torch.sum(res_gpu * res_gpu * w_gpu).item())
                        try:
                            L_np = np.linalg.cholesky(AtWA_np)
                            z_np = np.ascontiguousarray(
                                np.linalg.solve(L_np, rhs_np), dtype=np.float64
                            )
                            U_np = np.ascontiguousarray(L_np.T, dtype=np.float64)
                        except np.linalg.LinAlgError:
                            evals, evecs = np.linalg.eigh(AtWA_np)
                            evals = np.maximum(evals, 0.0)
                            sqrt_evals = np.sqrt(evals)
                            U_np = np.ascontiguousarray(
                                sqrt_evals[:, None] * evecs.T, dtype=np.float64
                            )
                            vt_rhs = evecs.T @ rhs_np
                            cutoff = (
                                1e-12 * float(sqrt_evals[-1])
                                if sqrt_evals[-1] > 0.0
                                else 0.0
                            )
                            safe_se = np.where(sqrt_evals > cutoff, sqrt_evals, 1.0)
                            z_np = np.ascontiguousarray(
                                np.where(sqrt_evals > cutoff, vt_rhs / safe_se, 0.0),
                                dtype=np.float64,
                            )
                        z_sq = float(np.dot(z_np, z_np))
                        U_flex = flumpy.from_numpy(U_np)
                        U_flex.reshape(flex.grid(n_par, n_par))
                        self.add_equations(
                            flumpy.from_numpy(z_np),
                            U_flex,
                            flex.double(n_par, 1.0),
                        )
                        rem_n = n_obs - n_par
                        rem_wr2 = max(0.0, tot_wr2 - z_sq)
                        self.add_residuals(
                            flex.double(rem_n, 1.0),
                            flex.double(rem_n, rem_wr2 / rem_n),
                        )
                        continue

                    res_np = np.ascontiguousarray(
                        res_gpu.cpu().numpy(), dtype=np.float64
                    )
                    w_np = np.ascontiguousarray(w_gpu.cpu().numpy(), dtype=np.float64)
                    residuals = flumpy.from_numpy(res_np)
                    weights = flumpy.from_numpy(w_np)
                    J_np = np.ascontiguousarray(J_gpu.cpu().numpy(), dtype=np.float64)
                    jacobian = flumpy.from_numpy(J_np)
                    jacobian.reshape(flex.grid(n_obs, n_par))
                    if (
                        J_gpu.device.type == "cpu"
                        and n_obs <= 2000
                        and len(getattr(self._scaler, "active_scalers", (1,))) == 1
                        and not need_explicit_jacobian
                    ):
                        sp = sparse.matrix(n_obs, n_par)
                        sp.assign_block(jacobian, 0, 0)
                        jacobian = sp
                else:
                    (
                        residuals,
                        jacobian,
                        weights,
                    ) = self._parameters.compute_residuals_and_gradients(block)

                self.add_equations(residuals, jacobian, weights)

        restraints = self._parameters.compute_restraints_residuals_and_gradients(
            self._parameters
        )
        if restraints:
            if objective_only:
                self.add_residuals(restraints[0], restraints[2])
            else:
                self.add_equations(restraints[0], restraints[1], restraints[2])

        logger.debug("added equations for all blocks")
        log_memory_usage()
        logger.debug("\n")


class ScalingGaussNewtonIterationsGPU(
    ScalingLstbxBuildUpMixinGPU, ScalingGaussNewtonIterations
):
    """GPU-accelerated Gauss-Newton minimizer for scaling."""

    pass


class ScalingLevenbergMarquardtIterationsGPU(
    ScalingLstbxBuildUpMixinGPU, ScalingLevenbergMarquardtIterations
):
    """GPU-accelerated Levenberg-Marquardt minimizer for scaling."""

    pass


def scaling_refinery_gpu(
    engine, scaler, target, prediction_parameterisation, max_iterations
):
    """Return the GPU-accelerated scaling refinery engine based on phil parameters."""
    if engine == "SimpleLBFGS":
        return ScalingSimpleLBFGSGPU(
            scaler,
            target=target,
            prediction_parameterisation=prediction_parameterisation,
            max_iterations=max_iterations,
        )
    elif engine == "GaussNewton":
        return ScalingGaussNewtonIterationsGPU(
            scaler,
            target=target,
            prediction_parameterisation=prediction_parameterisation,
            max_iterations=max_iterations,
        )
    elif engine == "LevMar":
        return ScalingLevenbergMarquardtIterationsGPU(
            scaler,
            target=target,
            prediction_parameterisation=prediction_parameterisation,
            max_iterations=max_iterations,
        )
    raise ValueError(f"Unsupported scaling_refinery engine: {engine}")
