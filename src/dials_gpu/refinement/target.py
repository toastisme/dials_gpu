from __future__ import annotations

from typing import Any

import numpy as np
import torch
from dxtbx import flumpy
from scitbx import sparse
from scitbx.array_family import flex

from dials.algorithms.refinement.target import (
    LaueLeastSquaresResidualWithRmsdCutoff,
    LeastSquaresPositionalResidualWithRmsdCutoff,
    LeastSquaresPositionalResidualWithRmsdCutoffSparse,
    SparseGradientsMixin,
    Target,
    TargetFactory,
    TOFLeastSquaresResidualWithRmsdCutoff,
    phil_scope,
    phil_str,
)
from dials.algorithms.refinement.target_stills import (
    LeastSquaresStillsResidualWithRmsdCutoff,
    LeastSquaresStillsResidualWithRmsdCutoffSparse,
)


class _GPUTargetMixin:
    """Mixin accelerating Target gradient, curvature, and Jacobian calculations on the GPU."""

    def update_matches(self, force=False):
        """Avoid deep-copying all 25 columns of obs when all reflections are used in refinement."""
        if not self._matches or force:
            obs = self._reflection_manager.get_obs()
            sel = obs.get_flags(obs.flags.used_in_refinement)
            if sel.count(False) == 0:
                self._matches = obs
            else:
                self._matches = obs.select(sel)

    def predict_for_reflection_table(self, reflections, skip_derivatives=False):
        """Perform prediction in-place when all reflections pass scan range trimming."""
        if isinstance(self, LeastSquaresStillsResidualWithRmsdCutoff):
            return self._predict_core(reflections)
        if "entering" not in reflections:
            reflections.calculate_entering_flags(self._experiments)

        inc = flex.size_t_range(len(reflections))
        to_keep = flex.bool(len(inc), False)

        for iexp, exp in enumerate(self._experiments):
            sel = reflections["id"] == iexp
            if (
                exp.goniometer is None
                or exp.scan is None
                or not exp.scan.has_property("oscillation")
            ):
                to_keep.set_selected(sel, True)
                continue

            phi = reflections["xyzobs.mm.value"].parts()[2]
            phi_min, phi_max = exp.scan.get_oscillation_range(deg=False)
            passed = (phi >= phi_min) & (phi <= phi_max)
            to_keep.set_selected(sel, passed)

        if to_keep.count(False) == 0:
            return self._predict_core(reflections, skip_derivatives)

        inc = inc.select(to_keep)
        sub_refl = reflections.select(inc)
        preds = self._predict_core(sub_refl, skip_derivatives)
        reflections.set_selected(inc, preds)
        return reflections

    def split_matches_into_blocks(self, nproc: int = 1):
        """Avoid deep-copying all 25 columns of self._matches and adding 'imatch'
        when gradient_calculation_blocksize is not set and nproc == 1.
        """
        if not self._gradient_calculation_blocksize and nproc == 1:
            self.update_matches()
            return [self._matches]
        self.update_matches()
        if self._matches is self._reflection_manager.get_obs():
            self._matches = self._reflection_manager.get_matches()
        return super().split_matches_into_blocks(nproc=nproc)

    def compute_functional_gradients_and_curvatures(self, block=None):
        """Calculate the value of the target function, its gradients, and curvatures on GPU."""
        self.update_matches()
        matches = block if block is not None else self._matches
        nref = len(matches)

        if nref == 0:
            self._curv = [1.0] * len(self._prediction_parameterisation)
            return 1.0e12, [1.0] * len(self._prediction_parameterisation)

        if (
            not hasattr(self._prediction_parameterisation, "get_gradients_gpu")
            or len(self._prediction_parameterisation) * nref > 25_000_000
        ):
            return super().compute_functional_gradients_and_curvatures(block)

        residuals, weights = self._extract_residuals_and_weights(matches)
        residuals2 = self._extract_squared_residuals(matches)
        L = 0.5 * flex.sum(weights * residuals2)

        # Compute gradients on GPU: tuple of 3 (P, N) tensors
        g0, g1, g2 = self._prediction_parameterisation.get_gradients_gpu(matches)
        device = g0.device

        res_t = torch.as_tensor(
            flumpy.to_numpy(residuals), dtype=torch.float64, device=device
        ).reshape(3, nref)
        w_t = torch.as_tensor(
            flumpy.to_numpy(weights), dtype=torch.float64, device=device
        ).reshape(3, nref)
        wr = w_t * res_t

        dL_dp_t = torch.mv(g0, wr[0]) + torch.mv(g1, wr[1]) + torch.mv(g2, wr[2])
        curvs_t = (
            torch.mv(g0.square(), w_t[0])
            + torch.mv(g1.square(), w_t[1])
            + torch.mv(g2.square(), w_t[2])
        )

        dL_dp = dL_dp_t.detach().cpu().tolist()
        curvs = curvs_t.detach().cpu().tolist()
        return (L, dL_dp, curvs)

    def compute_residuals_and_normal_equations_gpu(
        self, block=None, constr_manager=None
    ):
        """Compute residuals, weights, and packed upper-triangular normal matrix + RHS
        directly on GPU via torch.matmul without materializing or transferring (3*N, P) Jacobian.
        """
        self.update_matches()
        matches = block if block is not None else self._matches

        residuals, weights = self._extract_residuals_and_weights(matches)
        g0, g1, g2 = self._prediction_parameterisation.get_gradients_gpu(matches)
        device = g0.device
        n_params, n_refl = g0.shape

        if n_refl == 0 or n_params == 0:
            n_out = (
                constr_manager.get_num_constrained_parameters()
                if constr_manager is not None
                else n_params
            )
            a_packed = np.zeros(n_out * (n_out + 1) // 2, dtype=np.float64)
            rhs = np.zeros(n_out, dtype=np.float64)
            return residuals, weights, a_packed, rhs

        res_t = torch.as_tensor(
            flumpy.to_numpy(residuals), dtype=torch.float64, device=device
        ).reshape(3, n_refl)
        w_t = torch.as_tensor(
            flumpy.to_numpy(weights), dtype=torch.float64, device=device
        ).reshape(3, n_refl)

        wg = torch.mul(g0, w_t[0].unsqueeze(0))
        A_gpu = torch.matmul(wg, g0.T)
        torch.mul(g1, w_t[1].unsqueeze(0), out=wg)
        A_gpu.addmm_(wg, g1.T)
        torch.mul(g2, w_t[2].unsqueeze(0), out=wg)
        A_gpu.addmm_(wg, g2.T)

        wr = w_t * res_t
        rhs_gpu = -(
            torch.mv(g0, wr[0]) + torch.mv(g1, wr[1]) + torch.mv(g2, wr[2])
        )

        if constr_manager is not None:
            n_constrained = constr_manager.get_num_constrained_parameters()
            C_np = np.zeros((n_constrained, n_params), dtype=np.float64)
            n_unconstrained = len(constr_manager._constrained_idx)
            for row_i, col_i in enumerate(constr_manager._constrained_idx):
                C_np[row_i, col_i] = 1.0
            for k, c in enumerate(constr_manager._constraints):
                for col_i in c.indices:
                    C_np[n_unconstrained + k, col_i] = 1.0
            C_gpu = torch.as_tensor(C_np, dtype=torch.float64, device=device)
            A_gpu = torch.matmul(torch.matmul(C_gpu, A_gpu), C_gpu.T)
            rhs_gpu = torch.mv(C_gpu, rhs_gpu)

        A_np = A_gpu.detach().cpu().numpy()
        rhs_np = rhs_gpu.detach().cpu().numpy()
        iu = np.triu_indices(A_np.shape[0])
        a_packed = A_np[iu]
        return residuals, weights, a_packed, rhs_np

    def compute_residuals_and_dense_jacobian_gpu(self, block=None):
        """Compute residuals, weights, and a contiguous dense (3*N, P) flex.double Jacobian
        directly from GPU tensors without intermediate Python dict/list/sparse vector construction.
        """
        self.update_matches()
        matches = block if block is not None else self._matches

        residuals, weights = self._extract_residuals_and_weights(matches)
        g0, g1, g2 = self._prediction_parameterisation.get_gradients_gpu(matches)
        # Concatenate along reflection dimension -> (P, 3*N), then transpose -> (3*N, P)
        jac_t = torch.cat([g0, g1, g2], dim=1).T.contiguous()
        jac_np = np.ascontiguousarray(jac_t.detach().cpu().numpy(), dtype=np.float64)
        jacobian = flumpy.from_numpy(jac_np)
        return (residuals, jacobian, weights)

    def compute_residuals_and_gradients(
        self, block=None
    ) -> tuple[Any, flex.double | sparse.matrix, Any]:
        """Return (residuals, jacobian, weights). For dense targets, uses the fast GPU
        dense Jacobian path directly; for sparse targets, delegates to the sparse builder.
        """
        if not isinstance(self, SparseGradientsMixin) and hasattr(
            self._prediction_parameterisation, "get_gradients_gpu"
        ):
            return self.compute_residuals_and_dense_jacobian_gpu(block)
        return super().compute_residuals_and_gradients(block)


class LeastSquaresPositionalResidualWithRmsdCutoffGPU(
    _GPUTargetMixin, LeastSquaresPositionalResidualWithRmsdCutoff
):
    pass


class LeastSquaresPositionalResidualWithRmsdCutoffSparseGPU(
    _GPUTargetMixin, LeastSquaresPositionalResidualWithRmsdCutoffSparse
):
    pass


class LeastSquaresStillsResidualWithRmsdCutoffGPU(
    _GPUTargetMixin, LeastSquaresStillsResidualWithRmsdCutoff
):
    pass


class LeastSquaresStillsResidualWithRmsdCutoffSparseGPU(
    _GPUTargetMixin, LeastSquaresStillsResidualWithRmsdCutoffSparse
):
    pass


class TargetFactoryGPU(TargetFactory):
    @staticmethod
    def from_parameters_and_experiments(
        params,
        experiments,
        reflection_manager,
        predictor,
        pred_param,
        restraints_param,
        do_stills=False,
        do_sparse=False,
    ):
        if params.rmsd_cutoff == "fraction_of_bin_size":
            absolute_cutoffs = None
        elif params.rmsd_cutoff == "absolute":
            absolute_cutoffs = params.absolute_cutoffs
        else:
            raise RuntimeError(
                "Target function rmsd_cutoff option"
                + params.rmsd_cutoff
                + " not recognised"
            )

        if experiments.all_tof():
            targ = TOFLeastSquaresResidualWithRmsdCutoff
        elif experiments.all_laue():
            targ = LaueLeastSquaresResidualWithRmsdCutoff
        elif do_stills:
            if do_sparse:
                targ = LeastSquaresStillsResidualWithRmsdCutoffSparseGPU
            else:
                targ = LeastSquaresStillsResidualWithRmsdCutoffGPU
        else:
            if do_sparse:
                targ = LeastSquaresPositionalResidualWithRmsdCutoffSparseGPU
            else:
                targ = LeastSquaresPositionalResidualWithRmsdCutoffGPU

        return targ(
            experiments=experiments,
            predictor=predictor,
            reflection_manager=reflection_manager,
            prediction_parameterisation=pred_param,
            restraints_parameterisation=restraints_param,
            frac_binsize_cutoff=params.bin_size_fraction,
            absolute_cutoffs=absolute_cutoffs,
            gradient_calculation_blocksize=params.gradient_calculation_blocksize,
        )


__all__ = [
    "LaueLeastSquaresResidualWithRmsdCutoff",
    "LeastSquaresPositionalResidualWithRmsdCutoff",
    "LeastSquaresPositionalResidualWithRmsdCutoffGPU",
    "LeastSquaresPositionalResidualWithRmsdCutoffSparse",
    "LeastSquaresPositionalResidualWithRmsdCutoffSparseGPU",
    "LeastSquaresStillsResidualWithRmsdCutoff",
    "LeastSquaresStillsResidualWithRmsdCutoffGPU",
    "LeastSquaresStillsResidualWithRmsdCutoffSparse",
    "LeastSquaresStillsResidualWithRmsdCutoffSparseGPU",
    "SparseGradientsMixin",
    "TOFLeastSquaresResidualWithRmsdCutoff",
    "Target",
    "TargetFactory",
    "TargetFactoryGPU",
    "phil_scope",
    "phil_str",
]
