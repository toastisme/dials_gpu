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
            or len(self._prediction_parameterisation) * nref > 10_000_000
        ):
            return super().compute_functional_gradients_and_curvatures(block)

        residuals, weights = self._extract_residuals_and_weights(matches)
        residuals2 = self._extract_squared_residuals(matches)
        L = 0.5 * flex.sum(weights * residuals2)

        # Compute gradients on GPU: tuple of 3 (P, N) tensors
        g0, g1, g2 = self._prediction_parameterisation.get_gradients_gpu(matches)
        device = g0.device
        grads_cat = torch.cat([g0, g1, g2], dim=1)  # (P, 3*N)

        res_t = torch.as_tensor(flumpy.to_numpy(residuals), dtype=torch.float64, device=device)
        w_t = torch.as_tensor(flumpy.to_numpy(weights), dtype=torch.float64, device=device)
        w_resid_t = w_t * res_t  # (3*N,)

        dL_dp_t = torch.mv(grads_cat, w_resid_t)  # (P,)
        curvs_t = torch.mv(grads_cat * grads_cat, w_t)  # (P,)

        dL_dp = dL_dp_t.detach().cpu().tolist()
        curvs = curvs_t.detach().cpu().tolist()
        return (L, dL_dp, curvs)

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
