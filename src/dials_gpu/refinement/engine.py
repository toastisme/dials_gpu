from __future__ import annotations

import logging

from libtbx import easy_mp
from scitbx import sparse

from dials.algorithms.refinement.constraints import ConstraintManager
from dials.algorithms.refinement.engine import (
    DOF_TOO_LOW,
    MAX_ITERATIONS,
    MAX_TRIAL_ITERATIONS,
    OBJECTIVE_FLAT,
    OBJECTIVE_INCREASE,
    RMSD_CONVERGED,
    STEP_TOO_SMALL,
    TARGET_ACHIEVED,
    AdaptLbfgs,
    AdaptLstbx,
    DisableMPmixin,
    GaussNewtonIterations,
    Journal,
    LBFGScurvs,
    LevenbergMarquardtIterations,
    Refinery,
    SimpleLBFGS,
    refinery_phil_scope,
    refinery_phil_str,
)
from dials.algorithms.refinement.sparse_engine import (
    AdaptLstbxSparse,
    SparseLevenbergMarquardtIterations,
)

logger = logging.getLogger(__name__)


class AdaptLstbxGPU(AdaptLstbx):
    """GPU-accelerated AdaptLstbx that bypasses scitbx.sparse.matrix construction
    whenever dense normal equations (GaussNewton, LevMar) are used.
    """

    def prepare_for_step(self):
        super().prepare_for_step()
        self._last_prepared_x = tuple(self.x)
        self._last_prepared_params = tuple(self._parameters.get_param_vals())

    def build_up(self, objective_only=False):
        if (
            getattr(self, "_last_prepared_x", None) != tuple(self.x)
            or getattr(self, "_last_prepared_params", None)
            != tuple(self._parameters.get_param_vals())
            or getattr(self._target, "_matches", None) is None
        ):
            self.prepare_for_step()
        self.reset()

        if objective_only:
            residuals, weights = self._target.compute_residuals()
            self.add_residuals(residuals, weights)
        else:
            blocks = self._target.split_matches_into_blocks(nproc=1)
            use_fast_dense = (
                not isinstance(self, AdaptLstbxSparse)
                and "jacobian_structure" not in self.history
                and hasattr(self._target, "compute_residuals_and_dense_jacobian_gpu")
                and hasattr(self._target._prediction_parameterisation, "get_gradients_gpu")
            )

            for block in blocks:
                if use_fast_dense:
                    (
                        residuals,
                        jacobian,
                        weights,
                    ) = self._target.compute_residuals_and_dense_jacobian_gpu(block)
                    if self._constr_manager is not None:
                        jacobian = ConstraintManager.constrain_jacobian(
                            self._constr_manager, jacobian
                        )
                else:
                    (
                        residuals,
                        jacobian,
                        weights,
                    ) = self._target.compute_residuals_and_gradients(block)
                    if self._constr_manager is not None:
                        jacobian = self._constr_manager.constrain_jacobian(jacobian)

                self.add_equations(residuals, jacobian, weights)

            self._jacobian = jacobian

        # Restraints terms
        restraints = self._target.compute_restraints_residuals_and_gradients()
        if restraints:
            if objective_only:
                self.add_residuals(restraints[0], restraints[2])
            else:
                j = restraints[1]
                if self._constr_manager is not None:
                    if isinstance(j, sparse.matrix):
                        j = self._constr_manager.constrain_jacobian(j)
                    else:
                        j = ConstraintManager.constrain_jacobian(
                            self._constr_manager, j
                        )
                self.add_equations(restraints[0], j, restraints[2])


class GaussNewtonIterationsGPU(AdaptLstbxGPU, GaussNewtonIterations):
    """Gauss-Newton iterations using GPU gradient and normal equation build_up."""

    pass


class LevenbergMarquardtIterationsGPU(
    GaussNewtonIterationsGPU, LevenbergMarquardtIterations
):
    """Levenberg-Marquardt iterations using GPU gradient and normal equation build_up."""

    pass


class SparseLevenbergMarquardtIterationsGPU(
    SparseLevenbergMarquardtIterations, AdaptLstbxGPU
):
    """Sparse Levenberg-Marquardt iterations using Eigen normal equations and GPU gradients."""

    pass


class AdaptLbfgsGPU(AdaptLbfgs):
    """GPU-accelerated AdaptLbfgs that runs gradient and curvature evaluations
    in-process on the GPU without forking multiprocessing workers.
    """

    def compute_functional_gradients_and_curvatures(self):
        self.prepare_for_step()

        # Run in-process on GPU (never fork with an active CUDA context)
        blocks = self._target.split_matches_into_blocks(nproc=self._nproc)
        task_results = [
            self._target.compute_functional_gradients_and_curvatures(block)
            for block in blocks
        ]

        flist, glist, clist = zip(*task_results)
        f = sum(flist)
        g = [sum(g) for g in zip(*glist)]
        c = [sum(c) for c in zip(*clist)]

        restraints = (
            self._target.compute_restraints_functional_gradients_and_curvatures()
        )
        if restraints:
            f += restraints[0]
            g = [a + b for a, b in zip(g, restraints[1])]
            c = [a + b for a, b in zip(c, restraints[2])]

        if self._constr_manager is not None:
            g = self._constr_manager.constrain_gradient_vector(g)
            c = self._constr_manager.constrain_gradient_vector(c)

        return f, self.x.__class__(g), self.x.__class__(c)


class SimpleLBFGSGPU(AdaptLbfgsGPU, SimpleLBFGS):
    """L-BFGS refinery using GPU functional and gradient evaluation."""

    pass


class LBFGScurvsGPU(AdaptLbfgsGPU, LBFGScurvs):
    """L-BFGS with curvatures refinery using GPU functional, gradient, and curvature evaluation."""

    pass


__all__ = [
    "AdaptLbfgs",
    "AdaptLstbx",
    "AdaptLstbxGPU",
    "AdaptLstbxSparse",
    "DOF_TOO_LOW",
    "DisableMPmixin",
    "GaussNewtonIterations",
    "GaussNewtonIterationsGPU",
    "Journal",
    "LBFGScurvs",
    "LBFGScurvsGPU",
    "LevenbergMarquardtIterations",
    "LevenbergMarquardtIterationsGPU",
    "MAX_ITERATIONS",
    "MAX_TRIAL_ITERATIONS",
    "OBJECTIVE_FLAT",
    "OBJECTIVE_INCREASE",
    "RMSD_CONVERGED",
    "Refinery",
    "STEP_TOO_SMALL",
    "SimpleLBFGS",
    "SimpleLBFGSGPU",
    "SparseLevenbergMarquardtIterations",
    "SparseLevenbergMarquardtIterationsGPU",
    "TARGET_ACHIEVED",
    "refinery_phil_scope",
    "refinery_phil_str",
]
