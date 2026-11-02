from __future__ import annotations

import logging

from dxtbx import flumpy
import libtbx
import numpy as np
from scitbx.array_family import flex

from dials.algorithms.refinement import DialsRefineConfigError
from dials.algorithms.refinement.constraints import ConstraintManagerFactory
from dials.algorithms.refinement.engine import AdaptLstbx
from dials.algorithms.refinement.parameterisation.autoreduce import AutoReduce
from dials.algorithms.refinement.parameterisation.parameter_report import (
    ParameterReporter,
)
from dials.algorithms.refinement.refinement_helpers import (
    compute_radial_and_transverse_residuals,
)
from dials.algorithms.refinement.refiner import (
    RAD2DEG,
    Refiner,
    RefinerFactory,
    ScanVaryingRefiner,
    _copy_experiments_for_refining,
    _trim_scans_to_observations,
    phil_scope,
)
from dials.algorithms.refinement.reflection_manager import BlockCalculator
from dials.util.system import MEMORY_LIMIT
from dials_gpu.refinement.engine import (
    GaussNewtonIterationsGPU,
    LBFGScurvsGPU,
    LevenbergMarquardtIterationsGPU,
    SimpleLBFGSGPU,
    SparseLevenbergMarquardtIterationsGPU,
)
from dials_gpu.refinement.outlier_detection import (
    BlockCalculatorGPU,
    ReflectionManagerFactoryGPU,
)
from dials_gpu.refinement.parameterisation import build_prediction_parameterisation_gpu
from dials_gpu.refinement.predictor import ExperimentsPredictorFactoryGPU
from dials_gpu.refinement.target import TargetFactoryGPU

logger = logging.getLogger(__name__)


class _GPURefinerMixin:
    """Mixin for Refiner avoiding redundant 25-column deep copies and sorts."""

    def selection_used_for_refinement(self):
        obs = self._refman.get_obs()
        used = obs.get_flags(obs.flags.used_in_refinement)
        isel = obs["iobs"] if used.count(False) == 0 else obs["iobs"].select(used)
        selection = flex.bool(len(self._refman.get_indexed()), False)
        selection.set_selected(isel, True)
        return selection

    def predict_for_indexed(self):
        reflections = self.predict_for_reflection_table(
            self._refman.get_indexed(), skip_derivatives=True
        )
        iobs_np = flumpy.to_numpy(reflections["iobs"])
        if len(iobs_np) > 1 and np.any(iobs_np[:-1] > iobs_np[1:]):
            reflections.sort("iobs")
        mask = self.selection_used_for_refinement()
        reflections.set_flags(mask, reflections.flags.used_in_refinement)
        return reflections


class RefinerGPU(_GPURefinerMixin, Refiner):
    """GPU-accelerated Refiner."""

    pass


class ScanVaryingRefinerGPU(_GPURefinerMixin, ScanVaryingRefiner):
    """GPU-accelerated ScanVaryingRefiner."""

    pass


class RefinerFactoryGPU(RefinerFactory):
    """Factory class to create GPU-accelerated refiners."""

    @classmethod
    def _build_reflection_manager_and_predictor(cls, params, reflections, experiments):
        single_as_still = params.refinement.parameterisation.treat_single_image_as_still
        exps_are_stills = []
        for exp in experiments:
            if exp.scan is None:
                exps_are_stills.append(True)
            elif exp.scan.get_num_images() == 1:
                if single_as_still:
                    exps_are_stills.append(True)
                elif exp.scan.is_still():
                    exps_are_stills.append(True)
                else:
                    exps_are_stills.append(False)
            else:
                if exp.scan.is_still():
                    raise DialsRefineConfigError("Cannot refine a zero-width scan")
                exps_are_stills.append(False)

        if not all(exps_are_stills[0] == e for e in exps_are_stills):
            raise DialsRefineConfigError("Cannot refine a mixture of stills and scans")
        do_stills = exps_are_stills[0]

        if do_stills:
            params.refinement.parameterisation.scan_varying = False

        if params.refinement.parameterisation.scan_varying is libtbx.Auto:
            params.refinement.parameterisation.scan_varying = False

        if params.refinement.parameterisation.scan_varying:
            if params.refinement.parameterisation.trim_scan_to_observations:
                experiments = _trim_scans_to_observations(experiments, reflections)

            block_calculator = BlockCalculatorGPU(experiments, reflections)
            if params.refinement.parameterisation.compose_model_per == "block":
                reflections = block_calculator.per_width(
                    params.refinement.parameterisation.block_width, deg=True
                )
            elif params.refinement.parameterisation.compose_model_per == "image":
                reflections = block_calculator.per_image()

        logger.debug("\nBuilding reflection manager")
        logger.debug("Input reflection list size = %d observations", len(reflections))

        refman = ReflectionManagerFactoryGPU.from_parameters_reflections_experiments(
            params.refinement.reflections, reflections, experiments, do_stills
        )

        logger.debug(
            "Number of observations that pass initial inclusion criteria = %d",
            refman.get_accepted_refs_size(),
        )
        sample_size = refman.get_sample_size()
        if sample_size > 0:
            logger.debug("Working set size = %d observations", sample_size)
        logger.debug("Reflection manager built\n")

        params = cls.config_sparse(params, experiments)

        ref_predictor = ExperimentsPredictorFactoryGPU.from_experiments(
            experiments,
            force_stills=do_stills,
            spherical_relp=params.refinement.parameterisation.spherical_relp_model,
        )

        obs = refman.get_obs()
        ref_predictor(obs)
        x_obs, y_obs, phi_obs = obs["xyzobs.mm.value"].parts()
        x_calc, y_calc, phi_calc = obs["xyzcal.mm"].parts()
        obs["x_resid"] = x_calc - x_obs
        obs["y_resid"] = y_calc - y_obs
        obs["phi_resid"] = phi_calc - phi_obs
        refman.update_residuals()

        if (
            params.refinement.reflections.outlier.algorithm == "mcd"
            and params.refinement.reflections.outlier.mcd.positional_coordinates
            in ("radial_transverse", "deltatt_transverse")
        ):
            compute_radial_and_transverse_residuals(
                experiments,
                obs,
                two_theta=params.refinement.reflections.outlier.mcd.positional_coordinates
                == "deltatt_transverse"
                or params.refinement.reflections.outlier.mcd.positional_coordinates
                == "delpsidstar",
            )

        if params.refinement.reflections.outlier.block_width is libtbx.Auto:
            ca = refman.get_centroid_analyser()
            analysis = ca(calc_average_residuals=False, calc_periodograms=False)
        else:
            analysis = None

        refman.finalise(analysis)

        return (experiments, refman, ref_predictor, do_stills)

    @classmethod
    def _build_refiner(cls, params, experiments, refman, ref_predictor, do_stills):
        do_sparse = params.refinement.parameterisation.sparse

        logger.debug("Building prediction equation parameterisation")
        pred_param = build_prediction_parameterisation_gpu(
            params.refinement.parameterisation, experiments, refman, do_stills
        )

        cmf = ConstraintManagerFactory(params, pred_param)
        constraints_manager = cmf()

        autoreduce = AutoReduce(
            params.refinement.parameterisation.auto_reduction,
            pred_param,
            refman,
            constraints_manager,
            cmf,
        )
        autoreduce()

        constraints_manager = autoreduce.constraints_manager

        restraints_parameterisation = cls.config_restraints(
            params.refinement.parameterisation, pred_param
        )

        logger.debug("Prediction equation parameterisation built")
        logger.debug("Parameter order : name mapping")
        for i, e in enumerate(pred_param.get_param_names()):
            logger.debug("Parameter %03d : %s", i + 1, e)

        param_reporter = ParameterReporter(
            pred_param.get_detector_parameterisations(),
            pred_param.get_beam_parameterisations(),
            pred_param.get_crystal_orientation_parameterisations(),
            pred_param.get_crystal_unit_cell_parameterisations(),
            pred_param.get_goniometer_parameterisations(),
        )

        logger.debug("Building target function")
        target = cls.config_target(
            params.refinement.target,
            experiments,
            refman,
            ref_predictor,
            pred_param,
            restraints_parameterisation,
            do_stills,
            do_sparse,
        )
        logger.debug("Target function built")

        logger.debug("Building refinement engine")
        refinery = cls.config_refinery(params, target, pred_param, constraints_manager)
        logger.debug("Refinement engine built")

        nparam = len(pred_param)
        ndim = target.dim
        obs = refman.get_obs()
        nref = obs.get_flags(obs.flags.used_in_refinement).count(True)
        logger.info(
            "There are %s parameters to refine against %s reflections in %s dimensions",
            nparam,
            nref,
            ndim,
        )

        if not params.refinement.parameterisation.sparse and isinstance(
            refinery, AdaptLstbx
        ):
            dense_jacobian_gigabytes = (
                nparam * nref * ndim * flex.double.element_size()
            ) / 1e9
            avail_memory_gigabytes = MEMORY_LIMIT / 1e9
            if (
                dense_jacobian_gigabytes > 0.2 * avail_memory_gigabytes
                or dense_jacobian_gigabytes > 0.5
            ):
                logger.info(
                    "Storage of the Jacobian matrix requires %.1f GB",
                    dense_jacobian_gigabytes,
                )

        if params.refinement.parameterisation.scan_varying:
            refiner = ScanVaryingRefinerGPU
        else:
            refiner = RefinerGPU
        return refiner(
            experiments, pred_param, param_reporter, refman, target, refinery
        )

    @staticmethod
    def config_refinery(params, target, pred_param, constraints_manager):
        options = params.refinement.refinery

        if options.engine == "SimpleLBFGS":
            refinery = SimpleLBFGSGPU
        elif options.engine == "LBFGScurvs":
            refinery = LBFGScurvsGPU
        elif options.engine == "GaussNewton":
            refinery = GaussNewtonIterationsGPU
        elif options.engine == "LevMar":
            refinery = LevenbergMarquardtIterationsGPU
        elif options.engine == "SparseLevMar":
            refinery = SparseLevenbergMarquardtIterationsGPU
        else:
            raise RuntimeError(
                "Refinement engine " + options.engine + " not recognised"
            )

        logger.debug("Selected refinement engine type: %s", options.engine)

        engine = refinery(
            target=target,
            prediction_parameterisation=pred_param,
            constraints_manager=constraints_manager,
            log=options.log,
            tracking=options.journal,
            max_iterations=options.max_iterations,
        )

        if params.refinement.mp.nproc > 1:
            nproc = params.refinement.mp.nproc
            try:
                engine.set_nproc(nproc)
            except NotImplementedError:
                logger.warning(
                    "Could not set nproc=%s for refinement engine of type %s",
                    nproc,
                    options.engine,
                )

        return engine

    @staticmethod
    def config_target(
        params,
        experiments,
        reflection_manager,
        predictor,
        pred_param,
        restraints_param,
        do_stills,
        do_sparse,
    ):
        return TargetFactoryGPU.from_parameters_and_experiments(
            params,
            experiments,
            reflection_manager,
            predictor,
            pred_param,
            restraints_param,
            do_stills,
            do_sparse,
        )


__all__ = [
    "RAD2DEG",
    "Refiner",
    "RefinerFactory",
    "RefinerFactoryGPU",
    "RefinerGPU",
    "ScanVaryingRefiner",
    "ScanVaryingRefinerGPU",
    "_copy_experiments_for_refining",
    "_trim_scans_to_observations",
    "phil_scope",
]
