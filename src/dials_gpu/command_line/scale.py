"""
GPU-accelerated drop-in replacement for dials.scale (dials_gpu.scale).
Reproduces the exact dials.scale CLI and PHIL interface while running
scaling model evaluation, symmetry group reductions, target gradients/Jacobians,
outlier rejection, intensity combination, and error model refinement on the GPU.
"""

from __future__ import annotations

import logging
import sys

import torch  # Must be imported before scitbx/omptbx_ext to avoid OpenMP runtime conflict
from dxtbx.model import ExperimentList
from iotbx import phil

from dials.array_family import flex
from dials.command_line.scale import (
    __doc__ as _dials_scale_doc,
    _export_merged_mtz,
    _export_unmerged_mtz,
    phil_scope,
)
from dials.util import Sorry, log, show_mail_handle_errors
from dials.util.options import ArgumentParser, reflections_and_experiments_from_files
from dials.util.version import dials_version

from ..scaling.algorithm import ScaleAndFilterAlgorithmGPU, ScalingAlgorithmGPU

__doc__ = _dials_scale_doc
logger = logging.getLogger("dials")


def run_scaling(
    params: phil.scope_extract,
    experiments: ExperimentList,
    reflections: list[flex.reflection_table],
) -> tuple[ExperimentList, flex.reflection_table]:
    """Run scaling algorithms on the GPU: scale, cross-validate, or scale and filter."""
    if params.output.delete_integration_shoeboxes:
        for r in reflections:
            del r["shoebox"]

    if params.cross_validation.cross_validation_mode:
        from dials.algorithms.scaling.cross_validation.cross_validate import (
            cross_validate,
        )
        from dials.algorithms.scaling.cross_validation.crossvalidator import (
            DialsScaleCrossValidator,
        )

        class DialsScaleCrossValidatorGPU(DialsScaleCrossValidator):
            def run_script(self, params, config_no):
                val_params = self.set_free_set_offset(params, config_no)
                alg = ScalingAlgorithmGPU(
                    val_params,
                    self.experiments.deep_copy(),
                    [r.deep_copy() for r in self.reflections],
                )
                alg.run()
                return alg.scaler

        cross_validator = DialsScaleCrossValidatorGPU(experiments, reflections)
        cross_validate(params, cross_validator)
        logger.info(
            "Cross validation analysis does not produce scaling output files, rather\n"
            "it gives insight into the dataset. Choose an appropriate parameterisation\n"
            "and rerun scaling without cross_validation_mode.\n"
        )
        return (None, None)

    if params.filtering.method:
        algorithm = ScaleAndFilterAlgorithmGPU(params, experiments, reflections)
    else:
        algorithm = ScalingAlgorithmGPU(params, experiments, reflections)

    algorithm.run()
    experiments, joint_table = algorithm.finish()
    return experiments, joint_table


@show_mail_handle_errors()
def run(args: list[str] = None, phil: phil.scope = phil_scope) -> None:
    """Run GPU scaling from the command-line."""
    usage = """Usage: dials_gpu.scale integrated.refl integrated.expt
[integrated.refl(2) integrated.expt(2) ....] [options]"""

    parser = ArgumentParser(
        usage=usage,
        read_experiments=True,
        read_reflections=True,
        phil=phil,
        check_format=False,
        epilog=__doc__,
    )
    params, options = parser.parse_args(args=args, show_diff_phil=False)

    if not params.input.experiments or not params.input.reflections:
        parser.print_help()
        sys.exit()

    reflections, experiments = reflections_and_experiments_from_files(
        params.input.reflections, params.input.experiments
    )

    log.config(verbosity=options.verbose, logfile=params.output.log)
    logger.info(dials_version())

    diff_phil = parser.diff_phil.as_str()
    if diff_phil:
        logger.info("The following parameters have been modified:\n%s", diff_phil)

    try:
        scaled_experiments, joint_table = run_scaling(params, experiments, reflections)
    except (ValueError, RuntimeError) as e:
        raise Sorry(e)
    else:
        if scaled_experiments and joint_table:
            logger.info(
                "Saving the scaled experiments to %s", params.output.experiments
            )
            scaled_experiments.as_file(
                params.output.experiments, history_as_scaled=True
            )
            logger.info(
                "Saving the scaled reflections to %s", params.output.reflections
            )
            joint_table.as_file(params.output.reflections)

            if params.output.unmerged_mtz:
                _export_unmerged_mtz(params, scaled_experiments, joint_table)

            if params.output.merged_mtz:
                _export_merged_mtz(params, scaled_experiments, joint_table)

    logger.info(
        "See dials.github.io/dials_scale_user_guide.html for more info on scaling options"
    )


if __name__ == "__main__":
    run()
