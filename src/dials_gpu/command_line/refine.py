# DIALS_ENABLE_COMMAND_LINE_COMPLETION
"""
Refine the diffraction geometry of input experiments against the input indexed
reflections on the GPU using PyTorch. For rotation scans, the model may be either
static (the same for all reflections) or scan-varying (dependent on image number
in the scan).

Examples::

  dials_gpu.refine indexed.expt indexed.refl

  dials_gpu.refine indexed.expt indexed.refl scan_varying=(False/True/Auto)
"""

from __future__ import annotations

import copy
import logging
import sys
from collections import namedtuple

import torch  # noqa: F401 -- import before omptbx_ext to avoid OpenMP runtime conflict
from dxtbx.model.experiment_list import ExperimentList
from libtbx import Auto

import dials.util
import dials.util.log
from dials.algorithms.refinement import (
    DialsRefineConfigError,
    DialsRefineRuntimeError,
)
from dials.algorithms.refinement.corrgram import create_correlation_plots
from dials.array_family import flex
from dials.command_line.refine import (
    _find_disjoint_sets,
    phil_overrides,
    phil_scope,
    usage,
    working_phil,
    write_centroids_table,
)
from dials.util.options import ArgumentParser, reflections_and_experiments_from_files
from dials.util.version import dials_version
from dials_gpu.refinement import RefinerFactoryGPU

logger = logging.getLogger("dials.command_line.refine")


def run_macrocycle(params, reflections, experiments):
    """Run one macrocycle of GPU-accelerated refinement."""
    logger.info("Configuring refiner")
    refiner = RefinerFactoryGPU.from_parameters_data_experiments(
        params, reflections, experiments
    )

    nexp = len(experiments)
    if nexp == 1:
        logger.info("Performing refinement of a single Experiment...")
    else:
        logger.info(f"Performing refinement of {nexp} Experiments...")

    history = refiner.run()

    logger.info("Updating predictions for indexed reflections")
    preds = refiner.predict_for_indexed()

    for key in preds:
        if key in reflections.keys() or key in [
            "s1",
            "xyzcal.mm",
            "xyzcal.px",
            "entering",
            "delpsical.rad",
        ]:
            reflections[key] = preds[key]

    assert len(preds) == len(reflections)
    reflections.unset_flags(
        flex.size_t_range(len(reflections)),
        reflections.flags.excluded_for_refinement
        | reflections.flags.used_in_refinement
        | reflections.flags.centroid_outlier
        | reflections.flags.predicted,
    )
    reflections.set_flags(
        preds.get_flags(preds.flags.excluded_for_refinement),
        reflections.flags.excluded_for_refinement,
    )
    reflections.set_flags(
        preds.get_flags(preds.flags.centroid_outlier),
        reflections.flags.centroid_outlier,
    )
    reflections.set_flags(
        preds.get_flags(preds.flags.used_in_refinement),
        reflections.flags.used_in_refinement,
    )
    reflections.set_flags(
        preds.get_flags(preds.flags.predicted), reflections.flags.predicted
    )

    return refiner, reflections, history


def run_dials_refine(experiments, reflections, params):
    """Functional interface to tasks performed by the program dials_gpu.refine."""
    if params.refinement.mp.nproc > 1:
        logger.warning(
            "Setting nproc > 1 is only helpful in rare "
            "circumstances. It is not recommended for typical data processing "
            "tasks."
        )

    if params.refinement.parameterisation.scan_varying is not False:
        crystal_has_scan = {}
        for j, e in enumerate(experiments):
            if e.crystal in crystal_has_scan:
                if e.scan is not crystal_has_scan[e.crystal]:
                    logger.info(
                        "Duplicating crystal model for scan-varying refinement of experiment %d",
                        j,
                    )
                    e.crystal = copy.deepcopy(e.crystal)
            else:
                crystal_has_scan[e.crystal] = e.scan

    if params.output.correlation_plot.filename is not None:
        params.refinement.refinery.journal.track_parameter_correlation = True

    scan_varying = params.refinement.parameterisation.scan_varying
    if scan_varying is Auto:
        params.refinement.parameterisation.scan_varying = False

    sparse = params.refinement.parameterisation.sparse

    disjoint_sets = _find_disjoint_sets(experiments)

    bp = params.refinement.parameterisation.beam
    ucp = params.refinement.parameterisation.crystal.unit_cell
    op = params.refinement.parameterisation.crystal.orientation
    dp = params.refinement.parameterisation.detector
    gp = params.refinement.parameterisation.goniometer
    crosslinks = any(
        (
            bp.constraints,
            ucp.constraints,
            ucp.restraints.tie_to_group,
            op.constraints,
            dp.constraints,
            gp.constraints,
        )
    )

    RefinementSet = namedtuple(
        "RefinementSet", ["experiments", "reflections", "params", "original_ids"]
    )
    if len(disjoint_sets) == 1 or not params.separate_independent_sets:
        refinement_sets = [
            RefinementSet(experiments, reflections, params, disjoint_sets[0])
        ]
    elif crosslinks:
        logger.warning(
            "The experiments contain disjoint subsets that do not share models. "
            "However, these will not be refined independently, because restraints "
            "or constraints may link models between the subsets."
        )
        refinement_sets = [
            RefinementSet(experiments, reflections, params, disjoint_sets[0])
        ]
    else:
        if not params.refinement.reflections.outlier.separate_experiments:
            reflections = RefinerFactoryGPU.reflections_after_outlier_rejection(
                params, reflections, experiments
            )
            params.refinement.reflections.outlier.algorithm = "null"

        if hasattr(params, "input"):
            params.input.reflections = None
            params.input.experiments = None

        refinement_sets = []
        for ids in disjoint_sets:
            el = ExperimentList()
            for i in ids:
                el.append(experiments[i])

            refl = flex.reflection_table()
            new_id = 0
            for i in ids:
                refl_one_experiment = reflections.select(reflections["id"] == i)
                refl_one_experiment["id"] = flex.int(len(refl_one_experiment), new_id)
                new_id += 1
                refl.extend(refl_one_experiment)
            refinement_sets.append(RefinementSet(el, refl, copy.deepcopy(params), ids))

        logger.info(
            "The experiments have been separated into independent groups that "
            "do not share models.\nRefinement will occur separately for each group:"
        )
        header = ["Group", "Experiment ids"]
        rows = []
        for i, ids in enumerate(disjoint_sets):
            rows.append([str(i), " ".join(str(e) for e in ids)])
        logger.info(dials.util.tabulate(rows, header))

    refinement_results = []
    for rs in refinement_sets:
        experiments = rs.experiments
        reflections = rs.reflections
        params = rs.params
        if len(refinement_sets) > 1:
            logger.info(
                "\nSelected group of experiments to refine with original ids: "
                + " ".join([str(i) for i in rs.original_ids])
            )
        if params.n_static_macrocycles == 1:
            refiner, reflections, history = run_macrocycle(
                params, reflections, experiments
            )
            experiments = refiner.get_experiments()
        else:
            for i in range(params.n_static_macrocycles):
                logger.info("\nStatic refinement macrocycle %s", i + 1)
                refiner, reflections, history = run_macrocycle(
                    params, reflections, experiments
                )
                experiments = refiner.get_experiments()

        if scan_varying is Auto and refiner.experiment_type == "scans":
            logger.info("\nScan-varying refinement")
            params.refinement.parameterisation.scan_varying = True
            params.refinement.parameterisation.sparse = sparse
            refiner, reflections, history = run_macrocycle(
                params, reflections, experiments
            )
            experiments = refiner.get_experiments()

        refinement_results.append((experiments, reflections, refiner, history))

    if len(refinement_results) == 1:
        experiments, reflections, refiner, history = refinement_results[0]
    else:
        experiments = {}
        reflections = flex.reflection_table()
        table_headers = []
        table_rows = {}
        for (el, refl, refiner, _), ids in zip(refinement_results, disjoint_sets):
            header, rows = refiner.calc_exp_rmsd_table()
            id_col = flex.int(len(refl))
            for new_id, orig_id in enumerate(ids):
                experiments[orig_id] = el[new_id]
                id_col.set_selected(refl["id"] == new_id, orig_id)
                rows[new_id][0] = str(orig_id)
                table_rows[orig_id] = rows[new_id]
            refl["id"] = id_col
            reflections.extend(refl)
            table_headers.append(header)

        experiments = ExperimentList([experiments[i] for i in range(len(experiments))])

        logger.info(
            "\nIndependently-refined groups of experiments have been recombined"
        )

        refiner = None
        history = None

        rows = [table_rows[i] for i in range(len(table_rows))]
        header = table_headers[0]
        if not (h == header for h in table_headers[1:]):
            logger.warning(
                "Cannot calculate RMSDs by experiment as units are inconsistent"
            )
        else:
            logger.info("\nRMSDs by experiment:")
            logger.info(dials.util.tabulate(rows, header))

    return experiments, reflections, refiner, history


@dials.util.show_mail_handle_errors()
def run(args=None, phil=working_phil, usage=usage, epilog=__doc__):
    """Set up refinement from command line options, files and PHIL parameters.
    Run GPU refinement and save output files as specified.
    """
    parser = ArgumentParser(
        usage=usage,
        phil=phil,
        read_reflections=True,
        read_experiments=True,
        check_format=False,
        epilog=epilog,
    )

    params, options = parser.parse_args(args=args, show_diff_phil=False)

    reflections, experiments = reflections_and_experiments_from_files(
        params.input.reflections, params.input.experiments
    )

    dials.util.log.config(verbosity=options.verbose, logfile=params.output.log)

    nexp = len(experiments)
    if nexp == 0 or len(reflections) == 0:
        parser.print_help()
        return
    if len(reflections) > 1:
        sys.exit("Only one reflections list can be imported at present")
    reflections = reflections[0]

    msg = (
        "The supplied reflection table does not have the required data " + "column: {0}"
    )
    for key in ["xyzobs.mm.value", "xyzobs.mm.variance"]:
        if key not in reflections:
            sys.exit(msg.format(key))

    logger.info(dials_version())

    diff_phil = parser.diff_phil.as_str()
    if diff_phil:
        logger.info("The following parameters have been modified:\n")
        logger.info(diff_phil)

    try:
        experiments, reflections, refiner, history = run_dials_refine(
            experiments, reflections, params
        )
    except (DialsRefineConfigError, DialsRefineRuntimeError) as e:
        sys.exit(str(e))

    crystals = experiments.crystals()
    if len(crystals) == 1:
        logger.info("")
        logger.info("Final refined crystal model:")
        logger.info(crystals[0])

    if params.output.centroids:
        if not refiner:
            logger.warning(
                "Cannot write table of centroids as a single refiner object is not available"
            )
        else:
            logger.info(f"Writing table of centroids to '{params.output.centroids}'")
            write_centroids_table(refiner, params.output.centroids)

    if params.output.parameter_table:
        scans = experiments.scans()
        if len(scans) > 1:
            logger.info(
                "Writing a scan-varying parameter table is only supported "
                "for refinement of a single scan"
            )
        elif not refiner:
            logger.warning(
                "Cannot write scan-varying parameter table as a single refiner object is not available"
            )
        else:
            scan = scans[0]
            text = refiner.get_param_reporter().varying_params_vs_image_number(
                scan.get_array_range()
            )
            if text:
                logger.info(
                    "Writing scan-varying parameter table to %s",
                    params.output.parameter_table,
                )
                with open(params.output.parameter_table, "w") as f:
                    f.write(text)
            else:
                logger.info("No scan-varying parameter table to write")

    if params.output.matches:
        if not refiner:
            logger.warning(
                "Cannot write matches to file as a single refiner object is not available"
            )
        else:
            matches = refiner.get_matches()
            logger.info(
                "Saving matches (use for debugging purposes) to %s",
                params.output.matches,
            )
            matches.as_file(params.output.matches)

    if params.output.correlation_plot.filename is not None:
        if not refiner:
            logger.warning(
                "Cannot create correlation plots as a single refiner object is not available"
            )
        else:
            create_correlation_plots(refiner, params.output)

    if params.output.history:
        if not history:
            logger.warning(
                "Cannot write refinement step history as a single history object is not available"
            )
        else:
            logger.info(f"Saving refinement step history to {params.output.history}")
            history.to_json_file(params.output.history)

    output_experiments_filename = params.output.experiments
    logger.info(f"Saving refined experiments to {output_experiments_filename}")
    experiments.as_file(output_experiments_filename)

    if params.output.reflections:
        del experiments
        del refiner
        del history
        logger.info(
            "Saving reflections with updated predictions to %s",
            params.output.reflections,
        )
        if params.output.include_unused_reflections:
            reflections.as_file(params.output.reflections)
        else:
            sel = reflections.get_flags(reflections.flags.used_in_refinement)
            reflections.select(sel).as_file(params.output.reflections)


if __name__ == "__main__":
    run()
