from __future__ import annotations

import logging
import sys
import warnings

import libtbx
from dxtbx.model import ExperimentList

import dials.util.log
from dials.array_family import flex
from dials.command_line.integrate import (
    __doc__ as help_message,
    filter_reference_pixels,
    phil_scope,
    process_reference,
    sample_predictions,
    split_for_scan_range,
    working_phil,
)
from dials.util import show_mail_handle_errors
from dials.util.exclude_images import (
    expand_exclude_multiples,
    set_invalid_images,
)
from dials.util.options import ArgumentParser, reflections_and_experiments_from_files
from dials.util.version import dials_version

from dials_gpu.integration.integrator import create_integrator_gpu
from dials_gpu.integration.profile import ProfileModelFactoryGPU

logger = logging.getLogger("dials.command_line.integrate")


def run_integration(params, experiments, reference=None):
    """
    GPU-accelerated integration entry point matching `dials.command_line.integrate.run_integration`.
    """
    if params.integration.mp.nproc is libtbx.Auto:
        params.integration.mp.nproc = 1

    predicted = None
    rubbish = None

    for abs_params in params.absorption_correction:
        if abs_params.apply:
            if not (
                params.integration.debug.output
                and not params.integration.debug.separate_files
            ):
                raise ValueError(
                    "Shoeboxes must be saved to integration intermediates to apply an absorption correction. "
                    + "Set integration.debug.output=True, integration.debug.separate_files=False and "
                    + "integration.debug.delete_shoeboxes=True to temporarily store shoeboxes."
                )

    for i, exp in enumerate(experiments):
        mask = exp.imageset.external_lookup.mask
        if mask.filename is not None:
            if mask.data:
                logger.info("Using external mask: %s", mask.filename)
                for tile in mask.data:
                    logger.info(" Mask has %d pixels masked", tile.data().count(False))

    for i, exp in enumerate(experiments):
        summary = "\n".join(
            (
                "",
                "=" * 80,
                "",
                "Experiments",
                "",
                "Models for experiment %d" % i,
                "",
                str(exp.beam),
                str(exp.detector),
            )
        )
        if exp.goniometer:
            summary += str(exp.goniometer) + "\n"
        if exp.scan:
            summary += str(exp.scan) + "\n"
        summary += str(exp.crystal)
        logger.info(summary)

    logger.info("\n".join(("", "=" * 80, "")))
    logger.info("Initialising")

    profile_reference = None
    if reference:
        profile_reference, rubbish = process_reference(reference)

        if exp.goniometer is not None and exp.scan is not None:
            profile_reference = filter_reference_pixels(profile_reference, experiments)

        experiments, profile_reference = split_for_scan_range(
            experiments, profile_reference, params.scan_range
        )

    if params.exclude_images_multiple:
        params.exclude_images = expand_exclude_multiples(
            experiments,
            params.exclude_images_multiple,
            params.exclude_images,
        )
    if params.exclude_images:
        try:
            experiments = set_invalid_images(experiments, params.exclude_images)
        except ValueError as err:
            try:
                exclude_images = [
                    int(e)
                    for e in str(params.exclude_images)
                    .replace("[", "")
                    .replace("]", "")
                    .replace("'", "")
                    .split(",")
                ]
            except ValueError:
                raise (err)
            warnings.warn(
                "Providing exclude_images as a single list (e.g. 1,2,3,4,5 etc.) is deprecated.\n"
                + str(err),
                DeprecationWarning,
                stacklevel=2,
            )
            for experiment in experiments:
                for index in exclude_images:
                    experiment.imageset.mark_for_rejection(index, True)

    logger.info("\n".join(("", "=" * 80, "")))
    logger.info("Predicting reflections")
    predicted = flex.reflection_table.from_predictions_multi(
        experiments,
        dmin=params.prediction.d_min,
        dmax=params.prediction.d_max,
        margin=params.prediction.margin,
        force_static=params.prediction.force_static,
        padding=params.prediction.padding,
    )
    isets = {e.imageset: i for i, e in enumerate(experiments)}
    predicted["imageset_id"] = flex.int(predicted.size(), 0)
    if len(isets) > 1:
        for e in experiments:
            iset_id = isets[e.imageset]
            for id_ in predicted.experiment_identifiers().keys():
                identifier = predicted.experiment_identifiers()[id_]
                if identifier == e.identifier:
                    sel = predicted["id"] == id_
                    predicted["imageset_id"].set_selected(sel, iset_id)
                    break

    if profile_reference:
        matched, profile_reference, unmatched = predicted.match_with_reference(
            profile_reference
        )
        assert len(matched) == len(predicted)
        assert matched.count(True) <= len(profile_reference)
        if matched.count(True) == 0:
            raise ValueError(
                """
        Invalid input for reference reflections.
        Zero reference spots were matched to predictions
    """
            )
        elif unmatched:
            msg = (
                "Warning: %d reference spots were not matched to predictions"
                % unmatched.size()
            )
            border = "\n".join(("", "*" * 80, ""))
            logger.info("".join((border, msg, border)))
            rubbish.extend(unmatched)

        if len(experiments) > 1:
            f_reference = flex.reflection_table()
            f_predicted = flex.reflection_table()
            f_rubbish = flex.reflection_table()
            f_experiments = ExperimentList()
            good_expt_count = 0

            def refl_extend(src, dest, eid):
                old_id = eid
                new_id = good_expt_count
                tmp = src.select(src["id"] == old_id)
                tmp["id"] = flex.int(len(tmp), good_expt_count)
                if old_id in tmp.experiment_identifiers():
                    identifier = tmp.experiment_identifiers()[old_id]
                    del tmp.experiment_identifiers()[old_id]
                    tmp.experiment_identifiers()[new_id] = identifier
                dest.extend(tmp)

            for expt_id, experiment in enumerate(experiments):
                if (profile_reference["id"] == expt_id).count(True):
                    refl_extend(profile_reference, f_reference, expt_id)
                    refl_extend(predicted, f_predicted, expt_id)
                    refl_extend(rubbish, f_rubbish, expt_id)
                    f_experiments.append(experiment)
                    good_expt_count += 1
                else:
                    logger.info(
                        "Removing experiment %d: no reference reflections matched to predictions",
                        expt_id,
                    )

            profile_reference = f_reference
            predicted = f_predicted
            experiments = f_experiments
            rubbish = f_rubbish

    if not params.sampling.integrate_all_reflections:
        predicted = sample_predictions(experiments, predicted, params)

    # Compute the profile model on GPU
    experiments = ProfileModelFactoryGPU.create(params, experiments, profile_reference)
    for expr in experiments:
        if expr.profile is None:
            raise ValueError("No profile information in experiment list")
    del profile_reference

    if reference and "shoebox" in reference:
        del reference["shoebox"]
    if not params.output.include_bad_reference:
        rubbish = None

    predicted.compute_bbox(experiments)

    integrator = create_integrator_gpu(params, experiments, predicted)
    reflections = integrator.integrate()

    if not params.output.output_unintegrated_reflections:
        keep = reflections.get_flags(reflections.flags.integrated, all=False)
        logger.info(
            "Removing %d unintegrated reflections of %d total",
            keep.count(False),
            keep.size(),
        )
        reflections = reflections.select(keep)

    if rubbish is not None and params.output.include_bad_reference:
        mask = flex.bool(len(rubbish), True)
        rubbish.unset_flags(mask, rubbish.flags.integrated_sum)
        rubbish.unset_flags(mask, rubbish.flags.integrated_prf)
        rubbish.set_flags(mask, rubbish.flags.bad_reference)
        reflections.extend(rubbish)

    for abs_params in params.absorption_correction:
        if abs_params.apply:
            if abs_params.algorithm == "fuller_kapton":
                from dials.algorithms.integration.kapton_correction import (
                    multi_kapton_correction,
                )
            elif abs_params.algorithm == "kapton_2019":
                from dials.algorithms.integration.kapton_2019_correction import (
                    multi_kapton_correction,
                )
            elif abs_params.algorithm == "other":
                continue
            else:
                raise ValueError(
                    "absorption_correction.apply=True, "
                    "but no .algorithm has been selected!"
                )
            experiments, reflections = multi_kapton_correction(
                experiments, reflections, abs_params.fuller_kapton, logger=logger
            )()

    if params.significance_filter.enable:
        from dials.algorithms.integration.stills_significance_filter import (
            SignificanceFilter,
        )

        sig_filter = SignificanceFilter(params)
        filtered_refls = sig_filter(experiments, reflections)
        accepted_expts = ExperimentList()
        accepted_refls = flex.reflection_table()
        logger.info(
            "Removed %d reflections out of %d when applying significance filter",
            (reflections.size() - filtered_refls.size()),
            reflections.size(),
        )
        for expt_id, expt in enumerate(experiments):
            refls = filtered_refls.select(filtered_refls["id"] == expt_id)
            if refls:
                accepted_expts.append(expt)
                current_id = expt_id
                new_id = len(accepted_expts) - 1
                refls["id"] = flex.int(len(refls), new_id)
                if expt.identifier:
                    del refls.experiment_identifiers()[current_id]
                    refls.experiment_identifiers()[new_id] = expt.identifier
                accepted_refls.extend(refls)
            else:
                logger.info(
                    "Removed experiment %d which has no reflections left after applying significance filter",
                    expt_id,
                )

        if not accepted_refls:
            raise ValueError("No reflections left after applying significance filter")
        experiments = accepted_expts
        reflections = accepted_refls

    report = None
    if params.output.report is not None:
        report = integrator.report()

    return experiments, reflections, report


@show_mail_handle_errors()
def run(args=None, phil=working_phil):
    """Run the dials_gpu.integrate command line script."""
    usage = "usage: dials_gpu.integrate [options] models.expt"

    parser = ArgumentParser(
        usage=usage,
        phil=phil,
        epilog=help_message,
        read_experiments=True,
        read_reflections=True,
    )

    params, options = parser.parse_args(args=args, show_diff_phil=False)

    dials.util.log.config(verbosity=options.verbose, logfile=params.output.log)

    logger.info(dials_version())

    diff_phil = parser.diff_phil.as_str()
    if diff_phil:
        logger.info("The following parameters have been modified:\n%s", diff_phil)
    if params.output.phil is not None:
        with open(params.output.phil, "w") as outfile:
            outfile.write(parser.diff_phil.as_str())

    reference, experiments = reflections_and_experiments_from_files(
        params.input.reflections, params.input.experiments
    )
    params.input.reflections = []
    params.input.experiments = []

    if not reference and not experiments:
        parser.print_help()
        return
    if not experiments:
        sys.exit("No experiment list was specified")
    if not reference:
        reference = None
    elif len(reference) != 1:
        sys.exit("More than 1 reflection file was given")
    else:
        reference = reference[0]

    if reference and "shoebox" not in reference:
        sys.exit("Error: shoebox data missing from reflection table")

    try:
        experiments, reflections, report = run_integration(
            params, experiments, reference
        )
    except (ValueError, RuntimeError) as e:
        sys.exit(e)
    else:
        if params.integration.debug.delete_shoeboxes and "shoebox" in reflections:
            del reflections["shoebox"]

        logger.info(
            "Saving %d reflections to %s", reflections.size(), params.output.reflections
        )
        reflections.as_file(params.output.reflections)
        logger.info("Saving the experiments to %s", params.output.experiments)
        experiments.as_file(params.output.experiments, history_as_integrated=True)

        if report:
            report.as_file(params.output.report)


if __name__ == "__main__":
    run()
