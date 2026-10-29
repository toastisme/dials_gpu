# DIALS_ENABLE_COMMAND_LINE_COMPLETION

from __future__ import annotations

import logging

import libtbx
import libtbx.phil
from dxtbx.model import ExperimentList

from dials.algorithms.shoebox import MaskCode
from dials.algorithms.spot_finding import per_image_analysis
from dials.array_family import flex
from dials.command_line.find_spots import (
    help_message,
    phil_overrides,
    phil_scope,
    working_phil,
)
from dials.util import log, show_mail_handle_errors
from dials.util.ascii_art import spot_counts_per_image_plot
from dials.util.exclude_images import (
    expand_exclude_multiples,
    set_invalid_images,
)
from dials.util.multi_dataset_handling import generate_experiment_identifiers
from dials.util.options import ArgumentParser, flatten_experiments
from dials.util.version import dials_version
from dials_gpu.spot_finding.finder import SpotFinderFactoryGPU

logger = logging.getLogger("dials.command_line.find_spots")
flex_logger = logging.getLogger("dials.array_family.flex_ext")


def from_observations_gpu(
    experiments: ExperimentList,
    params: libtbx.phil.scope_extract | None = None,
    is_stills: bool = False,
) -> flex.reflection_table:
    """
    GPU-accelerated replacement for flex.reflection_table.from_observations.
    """
    if params is None:
        from dials.util.phil import parse

        params = phil_scope.fetch(source=parse("")).extract()

    if params.spotfinder.filter.min_spot_size is libtbx.Auto:
        detector = experiments[0].imageset.get_detector()
        if detector[0].get_type() == "SENSOR_PAD":
            params.spotfinder.filter.min_spot_size = 3
        else:
            params.spotfinder.filter.min_spot_size = 6
        flex_logger.info(
            "Setting spotfinder.filter.min_spot_size=%i",
            params.spotfinder.filter.min_spot_size,
        )

    if params.spotfinder.exclude_images_multiple:
        params.spotfinder.exclude_images = expand_exclude_multiples(
            experiments,
            params.spotfinder.exclude_images_multiple,
            params.spotfinder.exclude_images,
        )
    experiments = set_invalid_images(experiments, params.spotfinder.exclude_images)

    flex_logger.info("Configuring spot finder from input parameters (GPU)")
    spotfinder = SpotFinderFactoryGPU.from_parameters(
        experiments=experiments, params=params, is_stills=is_stills
    )

    return spotfinder.find_spots(experiments)


def do_spotfinding(
    experiments: ExperimentList,
    params: libtbx.phil.scope_extract,
) -> tuple[ExperimentList, flex.reflection_table] | flex.reflection_table:
    had_identifiers = False
    if all(i != "" for i in experiments.identifiers()):
        had_identifiers = True
    else:
        generate_experiment_identifiers(experiments)

    if params.maximum_trusted_value is not None:
        logger.info(
            "Overriding maximum trusted value to %.1f", params.maximum_trusted_value
        )
        input_trusted_ranges = {}
        for _d, detector in enumerate(experiments.detectors()):
            for _p, panel in enumerate(detector):
                trusted = panel.get_trusted_range()
                input_trusted_ranges[(_d, _p)] = trusted
                panel.set_trusted_range((trusted[0], params.maximum_trusted_value))

    # Loop through all the imagesets and find the strong spots on the GPU
    reflections = from_observations_gpu(experiments, params)

    # Add n_signal column - before deleting shoeboxes
    if "shoebox" in reflections:
        good = MaskCode.Foreground | MaskCode.Valid
        reflections["n_signal"] = reflections["shoebox"].count_mask_values(good)

        if not params.output.shoeboxes:
            del reflections["shoebox"]

    # ascii spot count per image plot - per imageset
    imagesets = []
    for i, experiment in enumerate(experiments):
        if experiment.imageset not in imagesets:
            imagesets.append(experiment.imageset)

    for i, imageset in enumerate(imagesets):
        selected = flex.bool(reflections.nrows(), False)
        for j, experiment in enumerate(experiments):
            if experiment.imageset is not imageset:
                continue
            selected.set_selected(reflections["id"] == j, True)
        ascii_plot = spot_counts_per_image_plot(reflections.select(selected))
        if len(ascii_plot):
            logger.info("\nHistogram of per-image spot count for imageset %i:", i)
            logger.info(ascii_plot)

    # Save the reflections to file
    logger.info("\n" + "-" * 80)
    if not had_identifiers:
        if not params.output.experiments:
            for k in reflections.experiment_identifiers().keys():
                del reflections.experiment_identifiers()[k]

    if len(reflections):
        reflections.as_file(params.output.reflections)
        logger.info(
            "Saved %s reflections to %s", len(reflections), params.output.reflections
        )
    else:
        logger.warning("No reflections found")

    if params.maximum_trusted_value is not None:
        for _d, detector in enumerate(experiments.detectors()):
            for _p, panel in enumerate(detector):
                trusted = input_trusted_ranges[(_d, _p)]
                panel.set_trusted_range(trusted)

    if params.output.experiments:
        logger.info(f"Saving experiments to {params.output.experiments}")
        experiments.as_file(params.output.experiments)

    if params.per_image_statistics:
        for i, experiment in enumerate(experiments):
            logger.info("Number of centroids per image for imageset %i:", i)
            refl = reflections.select(reflections["id"] == i)
            refl.centroid_px_to_mm([experiment])
            refl.map_centroids_to_reciprocal_space([experiment])
            stats = per_image_analysis.stats_per_image(
                experiment, refl, resolution_analysis=False
            )
            logger.info(str(stats))

    if params.output.experiments:
        return experiments, reflections
    else:
        return reflections


@show_mail_handle_errors()
def run(args=None, *, phil=working_phil, return_results=False):
    usage = (
        "usage: dials_gpu.find_spots [options] [param.phil] "
        "{models.expt | image1.file [image2.file ...]}"
    )

    parser = ArgumentParser(
        usage=usage,
        phil=phil,
        epilog=help_message,
        read_experiments_from_images=True,
        read_experiments=True,
    )

    params, options = parser.parse_args(args=args, show_diff_phil=False)
    log.config(verbosity=options.verbose, logfile=params.output.log)
    logger.info(dials_version())

    diff_phil = parser.diff_phil.as_str()
    if diff_phil != "":
        logger.info("The following parameters have been modified:\n")
        logger.info(diff_phil)

    experiments = flatten_experiments(params.input.experiments)

    if len(experiments) == 0:
        parser.print_help()
        return

    results = do_spotfinding(experiments, params)
    if return_results:
        return results


if __name__ == "__main__":
    run()
