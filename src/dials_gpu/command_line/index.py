# DIALS_ENABLE_COMMAND_LINE_COMPLETION

from __future__ import annotations

import concurrent.futures
import copy
import logging
import sys

from cctbx.sgtbx import space_group_info
from dxtbx.model import Crystal
from dxtbx.model.experiment_list import ExperimentList
from libtbx import Auto

from dials.algorithms.indexing import DialsIndexError
from dials.array_family import flex
from dials.command_line.index import (
    help_message,
    phil_overrides,
    phil_scope,
    working_phil,
)
from dials.util import log, show_mail_handle_errors
from dials.util.options import ArgumentParser, reflections_and_experiments_from_files
from dials.util.slice import slice_reflections
from dials.util.version import dials_version
from dials_gpu.indexing.indexer import IndexerGPU

logger = logging.getLogger("dials.command_line.index")


def _index_experiments(
    experiments,
    reflections,
    params,
    known_crystal_models=None,
    log_text=None,
):
    if log_text:
        logger.info(log_text)
    idxr = IndexerGPU.from_parameters(
        reflections,
        experiments,
        known_crystal_models=known_crystal_models,
        params=params,
    )
    idxr.index()
    idx_refl = copy.deepcopy(idxr.refined_reflections)
    idx_refl.extend(idxr.unindexed_reflections)
    return idxr.refined_experiments, idx_refl


def index(experiments, reflections, params):
    """
    Index the input experiments and reflections on the GPU using IndexerGPU.
    Reproduces dials.command_line.index.index() exactly.
    """
    if params.indexing.known_symmetry.A_matrix is not None:
        sg = params.indexing.known_symmetry.space_group
        if sg is None:
            sg = space_group_info("P1")
        crystal = Crystal(params.indexing.known_symmetry.A_matrix, sg.group())
        for expt in experiments:
            expt.crystal = crystal

    if experiments.crystals()[0] is not None:
        known_crystal_models = [expt.crystal for expt in experiments]
    else:
        known_crystal_models = None

    if len(reflections) == 0:
        raise ValueError("No reflection lists found in input")
    elif len(reflections) == 1:
        if "imageset_id" not in reflections[0]:
            reflections[0]["imageset_id"] = reflections[0]["id"]
    elif len(reflections) > 1:
        assert len(reflections) == len(experiments)
        for i in range(len(reflections)):
            reflections[i]["imageset_id"] = flex.int(len(reflections[i]), i)
            if i > 0:
                reflections[0].extend(reflections[i])
    reflections = reflections[0]

    if params.indexing.image_range:
        reflections = slice_reflections(reflections, params.indexing.image_range)

    if params.indexing.joint_indexing is Auto and len(experiments) > 1:
        if all(e.is_still() for e in experiments):
            params.indexing.joint_indexing = False
            logger.info("Disabling joint_indexing for still data")
        elif all(not e.is_still() for e in experiments):
            raise ValueError(
                "Unable to set joint_indexing automatically: set 'joint_indexing=True' for multi-axis data from a single crystal or 'joint_indexing=False' for data from multiple samples"
            )
        else:
            raise ValueError(
                "Unable to set joint_indexing automatically for a mixture of still and rotation data"
            )

    if len(experiments) == 1 or params.indexing.joint_indexing:
        indexed_experiments, indexed_reflections = _index_experiments(
            experiments,
            reflections,
            copy.deepcopy(params),
            known_crystal_models=known_crystal_models,
        )
    else:
        indexed_experiments = ExperimentList()

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=params.indexing.nproc
        ) as pool:
            futures = {}
            for iset_id, imgset in enumerate(experiments.imagesets()):
                refl = reflections.select(reflections["imageset_id"] == iset_id)
                i_expts = experiments.where(imageset=imgset)
                elist = ExperimentList([experiments[i] for i in i_expts])
                known_crystal_models_this = None
                if known_crystal_models:
                    known_crystal_models_this = [
                        known_crystal_models[i] for i in i_expts
                    ]
                refl["imageset_id"] = flex.int(len(refl), 0)
                futures[
                    pool.submit(
                        _index_experiments,
                        elist,
                        refl,
                        copy.deepcopy(params),
                        known_crystal_models=known_crystal_models_this,
                        log_text=f"Indexing imageset id {iset_id} ({iset_id + 1}/{len(experiments.imagesets())})",
                    )
                ] = iset_id

            tables_list = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    iset_id = futures[future]
                    idx_expts, idx_refl = future.result()
                except DialsIndexError as e:
                    logger.warning(str(e))
                else:
                    if idx_expts is None:
                        continue
                    idx_refl["imageset_id"] = flex.int(idx_refl.size(), iset_id)
                    tables_list.append(idx_refl)
                    indexed_experiments.extend(idx_expts)
        indexed_reflections = flex.reflection_table.concat(tables_list)

    return indexed_experiments, indexed_reflections


@show_mail_handle_errors()
def run(args=None, phil=working_phil):
    usage = "dials_gpu.index [options] models.expt strong.refl"

    parser = ArgumentParser(
        usage=usage,
        phil=phil,
        read_reflections=True,
        read_experiments=True,
        check_format=False,
        epilog=help_message,
    )

    params, options = parser.parse_args(args=args, show_diff_phil=False)

    log.config(verbosity=options.verbose, logfile=params.output.log)
    logger.info(dials_version())

    diff_phil = parser.diff_phil.as_str()
    if diff_phil != "":
        logger.info("The following parameters have been modified:\n")
        logger.info(diff_phil)

    reflections, experiments = reflections_and_experiments_from_files(
        params.input.reflections, params.input.experiments
    )

    if len(experiments) == 0:
        parser.print_help()
        return

    try:
        indexed_experiments, indexed_reflections = index(
            experiments, reflections, params
        )
    except (DialsIndexError, ValueError) as e:
        sys.exit(str(e))

    history = experiments.consolidate_histories()
    for experiment in indexed_experiments:
        experiment.history = history

    logger.info("Saving refined experiments to %s", params.output.experiments)
    assert indexed_experiments.is_consistent()
    indexed_experiments.as_file(params.output.experiments)

    logger.info("Saving refined reflections to %s", params.output.reflections)
    indexed_reflections.as_file(filename=params.output.reflections)


if __name__ == "__main__":
    run()
