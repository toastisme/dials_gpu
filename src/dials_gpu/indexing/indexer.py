from __future__ import annotations

import copy
import importlib.metadata
import logging
import math
import warnings

import libtbx
import scitbx.matrix
from cctbx import sgtbx
from dxtbx.imageset import ImageSequence, ImageSet
from dxtbx.model.experiment_list import Experiment, ExperimentList
from rstbx.dps_core.cell_assessment import SmallUnitCellVolume
from dials.array_family import flex

from dials.algorithms.indexing import DialsIndexError, DialsIndexRefineError, model_evaluation
from dials.algorithms.indexing.indexer import Indexer, phil_scope
from dials.algorithms.indexing.known_orientation import IndexerKnownOrientation
from dials.algorithms.indexing.lattice_search import (
    BasisVectorSearch,
    LatticeSearch,
    basis_vector_search_phil_scope,
)
from dials.algorithms.indexing.nave_parameters import NaveParameters
from dials.algorithms.indexing.stills_indexer import (
    StillsIndexer,
    StillsIndexerBasisVectorSearch,
    StillsIndexerKnownOrientation,
    StillsIndexerLatticeSearch,
    calc_2D_rmsd_and_displacements,
    plot_displacements,
)
from dials.algorithms.indexing.symmetry import SymmetryHandler
from dials.util.log import LoggingContext
from dials.util.multi_dataset_handling import generate_experiment_identifiers
from dials_gpu.indexing import assign_indices, combinations, non_primitive_basis, optimise
from dials_gpu.indexing.fft1d import FFT1DGPU
from dials_gpu.indexing.fft3d import FFT3DGPU
from dials_gpu.indexing.max_cell import find_max_cell_gpu
from dials_gpu.indexing.real_space_grid_search import RealSpaceGridSearchGPU
from dials_gpu.refinement import RefinerFactoryGPU
from dials_gpu.refinement.predictor import ExperimentsPredictorFactoryGPU

logger = logging.getLogger("dials.algorithms.indexing.indexer")

_GPU_BASIS_VECTOR_STRATEGIES = {
    "fft3d": FFT3DGPU,
    "fft1d": FFT1DGPU,
    "real_space_grid_search": RealSpaceGridSearchGPU,
}


class IndexerGPU(Indexer):
    """
    GPU-accelerated drop-in replacement for dials.algorithms.indexing.indexer.Indexer.
    """

    def __init__(self, reflections, experiments, params):
        self.reflections = reflections
        self.experiments = experiments

        self.params = params.indexing
        self.all_params = params
        self.refined_experiments = None
        self.hkl_offset = None

        if self.params.index_assignment.method == "local":
            self._assign_indices = assign_indices.AssignIndicesLocalGPU(
                epsilon=self.params.index_assignment.local.epsilon,
                delta=self.params.index_assignment.local.delta,
                l_min=self.params.index_assignment.local.l_min,
                nearest_neighbours=self.params.index_assignment.local.nearest_neighbours,
            )
        else:
            self._assign_indices = assign_indices.AssignIndicesGlobalGPU(
                tolerance=self.params.index_assignment.simple.hkl_tolerance
            )

        if self.all_params.refinement.reflections.outlier.algorithm in (
            "auto",
            libtbx.Auto,
        ):
            if (
                self.experiments[0].goniometer is None
                or self.experiments[0].scan is None
                or self.experiments[0].scan.is_still()
            ):
                self.all_params.refinement.reflections.outlier.algorithm = "sauter_poon"
            else:
                self.all_params.refinement.reflections.outlier.algorithm = "tukey"

        for expt in self.experiments[1:]:
            if expt.detector.is_similar_to(self.experiments[0].detector):
                expt.detector = self.experiments[0].detector
            if expt.goniometer is not None and expt.goniometer.is_similar_to(
                self.experiments[0].goniometer
            ):
                expt.goniometer = self.experiments[0].goniometer
                if expt.beam.is_similar_to(self.experiments[0].beam):
                    expt.beam = self.experiments[0].beam
                if self.params.combine_scans and expt.scan == self.experiments[0].scan:
                    expt.scan = self.experiments[0].scan

        if "flags" in self.reflections:
            strong_sel = self.reflections.get_flags(self.reflections.flags.strong)
            if strong_sel.count(True) > 0:
                self.reflections = self.reflections.select(strong_sel)
        if "flags" not in self.reflections or strong_sel.count(True) == 0:
            self.reflections.set_flags(
                flex.size_t_range(len(self.reflections)), self.reflections.flags.strong
            )

        self._setup_symmetry()
        self.d_min = None

        self.setup_indexing()

    def find_max_cell(self):
        params = self.params.max_cell_estimation
        if self.params.max_cell is libtbx.Auto:
            if self.params.known_symmetry.unit_cell is not None:
                uc_params = (
                    self._symmetry_handler.target_symmetry_primitive.unit_cell().parameters()
                )
                self.params.max_cell = params.multiplier * max(uc_params[:3])
                logger.info("Using max_cell: %.1f Angstrom", self.params.max_cell)
            else:
                convert_reflections_z_to_deg = True
                all_tof_experiments = False
                for expt in self.experiments:
                    if expt.scan is not None and expt.scan.has_property(
                        "time_of_flight"
                    ):
                        all_tof_experiments = True
                    elif all_tof_experiments:
                        raise ValueError(
                            "Cannot find max cell for ToF and non-ToF experiments at the same time"
                        )

                if all_tof_experiments:
                    if params.step_size < 100:
                        logger.info("Setting default ToF step size to 500 usec")
                        params.step_size = 500
                        convert_reflections_z_to_deg = False

                self.params.max_cell = find_max_cell_gpu(
                    self.reflections,
                    max_cell_multiplier=params.multiplier,
                    step_size=params.step_size,
                    nearest_neighbor_percentile=params.nearest_neighbor_percentile,
                    histogram_binning=params.histogram_binning,
                    nn_per_bin=params.nn_per_bin,
                    max_height_fraction=params.max_height_fraction,
                    filter_ice=params.filter_ice,
                    filter_overlaps=params.filter_overlaps,
                    overlaps_border=params.overlaps_border,
                    convert_reflections_z_to_deg=convert_reflections_z_to_deg,
                ).max_cell
                logger.info("Found max_cell: %.1f Angstrom", self.params.max_cell)

    def refine(self, experiments, reflections):
        if self.all_params.refinement.parameterisation.scan_varying:
            logger.warning(
                "scan_varying=True not supported in indexing: setting scan_varying=False"
            )
            self.all_params.refinement.parameterisation.scan_varying = False

        refiner = RefinerFactoryGPU.from_parameters_data_experiments(
            self.all_params, reflections, experiments
        )
        outliers = None
        refiner.run()

        properties_to_save = [
            "xyzcal.mm",
            "entering",
            "wavelength_cal",
            "s0_cal",
            "tof_cal",
        ]

        if outliers is not None:
            reflections["id"].set_selected(outliers, -1)

        predicted = refiner.predict_for_indexed()

        for i in properties_to_save:
            if i in predicted:
                reflections[i] = predicted[i]

        reflections.unset_flags(
            flex.bool(len(reflections), True), reflections.flags.centroid_outlier
        )
        assert (
            reflections.get_flags(reflections.flags.centroid_outlier).count(True) == 0
        )
        reflections.set_flags(
            predicted.get_flags(predicted.flags.centroid_outlier),
            reflections.flags.centroid_outlier,
        )
        reflections.set_flags(
            refiner.selection_used_for_refinement(),
            reflections.flags.used_in_refinement,
        )
        return refiner.get_experiments(), reflections

    @staticmethod
    def from_parameters(
        reflections, experiments, known_crystal_models=None, params=None
    ):
        if known_crystal_models is not None:
            if params.indexing.known_symmetry.space_group is None:
                params.indexing.known_symmetry.space_group = (
                    known_crystal_models[0].get_space_group().info()
                )
            idxr = IndexerKnownOrientationGPU(
                reflections, experiments, params, known_crystal_models
            )
        else:
            has_stills = False
            has_sequences = False
            for expt in experiments:
                if isinstance(expt.imageset, ImageSequence):
                    has_sequences = True
                else:
                    has_stills = True

            if has_stills and has_sequences:
                raise ValueError(
                    "Please provide only stills or only sequences, not both"
                )

            use_stills_indexer = has_stills

            if not (
                params.indexing.stills.indexer is libtbx.Auto
                or params.indexing.stills.indexer.lower() == "auto"
            ):
                if params.indexing.stills.indexer == "stills":
                    use_stills_indexer = True
                elif params.indexing.stills.indexer == "sequences":
                    use_stills_indexer = False
                else:
                    assert False

            if params.indexing.basis_vector_combinations.max_refine is libtbx.Auto:
                if use_stills_indexer:
                    params.indexing.basis_vector_combinations.max_refine = 5
                else:
                    params.indexing.basis_vector_combinations.max_refine = 50

            if use_stills_indexer:
                for experiment in experiments:
                    if isinstance(experiment.imageset, ImageSequence):
                        experiment.imageset = ImageSet(
                            experiment.imageset.data(), experiment.imageset.indices()
                        )
                    experiment.imageset.set_scan(None)
                    experiment.imageset.set_goniometer(None)
                    experiment.scan = None
                    experiment.goniometer = None

            indexer_type = None
            if params.indexing.method in _GPU_BASIS_VECTOR_STRATEGIES:
                indexer_type = (
                    StillsIndexerBasisVectorSearchGPU
                    if use_stills_indexer
                    else BasisVectorSearchGPU
                )
            else:
                for entry_point in importlib.metadata.entry_points(
                    group="dials.index.basis_vector_search"
                ):
                    if params.indexing.method == entry_point.name:
                        indexer_type = (
                            StillsIndexerBasisVectorSearchGPU
                            if use_stills_indexer
                            else BasisVectorSearchGPU
                        )

            if indexer_type is None:
                for entry_point in importlib.metadata.entry_points(
                    group="dials.index.lattice_search"
                ):
                    if params.indexing.method == entry_point.name:
                        indexer_type = (
                            StillsIndexerLatticeSearchGPU
                            if use_stills_indexer
                            else LatticeSearchGPU
                        )

            assert indexer_type is not None
            idxr = indexer_type(reflections, experiments, params=params)

        return idxr


class ModelEvaluationGPU(model_evaluation.ModelEvaluation):
    """GPU-accelerated ModelEvaluation using RefinerFactoryGPU."""

    def evaluate(self, experiments, reflections):
        with LoggingContext("dials.algorithms.refinement", level=logging.ERROR):
            indexed_reflections = reflections.select(reflections["id"] > -1)
            if (
                len(indexed_reflections)
                < self._params.indexing.basis_vector_combinations.n_indexed_threshold
            ):
                return
            try:
                refiner = RefinerFactoryGPU.from_parameters_data_experiments(
                    self._params, indexed_reflections, experiments
                )
                refiner.run()
            except (RuntimeError, ValueError):
                return
            else:
                rmsds = refiner.rmsds()
                xy_rmsds = math.sqrt(rmsds[0] ** 2 + rmsds[1] ** 2)
                px_size = experiments[0].detector[0].get_pixel_size()[0]
                if (
                    xy_rmsds / px_size
                    > self._params.indexing.basis_vector_combinations.xy_rmsd_threshold
                ):
                    return
                model_likelihood = 1.0 - xy_rmsds
                result = model_evaluation.Result(
                    model_likelihood=model_likelihood,
                    crystal=experiments.crystals()[0],
                    rmsds=rmsds,
                    n_indexed=len(indexed_reflections),
                    fraction_indexed=float(len(indexed_reflections)) / len(reflections),
                    hkl_offset=(0, 0, 0),
                )
                return result


def e_refine_gpu(params, experiments, reflections, graph_verbose=False):
    outlier_algorithm = params.refinement.reflections.outlier.algorithm
    if outlier_algorithm == "sauter_poon":
        params.refinement.reflections.outlier.algorithm = "null"

    try:
        refiner = RefinerFactoryGPU.from_parameters_data_experiments(
            params, reflections, experiments
        )
        refiner.run()
    finally:
        params.refinement.reflections.outlier.algorithm = outlier_algorithm

    if not graph_verbose:
        return refiner

    RR = refiner.predict_for_reflection_table(reflections)
    plot_displacements(reflections, RR, experiments)

    return refiner


class IndexerKnownOrientationGPU(IndexerKnownOrientation, IndexerGPU):
    pass


class LatticeSearchGPU(LatticeSearch, IndexerGPU):
    """
    GPU-accelerated LatticeSearch using AssignIndicesGlobalGPU, non_primitive_basis.correct_gpu,
    and hoisted base reflection selection in choose_best_orientation_matrix.
    """

    def choose_best_orientation_matrix(self, candidate_orientation_matrices):
        solution_scorer = self.params.basis_vector_combinations.solution_scorer
        if solution_scorer == "weighted":
            weighted_params = self.params.basis_vector_combinations.weighted
            solutions = model_evaluation.ModelRankWeighted(
                power=weighted_params.power,
                volume_weight=weighted_params.volume_weight,
                n_indexed_weight=weighted_params.n_indexed_weight,
                rmsd_weight=weighted_params.rmsd_weight,
            )
        else:
            filter_params = self.params.basis_vector_combinations.filter
            solutions = model_evaluation.ModelRankFilter(
                check_doubled_cell=filter_params.check_doubled_cell,
                likelihood_cutoff=filter_params.likelihood_cutoff,
                volume_cutoff=filter_params.volume_cutoff,
                n_indexed_cutoff=filter_params.n_indexed_cutoff,
            )

        args = []

        # Hoist candidate-independent reflection filtering outside the candidate loop
        sel = self.reflections["id"] == -1
        if self.d_min is not None:
            sel &= 1 / self.reflections["rlp"].norms() > self.d_min
        _, _, zo = self.reflections["xyzobs.mm.value"].parts()
        imageset_id = self.reflections["imageset_id"]
        for i_expt, expt in enumerate(self.experiments):
            if expt.scan is not None and expt.scan.has_property("oscillation"):
                start, end = expt.scan.get_oscillation_range()
                if (end - start) > 360:
                    sel.set_selected(
                        (imageset_id == i_expt)
                        & (zo > ((start * math.pi / 180) + 2 * math.pi)),
                        False,
                    )
        base_refl = self.reflections.select(sel)
        if "shoebox" in base_refl:
            del base_refl["shoebox"]

        threshold = self.params.basis_vector_combinations.sys_absent_threshold
        do_non_prim = bool(
            threshold
            and (
                self._symmetry_handler.target_symmetry_primitive is None
                or self._symmetry_handler.target_symmetry_primitive.unit_cell() is None
            )
        )

        for cm in candidate_orientation_matrices:
            experiments = ExperimentList()
            for expt in self.experiments:
                experiments.append(
                    Experiment(
                        imageset=expt.imageset,
                        beam=expt.beam,
                        detector=expt.detector,
                        goniometer=expt.goniometer,
                        scan=expt.scan,
                        crystal=cm,
                    )
                )
            refl = base_refl.copy()
            self.index_reflections(experiments, refl)
            if refl.get_flags(refl.flags.indexed).count(True) == 0:
                continue

            if do_non_prim:
                try:
                    non_primitive_basis.correct_gpu(
                        experiments, refl, self._assign_indices, threshold
                    )
                    if refl.get_flags(refl.flags.indexed).count(True) == 0:
                        continue
                except SmallUnitCellVolume:
                    logger.debug(
                        "correct_non_primitive_basis SmallUnitCellVolume error for unit cell %s:",
                        experiments[0].crystal.get_unit_cell(),
                    )
                    continue
                except RuntimeError as e:
                    if "Krivy-Gruber iteration limit exceeded" in str(e):
                        logger.debug(
                            "correct_non_primitive_basis Krivy-Gruber iteration limit exceeded error for unit cell %s:",
                            experiments[0].crystal.get_unit_cell(),
                        )
                        continue
                    raise
                if (
                    experiments[0].crystal.get_unit_cell().volume()
                    < self.params.min_cell_volume
                ):
                    continue

            if self.params.known_symmetry.space_group is not None:
                new_crystal, _ = self._symmetry_handler.apply_symmetry(
                    experiments[0].crystal
                )
                if new_crystal is None:
                    continue
                experiments[0].crystal.update(new_crystal)

            args.append((experiments, refl))
            if len(args) == self.params.basis_vector_combinations.max_refine:
                break

        evaluator = ModelEvaluationGPU(self.all_params)
        results = [evaluator.evaluate(expt, rfl) for expt, rfl in args]

        for soln in results:
            if soln is None:
                continue
            solutions.append(soln)

        if len(solutions):
            logger.info("Candidate solutions:")
            logger.info(str(solutions))
            best_model = solutions.best_model()
            logger.debug("best model_likelihood: %.2f", best_model.model_likelihood)
            logger.debug("best n_indexed: %i", best_model.n_indexed)
            self.hkl_offset = best_model.hkl_offset
            return best_model.crystal, best_model.n_indexed
        else:
            logger.info("No candidate solutions found within thresholds")
            return None, None


class BasisVectorSearchGPU(BasisVectorSearch, LatticeSearchGPU):
    """
    GPU-accelerated BasisVectorSearch using FFT3DGPU, FFT1DGPU, RealSpaceGridSearchGPU,
    optimise_basis_vectors_gpu, and candidate_orientation_matrices_gpu.
    """

    def __init__(self, reflections, experiments, params):
        LatticeSearchGPU.__init__(self, reflections, experiments, params)

        method = params.indexing.method
        strategy_class = _GPU_BASIS_VECTOR_STRATEGIES.get(method)
        if strategy_class is None:
            for entry_point in importlib.metadata.entry_points(
                group="dials.index.basis_vector_search"
            ):
                if entry_point.name == method:
                    strategy_class = entry_point.load()
                    break
        if not strategy_class:
            raise RuntimeError(f"Unknown basis vector search strategy: {method}")

        target_unit_cell = None
        if (
            self._symmetry_handler.target_symmetry_primitive is not None
            and self._symmetry_handler.target_symmetry_primitive.unit_cell() is not None
        ):
            target_unit_cell = (
                self._symmetry_handler.target_symmetry_primitive.unit_cell()
            )
        self._basis_vector_search_strategy = strategy_class(
            max_cell=self.params.max_cell,
            min_cell=self.params.min_cell,
            target_unit_cell=target_unit_cell,
            params=getattr(self.params, method),
        )

    def find_candidate_orientation_matrices(self, candidate_basis_vectors):
        candidate_crystal_models = combinations.candidate_orientation_matrices_gpu(
            candidate_basis_vectors,
            max_combinations=self.params.basis_vector_combinations.max_combinations,
        )
        if self._symmetry_handler.target_symmetry_reference_setting is not None:
            target_symmetry = self._symmetry_handler.target_symmetry_reference_setting
        elif self._symmetry_handler.target_symmetry_primitive is not None:
            target_symmetry = self._symmetry_handler.target_symmetry_primitive
        else:
            target_symmetry = None
        if target_symmetry is not None:
            candidate_crystal_models = combinations.filter_known_symmetry(
                candidate_crystal_models,
                target_symmetry,
                relative_length_tolerance=self.params.known_symmetry.relative_length_tolerance,
                absolute_angle_tolerance=self.params.known_symmetry.absolute_angle_tolerance,
                max_delta=self.params.known_symmetry.max_delta,
            )

        if self.refined_experiments is not None and len(self.refined_experiments) > 0:
            candidate_crystal_models = combinations.filter_similar_orientations(
                candidate_crystal_models,
                self.refined_experiments.crystals(),
                minimum_angular_separation=self.params.multiple_lattice_search.minimum_angular_separation,
            )

        return candidate_crystal_models

    def optimise_basis_vectors(self):
        optimised_basis_vectors = optimise.optimise_basis_vectors_gpu(
            self.reflections["rlp"].select(self._used_in_indexing),
            self.candidate_basis_vectors,
        )
        self.candidate_basis_vectors = [
            scitbx.matrix.col(v) for v in optimised_basis_vectors
        ]


class StillsIndexerGPU(StillsIndexer, IndexerGPU):
    """GPU-accelerated StillsIndexer using e_refine_gpu and ExperimentsPredictorFactoryGPU."""

    def index(self):
        experiments = ExperimentList()

        while True:
            self.d_min = self.params.refinement_protocol.d_min_start
            max_lattices = self.params.multiple_lattice_search.max_lattices
            if max_lattices is not None and len(experiments.crystals()) >= max_lattices:
                break
            if len(experiments) > 0:
                cutoff_fraction = self.params.multiple_lattice_search.recycle_unindexed_reflections_cutoff
                d_spacings = 1 / self.reflections["rlp"].norms()
                d_min_indexed = flex.min(d_spacings.select(self.indexed_reflections))
                min_reflections_for_indexing = cutoff_fraction * len(
                    self.reflections.select(d_spacings > d_min_indexed)
                )
                crystal_ids = self.reflections.select(d_spacings > d_min_indexed)["id"]
                if (crystal_ids == -1).count(True) < min_reflections_for_indexing:
                    logger.info(
                        "Finish searching for more lattices: %i unindexed reflections remaining.",
                        min_reflections_for_indexing,
                    )
                    break

            n_lattices_previous_cycle = len(experiments)

            if len(experiments) == 0:
                new = self.find_lattices()
                generate_experiment_identifiers(new)
                experiments.extend(new)
                if len(experiments) == 0:
                    raise DialsIndexError("No suitable lattice could be found.")
            else:
                try:
                    new = self.find_lattices()
                    generate_experiment_identifiers(new)
                    experiments.extend(new)
                except Exception as e:
                    logger.info("Indexing remaining reflections failed")
                    logger.debug(
                        "Indexing remaining reflections failed, exception:\n" + str(e)
                    )

            self.reflections["id"] = flex.int(len(self.reflections), -1)

            self.index_reflections(experiments, self.reflections)

            if len(experiments) == n_lattices_previous_cycle:
                break

            if (
                not self.params.stills.refine_candidates_with_known_symmetry
                and self.params.known_symmetry.space_group is not None
            ):
                self._apply_symmetry_post_indexing(
                    experiments, self.reflections, n_lattices_previous_cycle
                )

            if self._remove_similar_crystal_models(experiments):
                break

            self.indexed_reflections = self.reflections["id"] > -1
            if self.d_min is None:
                sel = self.reflections["id"] <= -1
            else:
                sel = flex.bool(len(self.reflections), False)
                lengths = 1 / self.reflections["rlp"].norms()
                isel = (lengths >= self.d_min).iselection()
                sel.set_selected(isel, True)
                sel.set_selected(self.reflections["id"] > -1, False)
            unindexed_reflections = self.reflections.select(sel)

            reflections_for_refinement = self.reflections.select(
                self.indexed_reflections
            )

            if len(self.params.stills.isoforms) > 0:
                logger.info("")
                logger.info("#" * 80)
                logger.info("Starting refinement")
                logger.info("#" * 80)
                logger.info("")

                isoform_experiments = ExperimentList()
                isoform_reflections = flex.reflection_table()
                self.all_params.refinement.parameterisation.crystal.fix = "cell"
                self.all_params.refinement.parameterisation.crystal.unit_cell.restraints.tie_to_target = []

                for expt_id, experiment in enumerate(experiments):
                    reflections = reflections_for_refinement.select(
                        reflections_for_refinement["id"] == expt_id
                    )
                    reflections["id"] = flex.int(len(reflections), 0)
                    refiners = []
                    for isoform in self.params.stills.isoforms:
                        iso_experiment = copy.deepcopy(experiment)
                        crystal = iso_experiment.crystal
                        if (
                            isoform.lookup_symbol
                            != crystal.get_space_group().type().lookup_symbol()
                        ):
                            logger.info(
                                "Crystal isoform lookup_symbol %s does not match isoform %s lookup_symbol %s",
                                crystal.get_space_group().type().lookup_symbol(),
                                isoform.name,
                                isoform.lookup_symbol,
                            )
                            continue
                        crystal.set_B(isoform.cell.fractionalization_matrix())

                        logger.info("Refining isoform %s", isoform.name)
                        refiners.append(
                            e_refine_gpu(
                                params=self.all_params,
                                experiments=ExperimentList([iso_experiment]),
                                reflections=reflections,
                                graph_verbose=False,
                            )
                        )

                    if len(refiners) == 0:
                        raise DialsIndexError(
                            "No isoforms had a lookup symbol that matched"
                        )
                    positional_rmsds = [
                        math.sqrt(P.rmsds()[0] ** 2 + P.rmsds()[1] ** 2)
                        for P in refiners
                    ]
                    logger.info(
                        "Positional rmsds for all isoforms:" + str(positional_rmsds)
                    )
                    minrmsd_mm = min(positional_rmsds)
                    minindex = positional_rmsds.index(minrmsd_mm)
                    logger.info(
                        "The smallest rmsd is %5.1f um from isoform %s",
                        1000.0 * minrmsd_mm,
                        self.params.stills.isoforms[minindex].name,
                    )
                    if self.params.stills.isoforms[minindex].rmsd_target_mm is not None:
                        logger.info(
                            "Asserting %f < %f",
                            minrmsd_mm,
                            self.params.stills.isoforms[minindex].rmsd_target_mm,
                        )
                        assert (
                            minrmsd_mm
                            < self.params.stills.isoforms[minindex].rmsd_target_mm
                        )
                    logger.info(
                        "Acceptable rmsd for isoform %s.",
                        self.params.stills.isoforms[minindex].name,
                    )
                    if len(self.params.stills.isoforms) == 2:
                        logger.info(
                            "Rmsd gain over the other isoform %5.1f um.",
                            1000.0 * abs(positional_rmsds[0] - positional_rmsds[1]),
                        )
                    R = refiners[minindex]
                    if self.params.stills.isoforms[minindex].beam_restraint is not None:
                        refined_beam = scitbx.matrix.col(
                            R.get_experiments()[0]
                            .detector[0]
                            .get_beam_centre_lab(experiments[0].beam.get_s0())[0:2]
                        )
                        known_beam = scitbx.matrix.col(
                            self.params.stills.isoforms[minindex].beam_restraint
                        )
                        logger.info(
                            "Asserting difference in refined beam center and expected beam center %f < %f",
                            (refined_beam - known_beam).length(),
                            self.params.stills.isoforms[minindex].rmsd_target_mm,
                        )
                        assert (
                            refined_beam - known_beam
                        ).length() < self.params.stills.isoforms[
                            minindex
                        ].rmsd_target_mm

                    experiment = R.get_experiments()[0]
                    experiment.crystal.identified_isoform = self.params.stills.isoforms[
                        minindex
                    ].name

                    isoform_experiments.append(experiment)
                    reflections["id"] = flex.int(len(reflections), expt_id)
                    isoform_reflections.extend(reflections)
                experiments = isoform_experiments
                reflections_for_refinement = isoform_reflections

            if self.params.refinement_protocol.mode == "repredict_only":
                refined_experiments, refined_reflections = (
                    experiments,
                    reflections_for_refinement,
                )
                ref_predictor = ExperimentsPredictorFactoryGPU.from_experiments(
                    experiments,
                    force_stills=True,
                    spherical_relp=self.all_params.refinement.parameterisation.spherical_relp_model,
                )
                ref_predictor(refined_reflections)
                refined_reflections["delpsical2"] = (
                    refined_reflections["delpsical.rad"] ** 2
                )
                for expt_id in range(len(refined_experiments)):
                    refls = refined_reflections.select(
                        refined_reflections["id"] == expt_id
                    )
                    nv = NaveParameters(
                        params=self.all_params,
                        experiments=refined_experiments[expt_id : expt_id + 1],
                        reflections=refls,
                        refinery=None,
                        graph_verbose=False,
                    )
                    experiments[expt_id].crystal = nv()
                ref_predictor = ExperimentsPredictorFactoryGPU.from_experiments(
                    experiments,
                    force_stills=True,
                    spherical_relp=self.all_params.refinement.parameterisation.spherical_relp_model,
                )
                ref_predictor(refined_reflections)

            elif self.params.refinement_protocol.mode is None:
                refined_experiments, refined_reflections = (
                    experiments,
                    reflections_for_refinement,
                )

            else:
                reflections_for_refinement["_reflection_id"] = flex.size_t(
                    range(reflections_for_refinement.size())
                )
                try:
                    refined_experiments, refined_reflections = self.refine(
                        experiments, reflections_for_refinement
                    )
                except Exception as e:
                    s = str(e)
                    if len(experiments) == 1:
                        raise DialsIndexRefineError(e)
                    logger.info("Refinement failed:")
                    logger.info(s)
                    models_to_remove = experiments.where(
                        crystal=experiments[-1].crystal
                    )
                    for model_id in sorted(models_to_remove, reverse=True):
                        del experiments[model_id]
                    break
                else:
                    sel = flex.bool(reflections_for_refinement.size(), True)
                    sel.set_selected(refined_reflections["_reflection_id"], False)
                    del refined_reflections["_reflection_id"]
                    del reflections_for_refinement["_reflection_id"]
                    unindexed_reflections.extend(reflections_for_refinement.select(sel))

            self._unit_cell_volume_sanity_check(experiments, refined_experiments)

            if -1 in set(refined_reflections["id"]):
                sel = refined_reflections["id"] < 0
                refined_reflections.unset_flags(
                    sel,
                    refined_reflections.flags.indexed,
                )
                refined_reflections["miller_index"].set_selected(sel, (0, 0, 0))
                unindexed_reflections.extend(refined_reflections.select(sel))
                refined_reflections.del_selected(sel)
            self.refined_reflections = refined_reflections
            unindexed_reflections.unset_flags(
                flex.bool(unindexed_reflections.size(), True),
                unindexed_reflections.flags.indexed,
            )
            unindexed_reflections["id"] = flex.int(unindexed_reflections.size(), -1)
            unindexed_reflections["miller_index"] = flex.miller_index(
                unindexed_reflections.size(), (0, 0, 0)
            )
            self.unindexed_reflections = unindexed_reflections

            for i, expt in enumerate(self.experiments):
                ref_sel = self.refined_reflections.select(
                    self.refined_reflections["imageset_id"] == i
                )
                ref_sel = ref_sel.select(ref_sel["id"] >= 0)
                for i_expt in set(ref_sel["id"]):
                    refined_expt = refined_experiments[i_expt]
                    expt.detector = refined_expt.detector
                    expt.beam = refined_expt.beam
                    expt.goniometer = refined_expt.goniometer
                    expt.scan = refined_expt.scan
                    refined_expt.imageset = expt.imageset

            if not (
                self.all_params.refinement.parameterisation.beam.fix == "all"
                and self.all_params.refinement.parameterisation.detector.fix == "all"
            ):
                self.reflections.map_centroids_to_reciprocal_space(self.experiments)

            experiments = refined_experiments
            self.refined_experiments = refined_experiments

        if self.refined_experiments is None:
            raise DialsIndexRefineError("None of the experiments could refine.")

        id_set = set(self.refined_reflections["id"])
        if len(id_set) < len(self.refined_experiments):
            filtered_refined_reflections = flex.reflection_table()
            for i in range(len(self.refined_experiments)):
                if i not in id_set:
                    del self.refined_experiments[i]
            for old, new in zip(sorted(id_set), range(len(id_set))):
                subset = self.refined_reflections.select(
                    self.refined_reflections["id"] == old
                )
                subset["id"] = flex.int(len(subset), new)
                filtered_refined_reflections.extend(subset)
            self.refined_reflections = filtered_refined_reflections

        if len(self.refined_experiments) > 1:
            from dials.algorithms.indexing.compare_orientation_matrices import (
                rotation_matrix_differences,
            )

            logger.info(
                rotation_matrix_differences(self.refined_experiments.crystals())
            )

        logger.info("Final refined crystal models:")
        for i, crystal_model in enumerate(self.refined_experiments.crystals()):
            n_indexed = 0
            for _ in experiments.where(crystal=crystal_model):
                n_indexed += (self.refined_reflections["id"] == i).count(True)
            logger.info("model %i (%i reflections):", i + 1, n_indexed)
            logger.info(crystal_model)

        if "xyzcal.mm" in self.refined_reflections:
            self._xyzcal_mm_to_px(self.experiments, self.refined_reflections)

    def choose_best_orientation_matrix(self, candidate_orientation_matrices):
        logger.info("*" * 80)
        logger.info("Selecting the best orientation matrix")
        logger.info("*" * 80)

        class CandidateInfo(libtbx.group_args):
            pass

        candidates = []
        params = self.all_params

        for icm, cm in enumerate(candidate_orientation_matrices):
            if icm >= self.params.basis_vector_combinations.max_refine:
                break
            sel = self.reflections["id"] == -1
            refl = self.reflections.select(sel)
            experiments = self.experiment_list_for_crystal(cm)
            self.index_reflections(experiments, refl)
            indexed = refl.select(refl["id"] >= 0)
            indexed = indexed.select(indexed.get_flags(indexed.flags.indexed))

            if (
                self.params.stills.refine_candidates_with_known_symmetry
                and self.params.known_symmetry.space_group is not None
            ):
                new_crystal, cb_op = self._symmetry_handler.apply_symmetry(cm)
                if new_crystal is None:
                    logger.info("Cannot convert to target symmetry, candidate %d", icm)
                    continue
                cm = new_crystal.change_basis(cb_op)
                experiments = self.experiment_list_for_crystal(cm)

                if not cb_op.is_identity_op():
                    indexed["miller_index"] = cb_op.apply(indexed["miller_index"])

            if params.indexing.stills.refine_all_candidates:
                try:
                    logger.info(
                        "$$$ stills_indexer::choose_best_orientation_matrix, candidate %d initial outlier identification",
                        icm,
                    )
                    acceptance_flags = self.identify_outliers(
                        params, experiments, indexed
                    )
                    indexed = indexed.select(acceptance_flags)

                    logger.info(
                        "$$$ stills_indexer::choose_best_orientation_matrix, candidate %d refinement before outlier rejection",
                        icm,
                    )
                    R = e_refine_gpu(
                        params=params,
                        experiments=experiments,
                        reflections=indexed,
                        graph_verbose=False,
                    )
                    ref_experiments = R.get_experiments()

                    acceptance_flags = self.identify_outliers(
                        params, ref_experiments, indexed
                    )

                    nv0 = NaveParameters(
                        params=params,
                        experiments=ref_experiments,
                        reflections=indexed,
                        refinery=R,
                        graph_verbose=False,
                    )
                    nv0()
                    acceptance_flags_nv0 = nv0.nv_acceptance_flags
                    indexed = indexed.select(acceptance_flags & acceptance_flags_nv0)

                    logger.info(
                        "$$$ stills_indexer::choose_best_orientation_matrix, candidate %d after positional and delta-psi outlier rejection",
                        icm,
                    )
                    R = e_refine_gpu(
                        params=params,
                        experiments=ref_experiments,
                        reflections=indexed,
                        graph_verbose=False,
                    )
                    ref_experiments = R.get_experiments()

                    nv = NaveParameters(
                        params=params,
                        experiments=ref_experiments,
                        reflections=indexed,
                        refinery=R,
                        graph_verbose=False,
                    )
                    crystal_model = nv()
                    assert len(crystal_model) == 1, (
                        "$$$ stills_indexer::choose_best_orientation_matrix, Only one crystal at this stage"
                    )
                    crystal_model = crystal_model[0]

                    if (
                        not self.params.stills.refine_candidates_with_known_symmetry
                        and self.params.known_symmetry.space_group is not None
                    ):
                        (
                            new_crystal,
                            cb_op_to_primitive,
                        ) = self._symmetry_handler.apply_symmetry(crystal_model)
                        if new_crystal is None:
                            logger.info(
                                "P1 refinement yielded model diverged from target, candidate %d",
                                icm,
                            )
                            continue

                    rmsd, _ = calc_2D_rmsd_and_displacements(
                        R.predict_for_reflection_table(indexed)
                    )
                except Exception as e:
                    logger.info(
                        "Couldn't refine candidate %d, %s: %s",
                        icm,
                        e.__class__.__name__,
                        str(e),
                    )
                else:
                    logger.info(
                        "$$$ stills_indexer::choose_best_orientation_matrix, candidate %d done",
                        icm,
                    )
                    candidates.append(
                        CandidateInfo(
                            crystal=crystal_model,
                            green_curve_area=nv.green_curve_area,
                            ewald_proximal_volume=nv.ewald_proximal_volume(),
                            n_indexed=len(indexed),
                            rmsd=rmsd,
                            indexed=indexed,
                            experiments=ref_experiments,
                        )
                    )
            else:
                ref_predictor = ExperimentsPredictorFactoryGPU.from_experiments(
                    experiments,
                    force_stills=True,
                    spherical_relp=params.refinement.parameterisation.spherical_relp_model,
                )
                rmsd, _ = calc_2D_rmsd_and_displacements(ref_predictor(indexed))
                candidates.append(
                    CandidateInfo(
                        crystal=cm,
                        n_indexed=len(indexed),
                        rmsd=rmsd,
                        indexed=indexed,
                        experiments=experiments,
                    )
                )
        if len(candidates) == 0:
            raise DialsIndexError("No suitable indexing solution found")

        logger.info("**** ALL CANDIDATES:")
        for i, XX in enumerate(candidates):
            logger.info("\n****Candidate %d %s", i, XX)
            cc = XX.crystal
            if hasattr(cc, "get_half_mosaicity_deg"):
                logger.info(
                    "  half mosaicity %5.2f deg.", (cc.get_half_mosaicity_deg())
                )
                logger.info("  domain size %.0f Ang.", (cc.get_domain_size_ang()))
        logger.info("\n**** BEST CANDIDATE:")

        results = flex.double([c.rmsd for c in candidates])
        best = candidates[flex.min_index(results)]
        logger.info(best)

        if params.indexing.basis_vector_combinations.xy_rmsd_threshold is libtbx.Auto:
            params.indexing.basis_vector_combinations.xy_rmsd_threshold = 2.0
        if params.indexing.stills.rmsd_min_px != 2.0:
            warnings.warn(
                "The parameter indexing.stills.rmsd_min_px is deprecated. Please use indexing.basis_vector_combinations.xy_rmsd_threshold instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            params.indexing.basis_vector_combinations.xy_rmsd_threshold = (
                params.indexing.stills.rmsd_min_px
            )
        if params.indexing.stills.refine_all_candidates:
            if best.rmsd > params.indexing.basis_vector_combinations.xy_rmsd_threshold:
                raise DialsIndexError(f"RMSD too high, {best.rmsd:f}")

            if len(candidates) > 1:
                for i in range(len(candidates)):
                    if i == flex.min_index(results):
                        continue
                    if best.ewald_proximal_volume > candidates[i].ewald_proximal_volume:
                        logger.info(
                            "Couldn't figure out which candidate is best; picked the one with the best RMSD."
                        )

        best.indexed["entering"] = flex.bool(best.n_indexed, False)

        return best.crystal, best.n_indexed

    def identify_outliers(self, params, experiments, indexed):
        if not params.indexing.stills.candidate_outlier_rejection:
            return flex.bool(len(indexed), True)

        logger.info("$$$ stills_indexer::identify_outliers")
        refiner = e_refine_gpu(params, experiments, indexed, graph_verbose=False)

        RR = refiner.predict_for_reflection_table(indexed)

        if params.refinement.reflections.outlier.algorithm != "sauter_poon":
            return refiner.selection_used_for_refinement()

        px_sz = experiments[0].detector[0].get_pixel_size()

        class Match:
            pass

        matches = []
        for item in RR.rows():
            m = Match()
            m.x_obs = item["xyzobs.px.value"][0] * px_sz[0]
            m.y_obs = item["xyzobs.px.value"][1] * px_sz[1]
            m.x_calc = item["xyzcal.px"][0] * px_sz[0]
            m.y_calc = item["xyzcal.px"][1] * px_sz[1]
            m.miller_index = item["miller_index"]
            matches.append(m)

        from rstbx.indexing_api import outlier_detection

        od = outlier_detection.find_outliers_from_matches(
            matches,
            verbose=self.all_params.refinement.reflections.outlier.sauter_poon.verbose,
            horizon_phil=self.hardcoded_phil,
        )

        if self.hardcoded_phil.indexing.outlier_detection.pdf is not None:
            od.make_graphs(canvas=self.hardcoded_phil.writer.R.c, left_margin=0.5)
            self.hardcoded_phil.writer.R.c.showPage()
            self.hardcoded_phil.writer.R.c.save()

        return od.get_cache_status()

    def refine(self, experiments, reflections):
        acceptance_flags = self.identify_outliers(
            self.all_params, experiments, reflections
        )
        reflections = reflections.select(acceptance_flags)

        R = e_refine_gpu(
            params=self.all_params,
            experiments=experiments,
            reflections=reflections,
            graph_verbose=False,
        )
        ref_experiments = R.get_experiments()

        acceptance_flags = self.identify_outliers(
            self.all_params, ref_experiments, reflections
        )

        nv0 = NaveParameters(
            params=self.all_params,
            experiments=ref_experiments,
            reflections=reflections,
            refinery=R,
            graph_verbose=False,
        )
        nv0()
        acceptance_flags_nv0 = nv0.nv_acceptance_flags
        reflections = reflections.select(acceptance_flags & acceptance_flags_nv0)

        R = e_refine_gpu(
            params=self.all_params,
            experiments=ref_experiments,
            reflections=reflections,
            graph_verbose=False,
        )
        ref_experiments = R.get_experiments()

        nv = NaveParameters(
            params=self.all_params,
            experiments=ref_experiments,
            reflections=reflections,
            refinery=R,
            graph_verbose=False,
        )
        nv()

        predicted = R.predict_for_indexed()
        xyzcal_mm = flex.vec3_double(len(reflections))
        xyzcal_mm.set_selected(predicted["iobs"], predicted["xyzcal.mm"])
        reflections["xyzcal.mm"] = xyzcal_mm
        reflections.set_flags(predicted["iobs"], reflections.flags.used_in_refinement)
        reflections["entering"] = flex.bool(len(reflections), False)

        if self.all_params.indexing.stills.set_domain_size_ang_value is not None:
            for exp in ref_experiments:
                exp.crystal.set_domain_size_ang(
                    self.all_params.indexing.stills.set_domain_size_ang_value
                )
        if self.all_params.indexing.stills.set_mosaic_half_deg_value is not None:
            for exp in ref_experiments:
                exp.crystal.set_half_mosaicity_deg(
                    self.all_params.indexing.stills.set_mosaic_half_deg_value
                )

        return ref_experiments, reflections


class StillsIndexerKnownOrientationGPU(
    IndexerKnownOrientationGPU, StillsIndexerGPU
):
    pass


class StillsIndexerBasisVectorSearchGPU(StillsIndexerGPU, BasisVectorSearchGPU):
    pass


class StillsIndexerLatticeSearchGPU(StillsIndexerGPU, LatticeSearchGPU):
    pass


__all__ = [
    "IndexerGPU",
    "IndexerKnownOrientationGPU",
    "LatticeSearchGPU",
    "BasisVectorSearchGPU",
    "StillsIndexerGPU",
    "StillsIndexerKnownOrientationGPU",
    "StillsIndexerBasisVectorSearchGPU",
    "StillsIndexerLatticeSearchGPU",
    "phil_scope",
    "basis_vector_search_phil_scope",
]
