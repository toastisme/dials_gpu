from __future__ import annotations

import importlib.metadata
import logging
import math

import libtbx
import scitbx.matrix
from cctbx import sgtbx
from dxtbx.imageset import ImageSequence, ImageSet
from dxtbx.model.experiment_list import Experiment, ExperimentList
from rstbx.dps_core.cell_assessment import SmallUnitCellVolume
from scitbx.array_family import flex

from dials.algorithms.indexing import DialsIndexError, model_evaluation
from dials.algorithms.indexing.indexer import Indexer, phil_scope
from dials.algorithms.indexing.known_orientation import IndexerKnownOrientation
from dials.algorithms.indexing.lattice_search import (
    BasisVectorSearch,
    LatticeSearch,
    basis_vector_search_phil_scope,
)
from dials.algorithms.indexing.stills_indexer import (
    StillsIndexer,
    StillsIndexerBasisVectorSearch,
    StillsIndexerKnownOrientation,
    StillsIndexerLatticeSearch,
)
from dials.algorithms.indexing.symmetry import SymmetryHandler
from dials_gpu.indexing import assign_indices, combinations, non_primitive_basis, optimise
from dials_gpu.indexing.fft1d import FFT1DGPU
from dials_gpu.indexing.fft3d import FFT3DGPU
from dials_gpu.indexing.max_cell import find_max_cell_gpu
from dials_gpu.indexing.real_space_grid_search import RealSpaceGridSearchGPU

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

        from libtbx import easy_mp

        evaluator = model_evaluation.ModelEvaluation(self.all_params)
        results = easy_mp.parallel_map(
            evaluator.evaluate,
            args,
            iterable_type=easy_mp.posiargs,
            processes=self.params.nproc,
            preserve_exception_message=True,
        )

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
    pass


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
