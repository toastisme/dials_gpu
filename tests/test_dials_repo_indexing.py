from __future__ import annotations

import collections
import copy
import logging
import pathlib
import random
import shutil
import subprocess

import pytest
import scitbx.matrix
from cctbx import crystal, sgtbx, uctbx
from cctbx.sgtbx import bravais_types
from cctbx.sgtbx.lattice_symmetry import metric_subgroups
from dxtbx.format import Format
from dxtbx.imageset import ImageSequence, ImageSet, ImageSetData
from dxtbx.model import (
    BeamFactory,
    Crystal,
    DetectorFactory,
    Experiment,
    ExperimentList,
    GoniometerFactory,
    ScanFactory,
)
from dxtbx.serialize import load
from libtbx.phil import parse
from scitbx import matrix
from scitbx.math import euler_angles_as_matrix

from dials.algorithms.indexing.basis_vector_search import (
    FFT1D,
    FFT3D,
    RealSpaceGridSearch,
    combinations as cpu_combinations,
)
from dials.algorithms.indexing.max_cell import find_max_cell as cpu_find_max_cell
from dials.algorithms.indexing.symmetry import find_matching_symmetry
from dials.algorithms.refinement.refiner import (
    RefinerFactory,
    phil_scope as refiner_phil_scope,
)
from dials.array_family import flex
from dials.command_line.index import phil_scope as index_phil_scope
from dials.command_line.slice_sequence import slice_experiments, slice_reflections
from dials_gpu.indexing.assign_indices import (
    AssignIndicesGlobalGPU,
    AssignIndicesLocalGPU,
)
from dials_gpu.indexing.combinations import (
    candidate_orientation_matrices_gpu,
    filter_known_symmetry,
    filter_similar_orientations,
)
from dials_gpu.indexing.fft1d import FFT1DGPU
from dials_gpu.indexing.fft3d import FFT3DGPU
from dials_gpu.indexing.indexer import (
    BasisVectorSearchGPU,
    StillsIndexerBasisVectorSearchGPU,
    StillsIndexerLatticeSearchGPU,
)
from dials_gpu.indexing.max_cell import find_max_cell_gpu
from dials_gpu.indexing.non_primitive_basis import (
    correct_gpu as non_primitive_correct_gpu,
    detect_gpu as non_primitive_detect_gpu,
)
from dials_gpu.indexing.optimise import (
    BasisVectorMinimiserGPU,
    BasisVectorTargetGPU,
    optimise_basis_vectors_gpu,
)
from dials_gpu.indexing.real_space_grid_search import RealSpaceGridSearchGPU


# =============================================================================
# Fixtures
# =============================================================================


def random_rotation(angle_min=0, angle_max=360):
    angles = [random.uniform(angle_min, angle_max) for _ in range(3)]
    return euler_angles_as_matrix(angles, deg=True)


@pytest.fixture(params=["P2", "P3", "P6", "R3:h", "I23"])
def setup_rlp(request):
    random.seed(42)
    space_group = request.param
    sgi = sgtbx.space_group_info(symbol=space_group)
    cs = sgi.any_compatible_crystal_symmetry(volume=10000)
    cs = cs.best_cell()
    cs = cs.minimum_cell()
    uc = cs.unit_cell()

    B = matrix.sqr(uc.fractionalization_matrix()).transpose()
    U = random_rotation()
    A = U * B

    ms = cs.build_miller_set(d_min=2, anomalous_flag=True).expand_to_p1()
    rlp = A.elems * ms.indices().as_vec3_double()

    return {"crystal_symmetry": cs, "rlp": rlp}


# =============================================================================
# 1. Equivalent of dials/tests/algorithms/indexing/basis_vector_search/test_strategies.py
# =============================================================================


class TestStrategiesGPU:
    def check_results(self, unit_cell, basis_vectors):
        for p in unit_cell.parameters()[:3]:
            found = False
            for v in basis_vectors:
                if p == pytest.approx(v.length(), abs=5e-1):
                    found = True
            assert found

    def test_fft1d(self, setup_rlp):
        max_cell = 1.3 * max(setup_rlp["crystal_symmetry"].unit_cell().parameters()[:3])
        strategy_gpu = FFT1DGPU(max_cell)
        basis_vectors_gpu, used_gpu = strategy_gpu.find_basis_vectors(setup_rlp["rlp"])
        self.check_results(setup_rlp["crystal_symmetry"].unit_cell(), basis_vectors_gpu)

        strategy_cpu = FFT1D(max_cell)
        basis_vectors_cpu, used_cpu = strategy_cpu.find_basis_vectors(setup_rlp["rlp"])
        assert list(used_gpu) == list(used_cpu)
        assert len(basis_vectors_gpu) == len(basis_vectors_cpu)
        for v_gpu, v_cpu in zip(basis_vectors_gpu, basis_vectors_cpu):
            assert v_gpu.elems == pytest.approx(v_cpu.elems, abs=1e-6) or (
                -v_gpu
            ).elems == pytest.approx(v_cpu.elems, abs=1e-6)

    def test_fft3d(self, setup_rlp):
        max_cell = 1.3 * max(setup_rlp["crystal_symmetry"].unit_cell().parameters()[:3])
        strategy_gpu = FFT3DGPU(max_cell)
        basis_vectors_gpu, used_gpu = strategy_gpu.find_basis_vectors(setup_rlp["rlp"])
        self.check_results(setup_rlp["crystal_symmetry"].unit_cell(), basis_vectors_gpu)

        strategy_cpu = FFT3D(max_cell)
        basis_vectors_cpu, used_cpu = strategy_cpu.find_basis_vectors(setup_rlp["rlp"])
        assert list(used_gpu) == list(used_cpu)
        assert len(basis_vectors_gpu) == len(basis_vectors_cpu)
        for v_gpu, v_cpu in zip(basis_vectors_gpu, basis_vectors_cpu):
            assert v_gpu.elems == pytest.approx(v_cpu.elems, abs=1e-6) or (
                -v_gpu
            ).elems == pytest.approx(v_cpu.elems, abs=1e-6)

    def test_real_space_grid_search(self, setup_rlp):
        max_cell = 1.3 * max(setup_rlp["crystal_symmetry"].unit_cell().parameters()[:3])
        strategy_gpu = RealSpaceGridSearchGPU(
            max_cell, target_unit_cell=setup_rlp["crystal_symmetry"].unit_cell()
        )
        basis_vectors_gpu, used_gpu = strategy_gpu.find_basis_vectors(setup_rlp["rlp"])
        self.check_results(setup_rlp["crystal_symmetry"].unit_cell(), basis_vectors_gpu)

        strategy_cpu = RealSpaceGridSearch(
            max_cell, target_unit_cell=setup_rlp["crystal_symmetry"].unit_cell()
        )
        basis_vectors_cpu, used_cpu = strategy_cpu.find_basis_vectors(setup_rlp["rlp"])
        assert list(used_gpu) == list(used_cpu)
        assert len(basis_vectors_gpu) == len(basis_vectors_cpu)
        for v_gpu, v_cpu in zip(basis_vectors_gpu, basis_vectors_cpu):
            assert v_gpu.elems == pytest.approx(v_cpu.elems, abs=1e-6) or (
                -v_gpu
            ).elems == pytest.approx(v_cpu.elems, abs=1e-6)


# =============================================================================
# 2. Equivalent of dials/tests/algorithms/indexing/basis_vector_search/test_combinations.py
# =============================================================================


def test_combinations_gpu(setup_rlp):
    max_cell = 1.3 * max(setup_rlp["crystal_symmetry"].unit_cell().parameters()[:3])
    strategy = FFT1DGPU(max_cell)
    basis_vectors, used = strategy.find_basis_vectors(setup_rlp["rlp"])

    target_symmetry_primitive = (
        setup_rlp["crystal_symmetry"]
        .primitive_setting()
        .customized_copy(space_group_info=sgtbx.space_group().info())
    )
    target_symmetry_sg_only = (
        setup_rlp["crystal_symmetry"]
        .primitive_setting()
        .customized_copy(unit_cell=None)
    )
    target_symmetry_ref = setup_rlp["crystal_symmetry"].as_reference_setting()

    for target_symmetry in (
        setup_rlp["crystal_symmetry"],
        target_symmetry_primitive,
        target_symmetry_sg_only,
        target_symmetry_ref,
    ):
        crystal_models = list(
            candidate_orientation_matrices_gpu(basis_vectors, max_combinations=50)
        )
        filtered_crystal_models = list(
            filter_known_symmetry(crystal_models, target_symmetry=target_symmetry)
        )

        assert filtered_crystal_models
        for model in filtered_crystal_models:
            best_subgroup = find_matching_symmetry(
                model.get_unit_cell(), target_symmetry.space_group()
            )
            if target_symmetry.unit_cell() is not None:
                assert best_subgroup["best_subsym"].unit_cell().is_similar_to(
                    setup_rlp["crystal_symmetry"]
                    .as_reference_setting()
                    .best_cell()
                    .unit_cell(),
                    relative_length_tolerance=0.1,
                    absolute_angle_tolerance=5,
                ) or best_subgroup[
                    "best_subsym"
                ].minimum_cell().unit_cell().is_similar_to(
                    setup_rlp["crystal_symmetry"]
                    .as_reference_setting()
                    .best_cell()
                    .minimum_cell()
                    .unit_cell(),
                    relative_length_tolerance=0.1,
                    absolute_angle_tolerance=5,
                )
            else:
                target_sg = (
                    target_symmetry.space_group_info().reference_setting().group()
                )
                subgroups = metric_subgroups(
                    model.get_crystal_symmetry(), max_delta=5, bravais_types_only=False
                )
                assert target_sg.build_derived_patterson_group() in [
                    g["ref_subsym"].space_group() for g in subgroups.result_groups
                ]


def test_filter_known_symmetry_no_matches_gpu(caplog):
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="dials")
    unit_cell = uctbx.unit_cell((10, 10, 10, 90, 90, 90))
    crystal_models = []
    B = scitbx.matrix.sqr(unit_cell.fractionalization_matrix()).transpose()
    for _ in range(10):
        crystal_models.append(
            Crystal(B, space_group=sgtbx.space_group(), reciprocal=True)
        )

    target_symmetry = crystal.symmetry(
        unit_cell=(15, 15, 15, 85, 85, 85), space_group=sgtbx.space_group()
    )
    list(filter_known_symmetry(crystal_models, target_symmetry=target_symmetry))
    assert "Rejecting crystal model inconsistent with input symmetry" in caplog.text
    assert (
        "No crystal models remaining after comparing with known symmetry" in caplog.text
    )


def test_filter_similar_orientations_gpu():
    space_group = sgtbx.space_group()
    unit_cell = space_group.info().any_compatible_unit_cell(volume=1000)

    crystal_models = []
    B = scitbx.matrix.sqr(unit_cell.fractionalization_matrix()).transpose()
    for i in range(10):
        U = euler_angles_as_matrix((i * 10, 0, 0), deg=True)
        crystal_models.append(Crystal(U * B, space_group=space_group, reciprocal=True))

    other_crystal_models = crystal_models[5:]
    crystal_models = crystal_models[:5]

    filtered = list(filter_similar_orientations(crystal_models, other_crystal_models))
    assert filtered == crystal_models

    cryst = copy.deepcopy(other_crystal_models[0])
    cryst.set_U(
        scitbx.matrix.sqr(cryst.get_U()) * euler_angles_as_matrix((3, 0, 0), deg=True)
    )
    crystal_models.append(cryst)
    filtered = filter_similar_orientations(crystal_models, other_crystal_models)
    assert list(filtered) == crystal_models[:5]

    filtered = filter_similar_orientations(
        crystal_models, other_crystal_models, minimum_angular_separation=2
    )
    assert list(filtered) == crystal_models


# =============================================================================
# 3. Equivalent of dials/tests/algorithms/indexing/basis_vector_search/test_optimise.py
# =============================================================================


def _gradient_fd(target, vector, eps=1e-6):
    grads = []
    for i in range(len(vector)):
        v = list(vector)
        v[i] -= eps
        tm, _ = target.compute_functional_and_gradients(v)
        v[i] += 2 * eps
        tp, _ = target.compute_functional_and_gradients(v)
        grads.append((tp - tm) / (2 * eps))
    return grads


def test_optimise_basis_vectors_gpu(setup_rlp):
    random.seed(42)
    max_cell = 1.3 * max(setup_rlp["crystal_symmetry"].unit_cell().parameters()[:3])
    rlp = setup_rlp["rlp"]
    strategy = FFT3DGPU(max_cell, n_points=256)
    basis_vectors, used = strategy.find_basis_vectors(rlp)

    for v in basis_vectors:
        target = BasisVectorTargetGPU(rlp)
        f, g = target.compute_functional_and_gradients(v)
        assert f == pytest.approx(target.compute_functional(v), rel=1e-12)
        g_fd = _gradient_fd(target, v)
        assert list(g) == pytest.approx(g_fd, rel=1)

        minimised = BasisVectorMinimiserGPU(rlp, v)
        assert v.length() == pytest.approx(minimised.x.norm(), rel=2e-2)
        assert v.elems == pytest.approx(minimised.x, abs=1.2)

    optimised = optimise_basis_vectors_gpu(rlp, basis_vectors)
    assert len(optimised) == len(basis_vectors)


# =============================================================================
# 4. Equivalent of dials/tests/algorithms/indexing/test_assign_indices.py
# =============================================================================


@pytest.fixture
def crystal_factory():
    def _crystal_factory(space_group_symbol):
        seed = 54321
        random.seed(seed)
        flex.set_random_seed(seed)

        space_group_info = sgtbx.space_group_info(symbol=space_group_symbol)
        space_group = space_group_info.group()
        unit_cell = space_group_info.any_compatible_unit_cell(
            volume=random.uniform(1e4, 1e6)
        )

        B = matrix.sqr(unit_cell.fractionalization_matrix()).transpose()
        U = random_rotation()
        A = U * B

        direct_matrix = A.inverse()
        return Crystal(
            direct_matrix[0:3],
            direct_matrix[3:6],
            direct_matrix[6:9],
            space_group=space_group,
        )

    return _crystal_factory


@pytest.fixture
def assign_experiment():
    beam = BeamFactory.make_beam(wavelength=0.97625, sample_to_source=(0, 0, 1))
    detector = DetectorFactory.simple(
        sensor="PAD",
        distance=265.27,
        beam_centre=(210.7602, 205.27684),
        fast_direction="+x",
        slow_direction="-y",
        pixel_size=(0.172, 0.172),
        image_size=(2463, 2527),
        trusted_range=(0, 1e8),
    )
    goniometer = GoniometerFactory.single_axis()
    scan = ScanFactory.make_scan(
        image_range=(1, 20),
        exposure_times=0.067,
        oscillation=(82, 0.15),
        epochs=[0] * 20,
    )
    isetdata = ImageSetData(
        reader=Format.Reader(None, ["path"] * len(scan)), masker=None
    )
    iset = ImageSequence(
        isetdata, beam=beam, detector=detector, goniometer=goniometer, scan=scan
    )
    return Experiment(
        imageset=iset, beam=beam, detector=detector, goniometer=goniometer, scan=scan
    )


class CompareGlobalLocalGPU:
    def __init__(self, experiment, reflections, expected_miller_indices):
        index_reflections_global = AssignIndicesGlobalGPU(gpu_threshold=0)
        index_reflections_local = AssignIndicesLocalGPU()

        self.reflections_global = copy.deepcopy(reflections)
        self.reflections_global["id"] = flex.int(len(self.reflections_global), -1)
        self.reflections_global["imageset_id"] = flex.int(
            len(self.reflections_global), 0
        )
        index_reflections_global(self.reflections_global, ExperimentList([experiment]))
        non_zero_sel = self.reflections_global["miller_index"] != (0, 0, 0)
        assert self.reflections_global["id"].select(~non_zero_sel).all_eq(-1)
        self.misindexed_global = (
            (expected_miller_indices == self.reflections_global["miller_index"])
            .select(non_zero_sel)
            .count(False)
        )
        self.correct_global = (
            expected_miller_indices == self.reflections_global["miller_index"]
        ).count(True)

        self.reflections_local = copy.deepcopy(reflections)
        self.reflections_local["id"] = flex.int(len(self.reflections_local), -1)
        self.reflections_local["imageset_id"] = flex.int(len(self.reflections_local), 0)
        index_reflections_local(self.reflections_local, ExperimentList([experiment]))
        non_zero_sel = self.reflections_local["miller_index"] != (0, 0, 0)
        assert self.reflections_local["id"].select(~non_zero_sel).all_eq(-1)
        self.misindexed_local = (
            (expected_miller_indices == self.reflections_local["miller_index"])
            .select(non_zero_sel)
            .count(False)
        )
        self.correct_local = (
            expected_miller_indices == self.reflections_local["miller_index"]
        ).count(True)


@pytest.mark.parametrize("space_group_symbol", bravais_types.acentric)
def test_assign_indices_gpu(space_group_symbol, assign_experiment, crystal_factory):
    cryst_model = crystal_factory(space_group_symbol)
    assign_experiment.crystal = cryst_model

    predicted_reflections = flex.reflection_table.from_predictions(assign_experiment)
    use_fraction = 0.3
    use_sel = flex.random_selection(
        len(predicted_reflections), int(use_fraction * len(predicted_reflections))
    )
    predicted_reflections = predicted_reflections.select(use_sel)
    miller_indices = predicted_reflections["miller_index"]
    predicted_reflections["xyzobs.mm.value"] = predicted_reflections["xyzcal.mm"]
    predicted_reflections["id"] = flex.int(len(predicted_reflections), 0)
    predicted_reflections.map_centroids_to_reciprocal_space(
        ExperimentList([assign_experiment])
    )

    result = CompareGlobalLocalGPU(
        assign_experiment, predicted_reflections, miller_indices
    )
    assert result.misindexed_local == 0
    assert result.misindexed_global == 0

    a, b, c = map(matrix.col, assign_experiment.crystal.get_real_space_vectors())
    relative_error = 0.02
    a *= 1 + relative_error
    b *= 1 + relative_error
    c *= 1 + relative_error

    cryst_model2 = Crystal(a, b, c, space_group=cryst_model.get_space_group())
    assign_experiment.crystal = cryst_model2

    result = CompareGlobalLocalGPU(
        assign_experiment, predicted_reflections, miller_indices
    )
    assert result.misindexed_local == 0
    assert result.correct_local > result.correct_global
    assert result.misindexed_local < (0.001 * len(result.reflections_local))

    A = matrix.sqr(cryst_model.get_A())
    A = random_rotation(angle_max=0.5) * A

    direct_matrix = A.inverse()
    cryst_model2 = Crystal(
        direct_matrix[0:3],
        direct_matrix[3:6],
        direct_matrix[6:9],
        space_group=cryst_model.get_space_group(),
    )
    assign_experiment.crystal = cryst_model2

    result = CompareGlobalLocalGPU(
        assign_experiment, predicted_reflections, miller_indices
    )
    assert result.misindexed_local <= result.misindexed_global
    assert result.misindexed_local < 0.01 * result.correct_local
    assert result.correct_local >= result.correct_global
    assert result.misindexed_local < (0.001 * len(result.reflections_local))


def test_index_reflections_gpu(dials_data):
    data_dir = dials_data("i04_weak_data")
    experiments = load.experiment_list(
        data_dir / "experiments.json", check_format=False
    )
    reflections = flex.reflection_table.from_file(data_dir / "full.pickle")
    reflections.centroid_px_to_mm(experiments)
    reflections.map_centroids_to_reciprocal_space(experiments)
    reflections["imageset_id"] = flex.int(len(reflections), 0)
    reflections["id"] = flex.int(len(reflections), -1)
    AssignIndicesGlobalGPU(tolerance=0.3, gpu_threshold=0)(reflections, experiments)
    assert "miller_index" in reflections
    counts = reflections["id"].counts()
    assert dict(counts) == {-1: 1390, 0: 114692}


def test_local_multiple_rotations_gpu(dials_data):
    experiments = load.experiment_list(
        dials_data("insulin_processed") / "indexed.expt",
        check_format=False,
    )
    experiments[0].scan.set_image_range((1, 1440))

    reflections = flex.reflection_table.from_predictions(experiments[0], dmin=4)
    reflections["imageset_id"] = flex.int(len(reflections), 0)
    reflections["id"] = flex.int(len(reflections), -1)
    reflections["xyzobs.px.value"] = reflections["xyzcal.px"]
    reflections["xyzobs.mm.value"] = reflections["xyzcal.mm"]
    predicted_miller_indices = reflections["miller_index"]

    reflections.centroid_px_to_mm(experiments)
    reflections.map_centroids_to_reciprocal_space(experiments)
    reflections["miller_index"] = flex.miller_index(len(reflections), (0, 0, 0))

    AssignIndicesLocalGPU()(reflections, experiments)
    assert (reflections["miller_index"] == (0, 0, 0)).count(True) == 0
    assert (reflections["miller_index"] == predicted_miller_indices).count(False) == 0

    experiments[0].scan.set_oscillation((0, 1 - 1 / 360), deg=True)
    reflections["miller_index"] = flex.miller_index(len(reflections), (0, 0, 0))
    reflections["id"] = flex.int(len(reflections), -1)
    reflections.centroid_px_to_mm(experiments)
    reflections.map_centroids_to_reciprocal_space(experiments)

    AssignIndicesLocalGPU()(reflections, experiments)
    indexed_sel = reflections["miller_index"] == (0, 0, 0)
    assert indexed_sel.count(True) < 10
    assert (
        (reflections["miller_index"] != predicted_miller_indices) & ~indexed_sel
    ).count(True) == 0


# =============================================================================
# 5. Equivalent of dials/tests/algorithms/indexing/test_max_cell.py
# =============================================================================


@pytest.fixture(params=bravais_types.acentric)
def setup_max_cell(request):
    random.seed(42)
    flex.set_random_seed(42)
    space_group_symbol = request.param
    sgi = sgtbx.space_group_info(space_group_symbol)
    cs = sgi.any_compatible_crystal_symmetry(volume=random.randint(10_000, 1_000_000))
    ms = cs.build_miller_set(anomalous_flag=True, d_min=3).expand_to_p1()

    B = scitbx.matrix.sqr(cs.unit_cell().fractionalization_matrix()).transpose()
    ms = ms.select(flex.random_permutation(ms.size())[: int(0.25 * ms.size())])

    refl = flex.reflection_table()
    refl["rlp"] = B.elems * ms.indices().as_vec3_double()
    refl["imageset_id"] = flex.int(len(refl))
    refl["xyzobs.mm.value"] = flex.vec3_double(len(refl))

    return {"crystal_symmetry": cs, "reflections": refl}


@pytest.mark.parametrize(
    "histogram_binning,nearest_neighbor_percentile", [("linear", None), ("log", 0.99)]
)
def test_max_cell_gpu(setup_max_cell, histogram_binning, nearest_neighbor_percentile):
    reflections = setup_max_cell["reflections"]
    crystal_symmetry = setup_max_cell["crystal_symmetry"]

    max_cell_multiplier = 1.3
    max_cell_gpu = find_max_cell_gpu(
        reflections,
        max_cell_multiplier=max_cell_multiplier,
        histogram_binning=histogram_binning,
        nearest_neighbor_percentile=nearest_neighbor_percentile,
    )
    max_cell_cpu = cpu_find_max_cell(
        reflections,
        max_cell_multiplier=max_cell_multiplier,
        histogram_binning=histogram_binning,
        nearest_neighbor_percentile=nearest_neighbor_percentile,
    )

    known_max_cell = max(
        crystal_symmetry.primitive_setting().unit_cell().parameters()[:3]
    )
    assert max_cell_gpu.max_cell > known_max_cell
    assert max_cell_gpu.max_cell == pytest.approx(max_cell_cpu.max_cell, rel=1e-6)


def test_max_cell_low_res_with_high_res_noise_gpu(setup_max_cell):
    reflections = copy.deepcopy(setup_max_cell["reflections"])
    crystal_symmetry = setup_max_cell["crystal_symmetry"]

    rlp = reflections["rlp"]
    reflections = reflections.select(1 / rlp.norms() > 4)

    n = int(0.1 * reflections.size())
    rlp_noise = flex.vec3_double(*(flex.random_double(n) for _ in range(3)))
    reflections["rlp"].extend(rlp_noise)
    reflections["imageset_id"].extend(flex.int(rlp_noise.size()))
    reflections["xyzobs.mm.value"].extend(flex.vec3_double(rlp_noise.size()))

    max_cell = find_max_cell_gpu(reflections, max_cell_multiplier=1.3)
    known_max_cell = max(
        crystal_symmetry.primitive_setting().unit_cell().parameters()[:3]
    )
    assert max_cell.max_cell > known_max_cell


# =============================================================================
# 6. Equivalent of dials/tests/algorithms/indexing/test_non_primitive_basis.py
# =============================================================================


@pytest.mark.parametrize("space_group_symbol", bravais_types.acentric)
def test_non_primitive_detect_gpu(space_group_symbol):
    sgi = sgtbx.space_group_info(space_group_symbol)
    cs = sgi.any_compatible_crystal_symmetry(volume=1000)
    ms = cs.build_miller_set(anomalous_flag=True, d_min=1).expand_to_p1()
    result = non_primitive_detect_gpu(ms.indices())
    if sgi.group().conventional_centring_type_symbol() != "P":
        assert result is not None
        assert isinstance(result, scitbx.matrix.sqr)
        assert result.n == (3, 3)
    else:
        assert result is None


@pytest.mark.parametrize("space_group_symbol", bravais_types.acentric)
def test_non_primitive_correct_gpu(space_group_symbol):
    sgi = sgtbx.space_group_info(space_group_symbol)
    cs = sgi.any_compatible_crystal_symmetry(volume=1000)
    ms = cs.build_miller_set(anomalous_flag=True, d_min=1).expand_to_p1()

    B = scitbx.matrix.sqr(cs.unit_cell().fractionalization_matrix()).transpose()
    cryst = Crystal(B, sgtbx.space_group())
    expts = ExperimentList([Experiment(crystal=cryst)])

    refl = flex.reflection_table()
    refl["miller_index"] = ms.indices()
    refl["rlp"] = B.elems * ms.indices().as_vec3_double()
    refl["imageset_id"] = flex.int(len(refl))
    refl["xyzobs.mm.value"] = flex.vec3_double(len(refl))

    non_primitive_correct_gpu(expts, refl, AssignIndicesGlobalGPU())

    cs_corrected = expts.crystals()[0].get_crystal_symmetry()
    assert cs_corrected.change_of_basis_op_to_primitive_setting().is_identity_op()
    assert (
        cs.change_of_basis_op_to_primitive_setting().apply(ms.indices())
        == refl["miller_index"]
    )


# =============================================================================
# 7. Equivalent of dials/tests/algorithms/indexing/lattice_search/test_lattice_search.py
# =============================================================================


@pytest.fixture
def i04_weak_data(dials_data):
    data_dir = dials_data("i04_weak_data")
    reflections = flex.reflection_table.from_file(data_dir / "full.pickle")
    experiments = load.experiment_list(
        data_dir / "experiments_import.json", check_format=False
    )
    return {
        "reflections": slice_reflections(reflections, [(1, 20)]),
        "experiments": slice_experiments(experiments, [(1, 20)]),
    }


@pytest.mark.parametrize(
    "indexing_method,space_group,unit_cell",
    (
        ("fft1d", None, None),
        ("fft3d", None, None),
        ("real_space_grid_search", "P422", (57.8, 57.8, 150.0, 90, 90, 90)),
    ),
)
def test_BasisVectorSearchGPU_i04_weak_data(
    i04_weak_data, indexing_method, space_group, unit_cell
):
    reflections = i04_weak_data["reflections"]
    experiments = i04_weak_data["experiments"]
    params = index_phil_scope.fetch().extract()
    params.indexing.refinement_protocol.n_macro_cycles = 2
    params.indexing.basis_vector_combinations.max_refine = 5
    params.indexing.method = indexing_method
    if unit_cell is not None:
        params.indexing.known_symmetry.unit_cell = uctbx.unit_cell(unit_cell)
    if space_group is not None:
        params.indexing.known_symmetry.space_group = sgtbx.space_group_info(space_group)
    idxr = BasisVectorSearchGPU(reflections, experiments, params)
    idxr.index()

    indexed_experiments = idxr.refined_experiments
    assert len(indexed_experiments) == 1
    assert indexed_experiments[0].crystal.get_unit_cell().parameters() == pytest.approx(
        (57.752, 57.776, 150.013, 90.0101, 89.976, 90.008), rel=1e-3
    )


@pytest.mark.parametrize(
    "indexing_method,space_group,unit_cell",
    (
        ("fft3d", "P422", (57.8, 57.8, 150.0, 90, 90, 90)),
        ("fft1d", "P422", (57.8, 57.8, 150.0, 90, 90, 90)),
        ("real_space_grid_search", "P422", (57.8, 57.8, 150.0, 90, 90, 90)),
        ("low_res_spot_match", "P422", (57.8, 57.8, 150.0, 90, 90, 90)),
    ),
)
def test_stills_indexer_methods_gpu_i04_weak_data(
    i04_weak_data, indexing_method, space_group, unit_cell
):
    reflections = slice_reflections(i04_weak_data["reflections"], [(1, 2)])
    experiments = slice_experiments(i04_weak_data["experiments"], [(1, 2)])
    for experiment in experiments:
        experiment.imageset = ImageSet(
            experiment.imageset.data(), experiment.imageset.indices()
        )
        experiment.imageset.set_scan(None)
        experiment.imageset.set_goniometer(None)
        experiment.scan = None
        experiment.goniometer = None
    params = index_phil_scope.fetch().extract()
    params.indexing.method = indexing_method
    params.indexing.basis_vector_combinations.max_refine = 5
    if unit_cell is not None:
        params.indexing.known_symmetry.unit_cell = uctbx.unit_cell(unit_cell)
    if space_group is not None:
        params.indexing.known_symmetry.space_group = sgtbx.space_group_info(space_group)
    try:
        idxr = StillsIndexerBasisVectorSearchGPU(reflections, experiments, params)
    except RuntimeError:
        idxr = StillsIndexerLatticeSearchGPU(reflections, experiments, params)
    idxr.index()

    indexed_experiments = idxr.refined_experiments
    assert len(indexed_experiments) == 1
    assert indexed_experiments[0].crystal.get_unit_cell().parameters() == pytest.approx(
        (57.752, 57.776, 150.013, 90.0101, 89.976, 90.008), rel=1e-2
    )


# =============================================================================
# 8. Equivalent of dials/tests/algorithms/indexing/test_index.py & test_phi_scan.py
# =============================================================================


def unit_cells_are_similar(
    uc1, uc2, relative_length_tolerance=0.01, absolute_angle_tolerance=1
):
    l1 = uc1.parameters()
    l2 = uc2.parameters()
    for i in range(3):
        if abs(min(l1[i], l2[i]) / max(l1[i], l2[i]) - 1) > relative_length_tolerance:
            return False
    for i in range(3, 6):
        if abs(l1[i] - l2[i]) > absolute_angle_tolerance:
            if abs(l1[i] - (180 - l2[i])) > absolute_angle_tolerance:
                return False
    return True


_indexing_result = collections.namedtuple(
    "indexing", ["indexed_reflections", "experiments", "rmsds"]
)


def run_indexing_gpu(
    reflections,
    experiment,
    working_directory,
    extra_args,
    expected_unit_cell,
    expected_rmsds,
    expected_hall_symbol,
    n_expected_lattices=1,
    relative_length_tolerance=0.005,
    absolute_angle_tolerance=0.5,
):
    commands = [shutil.which("dials_gpu.index")]
    if isinstance(reflections, list):
        commands.extend(reflections)
    else:
        commands.append(reflections)
    if isinstance(experiment, list):
        commands.extend(experiment)
    else:
        commands.append(experiment)
    commands.extend(extra_args)

    result = subprocess.run(commands, cwd=working_directory, capture_output=True)
    assert not result.returncode and not result.stderr, result.stderr.decode()

    out_expts = working_directory / "indexed.expt"
    out_refls = working_directory / "indexed.refl"
    assert out_expts.is_file()
    assert out_refls.is_file()

    experiments_list = load.experiment_list(out_expts, check_format=False)
    assert len(experiments_list.crystals()) == n_expected_lattices
    indexed_reflections = flex.reflection_table.from_file(out_refls)
    indexed_reflections.assert_experiment_identifiers_are_consistent(experiments_list)
    rmsds = None

    for i, experiment in enumerate(experiments_list):
        assert unit_cells_are_similar(
            experiment.crystal.get_unit_cell(),
            expected_unit_cell,
            relative_length_tolerance=relative_length_tolerance,
            absolute_angle_tolerance=absolute_angle_tolerance,
        ), (
            experiment.crystal.get_unit_cell().parameters(),
            expected_unit_cell.parameters(),
        )
        sg = experiment.crystal.get_space_group()
        assert sg.type().hall_symbol() == expected_hall_symbol, (
            sg.type().hall_symbol(),
            expected_hall_symbol,
        )
        reflections_i = indexed_reflections.select(indexed_reflections["id"] == i)
        mi = reflections_i["miller_index"]
        assert (mi != (0, 0, 0)).count(False) == 0
        reflections_i = reflections_i.select(mi != (0, 0, 0))
        reflections_i = reflections_i.select(
            reflections_i.get_flags(reflections_i.flags.used_in_refinement)
        )
        assert len(reflections_i) > 0
        obs_x, obs_y, obs_z = reflections_i["xyzobs.mm.value"].parts()
        calc_x, calc_y, calc_z = reflections_i["xyzcal.mm"].parts()
        rmsd_x = flex.mean(flex.pow2(obs_x - calc_x)) ** 0.5
        rmsd_y = flex.mean(flex.pow2(obs_y - calc_y)) ** 0.5
        rmsd_z = flex.mean(flex.pow2(obs_z - calc_z)) ** 0.5
        rmsds = (rmsd_x, rmsd_y, rmsd_z)
        if isinstance(expected_rmsds[0], (tuple, list)):
            expected = expected_rmsds[i]
        else:
            expected = expected_rmsds
        for actual, limit in zip(rmsds, expected):
            assert actual <= limit, f"Experiment {i}: {rmsds} {expected}"
        assert experiment.identifier != ""
        expt = ExperimentList()
        expt.append(experiment)
        reflections_i.assert_experiment_identifiers_are_consistent(expt)

    return _indexing_result(indexed_reflections, experiments_list, rmsds)


def test_index_i04_weak_data_fft3d_gpu(dials_data, tmp_path):
    data_dir = dials_data("i04_weak_data")
    pickle_path = data_dir / "full.pickle"
    sequence_path = data_dir / "experiments_import.json"
    extra_args = [
        "bin_size_fraction=0.25",
        "image_range=1,20",
        "image_range=250,270",
        "image_range=520,540",
    ]
    expected_unit_cell = uctbx.unit_cell((57.7, 57.7, 149.8, 90, 90, 90))
    expected_rmsds = (0.05, 0.04, 0.0005)
    expected_hall_symbol = " P 1"

    run_indexing_gpu(
        pickle_path,
        sequence_path,
        tmp_path,
        extra_args,
        expected_unit_cell,
        expected_rmsds,
        expected_hall_symbol,
    )


def test_index_i04_weak_data_fft1d_gpu(dials_data, tmp_path):
    data_dir = dials_data("i04_weak_data")
    pickle_path = data_dir / "full.pickle"
    sequence_path = data_dir / "experiments_import.json"
    extra_args = [
        "n_macro_cycles=2",
        "indexing.method=fft1d",
        "bin_size_fraction=0.25",
        "image_range=1,20",
        "image_range=250,270",
        "image_range=520,540",
    ]
    expected_unit_cell = uctbx.unit_cell((57.7, 57.7, 149.9, 90, 90, 90))
    expected_rmsds = (0.06, 0.05, 0.0005)
    expected_hall_symbol = " P 1"

    run_indexing_gpu(
        pickle_path,
        sequence_path,
        tmp_path,
        extra_args,
        expected_unit_cell,
        expected_rmsds,
        expected_hall_symbol,
    )


def test_index_trypsin_two_lattice_P212121_gpu(dials_data, tmp_path):
    data_dir = dials_data("semisynthetic_multilattice")
    strong_path = data_dir / "ag_strong_1_50.refl"
    sequence_path = data_dir / "ag_imported_1_50.expt"
    extra_args = [
        "indexing.method=real_space_grid_search",
        "reflections_per_degree=10",
        "n_macro_cycles=5",
        "known_symmetry.unit_cell=54.3,58.3,66.5,90,90,90",
        "known_symmetry.space_group=P212121",
        "image_range=0,10",
        "beam.fix=all",
        "detector.fix=all",
        "max_cell=70",
    ]
    expected_unit_cell = uctbx.unit_cell((54.3, 58.3, 66.5, 90, 90, 90))
    expected_rmsds = (0.31, 0.49, 0.002)
    expected_hall_symbol = " P 2ac 2ab"

    run_indexing_gpu(
        strong_path,
        sequence_path,
        tmp_path,
        extra_args,
        expected_unit_cell,
        expected_rmsds,
        expected_hall_symbol,
        n_expected_lattices=1,
        relative_length_tolerance=0.02,
        absolute_angle_tolerance=1,
    )


def test_index_trypsin_index_assignment_local_gpu(dials_data, tmp_path):
    data_dir = dials_data("semisynthetic_multilattice")
    reflections_path = data_dir / "ag_strong_1_50.refl"
    sequence_path = data_dir / "ag_imported_1_50.expt"
    extra_args = [
        "indexing.method=real_space_grid_search",
        "d_min_start=3",
        "n_macro_cycles=3",
        "known_symmetry.unit_cell=54.3,58.3,66.5,90,90,90",
        "known_symmetry.space_group=P212121",
        "image_range=0,10",
        "beam.fix=all",
        "detector.fix=all",
        "max_lattices=2",
        "index_assignment.method=local",
        "nearest_neighbours=50",
    ]
    expected_unit_cell = uctbx.unit_cell((54.3, 58.3, 66.5, 90, 90, 90))
    expected_rmsds = (0.32, 0.41, 0.004)
    expected_hall_symbol = " P 2ac 2ab"

    run_indexing_gpu(
        reflections_path,
        sequence_path,
        tmp_path,
        extra_args,
        expected_unit_cell,
        expected_rmsds,
        expected_hall_symbol,
        n_expected_lattices=2,
        relative_length_tolerance=0.02,
        absolute_angle_tolerance=1,
    )


def test_index_peak_search_clean_gpu(dials_data, tmp_path):
    data_dir = dials_data("i04_weak_data")
    pickle_path = data_dir / "first_image.pickle"
    sequence_path = data_dir / "experiments_import.json"
    extra_args = [
        "indexing.method=fft3d",
        "known_symmetry.space_group=P4",
        "known_symmetry.unit_cell=57.8,57.8,150,90,90,90",
        "peak_search=clean",
        "min_samples=15",
        "n_macro_cycles=4",
        "reciprocal_space_grid.d_min=4",
    ]
    expected_unit_cell = uctbx.unit_cell((57.8, 57.8, 150, 90, 90, 90))
    expected_rmsds = (0.06, 0.07, 0.003)
    expected_hall_symbol = " P 4"

    run_indexing_gpu(
        pickle_path,
        sequence_path,
        tmp_path,
        extra_args,
        expected_unit_cell,
        expected_rmsds,
        expected_hall_symbol,
    )


def test_phi_scan_gpu(dials_data, tmp_path):
    expected_unit_cell = uctbx.unit_cell(
        (11.624, 13.550, 30.103, 89.964, 93.721, 90.132)
    )
    expected_rmsds = (0.039, 0.035, 0.002)

    experiments_old = dials_data("indexing_test_data") / "phi_scan_old.expt"
    experiments_new = dials_data("indexing_test_data") / "phi_scan.expt"
    strong_pickle = dials_data("indexing_test_data") / "phi_scan_strong.pickle"

    result_old = run_indexing_gpu(
        strong_pickle,
        experiments_old,
        tmp_path,
        extra_args=[],
        expected_unit_cell=expected_unit_cell,
        expected_rmsds=expected_rmsds,
        expected_hall_symbol=" P 1",
    )
    result_new = run_indexing_gpu(
        strong_pickle,
        experiments_new,
        tmp_path,
        extra_args=[],
        expected_unit_cell=expected_unit_cell,
        expected_rmsds=expected_rmsds,
        expected_hall_symbol=" P 1",
    )

    assert result_old.rmsds == pytest.approx(result_new.rmsds, abs=1e-6)
    assert result_old.experiments[
        0
    ].crystal.get_unit_cell().parameters() == pytest.approx(
        result_new.experiments[0].crystal.get_unit_cell().parameters(), abs=1e-6
    )
