from __future__ import annotations

import copy
import json
import math
import os
import pickle
import random
import shutil
import subprocess
from unittest import mock

import numpy as np
import pytest
from dxtbx.model import Beam, Crystal, Detector, Experiment, ExperimentList, Goniometer, Scan
from dxtbx.model.experiment_list import ExperimentListFactory
from dxtbx.serialize import load
from scitbx import matrix

import dials.command_line.integrate
from dials.algorithms.background.glm import Creator as GLMCreator
from dials.algorithms.background.simple import (
    Constant2dModeller,
    Constant3dModeller,
    Creator as SimpleCreator,
    Linear2dModeller,
    Linear3dModeller,
    NSigmaOutlierRejector,
    TruncatedOutlierRejector,
    TukeyOutlierRejector,
)
from dials.algorithms.integration import Corrections, CorrectionsMulti
from dials.algorithms.integration.filtering import IceRingFilter, PowderRingFilter
from dials.algorithms.integration.fit import ProfileFitter
from dials.algorithms.integration.integrator import (
    Executor,
    IntegratorExecutor,
    JobList,
    Parameters,
    Processor2D,
    Processor3D,
    ProcessorFlat3D,
    ProcessorSingle2D,
    ProcessorStills,
    generate_phil_scope,
    phil_scope as integrator_phil_scope,
)
from dials.algorithms.integration.overlaps_filter import OverlapsFilter, OverlapsFilterMultiExpt
from dials.algorithms.integration.processor import _Job, execute_parallel_task
from dials.algorithms.profile_model.gaussian_rs import (
    BBoxCalculator3D,
    CoordinateSystem,
    MaskCalculator3D,
    Model as ProfileModel,
    PartialityCalculator3D,
)
from dials.algorithms.profile_model.gaussian_rs.calculator import (
    ComputeEsdBeamDivergence,
    ComputeEsdReflectingRange,
    FractionOfObservedIntensity,
    ProfileModelCalculator,
)
from dials.algorithms.shoebox import MaskCode
from dials.array_family import flex
from dials.util.options import Importer, flatten_experiments, flatten_reflections
from dials.util.phil import parse

import dials_gpu.command_line.integrate
from dials_gpu.integration import (
    AutoBackgroundGPU,
    ComputeEsdBeamDivergenceGPU,
    ComputeEsdReflectingRangeGPU,
    Constant2dModellerGPU,
    Constant3dModellerGPU,
    CorrectionsGPU,
    FractionOfObservedIntensityGPU,
    FrameSlicedIntegratorExecutorGPU,
    GLMBackgroundGPU,
    GaussianRSProfileModelGPU,
    Integrator2DGPU,
    Integrator3DGPU,
    IntegratorExecutorGPU,
    IntegratorFlat3DGPU,
    IntegratorSingle2DGPU,
    IntegratorStillsGPU,
    Linear2dModellerGPU,
    Linear3dModellerGPU,
    NSigmaOutlierRejectorGPU,
    NullBackgroundGPU,
    ProfileFitterGPU,
    ProfileModelCalculatorGPU,
    ProfileModelFactoryGPU,
    SimpleBackgroundGPU,
    SimpleCentroidGPU,
    SummationGPU,
    TruncatedOutlierRejectorGPU,
    TukeyOutlierRejectorGPU,
    compute_centroid_gpu,
    compute_corrections_gpu,
    compute_d_gpu,
    compute_partiality_gpu,
    compute_summed_intensity_gpu,
    compute_zeta_gpu,
    create_integrator_gpu,
    lp_correction_gpu,
    qe_correction_gpu,
)


def _get_dials_gpu_integrate_bin() -> str:
    return (
        shutil.which("dials_gpu.integrate")
        or "/home/davidm/dials/conda_base/bin/dials_gpu.integrate"
    )


# ==============================================================================
# 1. Port of tests/algorithms/integration/test_corrections.py
# ==============================================================================


def test_lp_correction_gpu(dials_data):
    expts = ExperimentListFactory.from_json_file(
        dials_data("centroid_test_data") / "experiments.json", check_format=False
    )
    beam = expts[0].beam
    goniometer = expts[0].goniometer
    detector = expts[0].detector
    correct_cpu = Corrections(beam, goniometer, detector)
    correct_gpu = CorrectionsGPU(beam, goniometer, detector)

    s0 = matrix.col(beam.get_s0())
    m2 = matrix.col(goniometer.get_rotation_axis())
    pn = matrix.col(beam.get_polarization_normal())
    pf = beam.get_polarization_fraction()

    s1_list = flex.vec3_double()
    for _ in range(1000):
        s1 = matrix.col(
            (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
        ).normalize() * s0.length()
        s1_list.append(s1)
        lp1 = correct_cpu.lp(s1)
        lp2 = correct_gpu.lp(s1)
        L = abs(m2.dot(s1.cross(s0))) / (s1.length() * s0.length())
        P1 = (1 - 2 * pf) * (1.0 - (pn.dot(s1) / s1.length()) ** 2)
        P2 = pf * (1.0 + (s1.dot(s0) / (s1.length() * s0.length())) ** 2)
        P = P1 + P2
        lp_manual = L / P
        assert pytest.approx(lp1, abs=1e-7) == lp_manual
        assert pytest.approx(lp2, abs=1e-7) == lp_manual

    lp_arr_cpu = flex.double([correct_cpu.lp(s1) for s1 in s1_list])
    lp_arr_gpu = correct_gpu.lp(s1_list)
    np.testing.assert_allclose(
        lp_arr_gpu.as_numpy_array(), lp_arr_cpu.as_numpy_array(), rtol=1e-12, atol=1e-12
    )


def test_qe_correction_gpu(dials_data):
    expts = ExperimentListFactory.from_json_file(
        dials_data("centroid_test_data") / "experiments.json", check_format=False
    )
    beam = expts[0].beam
    goniometer = expts[0].goniometer
    detector = expts[0].detector
    correct_cpu = Corrections(beam, goniometer, detector)
    correct_gpu = CorrectionsGPU(beam, goniometer, detector)

    s1_list = flex.vec3_double(
        [(random.uniform(-1, 1), random.uniform(-1, 1), -1.0) for _ in range(200)]
    )
    panels = flex.size_t(200, 0)

    qe_cpu = flex.double([correct_cpu.qe(s1, int(p)) for s1, p in zip(s1_list, panels)])
    qe_gpu = correct_gpu.qe(s1_list, panels)
    np.testing.assert_allclose(
        qe_gpu.as_numpy_array(), qe_cpu.as_numpy_array(), rtol=1e-12, atol=1e-12
    )


# ==============================================================================
# 2. Port of tests/algorithms/integration/test_filtering.py
# ==============================================================================


def test_powder_ring_filter():
    from cctbx import sgtbx, uctbx

    unit_cell = uctbx.unit_cell((4.498, 4.498, 7.338, 90, 90, 120))
    space_group = sgtbx.space_group_info(number=194).group()
    d_spacings = flex.double([1.9220704466392748])
    d_min = flex.min(d_spacings)
    ice_filter = PowderRingFilter(unit_cell, space_group, d_min, width=0.004)
    assert min(uctbx.d_star_sq_as_d(ice_filter.d_star_sq)) < d_min
    assert all(ice_filter(d_spacings))


def test_ice_ring_filter():
    filter_obj = IceRingFilter()
    d = flex.double([3.897, 3.669, 3.441, 2.671, 2.249, 2.50])
    result = filter_obj(d)
    assert list(result[:5]) == [True, True, True, True, True]
    assert result[5] is False


# ==============================================================================
# 3. Port of tests/algorithms/integration/profile/test_profile_fitting.py
# ==============================================================================


def _evaluate_gaussian(x, a, x0, sx):
    return a * math.exp(-((x - x0) ** 2) / (2.0 * sx**2))


def _gaussian_3d(size, a, x0, sx):
    result = flex.double(flex.grid(size))
    for k in range(size[0]):
        for j in range(size[1]):
            for i in range(size[2]):
                result[k, j, i] = (
                    _evaluate_gaussian(k, 1, x0[0], sx[0])
                    * _evaluate_gaussian(j, 1, x0[1], sx[1])
                    * _evaluate_gaussian(i, 1, x0[2], sx[2])
                )
    result /= flex.sum(result)
    result *= a
    return result


def test_profile_fitting_zero_intensity_gpu():
    np.random.seed(0)
    p = _gaussian_3d((9, 9, 9), 1, (4, 4, 4), (1.5, 1.5, 1.5))
    s = flex.double(flex.grid(9, 9, 9), 0)
    b = flex.double(np.random.poisson(10, size=(9, 9, 9)).astype(np.float64))
    m = flex.bool(flex.grid(9, 9, 9), True)
    c = s + b
    b_model = flex.double(flex.grid(9, 9, 9), 10.0)

    fit_cpu = ProfileFitter(c, b_model, m, p)
    fit_gpu = ProfileFitterGPU(c, b_model, m, p)

    assert fit_gpu.intensity()[0] == pytest.approx(fit_cpu.intensity()[0], abs=1e-7)
    assert fit_gpu.variance()[0] == pytest.approx(fit_cpu.variance()[0], abs=1e-7)
    assert fit_gpu.correlation() == pytest.approx(fit_cpu.correlation(), abs=1e-7)
    assert fit_gpu.niter() == fit_cpu.niter()


def test_profile_fitting_known_intensity_gpu():
    np.random.seed(42)
    p = _gaussian_3d((9, 9, 9), 1, (4, 4, 4), (1.5, 1.5, 1.5))
    s = flex.double(np.random.poisson(500 * p.as_numpy_array()).astype(np.float64))
    b = flex.double(np.random.poisson(10, size=(9, 9, 9)).astype(np.float64))
    m = flex.bool(flex.grid(9, 9, 9), True)
    c = s + b
    b_model = flex.double(flex.grid(9, 9, 9), 10.0)

    fit_cpu = ProfileFitter(c, b_model, m, p)
    fit_gpu = ProfileFitterGPU(c, b_model, m, p)

    assert fit_gpu.intensity()[0] == pytest.approx(fit_cpu.intensity()[0], abs=1e-6)
    assert fit_gpu.variance()[0] == pytest.approx(fit_cpu.variance()[0], abs=1e-6)
    assert fit_gpu.correlation() == pytest.approx(fit_cpu.correlation(), abs=1e-6)


def test_profile_fitting_deconvolution_3_spots_gpu():
    from dxtbx import flumpy

    np.random.seed(123)
    p = flex.double(flex.grid(3, 9, 9, 9), 0.0)
    p0 = _gaussian_3d((9, 9, 9), 1, (2.5, 2.5, 2.5), (1.5, 1.5, 1.5))
    p1 = _gaussian_3d((9, 9, 9), 1, (4.0, 4.0, 4.0), (1.5, 1.5, 1.5))
    p2 = _gaussian_3d((9, 9, 9), 1, (5.5, 5.5, 5.5), (1.5, 1.5, 1.5))
    p_np = flumpy.to_numpy(p)
    p_np[0] = p0.as_numpy_array()
    p_np[1] = p1.as_numpy_array()
    p_np[2] = p2.as_numpy_array()

    I_true = [300.0, 500.0, 700.0]
    c_np = (
        I_true[0] * p_np[0]
        + I_true[1] * p_np[1]
        + I_true[2] * p_np[2]
        + 10.0
    )
    c = flex.double(np.ascontiguousarray(c_np, dtype=np.float64))
    b_model = flex.double(flex.grid(9, 9, 9), 10.0)
    m = flex.bool(flex.grid(9, 9, 9), True)

    fit_cpu = ProfileFitter(c, b_model, m, p)
    fit_gpu = ProfileFitterGPU(c, b_model, m, p)

    np.testing.assert_allclose(
        np.asarray(fit_gpu.intensity()),
        np.asarray(fit_cpu.intensity()),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(fit_gpu.intensity()),
        np.array(I_true),
        rtol=1e-3,
        atol=1e-3,
    )


# ==============================================================================
# 4. Port of tests/algorithms/background/test_modeller.py
# ==============================================================================


def test_background_modellers_and_rejectors_gpu():
    # Generate synthetic shoeboxes and compare SimpleBackgroundGPU & GLMBackgroundGPU against CPU
    from dials.command_line.integrate import phil_scope as int_phil

    r = flex.reflection_table()
    n_sb = 20
    r["id"] = flex.int(n_sb, 0)
    r["panel"] = flex.size_t(n_sb, 0)
    bboxes = [(10, 21, 10, 21, 0, 5) for _ in range(n_sb)]
    r["bbox"] = flex.int6(bboxes)
    r["shoebox"] = flex.shoebox(r["panel"], r["bbox"], allocate=True)

    rng = np.random.default_rng(42)
    for i in range(n_sb):
        sb = r["shoebox"][i]
        d_np = sb.data.as_numpy_array()
        m_np = sb.mask.as_numpy_array()
        bg_level = float((i + 1) * 5)
        d_np[:] = rng.poisson(bg_level, size=d_np.shape).astype(np.float32)
        m_np[:] = 3  # Valid | Background
        # Put a foreground peak in the middle
        m_np[1:4, 4:7, 4:7] = 5  # Valid | Foreground
        d_np[1:4, 4:7, 4:7] += 100.0
        # Put an outlier pixel in background
        d_np[0, 0, 0] = bg_level * 20.0 + 100.0

    # Test GLM constant3d and constant2d
    for glm_model in ("constant3d", "constant2d"):
        params = int_phil.fetch(
            source=parse(
                f"integration.background.algorithm=glm\nintegration.background.glm.model.algorithm={glm_model}"
            )
        ).extract()
        r_cpu = copy.deepcopy(r)
        r_gpu = copy.deepcopy(r)
        s_cpu = GLMBackgroundGPU(params, None)._cpu_fallback or dials.extensions.Background.load("glm")(params, None)
        ok_cpu = s_cpu.compute_background(r_cpu)
        ok_gpu = GLMBackgroundGPU(params, None).compute_background(r_gpu)
        assert list(ok_gpu) == list(ok_cpu)
        np.testing.assert_allclose(
            r_gpu["background.mean"].as_numpy_array(),
            r_cpu["background.mean"].as_numpy_array(),
            rtol=1e-4,
            atol=1e-4,
        )

    # Test Simple background with all rejectors & modellers
    for outlier_alg in ("null", "nsigma", "tukey", "truncated"):
        for model_alg in ("constant3d", "constant2d", "linear3d", "linear2d"):
            params = int_phil.fetch(
                source=parse(
                    f"integration.background.algorithm=simple\n"
                    f"integration.background.simple.outlier.algorithm={outlier_alg}\n"
                    f"integration.background.simple.model.algorithm={model_alg}"
                )
            ).extract()
            r_cpu = copy.deepcopy(r)
            r_gpu = copy.deepcopy(r)
            s_cpu = dials.extensions.Background.load("simple")(params, None)
            ok_cpu = s_cpu.compute_background(r_cpu)
            ok_gpu = SimpleBackgroundGPU(params, None).compute_background(r_gpu)
            assert list(ok_gpu) == list(ok_cpu)
            np.testing.assert_allclose(
                r_gpu["background.mean"].as_numpy_array(),
                r_cpu["background.mean"].as_numpy_array(),
                rtol=1e-4,
                atol=1e-4,
            )
            np.testing.assert_allclose(
                r_gpu["background.mse"].as_numpy_array(),
                r_cpu["background.mse"].as_numpy_array(),
                rtol=1e-4,
                atol=1e-4,
            )


# ==============================================================================
# 5. Port of tests/algorithms/profile_model/test_calculator.py & test_profile_model.py
# ==============================================================================


def test_profile_model_calculator_gpu(dials_data):
    data_dir = dials_data("centroid_test_data")
    expts = load.experiment_list(data_dir / "experiments.json", check_format=True)
    refls = flex.reflection_table.from_file(data_dir / "integrated.pickle")

    # Compute zeta on reflections
    refls["zeta"] = compute_zeta_gpu(expts[0], refls["s1"])

    for centroid_def in ("s1", "com"):
        for alg in ("basic", "extended"):
            calc_cpu = ProfileModelCalculator(
                copy.deepcopy(refls),
                expts[0].crystal,
                expts[0].beam,
                expts[0].detector,
                expts[0].goniometer,
                expts[0].scan,
                min_zeta=0.05,
                algorithm=alg,
                centroid_definition=centroid_def,
            )
            calc_gpu = ProfileModelCalculatorGPU(
                copy.deepcopy(refls),
                expts[0].crystal,
                expts[0].beam,
                expts[0].detector,
                expts[0].goniometer,
                expts[0].scan,
                min_zeta=0.05,
                algorithm=alg,
                centroid_definition=centroid_def,
            )
            assert calc_gpu.sigma_b() == pytest.approx(calc_cpu.sigma_b(), rel=1e-6, abs=1e-8)
            assert calc_gpu.sigma_m() == pytest.approx(calc_cpu.sigma_m(), rel=1e-5, abs=1e-7)


# ==============================================================================
# 6. Port of tests/algorithms/integration/test_integrator.py & test_interface.py
# ==============================================================================


def test_integrator_executor_gpu(dials_data):
    data_dir = dials_data("centroid_test_data")
    expts = load.experiment_list(data_dir / "indexed.expt", check_format=True)
    refls = flex.reflection_table.from_file(data_dir / "indexed.refl")

    params = dials.command_line.integrate.phil_scope.fetch(
        source=parse("integration.mp.nproc=1\nprofile.fitting=False")
    ).extract()

    expts_cpu, refls_cpu, _ = dials.command_line.integrate.run_integration(
        copy.deepcopy(params), copy.deepcopy(expts), copy.deepcopy(refls)
    )
    expts_gpu, refls_gpu, _ = dials_gpu.command_line.integrate.run_integration(
        copy.deepcopy(params), copy.deepcopy(expts), copy.deepcopy(refls)
    )

    assert len(refls_gpu) == len(refls_cpu)
    for col in (
        "intensity.sum.value",
        "intensity.sum.variance",
        "background.sum.value",
        "background.mean",
        "xyzobs.px.value",
        "xyzobs.px.variance",
        "xyzobs.mm.value",
        "xyzobs.mm.variance",
        "lp",
    ):
        tol = 1e-2 if "sum" in col else 1e-5
        np.testing.assert_allclose(
            refls_gpu[col].as_numpy_array(),
            refls_cpu[col].as_numpy_array(),
            rtol=tol,
            atol=tol,
        )


def test_time_and_memory_info():
    jobs = JobList()
    jobs.add((0, 1), (0, 9), 5, 0)
    r = flex.reflection_table()
    r["id"] = flex.int(10, 0)
    r["bbox"] = flex.int6([(0, 10, 0, 10, 0, 5)] * 10)
    r["flags"] = flex.size_t(10, flex.reflection_table.flags.integrated)
    mem = jobs.shoebox_memory(r, False)
    assert len(mem) == len(jobs)


# ==============================================================================
# 7. Port of tests/algorithms/integration/test_filter_overlaps.py
# ==============================================================================


def test_filter_overlaps(dials_data):
    data_dir = dials_data("centroid_test_data")
    expts = load.experiment_list(data_dir / "indexed.expt", check_format=True)
    refls = flex.reflection_table.from_file(data_dir / "indexed.refl")

    params = dials.command_line.integrate.phil_scope.fetch(
        source=parse(
            "integration.mp.nproc=1\n"
            "integration.debug.output=True\n"
            "integration.debug.separate_files=False\n"
            "integration.overlaps_filter.foreground_foreground.enable=True\n"
            "integration.overlaps_filter.foreground_background.enable=True\n"
            "profile.fitting=False"
        )
    ).extract()

    _, refls_cpu, _ = dials.command_line.integrate.run_integration(
        copy.deepcopy(params), copy.deepcopy(expts), copy.deepcopy(refls)
    )
    _, refls_gpu, _ = dials_gpu.command_line.integrate.run_integration(
        copy.deepcopy(params), copy.deepcopy(expts), copy.deepcopy(refls)
    )
    assert len(refls_gpu) == len(refls_cpu)


# ==============================================================================
# 8. Port of tests/command_line/test_integrate.py
# ==============================================================================


def test_basic_integrate_gpu(dials_data, tmp_path):
    exp = load.experiment_list(
        dials_data("centroid_test_data") / "experiments.json", check_format=True
    )
    exp[0].identifier = "foo"
    exp.as_json(tmp_path / "modified_input.json")

    result = subprocess.run(
        [
            _get_dials_gpu_integrate_bin(),
            "nproc=1",
            "modified_input.json",
            "profile.fitting=False",
            "integration.integrator=3d",
            "prediction.padding=0",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr

    experiments = load.experiment_list(
        tmp_path / "integrated.expt", check_format=False
    )
    assert experiments[0].identifier == "foo"

    table = flex.reflection_table.from_file(tmp_path / "integrated.refl")
    mask = table.get_flags(table.flags.integrated, all=False)
    assert len(table) == 1666
    assert mask.count(True) == 1666
    assert dict(table.experiment_identifiers()) == {0: "foo"}


@pytest.mark.parametrize("integrator", ["2d", "single2d", "3d", "flat3d"])
def test_EntryPoint_gpu(dials_data, tmp_path, integrator):
    shutil.copy(
        dials_data("centroid_test_data") / "experiments.json",
        tmp_path / "experiments.json",
    )
    for i in range(1, 10):
        shutil.copy(
            dials_data("centroid_test_data") / f"centroid_000{i}.cbf",
            tmp_path,
        )
    shutil.copy(
        dials_data("centroid_test_data") / "indexed.refl",
        tmp_path / "indexed.refl",
    )

    result = subprocess.run(
        [
            _get_dials_gpu_integrate_bin(),
            "nproc=1",
            "experiments.json",
            "indexed.refl",
            "profile.fitting=False",
            f"integration.integrator={integrator}",
            "prediction.padding=0",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr


def test_integration_with_sampling_gpu(dials_data, tmp_path):
    exp = load.experiment_list(
        dials_data("centroid_test_data") / "experiments.json", check_format=True
    )
    exp[0].identifier = "foo"
    exp.as_json(tmp_path / "modified_input.json")

    result = subprocess.run(
        [
            _get_dials_gpu_integrate_bin(),
            "nproc=1",
            "modified_input.json",
            "profile.fitting=False",
            "sampling.integrate_all_reflections=False",
            "sampling.random_seed=42",
            "prediction.padding=0",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    table = flex.reflection_table.from_file(tmp_path / "integrated.refl")
    assert len(table) == 839


def test_integration_with_sample_size_gpu(dials_data, tmp_path):
    exp = load.experiment_list(
        dials_data("centroid_test_data") / "experiments.json", check_format=True
    )
    exp[0].identifier = "foo"
    exp.as_json(tmp_path / "modified_input.json")

    result = subprocess.run(
        [
            _get_dials_gpu_integrate_bin(),
            "nproc=1",
            "modified_input.json",
            "profile.fitting=False",
            "sampling.integrate_all_reflections=False",
            "sampling.random_seed=42",
            "sampling.minimum_sample_size=500",
            "prediction.padding=0",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    table = flex.reflection_table.from_file(tmp_path / "integrated.refl")
    assert len(table) == 415


@pytest.mark.parametrize(
    "ScanRange,ExpectedOscillation",
    [
        ((0, 9), (0.0, 1.8)),
        ((1, 5), (0.2, 1.0)),
    ],
)
def test_integrate_scan_range_gpu(
    dials_data, tmp_path, ScanRange, ExpectedOscillation
):
    expt_path = dials_data("centroid_test_data") / "indexed.expt"
    refl_path = dials_data("centroid_test_data") / "indexed.refl"

    params_cpu = dials.command_line.integrate.working_phil.fetch(
        source=parse(
            f"integration.mp.nproc=1\nintegration.profile.fitting=False\nprediction.padding=0\nscan_range={ScanRange[0]},{ScanRange[1]}"
        )
    ).extract()
    expts_cpu, refls_cpu, _ = dials.command_line.integrate.run_integration(
        params_cpu,
        load.experiment_list(expt_path, check_format=True),
        flex.reflection_table.from_file(refl_path),
    )

    scan_range_strs = [f"scan_range={ScanRange[0]},{ScanRange[1]}"]
    result = subprocess.run(
        [
            _get_dials_gpu_integrate_bin(),
            "nproc=1",
            str(expt_path),
            str(refl_path),
            "profile.fitting=False",
            "prediction.padding=0",
        ]
        + scan_range_strs,
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    experiments = load.experiment_list(tmp_path / "integrated.expt", check_format=False)
    assert experiments[0].scan.get_oscillation_range() == pytest.approx(
        ExpectedOscillation
    )
    table = flex.reflection_table.from_file(tmp_path / "integrated.refl")
    assert len(table) == len(refls_cpu)


def test_frame_slice_shoeboxes_not_available_for_flat3d_gpu(dials_data, tmp_path):
    result = subprocess.run(
        [
            _get_dials_gpu_integrate_bin(),
            "nproc=1",
            dials_data("centroid_test_data") / "indexed.expt",
            dials_data("centroid_test_data") / "indexed.refl",
            "frame_slice_shoeboxes=True",
            "integrator=flat3d",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert result.returncode
    assert b"cannot be used with integrator=flat3d" in result.stderr


def test_multi_sweep_gpu(dials_data, tmp_path):
    expts = str(dials_data("centroid_test_data") / "multi_sweep_indexed.expt")
    refls = str(dials_data("centroid_test_data") / "multi_sweep_indexed.refl")
    result = subprocess.run(
        [
            _get_dials_gpu_integrate_bin(),
            "nproc=1",
            expts,
            refls,
            "prediction.padding=0",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "integrated.refl").is_file()
    assert (tmp_path / "integrated.expt").is_file()

    experiments = load.experiment_list(tmp_path / "integrated.expt")
    for i, expt in enumerate(experiments):
        assert expt.identifier == str(100 + i)

    table = flex.reflection_table.from_file(tmp_path / "integrated.refl")
    assert len(table) == 3530
    assert dict(table.experiment_identifiers()) == {0: "100", 1: "101"}


def test_multi_lattice_gpu(dials_data, tmp_path):
    expt = str(dials_data("trypsin_multi_lattice") / "refined.expt")
    refl = str(dials_data("trypsin_multi_lattice") / "refined.refl")

    result = subprocess.run(
        [
            _get_dials_gpu_integrate_bin(),
            "nproc=1",
            "d_min=4",
            "prediction.padding=0",
            expt,
            refl,
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "integrated.refl").is_file()
    assert (tmp_path / "integrated.expt").is_file()

    table = flex.reflection_table.from_file(tmp_path / "integrated.refl")
    assert len(table) == 4020
    mask = table.get_flags(table.flags.integrated_prf)
    table = table.select(mask)
    exp_id = list(set(table["id"]))
    assert len(exp_id) == 6


def test_integrate_with_kapton_gpu(dials_data, tmp_path):
    data_dir = dials_data("integration_test_data")
    refl_path = str(data_dir / "kapton-idx-20161021225550223_indexed.refl")
    expt_path = str(data_dir / "kapton-idx-20161021225550223_refined.expt")
    image_path = str(data_dir / "kapton-20161021225550223.pickle")
    mask_path = str(data_dir / "kapton-mask.pickle")

    shutil.copy(refl_path, tmp_path)
    shutil.copy(image_path, tmp_path)

    templ_phil = """
      output {
        experiments = 'idx-20161021225550223_integrated_experiments_%s.expt'
        reflections = 'idx-20161021225550223_integrated_%s.refl'
      }
      integration {
        lookup.mask = '%s'
        integrator = stills
        profile.fitting = False
        background.algorithm = simple
        debug {
          output = True
          separate_files = False
          split_experiments = False
        }
      }
      profile {
        gaussian_rs.min_spots.overall = 0
      }
      absorption_correction {
        apply = %s
        algorithm = fuller_kapton
        fuller_kapton {
          smart_sigmas = True
        }
      }
"""
    (tmp_path / "integrate_without_kapton.phil").write_text(
        templ_phil % ("nokapton", "nokapton", mask_path, "False")
    )
    (tmp_path / "integrate_with_kapton.phil").write_text(
        templ_phil % ("kapton", "kapton", mask_path, "True")
    )

    for phil in "integrate_without_kapton.phil", "integrate_with_kapton.phil":
        result = subprocess.run(
            [_get_dials_gpu_integrate_bin(), "nproc=1", refl_path, expt_path, phil],
            cwd=tmp_path,
            capture_output=True,
        )
        assert not result.returncode and not result.stderr

    results = []
    for mode in "kapton", "nokapton":
        table = flex.reflection_table.from_file(
            tmp_path / f"idx-20161021225550223_integrated_{mode}.refl"
        )
        millers = table["miller_index"]
        test_indices = {"zero": (-5, 2, -6), "low": (-2, -20, 7), "high": (-1, -10, 4)}
        test_rows = {k: millers.first_index(v) for k, v in test_indices.items()}
        test_I_sigsqI = {
            k: (table[v]["intensity.sum.value"], table[v]["intensity.sum.variance"])
            for k, v in test_rows.items()
        }
        results.append(test_I_sigsqI)
    assert results[0]["zero"][0] == results[1]["zero"][0]
    assert results[0]["zero"][1] - results[1]["zero"][1] < 0.0001
    assert False not in [results[0]["low"][i] > results[1]["low"][i] for i in (0, 1)]
    assert False not in [results[0]["high"][i] > results[1]["high"][i] for i in (0, 1)]


def test_reference_shoeboxes_freed_after_profile_modelling_gpu(dials_data):
    data = dials_data("centroid_test_data")
    experiments = load.experiment_list(data / "indexed.expt", check_format=True)
    reference = flex.reflection_table.from_file(data / "indexed.refl")
    assert "shoebox" in reference
    params = dials_gpu.command_line.integrate.working_phil.extract()

    def fake_create_integrator(params, experiments, predicted):
        assert "shoebox" not in reference
        raise RuntimeError("Stop before integrating")

    with mock.patch.object(
        dials_gpu.command_line.integrate, "create_integrator_gpu", fake_create_integrator
    ):
        with pytest.raises(RuntimeError, match="Stop before integrating"):
            dials_gpu.command_line.integrate.run_integration(
                params, experiments, reference
            )
    assert "shoebox" not in reference


def test_input_data_not_pickled_with_params_gpu(dials_data, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = dials_data("centroid_test_data")
    sizes = {}

    def fake_run_integration(params, experiments, reference):
        sizes["profile"] = len(pickle.dumps(params.profile))
        sizes["integration"] = len(pickle.dumps(params.integration))
        raise RuntimeError("Stop before integrating")

    with mock.patch.object(
        dials_gpu.command_line.integrate, "run_integration", fake_run_integration
    ):
        with pytest.raises(SystemExit):
            dials_gpu.command_line.integrate.run(
                [str(data / "indexed.expt"), str(data / "indexed.refl")]
            )
    assert sizes["profile"] < 50_000
    assert sizes["integration"] < 50_000
