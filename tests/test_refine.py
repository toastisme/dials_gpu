from __future__ import annotations

import copy
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import torch
from dxtbx import flumpy
from dxtbx.model.experiment_list import ExperimentListFactory

from dials.algorithms.refinement import RefinerFactory
from dials.array_family import flex
from dials.command_line.index import phil_scope as index_phil_scope
from dials.command_line.refine import working_phil as refine_phil_scope
from dials.util.phil import parse
from dials_gpu.command_line.find_spots import from_observations_gpu
from dials_gpu.command_line.index import index as index_experiments_gpu
from dials_gpu.refinement import RefinerFactoryGPU

TEST_I19_NXS = pathlib.Path(
    "/home/davidm/diffraction/test_data/i19/301_Nidppe_f3_phi0/301_NiD_00001_w00017.nxs"
)


@pytest.fixture(scope="module")
def i19_indexed_fixture():
    if not TEST_I19_NXS.exists():
        pytest.skip("i19 .nxs test dataset not present")
    flex.set_random_seed(42)
    expts = ExperimentListFactory.from_filenames([str(TEST_I19_NXS)])
    refl = from_observations_gpu(expts)
    from dials_gpu.spot_finding.finder import _cleanup_spotfinding_caches
    from dials_gpu.spot_finding.threshold import (
        _TRITON_BATCH_SCRATCH_CACHE,
        _TRITON_SCRATCH_CACHE,
    )

    _cleanup_spotfinding_caches()
    _TRITON_SCRATCH_CACHE.clear()
    _TRITON_BATCH_SCRATCH_CACHE.clear()
    torch.cuda.empty_cache()

    idx_params = index_phil_scope.fetch(source=parse("indexing.method=fft3d")).extract()
    idx_expts, idx_refl = index_experiments_gpu(
        copy.deepcopy(expts), [copy.deepcopy(refl)], idx_params
    )
    return idx_expts, idx_refl


@pytest.mark.parametrize(
    "phil_str",
    [
        "refinement.parameterisation.scan_varying=False\nrefinement.reflections.outlier.algorithm=mcd",
        "refinement.parameterisation.scan_varying=True\nrefinement.reflections.outlier.algorithm=mcd",
        "refinement.parameterisation.scan_varying=False\nrefinement.reflections.outlier.algorithm=tukey",
        "refinement.parameterisation.scan_varying=False\nrefinement.reflections.outlier.algorithm=sauter_poon",
        "refinement.parameterisation.scan_varying=False\nrefinement.reflections.outlier.algorithm=null",
        "refinement.parameterisation.scan_varying=False\nrefinement.refinery.engine=GaussNewton\nrefinement.reflections.outlier.algorithm=null",
        "refinement.parameterisation.scan_varying=False\nrefinement.refinery.engine=SimpleLBFGS\nrefinement.reflections.outlier.algorithm=null\nrefinement.mp.nproc=1",
        "refinement.parameterisation.scan_varying=False\nrefinement.refinery.engine=LBFGScurvs\nrefinement.reflections.outlier.algorithm=null\nrefinement.mp.nproc=1",
        "refinement.parameterisation.scan_varying=False\nrefinement.refinery.engine=SparseLevMar\nrefinement.reflections.outlier.algorithm=null",
    ],
)
def test_i19_refiner_equivalence(i19_indexed_fixture, phil_str: str):
    assert torch.cuda.is_available(), "CUDA GPU is required for GPU tests"
    base_expts, base_refl = i19_indexed_fixture

    expts_cpu = copy.deepcopy(base_expts)
    refl_cpu = copy.deepcopy(base_refl)
    params_cpu = refine_phil_scope.fetch(source=parse(phil_str)).extract()
    flex.set_random_seed(42)
    refiner_cpu = RefinerFactory.from_parameters_data_experiments(
        params_cpu, refl_cpu, expts_cpu
    )
    refiner_cpu.run()

    expts_gpu = copy.deepcopy(base_expts)
    refl_gpu = copy.deepcopy(base_refl)
    params_gpu = refine_phil_scope.fetch(source=parse(phil_str)).extract()
    flex.set_random_seed(42)
    refiner_gpu = RefinerFactoryGPU.from_parameters_data_experiments(
        params_gpu, refl_gpu, expts_gpu
    )
    refiner_gpu.run()

    if "SimpleLBFGS" in phil_str:
        rel_tol, abs_tol = 4e-2, 1e-3
    elif "LBFGScurvs" in phil_str:
        rel_tol, abs_tol = 2e-5, 1e-6
    else:
        rel_tol, abs_tol = 1e-5, 1e-7
    for r_cpu, r_gpu in zip(refiner_cpu.rmsds(), refiner_gpu.rmsds()):
        assert r_gpu == pytest.approx(r_cpu, rel=rel_tol, abs=abs_tol)

    uc_cpu = refiner_cpu.get_experiments()[0].crystal.get_unit_cell().parameters()
    uc_gpu = refiner_gpu.get_experiments()[0].crystal.get_unit_cell().parameters()
    for p_cpu, p_gpu in zip(uc_cpu, uc_gpu):
        assert p_gpu == pytest.approx(p_cpu, rel=rel_tol, abs=abs_tol)


def test_i19_cli_refine_equivalence(i19_indexed_fixture, tmp_path):
    assert torch.cuda.is_available(), "CUDA GPU is required for GPU tests"
    base_expts, base_refl = i19_indexed_fixture

    expt_in = tmp_path / "indexed.expt"
    refl_in = tmp_path / "indexed.refl"
    base_expts.as_file(expt_in)
    base_refl.as_file(refl_in)

    bin_dir = pathlib.Path(sys.executable).parent
    cpu_cmd = [
        str(bin_dir / "dials.refine"),
        str(expt_in),
        str(refl_in),
        "output.experiments=cpu_refined.expt",
        "output.reflections=cpu_refined.refl",
        "output.log=cpu_refine.log",
    ]
    gpu_cmd = [
        str(bin_dir / "dials_gpu.refine"),
        str(expt_in),
        str(refl_in),
        "output.experiments=gpu_refined.expt",
        "output.reflections=gpu_refined.refl",
        "output.log=gpu_refine.log",
    ]

    res_cpu = subprocess.run(cpu_cmd, cwd=tmp_path, capture_output=True, text=True)
    assert res_cpu.returncode == 0, res_cpu.stderr

    res_gpu = subprocess.run(gpu_cmd, cwd=tmp_path, capture_output=True, text=True)
    assert res_gpu.returncode == 0, res_gpu.stderr

    expts_cpu = ExperimentListFactory.from_json_file(
        tmp_path / "cpu_refined.expt", check_format=False
    )
    expts_gpu = ExperimentListFactory.from_json_file(
        tmp_path / "gpu_refined.expt", check_format=False
    )
    assert expts_gpu[0].crystal.is_similar_to(expts_cpu[0].crystal)
    assert expts_gpu[0].beam.is_similar_to(expts_cpu[0].beam)
    assert expts_gpu[0].detector.is_similar_to(expts_cpu[0].detector)

    refl_cpu = flex.reflection_table.from_file(tmp_path / "cpu_refined.refl")
    refl_gpu = flex.reflection_table.from_file(tmp_path / "gpu_refined.refl")
    assert len(refl_gpu) == len(refl_cpu)
    xyzcal_cpu = flumpy.to_numpy(refl_cpu["xyzcal.mm"])
    xyzcal_gpu = flumpy.to_numpy(refl_gpu["xyzcal.mm"])
    np.testing.assert_allclose(xyzcal_gpu, xyzcal_cpu, rtol=1e-5, atol=1e-6)
