from __future__ import annotations

import copy
import pathlib

import numpy as np
import pytest
import torch
from cctbx import sgtbx, uctbx
from dxtbx import flumpy
from dxtbx.model.experiment_list import ExperimentListFactory

from dials.array_family import flex
from dials.command_line.index import index as index_experiments_cpu, phil_scope
from dials.util.phil import parse
from dials_gpu.command_line.find_spots import from_observations_gpu
from dials_gpu.command_line.index import index as index_experiments_gpu

TEST_I19_NXS = pathlib.Path(
    "/home/davidm/diffraction/test_data/i19/301_Nidppe_f3_phi0/301_NiD_00001_w00017.nxs"
)


@pytest.fixture(scope="module")
def i19_spotfinding_fixture():
    if not TEST_I19_NXS.exists():
        pytest.skip("i19 .nxs test dataset not present")
    expts = ExperimentListFactory.from_filenames([str(TEST_I19_NXS)])
    refl = from_observations_gpu(expts)
    return expts, refl


@pytest.mark.parametrize(
    "phil_str",
    [
        "indexing.method=fft3d",
        "indexing.method=fft1d",
        "indexing.method=real_space_grid_search\nindexing.known_symmetry.unit_cell=11.39,13.30,15.82,90,99.02,90\nindexing.known_symmetry.space_group=P21/n",
        "indexing.method=fft3d\nindexing.index_assignment.method=local",
    ],
)
def test_i19_index_equivalence(i19_spotfinding_fixture, phil_str: str):
    assert torch.cuda.is_available(), "CUDA GPU is required for GPU tests"
    base_expts, base_refl = i19_spotfinding_fixture

    expts_cpu = copy.deepcopy(base_expts)
    refl_cpu = copy.deepcopy(base_refl)
    params_cpu = phil_scope.fetch(source=parse(phil_str)).extract()

    expts_gpu = copy.deepcopy(base_expts)
    refl_gpu = copy.deepcopy(base_refl)
    params_gpu = phil_scope.fetch(source=parse(phil_str)).extract()

    idx_expts_cpu, idx_refl_cpu = index_experiments_cpu(
        expts_cpu, [refl_cpu], params_cpu
    )
    idx_expts_gpu, idx_refl_gpu = index_experiments_gpu(
        expts_gpu, [refl_gpu], params_gpu
    )

    assert len(idx_expts_cpu) == len(idx_expts_gpu)
    for c_cpu, c_gpu in zip(idx_expts_cpu.crystals(), idx_expts_gpu.crystals()):
        assert c_gpu.get_unit_cell().parameters() == pytest.approx(
            c_cpu.get_unit_cell().parameters(), rel=1e-4, abs=1e-3
        )
        assert (
            c_gpu.get_space_group().type().hall_symbol()
            == c_cpu.get_space_group().type().hall_symbol()
        )

    n_indexed_cpu = idx_refl_cpu.get_flags(idx_refl_cpu.flags.indexed).count(True)
    n_indexed_gpu = idx_refl_gpu.get_flags(idx_refl_gpu.flags.indexed).count(True)
    assert n_indexed_gpu == n_indexed_cpu
    assert np.array_equal(
        flumpy.to_numpy(idx_refl_gpu["miller_index"]),
        flumpy.to_numpy(idx_refl_cpu["miller_index"]),
    )
