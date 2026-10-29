from __future__ import annotations

import pathlib
import numpy as np
import pytest
import torch
from dxtbx import flumpy
from dxtbx.model.experiment_list import ExperimentListFactory

from dials.algorithms.spot_finding.factory import SpotFinderFactory
from dials.array_family import flex
from dials.command_line.find_spots import do_spotfinding as do_spotfinding_cpu, phil_scope
from dials.util.phil import parse
from dials_gpu.command_line.find_spots import do_spotfinding as do_spotfinding_gpu, from_observations_gpu

TEST_CBF = pathlib.Path("/home/davidm/dials_gpu/test_data/C2sum_1_0001.cbf.gz")
TEST_I19_NXS = pathlib.Path(
    "/home/davidm/diffraction/test_data/i19/301_Nidppe_f3_phi0/301_NiD_00001_w00017.nxs"
)


def _sort_reflection_table(refl: flex.reflection_table) -> flex.reflection_table:
    if len(refl) <= 1:
        return refl
    xyz = flumpy.to_numpy(refl["xyzobs.px.value"])
    order = np.lexsort((xyz[:, 0], xyz[:, 1], xyz[:, 2])).astype(np.uint64)
    return refl.select(flumpy.from_numpy(np.ascontiguousarray(order)))


def _assert_reflection_tables_equal(cpu_refl: flex.reflection_table, gpu_refl: flex.reflection_table):
    assert len(cpu_refl) == len(gpu_refl)
    assert set(cpu_refl.keys()) == set(gpu_refl.keys())
    cpu_refl = _sort_reflection_table(cpu_refl)
    gpu_refl = _sort_reflection_table(gpu_refl)

    for k in sorted(cpu_refl.keys()):
        if k == "shoebox":
            s_cpu = cpu_refl["shoebox"]
            s_gpu = gpu_refl["shoebox"]
            for i in range(len(s_cpu)):
                assert np.array_equal(
                    flumpy.to_numpy(s_cpu[i].data), flumpy.to_numpy(s_gpu[i].data)
                )
                assert np.array_equal(
                    flumpy.to_numpy(s_cpu[i].mask), flumpy.to_numpy(s_gpu[i].mask)
                )
                assert np.array_equal(
                    flumpy.to_numpy(s_cpu[i].background),
                    flumpy.to_numpy(s_gpu[i].background),
                )
        elif k == "bbox":
            for p in range(6):
                assert np.array_equal(
                    flumpy.to_numpy(cpu_refl["bbox"].parts()[p]),
                    flumpy.to_numpy(gpu_refl["bbox"].parts()[p]),
                )
        else:
            a = flumpy.to_numpy(cpu_refl[k])
            b = flumpy.to_numpy(gpu_refl[k])
            if np.issubdtype(a.dtype, np.floating):
                assert np.allclose(a, b, rtol=1e-10, atol=1e-10)
            else:
                assert np.array_equal(a, b)


@pytest.mark.parametrize(
    "phil_str",
    [
        "",
        "spotfinder.threshold.algorithm=dispersion",
        "spotfinder.threshold.dispersion.gain=1.5\nspotfinder.threshold.dispersion.global_threshold=Auto",
        "spotfinder.threshold.algorithm=radial_profile\nspotfinder.threshold.radial_profile.blur=narrow",
        "spotfinder.region_of_interest=400,2100,400,2100\nspotfinder.filter.d_min=2.5\nspotfinder.filter.d_max=20.0",
        "spotfinder.filter.max_separation=1.5\nspotfinder.filter.min_spot_size=5",
        "output.shoeboxes=False\nspotfinder.force_2d=True",
    ],
)
def test_find_spots_equivalence(phil_str: str, tmp_path: pathlib.Path):
    assert torch.cuda.is_available(), "CUDA GPU is required for GPU tests"
    expts_cpu = ExperimentListFactory.from_filenames([str(TEST_CBF)])
    expts_gpu = ExperimentListFactory.from_filenames([str(TEST_CBF)])

    params_cpu = phil_scope.fetch(source=parse(phil_str)).extract()
    params_gpu = phil_scope.fetch(source=parse(phil_str)).extract()
    params_cpu.output.reflections = str(tmp_path / "cpu.refl")
    params_gpu.output.reflections = str(tmp_path / "gpu.refl")

    cpu_refl = do_spotfinding_cpu(expts_cpu, params_cpu)
    gpu_refl = do_spotfinding_gpu(expts_gpu, params_gpu)

    _assert_reflection_tables_equal(cpu_refl, gpu_refl)


@pytest.mark.skipif(not TEST_I19_NXS.exists(), reason="i19 .nxs test dataset not present")
def test_i19_nxs_batch_volume_equivalence(tmp_path: pathlib.Path):
    assert torch.cuda.is_available(), "CUDA GPU is required for GPU tests"
    expts_cpu = ExperimentListFactory.from_filenames([str(TEST_I19_NXS)])
    expts_gpu = ExperimentListFactory.from_filenames([str(TEST_I19_NXS)])

    for idx, phil_str in enumerate(
        [
            "spotfinder.mp.nproc=1",
            "spotfinder.mp.nproc=1\nspotfinder.threshold.dispersion.sigma_background=5.0\nspotfinder.threshold.dispersion.sigma_strong=2.5\nspotfinder.threshold.dispersion.kernel_size=4,4",
        ]
    ):
        params_cpu = phil_scope.fetch(source=parse(phil_str)).extract()
        params_gpu = phil_scope.fetch(source=parse(phil_str)).extract()
        params_cpu.output.reflections = str(tmp_path / f"cpu_{idx}.refl")
        params_gpu.output.reflections = str(tmp_path / f"gpu_{idx}.refl")

        cpu_refl = do_spotfinding_cpu(expts_cpu, params_cpu)
        gpu_refl = do_spotfinding_gpu(expts_gpu, params_gpu)

        _assert_reflection_tables_equal(cpu_refl, gpu_refl)
