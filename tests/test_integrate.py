from __future__ import annotations

import copy
import pathlib
import shutil
import subprocess

import numpy as np
import pytest
import torch
from dxtbx.model.experiment_list import ExperimentListFactory

from dials.array_family import flex
from dials.command_line.index import phil_scope as index_phil_scope
from dials.command_line.integrate import (
    phil_scope as integrate_phil_scope,
    run_integration as cpu_run_integration,
)
from dials.util.phil import parse
from dials_gpu.command_line.find_spots import from_observations_gpu
from dials_gpu.command_line.index import index as index_experiments_gpu
from dials_gpu.command_line.integrate import run_integration as gpu_run_integration

TEST_I19_NXS = pathlib.Path(
    "/home/davidm/diffraction/test_data/i19/301_Nidppe_f3_phi0/301_NiD_00001_w00017.nxs"
)


@pytest.fixture(scope="module")
def i19_indexed_fixture():
    if not TEST_I19_NXS.exists():
        pytest.skip("i19 .nxs test dataset not present")
    expts = ExperimentListFactory.from_filenames([str(TEST_I19_NXS)])
    refl = from_observations_gpu(expts)
    idx_params = index_phil_scope.fetch(source=parse("indexing.method=fft3d")).extract()
    idx_expts, idx_refl = index_experiments_gpu(
        copy.deepcopy(expts), [copy.deepcopy(refl)], idx_params
    )
    return idx_expts, idx_refl


@pytest.mark.parametrize(
    "phil_str",
    [
        "integration.background.algorithm=glm\nintegration.background.glm.model.algorithm=constant3d\nintegration.mp.nproc=1",
        "integration.background.algorithm=glm\nintegration.background.glm.model.algorithm=constant2d\nintegration.mp.nproc=1",
        "integration.background.algorithm=simple\nintegration.background.simple.outlier.algorithm=null\nintegration.background.simple.model.algorithm=constant3d\nintegration.mp.nproc=1",
        "integration.background.algorithm=simple\nintegration.background.simple.outlier.algorithm=null\nintegration.background.simple.model.algorithm=constant2d\nintegration.mp.nproc=1",
        "integration.background.algorithm=simple\nintegration.background.simple.outlier.algorithm=null\nintegration.background.simple.model.algorithm=linear3d\nintegration.mp.nproc=1",
        "integration.background.algorithm=simple\nintegration.background.simple.outlier.algorithm=null\nintegration.background.simple.model.algorithm=linear2d\nintegration.mp.nproc=1",
        "integration.background.algorithm=simple\nintegration.background.simple.outlier.algorithm=nsigma\nintegration.background.simple.model.algorithm=constant3d\nintegration.mp.nproc=1",
        "integration.background.algorithm=simple\nintegration.background.simple.outlier.algorithm=tukey\nintegration.background.simple.model.algorithm=constant3d\nintegration.mp.nproc=1",
        "integration.background.algorithm=simple\nintegration.background.simple.outlier.algorithm=truncated\nintegration.background.simple.model.algorithm=constant3d\nintegration.mp.nproc=1",
        "integration.background.algorithm=null\nintegration.mp.nproc=1",
        "integration.profile.fitting=False\nintegration.mp.nproc=1",
        "integration.integrator=flat3d\nintegration.profile.fitting=False\nintegration.mp.nproc=1",
        "integration.integrator=2d\nintegration.profile.fitting=False\nintegration.mp.nproc=1",
        "integration.integrator=single2d\nintegration.profile.fitting=False\nintegration.mp.nproc=1",
        "profile.gaussian_rs.sigma_m_algorithm=basic\nintegration.mp.nproc=1",
        "profile.gaussian_rs.centroid_definition=com\nintegration.mp.nproc=1",
    ],
)
def test_i19_integrate_equivalence(i19_indexed_fixture, phil_str: str):
    assert torch.cuda.is_available(), "CUDA GPU is required for GPU tests"
    base_expts, base_refl = i19_indexed_fixture

    params_cpu = integrate_phil_scope.fetch(source=parse(phil_str)).extract()
    params_gpu = integrate_phil_scope.fetch(source=parse(phil_str)).extract()

    expts_cpu, refl_cpu, _ = cpu_run_integration(
        params_cpu, copy.deepcopy(base_expts), copy.deepcopy(base_refl)
    )
    expts_gpu, refl_gpu, _ = gpu_run_integration(
        params_gpu, copy.deepcopy(base_expts), copy.deepcopy(base_refl)
    )

    assert len(expts_gpu) == len(expts_cpu)
    assert len(refl_gpu) == len(refl_cpu)
    assert expts_gpu[0].profile.sigma_b() == pytest.approx(
        expts_cpu[0].profile.sigma_b(), rel=1e-6, abs=1e-8
    )
    assert expts_gpu[0].profile.sigma_m() == pytest.approx(
        expts_cpu[0].profile.sigma_m(), rel=1e-5, abs=1e-7
    )

    for col in [
        "intensity.sum.value",
        "intensity.sum.variance",
        "background.sum.value",
        "background.sum.variance",
        "background.mean",
        "lp",
        "partiality",
        "d",
        "zeta",
    ]:
        if col in refl_cpu and col in refl_gpu:
            tol = 2e-3 if "sum" in col else 1e-4
            np.testing.assert_allclose(
                refl_gpu[col].as_numpy_array(),
                refl_cpu[col].as_numpy_array(),
                rtol=tol,
                atol=tol,
                err_msg=f"Mismatch in column {col}",
            )

    if "intensity.prf.value" in refl_cpu and "intensity.prf.value" in refl_gpu:
        prf_sel = refl_cpu.get_flags(refl_cpu.flags.integrated_prf)
        assert list(refl_gpu.get_flags(refl_gpu.flags.integrated_prf)) == list(prf_sel)
        np.testing.assert_allclose(
            refl_gpu["intensity.prf.value"].select(prf_sel).as_numpy_array(),
            refl_cpu["intensity.prf.value"].select(prf_sel).as_numpy_array(),
            rtol=1e-3,
            atol=1e-3,
        )


def test_i19_cli_integrate_equivalence(i19_indexed_fixture, tmp_path):
    base_expts, base_refl = i19_indexed_fixture
    in_expt = tmp_path / "indexed.expt"
    in_refl = tmp_path / "indexed.refl"
    base_expts.as_file(str(in_expt))
    base_refl.as_file(str(in_refl))

    for cli_name in ("dials_gpu.integrate", "dials_gpu.interate"):
        cli_bin = shutil.which(cli_name) or f"/home/davidm/dials/conda_base/bin/{cli_name}"
        out_expt = tmp_path / f"{cli_name}.expt"
        out_refl = tmp_path / f"{cli_name}.refl"
        res = subprocess.run(
            [
                cli_bin,
                str(in_expt),
                str(in_refl),
                f"output.experiments={out_expt}",
                f"output.reflections={out_refl}",
                "integration.mp.nproc=1",
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0, f"{cli_name} failed:\n{res.stderr}\n{res.stdout}"
        assert out_expt.exists()
        assert out_refl.exists()
        r_table = flex.reflection_table.from_file(str(out_refl))
        assert len(r_table) > 100
        assert (
            r_table.get_flags(r_table.flags.integrated_sum).count(True) > 50
        )
