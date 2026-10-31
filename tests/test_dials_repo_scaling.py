"""
Port of the DIALS scaling test suite (/home/davidm/diffraction/dials/modules/dials/tests/algorithms/scaling/)
running against dials_gpu.scaling and dials_gpu.scale.
"""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import subprocess
from math import exp, pi, sqrt
from unittest.mock import MagicMock, Mock, patch

import iotbx.merging_statistics
import iotbx.mtz
import numpy as np
import pytest
import torch
from cctbx import sgtbx, uctbx
from cctbx.sgtbx import space_group
from dxtbx import flumpy
from dxtbx.model import (
    Beam,
    Crystal,
    Detector,
    Experiment,
    ExperimentList,
    Goniometer,
    GoniometerFactory,
    Scan,
)
from dxtbx.serialize import load
from libtbx import phil
from scitbx import sparse

from dials.algorithms.scaling.algorithm import prepare_input
from dials.algorithms.scaling.active_parameter_managers import (
    multi_active_parameter_manager,
)
from dials.algorithms.scaling.basis_functions import RefinerCalculator
from dials.algorithms.scaling.combine_intensities import combine_intensities
from dials.algorithms.scaling.error_model.error_model import (
    BasicErrorModel,
    ErrorModelA_APM,
    ErrorModelB_APM,
)
from dials.algorithms.scaling.error_model.error_model_target import (
    ErrorModelTargetA,
    ErrorModelTargetB,
)
from dials.algorithms.scaling.Ih_table import IhTable, IhTableBlock, map_indices_to_asu
from dials.algorithms.scaling.model.components.scale_components import (
    SHScaleComponent,
    SingleBScaleFactor,
    SingleScaleFactor,
)
from dials.algorithms.scaling.model.components.smooth_scale_components import (
    GaussianSmoother1D,
    GaussianSmoother2D,
    GaussianSmoother3D,
    SmoothBScaleComponent1D,
    SmoothScaleComponent1D,
    SmoothScaleComponent2D,
    SmoothScaleComponent3D,
)
from dials.algorithms.scaling.model.model import PhysicalScalingModel
from dials.algorithms.scaling.outlier_rejection import (
    determine_Esq_outlier_index_arrays,
)
from dials.algorithms.scaling.parameter_handler import scaling_active_parameter_manager
from dials.algorithms.scaling.scaling_library import create_scaling_model
from dials.algorithms.scaling.scaling_utilities import (
    calculate_prescaling_correction,
    quasi_normalisation,
    set_wilson_outliers,
)
from dials.algorithms.scaling.target_function import ScalingTarget
from dials.array_family import flex
from dials.command_line import merge, report
from dials.util.options import ArgumentParser
from dials_scaling_ext import create_sph_harm_table

import dials_gpu.command_line.scale as gpu_scale
from dials_gpu.scaling import (
    ErrorModelRefineryGPU,
    ErrorModelTargetAGPU,
    ErrorModelTargetBGPU,
    ErrorModelTargetRegressionGPU,
    GaussianSmoother1DGPU,
    GaussianSmoother2DGPU,
    GaussianSmoother3DGPU,
    MultiDatasetIntensityCombinerGPU,
    MultiScalerGPU,
    NormDevOutlierRejectionGPU,
    RefinerCalculatorGPU,
    ScaleAndFilterAlgorithmGPU,
    ScalingAlgorithmGPU,
    ScalingTargetFixedIHGPU,
    ScalingTargetGPU,
    SimpleNormDevOutlierRejectionGPU,
    SingleDatasetIntensityCombinerGPU,
    SingleScalerGPU,
    TargetedOutlierRejectionGPU,
    TargetScalerGPU,
    align_axis_along_z_gpu,
    calc_crystal_frame_vectors_gpu,
    calc_Ih_gpu,
    calc_lookup_index_gpu,
    calc_sf_variances_gpu,
    calc_theta_phi_gpu,
    combine_intensities_gpu,
    create_scaler_gpu,
    determine_outlier_index_arrays_gpu,
    evaluate_component_gpu,
    reject_outliers_gpu,
    run_error_model_refinement_gpu,
)


def _get_dials_gpu_scale_bin() -> str:
    return (
        shutil.which("dials_gpu.scale")
        or "/home/davidm/dials/conda_base/bin/dials_gpu.scale"
    )


def run_one_scaling_gpu(working_directory, argument_list):
    """Run the dials_gpu.scale algorithm via CLI."""
    command = [_get_dials_gpu_scale_bin()] + [str(a) for a in argument_list]
    result = subprocess.run(command, cwd=working_directory, capture_output=True)
    assert not result.returncode and not result.stderr, result.stderr.decode()
    assert (working_directory / "scaled.expt").is_file()
    assert (working_directory / "scaled.refl").is_file()
    assert (working_directory / "dials.scale.html").is_file()

    table = flex.reflection_table.from_file(working_directory / "scaled.refl")
    assert "inverse_scale_factor" in table
    assert "inverse_scale_factor_variance" in table


def get_merging_stats(
    scaled_unmerged_mtz,
    anomalous=False,
    n_bins=20,
    use_internal_variance=False,
    eliminate_sys_absent=False,
    data_labels=None,
):
    """Return a merging statistics result from an mtz file."""
    i_obs = iotbx.merging_statistics.select_data(
        str(scaled_unmerged_mtz), data_labels=data_labels
    )
    i_obs = i_obs.customized_copy(anomalous_flag=False, info=i_obs.info())
    result = iotbx.merging_statistics.dataset_statistics(
        i_obs=i_obs,
        n_bins=n_bins,
        anomalous=anomalous,
        use_internal_variance=use_internal_variance,
        eliminate_sys_absent=eliminate_sys_absent,
    )
    return result


# ==============================================================================
# 1. Port of test_gaussian_smoother.py
# ==============================================================================


@pytest.mark.parametrize(
    "n_params",
    [2, 3, 5],
)
def test_1D_smoother_gpu(n_params):
    """Test the 1D smoother GPU implementation against GaussianSmoother1D."""
    assert n_params > 1
    if n_params == 2:
        n_intervals = 1
    elif n_params == 3:
        n_intervals = 2
    else:
        n_intervals = n_params - 2

    smoother_cpu = GaussianSmoother1D([0.0, 10.0], n_intervals)
    smoother_gpu = GaussianSmoother1DGPU([0.0, 10.0], n_intervals)
    assert list(smoother_gpu.positions()) == pytest.approx(list(smoother_cpu.positions()))
    assert smoother_gpu.spacing() == pytest.approx(smoother_cpu.spacing())

    params = flex.double([1.0 + 0.3 * i for i in range(n_params)])
    v1_cpu, w1_cpu, sw1_cpu = smoother_cpu.value_weight(2.5, params)
    v1_gpu, w1_gpu, sw1_gpu = smoother_gpu.value_weight(2.5, params)
    assert v1_gpu == pytest.approx(v1_cpu)
    assert sw1_gpu == pytest.approx(sw1_cpu)
    assert list(w1_gpu) == pytest.approx(list(w1_cpu))

    x_pts = flex.double([0.5, 2.5, 5.0, 7.5, 9.5])
    vals_cpu, weights_cpu, sumweights_cpu = smoother_cpu.multi_value_weight(x_pts, params)
    vals_gpu, weights_gpu, sumweights_gpu = smoother_gpu.multi_value_weight(x_pts, params)
    assert list(vals_gpu) == pytest.approx(list(vals_cpu))
    assert list(sumweights_gpu) == pytest.approx(list(sumweights_cpu))
    np.testing.assert_allclose(
        weights_gpu.as_dense_matrix().as_numpy_array(),
        weights_cpu.as_dense_matrix().as_numpy_array(),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "nx_params,ny_params",
    [(2, 2), (2, 3), (2, 5), (5, 2), (5, 3), (3, 3), (5, 5)],
)
def test_2D_smoother_gpu(nx_params, ny_params):
    """Test the 2D smoother GPU implementation against GaussianSmoother2D."""
    nx_intervals = 1 if nx_params == 2 else (2 if nx_params == 3 else nx_params - 2)
    ny_intervals = 1 if ny_params == 2 else (2 if ny_params == 3 else ny_params - 2)

    smoother_cpu = GaussianSmoother2D([0.0, 10.0], nx_intervals, [0.0, 10.0], ny_intervals)
    smoother_gpu = GaussianSmoother2DGPU([0.0, 10.0], nx_intervals, [0.0, 10.0], ny_intervals)
    assert list(smoother_gpu.x_positions()) == pytest.approx(list(smoother_cpu.x_positions()))
    assert list(smoother_gpu.y_positions()) == pytest.approx(list(smoother_cpu.y_positions()))

    params = flex.double([1.0 + 0.1 * i for i in range(nx_params * ny_params)])
    x_pts = flex.double([0.5, 2.5, 5.0, 7.5, 9.5])
    y_pts = flex.double([1.0, 3.0, 5.0, 7.0, 9.0])
    vals_cpu, weights_cpu, sumweights_cpu = smoother_cpu.multi_value_weight(x_pts, y_pts, params)
    vals_gpu, weights_gpu, sumweights_gpu = smoother_gpu.multi_value_weight(x_pts, y_pts, params)
    assert list(vals_gpu) == pytest.approx(list(vals_cpu))
    assert list(sumweights_gpu) == pytest.approx(list(sumweights_cpu))
    np.testing.assert_allclose(
        weights_gpu.as_dense_matrix().as_numpy_array(),
        weights_cpu.as_dense_matrix().as_numpy_array(),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "nx_params,ny_params,nz_params",
    [(2, 2, 2), (2, 3, 5), (5, 2, 3), (3, 3, 3), (5, 5, 5)],
)
def test_3D_smoother_gpu(nx_params, ny_params, nz_params):
    """Test the 3D smoother GPU implementation against GaussianSmoother3D."""
    nx_intervals = 1 if nx_params == 2 else (2 if nx_params == 3 else nx_params - 2)
    ny_intervals = 1 if ny_params == 2 else (2 if ny_params == 3 else ny_params - 2)
    nz_intervals = 1 if nz_params == 2 else (2 if nz_params == 3 else nz_params - 2)

    smoother_cpu = GaussianSmoother3D(
        [0.0, 10.0], nx_intervals, [0.0, 10.0], ny_intervals, [0.0, 10.0], nz_intervals
    )
    smoother_gpu = GaussianSmoother3DGPU(
        [0.0, 10.0], nx_intervals, [0.0, 10.0], ny_intervals, [0.0, 10.0], nz_intervals
    )
    params = flex.double([1.0 + 0.05 * i for i in range(nx_params * ny_params * nz_params)])
    x_pts = flex.double([0.5, 2.5, 5.0, 7.5, 9.5])
    y_pts = flex.double([1.0, 3.0, 5.0, 7.0, 9.0])
    z_pts = flex.double([1.5, 3.5, 5.5, 7.5, 8.5])
    vals_cpu, weights_cpu, sumweights_cpu = smoother_cpu.multi_value_weight(
        x_pts, y_pts, z_pts, params
    )
    vals_gpu, weights_gpu, sumweights_gpu = smoother_gpu.multi_value_weight(
        x_pts, y_pts, z_pts, params
    )
    assert list(vals_gpu) == pytest.approx(list(vals_cpu))
    assert list(sumweights_gpu) == pytest.approx(list(sumweights_cpu))
    np.testing.assert_allclose(
        weights_gpu.as_dense_matrix().as_numpy_array(),
        weights_cpu.as_dense_matrix().as_numpy_array(),
        rtol=1e-12,
        atol=1e-12,
    )


# ==============================================================================
# 2. Port of test_scale_components.py & test_basis_function.py
# ==============================================================================


def test_scale_components_gpu():
    """Port of test_scale_components.py running on GPU via evaluate_component_gpu."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. SingleScaleFactor
    KSF = SingleScaleFactor(flex.double([2.0]))
    rt = flex.reflection_table()
    rt["d"] = flex.double([1.0, 1.0])
    rt["id"] = flex.int([0, 0])
    KSF.data = {"id": rt["id"]}
    KSF.update_reflection_data()
    s_gpu, d_gpu = evaluate_component_gpu(KSF, block_id=0, compute_derivatives=True, device=device)
    assert s_gpu.cpu().numpy().tolist() == pytest.approx([2.0, 2.0])
    assert d_gpu.cpu().numpy().ravel().tolist() == pytest.approx([1.0, 1.0])

    # 2. SingleBScaleFactor
    BSF = SingleBScaleFactor(flex.double([0.0]))
    BSF.data = {"d": rt["d"]}
    BSF.update_reflection_data()
    s_gpu, d_gpu = evaluate_component_gpu(BSF, block_id=0, compute_derivatives=True, device=device)
    assert s_gpu.cpu().numpy().tolist() == pytest.approx([1.0, 1.0])
    assert d_gpu.cpu().numpy().ravel().tolist() == pytest.approx([0.5, 0.5])

    # 3. SHScaleComponent
    initial_param = 0.1
    initial_val = 0.2
    SF_sh = SHScaleComponent(flex.double([initial_param] * 3))
    harmonic_values = sparse.matrix(3, 1)
    harmonic_values[0, 0] = initial_val
    harmonic_values[1, 0] = initial_val
    harmonic_values[2, 0] = initial_val
    SF_sh.data = {"sph_harm_table": harmonic_values}
    SF_sh.update_reflection_data()
    s_gpu, d_gpu = evaluate_component_gpu(SF_sh, block_id=0, compute_derivatives=True, device=device)
    assert s_gpu.cpu().numpy().tolist() == pytest.approx([1.0 + 3.0 * initial_val * initial_param])
    assert d_gpu.cpu().numpy().ravel().tolist() == pytest.approx([initial_val] * 3)

    # 4. SmoothScaleComponent1D
    SF_1d = SmoothScaleComponent1D(flex.double(5, 1.1))
    SF_1d.data = {"x": flex.double([0.5, 1.0, 2.5, 0.0])}
    SF_1d.update_reflection_data()
    SF_1d.smoother.set_smoothing(4, 1.0)
    s_cpu, d_cpu = SF_1d.calculate_scales_and_derivatives()
    s_gpu, d_gpu = evaluate_component_gpu(SF_1d, block_id=0, compute_derivatives=True, device=device)
    assert s_gpu.cpu().numpy().tolist() == pytest.approx(list(s_cpu))
    np.testing.assert_allclose(
        d_gpu.cpu().numpy(), d_cpu.as_dense_matrix().as_numpy_array(), rtol=1e-12, atol=1e-12
    )

    # 5. SmoothBScaleComponent1D
    SF_b1d = SmoothBScaleComponent1D(flex.double(5, 0.0))
    SF_b1d.data = {
        "x": flex.double([0.5, 1.0, 2.5, 0.0]),
        "d": flex.double([1.0, 1.0, 1.0, 1.0]),
    }
    SF_b1d.update_reflection_data()
    SF_b1d.smoother.set_smoothing(4, 1.0)
    s_cpu, d_cpu = SF_b1d.calculate_scales_and_derivatives()
    s_gpu, d_gpu = evaluate_component_gpu(SF_b1d, block_id=0, compute_derivatives=True, device=device)
    assert s_gpu.cpu().numpy().tolist() == pytest.approx(list(s_cpu))
    np.testing.assert_allclose(
        d_gpu.cpu().numpy(), d_cpu.as_dense_matrix().as_numpy_array(), rtol=1e-12, atol=1e-12
    )

    # 6. SmoothScaleComponent2D
    SF_2d = SmoothScaleComponent2D(flex.double(30, 1.1), shape=(6, 5))
    norm_rot = flex.double(30, 0.5)
    norm_time = flex.double(30, 0.5)
    norm_rot[0] = 0.0
    norm_time[0] = 0.0
    norm_rot[29] = 3.99
    norm_time[29] = 2.99
    SF_2d.data = {"x": norm_rot, "y": norm_time}
    SF_2d.update_reflection_data()
    SF_2d.smoother.set_smoothing(4, 1.0)
    s_cpu, d_cpu = SF_2d.calculate_scales_and_derivatives()
    s_gpu, d_gpu = evaluate_component_gpu(SF_2d, block_id=0, compute_derivatives=True, device=device)
    assert s_gpu.cpu().numpy().tolist() == pytest.approx(list(s_cpu))
    np.testing.assert_allclose(
        d_gpu.cpu().numpy(), d_cpu.as_dense_matrix().as_numpy_array(), rtol=1e-12, atol=1e-12
    )

    # 7. SmoothScaleComponent3D
    SF_3d = SmoothScaleComponent3D(flex.double(150, 1.1), shape=(6, 5, 5))
    norm_rot = flex.double(150, 0.5)
    norm_time = flex.double(150, 0.5)
    norm_z = flex.double(150, 0.5)
    norm_rot[0] = 0.0
    norm_time[0] = 0.0
    norm_z[0] = 0.0
    norm_rot[149] = 3.99
    norm_time[149] = 2.99
    norm_z[149] = 2.99
    SF_3d.data = {"x": norm_rot, "y": norm_time, "z": norm_z}
    SF_3d.update_reflection_data()
    SF_3d.smoother.set_smoothing(3, 1.0)
    s_cpu, d_cpu = SF_3d.calculate_scales_and_derivatives()
    s_gpu, d_gpu = evaluate_component_gpu(SF_3d, block_id=0, compute_derivatives=True, device=device)
    assert s_gpu.cpu().numpy().tolist() == pytest.approx(list(s_cpu))
    np.testing.assert_allclose(
        d_gpu.cpu().numpy(), d_cpu.as_dense_matrix().as_numpy_array(), rtol=1e-12, atol=1e-12
    )


def test_RefinerCalculator_gpu():
    """Port of test_basis_function.py::test_RefinerCalculator."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rt = flex.reflection_table()
    rt["d"] = flex.double([2.0, 0.8, 2.0])
    rt["xyzobs.px.value"] = flex.vec3_double([(0.0, 0.0, 0.0), (0.0, 0.0, 5.0), (0.0, 0.0, 10.0)])
    rt["id"] = flex.int(3, 0)

    components = {
        "scale": SingleScaleFactor(flex.double([1.0])),
        "decay": SingleBScaleFactor(flex.double([0.0])),
        "abs": SingleScaleFactor(flex.double([1.0])),
    }
    components["scale"].data = {"id": rt["id"]}
    components["decay"].data = {"d": rt["d"]}
    components["abs"].data = {"id": rt["id"]}
    for component in components.values():
        component.update_reflection_data()

    apm = scaling_active_parameter_manager(components, ["decay", "scale"])
    apm.set_param_vals(flex.double([2.0, 1.0]))

    s_cpu, d_cpu = RefinerCalculator.calculate_scales_and_derivatives(apm, 0)
    s_gpu, d_gpu = RefinerCalculatorGPU.calculate_scales_and_derivatives_gpu(
        apm, 0, device=device
    )
    assert s_gpu.cpu().numpy().tolist() == pytest.approx(list(s_cpu))
    np.testing.assert_allclose(
        d_gpu.cpu().numpy(), d_cpu.as_dense_matrix().as_numpy_array(), rtol=1e-12, atol=1e-12
    )

    # Single active component
    components["decay"].parameters = flex.double([0.0])
    components["scale"].parameters = flex.double([1.0])
    components["abs"].parameters = flex.double([1.0])
    for c in components.values():
        c.calculate_scales_and_derivatives()

    apm = scaling_active_parameter_manager(components, ["scale"])
    apm.set_param_vals(flex.double(components["scale"].n_params, 2.0))
    s_cpu, d_cpu = RefinerCalculator.calculate_scales_and_derivatives(apm, 0)
    s_gpu, d_gpu = RefinerCalculatorGPU.calculate_scales_and_derivatives_gpu(
        apm, 0, device=device
    )
    assert s_gpu.cpu().numpy().tolist() == pytest.approx(list(s_cpu))
    np.testing.assert_allclose(
        d_gpu.cpu().numpy(), d_cpu.as_dense_matrix().as_numpy_array(), rtol=1e-12, atol=1e-12
    )


# ==============================================================================
# 3. Port of test_Ih_table.py & test_target_function.py
# ==============================================================================


def test_Ih_table_and_target_function_methods_gpu():
    """Port of test_Ih_table.py and test_target_function.py."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sg = space_group("C 2y")

    reflections = flex.reflection_table()
    reflections["intensity"] = flex.double([100.0, 100.0, 80.0, 60.0, 30.0, 40.0, 60.0])
    reflections["inverse_scale_factor"] = flex.double(7, 2.0)
    reflections["variance"] = flex.double([90.0, 100.0, 90.0, 60.0, 30.0, 50.0, 50.0])
    reflections["miller_index"] = flex.miller_index(
        [(1, 0, 0), (0, 0, 1), (-1, 0, 0), (0, 2, 0), (0, 4, 0), (0, 0, -2), (0, 0, 2)]
    )
    reflections.set_flags(flex.bool(7, True), reflections.flags.integrated)

    table = IhTable([reflections], sg, nblocks=1)
    block = table.blocked_data_list[0]
    block.calc_Ih()
    expected_ih = list(block.Ih_values)
    Ih_gpu, _, _ = calc_Ih_gpu(block, device=device)
    assert Ih_gpu.cpu().numpy().tolist() == pytest.approx(expected_ih)

    # Finite difference gradient and Jacobian checks from test_target_function.py
    crystal = Crystal.from_dict(
        {
            "__id__": "crystal",
            "real_space_a": [1.0, 0.0, 0.0],
            "real_space_b": [0.0, 1.0, 0.0],
            "real_space_c": [0.0, 0.0, 2.0],
            "space_group_hall_symbol": " C 2y",
        }
    )
    scan = Scan(image_range=[0, 60], oscillation=[0.0, 1.0])
    single_exp = Experiment(scan=scan, crystal=crystal)

    phil_scope = phil.parse(
        """
      include scope dials.algorithms.scaling.model.model.model_phil_scope
      include scope dials.algorithms.scaling.scaling_options.phil_scope
  """,
        process_includes=True,
    )
    parser = ArgumentParser(phil=phil_scope, check_format=False)
    physical_param, _ = parser.parse_args(args=[], quick_parse=True, show_diff_phil=False)
    physical_param.model = "physical"
    physical_param.physical.absorption_correction = False

    small_rt = flex.reflection_table()
    small_rt["intensity"] = flex.double([75.0, 10.0, 100.0, 50.0, 65.0])
    small_rt["variance"] = flex.double([50.0, 10.0, 100.0, 50.0, 65.0])
    small_rt["inverse_scale_factor"] = flex.double(5, 1.0)
    small_rt["miller_index"] = flex.miller_index(
        [(1, 0, 0), (0, 0, 1), (1, 0, 0), (0, 0, 1), (0, 0, 2)]
    )
    small_rt["d"] = flex.double([2.0, 0.8, 2.0, 0.8, 1.5])
    small_rt["xyzobs.px.value"] = flex.vec3_double(
        [(0.0, 0.0, i) for i in [0.0, 5.0, 10.0, 2.0, 8.0]]
    )
    small_rt["s1"] = flex.vec3_double([(0.0, 0.1, 1.0)] * 2 + [(0.0, 0.1, 20.0)] * 3)

    model = PhysicalScalingModel.from_data(physical_param, single_exp, small_rt)
    model.configure_components(small_rt, single_exp, physical_param)
    model.components["scale"].update_reflection_data()
    model.components["decay"].update_reflection_data()

    target_cpu = ScalingTarget()
    target_gpu = ScalingTargetGPU()
    apm = multi_active_parameter_manager(
        target_gpu,
        [model.components],
        [["scale", "decay"]],
        scaling_active_parameter_manager,
    )
    Ih_table = IhTable([small_rt], single_exp.crystal.get_space_group())

    from dials.algorithms.scaling.scaler import SingleScaler

    with patch.object(SingleScalerGPU, "__init__", lambda x, y, z, k: None):
        scaler = SingleScalerGPU(None, None, None)
        scaler._Ih_table = Ih_table

        SingleScaler.update_for_minimisation(scaler, apm, 0)
        f_cpu, g_cpu = target_cpu.compute_functional_gradients(scaler.Ih_table.blocked_data_list[0])
        scaler.update_for_minimisation(apm, 0)
        f_gpu, g_gpu = target_gpu.compute_functional_gradients(scaler.Ih_table.blocked_data_list[0])
        assert f_gpu == pytest.approx(f_cpu)
        assert list(g_gpu) == pytest.approx(list(g_cpu))

        SingleScaler.update_for_minimisation(scaler, apm, 0)
        r_cpu, j_cpu, w_cpu = target_cpu.compute_residuals_and_gradients(
            scaler.Ih_table.blocked_data_list[0]
        )
        scaler.update_for_minimisation(apm, 0)
        r_gpu, j_gpu, w_gpu = target_gpu.compute_residuals_and_gradients(
            scaler.Ih_table.blocked_data_list[0]
        )
        assert list(r_gpu) == pytest.approx(list(r_cpu))
        assert list(w_gpu) == pytest.approx(list(w_cpu))
        np.testing.assert_allclose(
            j_gpu.as_dense_matrix().as_numpy_array(),
            j_cpu.as_dense_matrix().as_numpy_array(),
            rtol=1e-12,
            atol=1e-12,
        )


# ==============================================================================
# 4. Port of test_outlier_rejection.py
# ==============================================================================


def test_dials_repo_outlier_rejection():
    """Port of test_outlier_rejection.py running against GPU outlier rejection classes."""
    sg = space_group("C 2y")

    # 1. Small outliers test
    rt_small = flex.reflection_table()
    rt_small["intensity"] = flex.double(
        [3560.84231, 3433.66407, 3830.64235, 0.20552, 3786.59537]
        + [4009.98652, 0.00000, 3578.91470, 3549.19151, 3379.58616]
        + [3686.38610, 3913.42869, 0.00000, 3608.84869, 3681.11110]
    )
    rt_small["variance"] = flex.double(
        [10163.98104, 9577.90389, 9702.84868, 3.77427, 8244.70685]
        + [9142.38221, 1.51118, 9634.53782, 9870.73103, 9078.23488]
        + [8977.26984, 8712.91360, 1.78802, 7473.26521, 10075.49862]
    )
    rt_small["inverse_scale_factor"] = flex.double(rt_small.size(), 1.0)
    rt_small["miller_index"] = flex.miller_index([(0, 0, 1)] * rt_small.size())
    rej_small = NormDevOutlierRejectionGPU(IhTable([rt_small], space_group("P 1")), zmax=6.0)
    rej_small.run()
    outliers_small = rej_small.final_outlier_arrays
    assert len(outliers_small) == 1
    assert set(outliers_small[0]) == {3, 6, 12}

    # 2. Standard, Simple, and Targeted outlier rejection
    rt = flex.reflection_table()
    rt["intensity"] = flex.double(
        [1.0, 1.0, 1.0, 1.0, 20.0, 1.0, 1.0, 20.0, 400.0, 500.0, 10.0, 10.0]
    )
    rt["variance"] = flex.double(12, 1.0)
    rt["inverse_scale_factor"] = flex.double(12, 1.0)
    rt["miller_index"] = flex.miller_index(
        [
            (0, 0, 1),
            (0, 0, 1),
            (0, 0, 1),
            (0, 0, 1),
            (0, 0, 1),
            (0, 0, 2),
            (0, 0, 2),
            (0, 0, 2),
            (0, 0, 2),
            (0, 0, 2),
            (0, 0, 23),
            (0, 0, 3),
        ]
    )
    rt.set_flags(
        flex.bool([True] + [False] * 11),
        rt.flags.excluded_for_scaling,
    )
    rt.set_flags(flex.bool(12, False), rt.flags.user_excluded_in_scaling)
    ih_table = IhTable([rt], sg, nblocks=1)

    target = flex.reflection_table()
    target["intensity"] = flex.double([500])
    target["variance"] = flex.double([1.0])
    target["inverse_scale_factor"] = flex.double([1.0])
    target["miller_index"] = flex.miller_index([(0, 0, 2)])
    target.set_flags(flex.bool([False]), target.flags.excluded_for_scaling)
    target.set_flags(flex.bool([False]), target.flags.user_excluded_in_scaling)
    target_ih = IhTable([target], sg, nblocks=1)

    outliers_std = determine_outlier_index_arrays_gpu(ih_table, "standard")
    assert set(outliers_std[0]) == {4, 7, 8, 9}

    outliers_simp = determine_outlier_index_arrays_gpu(ih_table, "simple")
    assert list(outliers_simp[0]) == [4, 5, 6, 7, 8, 9]

    outliers_targ = determine_outlier_index_arrays_gpu(ih_table, "target", target=target_ih)
    assert list(outliers_targ[0]) == [5, 6, 7, 8]

    # 3. Multi-dataset outlier rejection
    rt1 = flex.reflection_table()
    rt1["intensity"] = flex.double([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 20.0, 400.0, 10.0])
    rt1["variance"] = flex.double(9, 1.0)
    rt1["inverse_scale_factor"] = flex.double(9, 1.0)
    rt1["miller_index"] = flex.miller_index(
        [
            (0, 0, 1),
            (0, 0, 1),
            (0, 0, 1),
            (0, 0, 1),
            (0, 0, 2),
            (0, 0, 2),
            (0, 0, 2),
            (0, 0, 2),
            (0, 0, 3),
        ]
    )
    rt1.set_flags(flex.bool([True] + [False] * 8), rt1.flags.excluded_for_scaling)
    rt1.set_flags(flex.bool(9, False), rt1.flags.user_excluded_in_scaling)

    rt2 = flex.reflection_table()
    rt2["intensity"] = flex.double([10.0, 20.0, 500.0])
    rt2["variance"] = flex.double(3, 1.0)
    rt2["inverse_scale_factor"] = flex.double(3, 1.0)
    rt2["miller_index"] = flex.miller_index([(0, 0, 23), (0, 0, 1), (0, 0, 2)])
    rt2.set_flags(flex.bool(3, False), rt2.flags.excluded_for_scaling)
    rt2.set_flags(flex.bool(3, False), rt2.flags.user_excluded_in_scaling)

    ih_multi = IhTable([rt1, rt2], sg, nblocks=1)
    rej_multi_simp = SimpleNormDevOutlierRejectionGPU(ih_multi, 6.0)
    rej_multi_simp.run()
    outliers_multi_simp = rej_multi_simp.final_outlier_arrays
    assert list(outliers_multi_simp[0]) == [4, 5, 6, 7]
    assert list(outliers_multi_simp[1]) == [1, 2]

    rej_multi_std = NormDevOutlierRejectionGPU(ih_multi, 6.0)
    rej_multi_std.run()
    outliers_multi_std = rej_multi_std.final_outlier_arrays
    assert set(outliers_multi_std[0]) == {6, 7}
    assert set(outliers_multi_std[1]) == {1, 2}


# ==============================================================================
# 5. Port of test_combine_intensities.py
# ==============================================================================


def _generate_simple_table_for_combine(prf=True):
    reflections = flex.reflection_table()
    reflections["miller_index"] = flex.miller_index(
        [(0, 0, 1)] * 5
        + [(0, 0, 2)] * 5
        + [(0, 0, 3)] * 5
        + [(0, 0, 4)] * 5
        + [(0, 0, 5)] * 5
    )
    reflections["inverse_scale_factor"] = flex.double(25, 1.0)
    reflections["intensity.sum.value"] = flex.double(
        [
            10000.0,
            11000.0,
            9000.0,
            8000.0,
            12000.0,
            500.0,
            5600.0,
            5500.0,
            2000.0,
            6000.0,
            100.0,
            50.0,
            150.0,
            75.0,
            125.0,
            30.0,
            10.0,
            2.0,
            35.0,
            79.0,
            1.0,
            10.0,
            20.0,
            10.0,
            5.0,
        ]
    )
    reflections["intensity.sum.variance"] = flex.double(
        [10000] * 5 + [5000] * 5 + [100] * 5 + [30] * 5 + [10] * 5
    )
    reflections.set_flags(flex.bool(25, False), reflections.flags.outlier_in_scaling)
    reflections.set_flags(flex.bool(25, True), reflections.flags.integrated)
    reflections["lp"] = flex.double(25, 0.5)
    if prf:
        reflections["intensity.prf.value"] = flex.double(
            [
                10000.0,
                16000.0,
                12000.0,
                6000.0,
                9000.0,
                5000.0,
                2000.0,
                1500.0,
                1300.0,
                9000.0,
                100.0,
                80.0,
                120.0,
                90.0,
                100.0,
                30.0,
                40.0,
                50.0,
                30.0,
                30.0,
                10.0,
                12.0,
                9.0,
                8.0,
                10.0,
            ]
        )
        reflections["intensity.prf.variance"] = flex.double(
            [1000] * 5 + [500] * 5 + [10] * 5 + [3] * 5 + [1] * 5
        )
    reflections = calculate_prescaling_correction(reflections)
    return reflections


def test_dials_repo_combine_intensities():
    """Port of test_combine_intensities.py running on GPU."""
    exp = Experiment()
    exp.crystal = Crystal.from_dict(
        {
            "__id__": "crystal",
            "real_space_a": [1.0, 0.0, 0.0],
            "real_space_b": [0.0, 1.0, 0.0],
            "real_space_c": [0.0, 0.0, 1.0],
            "space_group_hall_symbol": " P 1",
        }
    )

    reflections = flex.reflection_table()
    reflections["intensity.sum.value"] = flex.double([100.0, 100.0, 100.0, 100.0])
    reflections["intensity.prf.value"] = flex.double([200.0, 200.0, 200.0, 200.0])
    reflections["intensity.sum.variance"] = flex.double(4, 100)
    reflections["intensity.prf.variance"] = flex.double(4, 200)
    reflections["prescaling_correction"] = flex.double(4, 1.0)
    reflections.set_flags(
        flex.bool([False, False, True, True]), reflections.flags.integrated_prf
    )
    reflections.set_flags(
        flex.bool([True, False, False, True]), reflections.flags.integrated_sum
    )

    intensities, _ = combine_intensities_gpu(reflections, Imid=100.0)
    assert list(intensities) == pytest.approx([100.0, 100.0, 200.0, 150.0])

    # SingleDatasetIntensityCombinerGPU
    rt = _generate_simple_table_for_combine()
    scaler = Mock()
    scaler.reflection_table = rt
    scaler.suitable_refl_for_scaling_sel = flex.bool(rt.size(), True)
    scaler.outliers = flex.bool(rt.size(), False)
    scaler.free_set_selection = flex.bool(rt.size(), False)
    scaler.experiment = exp
    scaler.params.reflection_selection.combine.Imid = None

    combiner = SingleDatasetIntensityCombinerGPU(scaler)
    assert combiner.max_key == pytest.approx(1200.0)
    intensity, variance = combiner.calculate_suitable_combined_intensities()
    assert list(intensity[0:5]) == pytest.approx(
        list(rt["intensity.sum.value"][0:5] * rt["prescaling_correction"][0:5]),
        rel=2e-2,
    )
    assert list(intensity[20:25]) == pytest.approx(
        list(rt["intensity.prf.value"][20:25] * rt["prescaling_correction"][20:25]),
        rel=2e-2,
    )

    # MultiDatasetIntensityCombinerGPU
    r1 = _generate_simple_table_for_combine()
    r1["partiality"] = flex.double(25, 1.0)
    r2 = _generate_simple_table_for_combine(prf=False)
    scaler1 = Mock()
    scaler1.reflection_table = r1
    scaler1.suitable_refl_for_scaling_sel = flex.bool(r1.size(), True)
    scaler1.outliers = flex.bool(r1.size(), False)
    scaler1.free_set_selection = flex.bool(r1.size(), False)
    scaler1.experiment = exp
    scaler1.params.reflection_selection.combine.Imid = None
    scaler2 = Mock()
    scaler2.reflection_table = r2
    scaler2.suitable_refl_for_scaling_sel = flex.bool(r2.size(), True)
    scaler2.outliers = flex.bool(r2.size(), False)
    scaler2.free_set_selection = flex.bool(r2.size(), False)
    scaler2.experiment = exp
    scaler2.params.reflection_selection.combine.Imid = None

    multiscaler = Mock()
    multiscaler.active_scalers = [scaler1, scaler2]
    multiscaler.experiment = exp
    multiscaler.params.reflection_selection.combine.Imid = None

    combiner_multi = MultiDatasetIntensityCombinerGPU(multiscaler)
    assert combiner_multi.max_key == pytest.approx(1200.0)


# ==============================================================================
# 6. Port of test_error_model.py
# ==============================================================================


def test_dials_repo_error_model():
    """Port of test_error_model.py running on GPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sg = space_group("P 1")

    reflections = flex.reflection_table()
    reflections["intensity"] = flex.double([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    reflections["inverse_scale_factor"] = flex.double([1.0] * 5 + [2.0] * 5)
    reflections["variance"] = flex.double([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    reflections["miller_index"] = flex.miller_index(
        [(1, 0, 0), (0, 0, 1), (-3, 0, 0), (0, 2, 0), (1, 0, 0)]
        + [(0, 0, -20), (0, 0, 2), (10, 0, 0), (10, 0, 0), (1, 0, 0)]
    )
    reflections.set_flags(flex.bool(10, True), reflections.flags.integrated)

    Ih_table = IhTable([reflections], sg, nblocks=1)
    block = Ih_table.blocked_data_list[0]
    params = gpu_scale.phil_scope.extract()
    params.weighting.error_model.basic.n_bins = 2
    params.weighting.error_model.basic.min_Ih = 1.0
    BasicErrorModel.min_reflections_required = 1
    error_model = BasicErrorModel(basic_params=params.weighting.error_model.basic)
    error_model.min_reflections_required = 1
    error_model.configure_for_refinement(block)
    error_model.parameters = [1.0, 0.05]

    parameterisation = ErrorModelB_APM(error_model)
    target_cpu = ErrorModelTargetB(error_model)
    target_gpu = ErrorModelTargetBGPU(error_model)
    target_cpu.predict(parameterisation)
    target_gpu.predict(parameterisation)

    _, g_cpu = target_cpu.compute_functional_gradients(parameterisation)
    _, g_gpu = target_gpu.compute_functional_gradients(parameterisation)
    assert list(g_gpu) == pytest.approx(list(g_cpu))


# ==============================================================================
# 7. Port of test_scaling_utilities.py
# ==============================================================================


def test_dials_repo_scaling_utilities():
    """Port of test_scaling_utilities.py running on GPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gonio = GoniometerFactory.from_dict(
        {
            "axes": [[1.0, 0.0, 0.0]],
            "angles": [0.0],
            "names": ["GON_PHI"],
            "scan_axis": 0,
        }
    )
    exp = Experiment(
        beam=Beam(s0=(0.0, 0.0, 2.0)),
        goniometer=gonio,
        scan=Scan(image_range=[1, 90], oscillation=[0.0, 1.0]),
    )
    rt = flex.reflection_table()
    s1_vec = (1.0 / sqrt(2.0), 0.0, 1.0 / sqrt(2.0))
    rt["s1"] = flex.vec3_double([s1_vec, s1_vec, s1_vec])
    rt["phi"] = flex.double([0.0, 45.0, 90.0])
    rt["xyzobs.px.value"] = flex.vec3_double(
        [(0.0, 0.0, 0.0), (0.0, 0.0, 45.0), (0.0, 0.0, 90.0)]
    )

    rt = calc_crystal_frame_vectors_gpu(rt, exp, device=device)
    expected_s0c = [
        (0.0, 0.0, -1.0),
        (0.0, -1.0 / sqrt(2.0), -1.0 / sqrt(2.0)),
        (0.0, -1.0, 0.0),
    ]
    for v1, v2 in zip(rt["s0c"], expected_s0c):
        assert v1 == pytest.approx(v2)
    expected_s1c = [
        (1.0 / sqrt(2.0), 0.0, 1.0 / sqrt(2.0)),
        (1.0 / sqrt(2.0), 0.5, 0.5),
        (1.0 / sqrt(2.0), 1.0 / sqrt(2.0), 0.0),
    ]
    for v1, v2 in zip(rt["s1c"], expected_s1c):
        assert v1 == pytest.approx(v2)

    rt["s0c"] = align_axis_along_z_gpu((1.0, 0.0, 0.0), rt["s0c"], device=device)
    rt["s1c"] = align_axis_along_z_gpu((1.0, 0.0, 0.0), rt["s1c"], device=device)
    theta_phi_0 = calc_theta_phi_gpu(rt["s0c"], device=device)
    theta_phi_1 = calc_theta_phi_gpu(rt["s1c"], device=device)
    sph_h_t = create_sph_harm_table(theta_phi_0, theta_phi_1, 2)
    Y10 = ((3.0 / (8.0 * pi)) ** 0.5) / 2.0
    Y20 = -1.0 * ((5.0 / (256.0 * pi)) ** 0.5)
    assert sph_h_t[1, 0] == pytest.approx(Y10)
    assert sph_h_t[5, 0] == pytest.approx(Y20)


# ==============================================================================
# 8. Port of test_scale.py (CLI and end-to-end DIALS regression tests)
# ==============================================================================


def test_scale_script_prepare_input_gpu():
    """Port of test_scale.py::test_scale_script_prepare_input."""
    params, exp, reflections = _generate_test_input()
    reflections.append(reflections[0].copy())
    with pytest.raises(ValueError):
        _ = ScalingAlgorithmGPU(params, exp, reflections)

    params, exp, reflections = _generate_test_input(n=1)
    params.cut_data.d_min = 1.5
    params, _, script_reflections = prepare_input(params, exp, reflections)
    r = script_reflections[0]
    assert list(r.get_flags(r.flags.user_excluded_in_scaling)) == [
        False,
        False,
        True,
        True,
    ]


def _generate_test_input(n=1):
    experiments = ExperimentList()
    crystal = Crystal.from_dict(
        {
            "__id__": "crystal",
            "real_space_a": [1.0, 0.0, 0.0],
            "real_space_b": [0.0, 1.0, 0.0],
            "real_space_c": [0.0, 0.0, 2.0],
            "space_group_hall_symbol": " C 2y",
        }
    )
    scan = Scan(image_range=[0, 90], oscillation=[0.0, 1.0])
    beam = Beam(s0=(0.0, 0.0, 1.01))
    goniometer = Goniometer((1.0, 0.0, 0.0))
    detector = Detector()
    reflections = []
    for _ in range(n):
        experiments.append(
            Experiment(
                beam=beam,
                scan=scan,
                goniometer=goniometer,
                detector=detector,
                crystal=crystal,
            )
        )
        r = flex.reflection_table()
        r["intensity.sum.value"] = flex.double([1.0, 2.0, 3.0, 4.0])
        r["intensity.sum.variance"] = flex.double([1.0, 2.0, 3.0, 4.0])
        r["miller_index"] = flex.miller_index([(0, 0, 1), (0, 0, 1), (0, 0, 2), (0, 0, 2)])
        r["xyzobs.px.value"] = flex.vec3_double([(0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4)])
        r["id"] = flex.int([0, 0, 0, 0])
        reflections.append(r)
    return gpu_scale.phil_scope.extract(), experiments, reflections


def test_targeted_scaling_against_mtz_gpu(dials_data, tmp_path):
    """Port of test_scale.py::test_targeted_scaling_against_mtz."""
    location = dials_data("l_cysteine_4_sweeps_scaled")
    refl = location / "scaled_35.refl"
    expt = location / "scaled_35.expt"
    command = [_get_dials_gpu_scale_bin(), str(refl), str(expt), "unmerged_mtz=unmerged.mtz"]

    result = subprocess.run(command, cwd=tmp_path, capture_output=True)
    assert not result.returncode and not result.stderr, result.stderr.decode()
    assert (tmp_path / "scaled.expt").is_file()
    assert (tmp_path / "scaled.refl").is_file()
    assert (tmp_path / "unmerged.mtz").is_file()

    refl = location / "scaled_30.refl"
    expt = location / "scaled_30.expt"
    reference = tmp_path / "unmerged.mtz"
    command = [
        _get_dials_gpu_scale_bin(),
        str(refl),
        str(expt),
        f"reference={reference}",
        "unmerged_mtz=unmerged_2.mtz",
    ]

    result = subprocess.run(command, cwd=tmp_path, capture_output=True)
    assert not result.returncode and not result.stderr, result.stderr.decode()
    assert (tmp_path / "scaled.expt").is_file()
    assert (tmp_path / "scaled.refl").is_file()
    assert (tmp_path / "unmerged_2.mtz").is_file()
    expts = load.experiment_list(tmp_path / "scaled.expt", check_format=False)
    assert len(expts) == 1
    stats = get_merging_stats(tmp_path / "unmerged_2.mtz")
    assert stats.overall.r_pim < 0.024
    assert stats.overall.cc_one_half > 0.995
    assert stats.overall.n_obs > 3150


@pytest.mark.parametrize(
    "option",
    [
        None,
        "reflection_selection.method=random",
        "reflection_selection.method=intensity_ranges",
        "reflection_selection.method=use_all",
        "intensity_choice=sum",
        "intensity=profile",
    ],
)
def test_scale_single_dataset_with_options_gpu(dials_data, tmp_path, option):
    """Port of test_scale.py::test_scale_single_dataset_with_options."""
    data_dir = dials_data("l_cysteine_dials_output")
    refl_1 = data_dir / "20_integrated.pickle"
    expt_1 = data_dir / "20_integrated_experiments.json"
    args = [refl_1, expt_1]
    if option:
        args.append(option)
    run_one_scaling_gpu(tmp_path, args)


@pytest.fixture(scope="session")
def vmxi_protk_reindexed(dials_data, tmp_path_factory):
    """Reindex the protk data to be in the correct space group."""
    location = dials_data("vmxi_proteinase_k_sweeps")
    command = [
        shutil.which("dials.reindex") or "/home/davidm/dials/conda_base/bin/dials.reindex",
        str(location / "experiments_0.json"),
        str(location / "reflections_0.pickle"),
        "space_group=P422",
    ]
    tmp_path = tmp_path_factory.mktemp("vmxi_protk_reindexed")
    subprocess.run(command, cwd=tmp_path, capture_output=True, check=True)
    return tmp_path / "reindexed.expt", tmp_path / "reindexed.refl"


@pytest.mark.parametrize(
    ("options", "expected", "tolerances"),
    [
        (["error_model=None"], None, None),
        (
            ["error_model=basic", "basic.minimisation=individual"],
            (0.61, 0.049),
            (0.05, 0.005),
        ),
        (["error_model.basic.a=0.61"], (0.61, 0.049), (1e-6, 0.005)),
        (["error_model.basic.b=0.049"], (0.61, 0.049), (0.05, 1e-6)),
        (
            ["error_model.basic.b=0.02", "error_model.basic.a=1.5"],
            (1.50, 0.02),
            (1e-6, 1e-6),
        ),
        (
            ["error_model=basic", "basic.minimisation=regression"],
            (0.995, 0.051),
            (0.05, 0.005),
        ),
    ],
)
def test_error_model_options_gpu(
    vmxi_protk_reindexed, tmp_path, options, expected, tolerances
):
    """Port of test_scale.py::test_error_model_options."""
    expt_1, refl_1 = vmxi_protk_reindexed
    args = [refl_1, expt_1] + list(options)
    run_one_scaling_gpu(tmp_path, args)
    expts = load.experiment_list(tmp_path / "scaled.expt", check_format=False)
    config = expts[0].scaling_model.configdict
    if not expected:
        assert "error_model_parameters" not in config
    else:
        params = expts[0].scaling_model.configdict["error_model_parameters"]
        assert params[0] == pytest.approx(expected[0], abs=tolerances[0])
        assert params[1] == pytest.approx(expected[1], abs=tolerances[1])


def test_scale_physical_gpu(dials_data, tmp_path):
    """Port of test_scale.py::test_scale_physical."""
    data_dir = dials_data("l_cysteine_dials_output")
    refl_1 = data_dir / "20_integrated.pickle"
    expt_1 = data_dir / "20_integrated_experiments.json"
    extra_args = [
        "model=physical",
        "merged_mtz=merged.mtz",
        "error_model=None",
        "intensity_choice=profile",
        "unmerged_mtz=unmerged.mtz",
        "crystal_name=foo",
        "project_name=bar",
        "use_free_set=1",
        "outlier_rejection=simple",
        "json=scaling.json",
        "additional_stats=True",
    ]
    run_one_scaling_gpu(tmp_path, [refl_1, expt_1] + extra_args)
    unmerged_mtz = tmp_path / "unmerged.mtz"
    merged_mtz = tmp_path / "merged.mtz"
    assert unmerged_mtz.is_file()
    assert merged_mtz.is_file()
    assert (tmp_path / "scaling.json").is_file()

    result = get_merging_stats(unmerged_mtz)
    assert result.overall.r_pim < 0.0255
    assert result.overall.cc_one_half > 0.9955
    assert result.overall.n_obs > 2300


def test_scale_set_absorption_level_gpu(dials_data, tmp_path):
    """Port of test_scale.py::test_scale_set_absorption_level."""
    location = dials_data("l_cysteine_dials_output")
    refl = location / "20_integrated.pickle"
    expt = location / "20_integrated_experiments.json"

    command = [
        _get_dials_gpu_scale_bin(),
        str(refl),
        str(expt),
        "absorption_level=medium",
        "unmerged_mtz=unmerged.mtz",
    ]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True)
    assert not result.returncode and not result.stderr, result.stderr.decode()
    expts = load.experiment_list(tmp_path / "scaled.expt", check_format=False)
    assert expts[0].scaling_model.configdict["lmax"] == 6
    assert expts[0].scaling_model.configdict["abs_surface_weight"] == 5e4
    abs_params = expts[0].scaling_model.components["absorption"].parameters
    stats = get_merging_stats(tmp_path / "unmerged.mtz")
    assert stats.overall.r_pim < 0.024
    assert stats.overall.cc_one_half > 0.995
    assert stats.overall.n_obs > 2300

    command = [
        _get_dials_gpu_scale_bin(),
        str(tmp_path / "scaled.refl"),
        str(tmp_path / "scaled.expt"),
        "error_model=None",
        "physical.correction.fix=absorption",
    ]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True)
    assert not result.returncode and not result.stderr, result.stderr.decode()
    expts = load.experiment_list(tmp_path / "scaled.expt", check_format=False)
    new_abs_params = expts[0].scaling_model.components["absorption"].parameters
    assert abs_params == new_abs_params


def test_scale_normal_equations_failure_gpu(dials_data, tmp_path):
    """Port of test_scale.py::test_scale_normal_equations_failure."""
    location = dials_data("l_cysteine_dials_output")
    refl = location / "20_integrated.pickle"
    expt = location / "20_integrated_experiments.json"

    command = [_get_dials_gpu_scale_bin(), str(refl), str(expt), "exclude_images=800:1400"]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True)
    assert not result.returncode and not result.stderr, result.stderr.decode()
    assert (tmp_path / "scaled.refl").is_file()
    assert (tmp_path / "scaled.expt").is_file()


def test_scale_and_filter_image_group_single_dataset_gpu(dials_data, tmp_path):
    """Port of test_scale.py::test_scale_and_filter_image_group_single_dataset."""
    data_dir = dials_data("l_cysteine_dials_output")
    command = [
        _get_dials_gpu_scale_bin(),
        str(data_dir / "20_integrated.pickle"),
        str(data_dir / "20_integrated_experiments.json"),
        "filtering.method=deltacchalf",
        "stdcutoff=3.0",
        "mode=image_group",
        "max_cycles=1",
        "scale_and_filter_results=analysis_results.json",
        "error_model=None",
    ]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True)
    assert not result.returncode and not result.stderr, result.stderr.decode()
    assert (tmp_path / "scaled.refl").is_file()
    assert (tmp_path / "scaled.expt").is_file()
    assert (tmp_path / "analysis_results.json").is_file()


def test_scale_dose_decay_model_gpu(dials_data, tmp_path):
    """Port of test_scale.py::test_scale_dose_decay_model."""
    location = dials_data("multi_crystal_proteinase_k")
    command = [_get_dials_gpu_scale_bin(), "d_min=2.0", "model=dose_decay"]
    for i in [1, 2, 3, 4, 5, 7, 10]:
        command.append(str(location / f"experiments_{i}.json"))
        command.append(str(location / f"reflections_{i}.pickle"))

    result = subprocess.run(command, cwd=tmp_path, capture_output=True)
    assert not result.returncode and not result.stderr, result.stderr.decode()
    assert (tmp_path / "scaled.refl").is_file()
    assert (tmp_path / "scaled.expt").is_file()
    expts = load.experiment_list(tmp_path / "scaled.expt", check_format=False)
    assert expts[0].scaling_model.id_ == "dose_decay"


def test_scale_array_gpu(dials_data, tmp_path):
    """Port of test_scale.py::test_scale_array."""
    data_dir = dials_data("l_cysteine_dials_output")
    refl = data_dir / "20_integrated.pickle"
    expt = data_dir / "20_integrated_experiments.json"
    extra_args = ["model=array", "array.absorption_correction=0", "full_matrix=0"]
    run_one_scaling_gpu(tmp_path, [refl, expt] + extra_args)


def test_multi_scale_gpu(dials_data, tmp_path):
    """Port of test_scale.py::test_multi_scale."""
    data_dir = dials_data("l_cysteine_dials_output")
    refl_1 = data_dir / "20_integrated.pickle"
    expt_1 = data_dir / "20_integrated_experiments.json"
    refl_2 = data_dir / "25_integrated.pickle"
    expt_2 = data_dir / "25_integrated_experiments.json"
    extra_args = [
        "unmerged_mtz=unmerged.mtz",
        "error_model=None",
        "intensity_choice=profile",
        "outlier_rejection=simple",
    ]
    run_one_scaling_gpu(tmp_path, [refl_1, refl_2, expt_1, expt_2] + extra_args)
    result = get_merging_stats(tmp_path / "unmerged.mtz")
    assert abs(result.overall.n_obs - 5526) < 30
    assert result.overall.r_pim < 0.0221
    assert result.overall.cc_one_half > 0.9975


def test_targeted_scaling_gpu(dials_data, tmp_path):
    """Port of test_scale.py::test_targeted_scaling."""
    location = dials_data("l_cysteine_4_sweeps_scaled")
    target_refl = location / "scaled_35.refl"
    target_expt = location / "scaled_35.expt"

    data_dir = dials_data("l_cysteine_dials_output")
    refl_1 = data_dir / "20_integrated.pickle"
    expt_1 = data_dir / "20_integrated_experiments.json"
    refl_2 = data_dir / "25_integrated.pickle"
    expt_2 = data_dir / "25_integrated_experiments.json"

    run_one_scaling_gpu(tmp_path, [target_refl, refl_1, target_expt, expt_1, "model=KB"])
    scaled_exp = tmp_path / "scaled.expt"
    scaled_refl = tmp_path / "scaled.refl"
    experiments_list = load.experiment_list(scaled_exp, check_format=False)
    assert len(experiments_list.scaling_models()) == 2
    assert experiments_list.scaling_models()[0].id_ == "physical"
    assert experiments_list.scaling_models()[1].id_ == "KB"

    run_one_scaling_gpu(
        tmp_path, [refl_2, scaled_refl, expt_2, scaled_exp, "model=KB", "only_target=True"]
    )
    experiments_list = load.experiment_list(tmp_path / "scaled.expt", check_format=False)
    assert len(experiments_list.scaling_models()) == 1
    assert experiments_list.scaling_models()[0].id_ == "KB"


def test_shared_absorption_surface_gpu(dials_data, tmp_path):
    """Port of test_scale.py::test_shared_absorption_surface."""
    data_dir = dials_data("l_cysteine_dials_output")
    refl_1 = data_dir / "20_integrated.pickle"
    expt_1 = data_dir / "20_integrated_experiments.json"
    refl_2 = data_dir / "25_integrated.pickle"
    expt_2 = data_dir / "25_integrated_experiments.json"

    run_one_scaling_gpu(tmp_path, [refl_1, expt_1, refl_2, expt_2, "share.absorption=True"])
    expts = load.experiment_list(tmp_path / "scaled.expt", check_format=False)
    assert (
        expts.scaling_models()[0].components["absorption"].parameters
        == expts.scaling_models()[1].components["absorption"].parameters
    )


def test_few_reflections_gpu(dials_data):
    """Port of test_scale.py::test_few_reflections."""
    data_dir = dials_data("l_cysteine_dials_output")
    experiments = ExperimentList.from_file(data_dir / "11_integrated.expt")
    experiments.extend(ExperimentList.from_file(data_dir / "23_integrated.expt"))
    refls = "11_integrated.refl", "23_integrated.refl"
    reflections = [flex.reflection_table.from_file(data_dir / refl) for refl in refls]

    scale_params = gpu_scale.phil_scope.fetch(source=phil.parse("output.html=None")).extract()
    scaled_expt, scaled_refl = gpu_scale.run_scaling(scale_params, experiments, reflections)

    merge_params = merge.phil_scope.fetch(source=phil.parse("")).extract()
    merge.merge_data_to_mtz(merge_params, scaled_expt, [scaled_refl])
