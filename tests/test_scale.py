"""
Comprehensive GPU vs CPU tests for dials_gpu.scale and dials_gpu.scaling
(Gaussian smoothers, scale components, RefinerCalculatorGPU, IhTable GPU reductions,
ScalingTargetGPU, outlier rejection, intensity combination, error model,
SingleScalerGPU / MultiScalerGPU / TargetScalerGPU, and end-to-end i19 scaling).
"""

from __future__ import annotations

import copy
import glob
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from cctbx import sgtbx
from dxtbx import flumpy
from dxtbx.model import Beam, Crystal, Detector, Experiment, ExperimentList, Goniometer, Scan
from scitbx import sparse

from dials.algorithms.scaling.active_parameter_managers import (
    active_parameter_manager,
    multi_active_parameter_manager,
)
from dials.algorithms.scaling.basis_functions import RefinerCalculator
from dials.algorithms.scaling.combine_intensities import (
    SingleDatasetIntensityCombiner,
    combine_intensities,
)
from dials.algorithms.scaling.error_model.engine import run_error_model_refinement
from dials.algorithms.scaling.error_model.error_model import (
    BasicErrorModel,
    phil_scope as em_phil_scope,
)
from dials.algorithms.scaling.Ih_table import IhTable
from dials.algorithms.scaling.model.components.analytical_component import (
    AnalyticalComponent,
)
from dials.algorithms.scaling.model.components.scale_components import (
    LinearDoseDecay,
    QuadraticDoseDecay,
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
from dials.algorithms.scaling.outlier_rejection import (
    determine_outlier_index_arrays,
    reject_outliers,
)
from dials.algorithms.scaling.parameter_handler import scaling_active_parameter_manager
from dials.algorithms.scaling.scaler import calc_sf_variances
from dials.algorithms.scaling.scaling_utilities import (
    align_axis_along_z,
    calc_crystal_frame_vectors,
    sph_harm_table,
)
from dials.algorithms.scaling.target_function import ScalingTarget, ScalingTargetFixedIH
from dials.array_family import flex
from dials.command_line.scale import phil_scope as scale_phil_scope
from dials.command_line.scale import run_scaling as cpu_run_scaling
from dials_scaling_ext import (
    calc_lookup_index,
    calc_theta_phi,
    create_sph_harm_lookup_table,
    rotate_vectors_about_axis,
)

from dials_gpu.command_line.scale import run_scaling as gpu_run_scaling
from dials_gpu.scaling import (
    GaussianSmoother1DGPU,
    GaussianSmoother2DGPU,
    GaussianSmoother3DGPU,
    RefinerCalculatorGPU,
    ScalingTargetFixedIHGPU,
    ScalingTargetGPU,
    SingleDatasetIntensityCombinerGPU,
    align_axis_along_z_gpu,
    calc_Ih_gpu,
    calc_crystal_frame_vectors_gpu,
    calc_lookup_index_gpu,
    calc_sf_variances_gpu,
    calc_theta_phi_gpu,
    combine_intensities_gpu,
    determine_outlier_index_arrays_gpu,
    eval_smoother_1d_gpu,
    eval_smoother_2d_gpu,
    eval_smoother_3d_gpu,
    evaluate_component_gpu,
    get_block_group_ids_gpu,
    reject_outliers_gpu,
    rotate_vectors_about_axis_gpu,
    run_error_model_refinement_gpu,
    sum_in_groups_gpu,
)

I19_DIR = "/home/davidm/diffraction/test_data/i19/301_Nidppe_f3_phi0"


def _make_synthetic_experiment():
    exp = Experiment()
    exp.beam = Beam((0.0, 0.0, 1.0), 0.7)
    exp.detector = Detector()
    exp.goniometer = Goniometer((1.0, 0.0, 0.0))
    exp.scan = Scan((1, 50), (0.0, 1.0))
    exp.crystal = Crystal(
        (10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0), space_group_symbol="P 1"
    )
    exp.identifier = "0"
    return exp


def _make_synthetic_scaling_reflections(n_groups=120, mult=4, seed=42):
    rng = np.random.default_rng(seed)
    n_refl = n_groups * mult
    refl = flex.reflection_table()
    hkl_list = []
    intensities = []
    variances = []
    d_vals = []
    phi_vals = []
    x_vals = []
    y_vals = []
    s1_vals = []
    for g in range(n_groups):
        h = (g % 7) + 1
        k = ((g // 7) % 7) + 1
        l = (g // 49) + 1
        true_Ih = float(rng.uniform(50.0, 1000.0))
        d_val = 10.0 / np.sqrt(h * h + k * k + l * l)
        for m in range(mult):
            hkl_list.append((h, k, l))
            phi = float(rng.uniform(0.0, 49.0))
            scale = 1.0 + 0.15 * np.sin(phi / 15.0)
            var = true_Ih + 10.0
            obs_I = scale * true_Ih + float(rng.normal(0.0, np.sqrt(var) * 0.5))
            intensities.append(max(obs_I, 1.0))
            variances.append(var)
            d_vals.append(d_val)
            phi_vals.append(phi)
            x_vals.append(float(rng.uniform(0.0, 100.0)))
            y_vals.append(float(rng.uniform(0.0, 100.0)))
            v = rng.normal(size=3)
            v = v / np.linalg.norm(v)
            s1_vals.append(tuple(v))

    refl["miller_index"] = flex.miller_index(hkl_list)
    refl["intensity"] = flex.double(intensities)
    refl["variance"] = flex.double(variances)
    refl["intensity.sum.value"] = flex.double(intensities)
    refl["intensity.sum.variance"] = flex.double(variances)
    refl["intensity.prf.value"] = flex.double(np.array(intensities) * 0.99)
    refl["intensity.prf.variance"] = flex.double(np.array(variances) * 0.95)
    refl["d"] = flex.double(d_vals)
    refl["xyzobs.px.value"] = flex.vec3_double(
        [(x, y, z) for x, y, z in zip(x_vals, y_vals, phi_vals)]
    )
    refl["s1"] = flex.vec3_double(s1_vals)
    refl["lp"] = flex.double(n_refl, 1.0)
    refl["partiality"] = flex.double(n_refl, 1.0)
    refl["prescaling_correction"] = flex.double(n_refl, 1.0)
    refl["inverse_scale_factor"] = flex.double(n_refl, 1.0)
    refl["id"] = flex.int(n_refl, 0)
    refl.experiment_identifiers()[0] = "0"
    refl.set_flags(flex.bool(n_refl, True), refl.flags.integrated)
    return refl


def test_gaussian_smoothers_gpu_vs_cpu():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(123)

    # 1D Smoother across all interval regimes (1, 2, 5) and first_fixed=False/True
    for n_intervals in (1, 2, 5):
        s_cpu = GaussianSmoother1D([0.0, 50.0], n_intervals)
        s_gpu = GaussianSmoother1DGPU([0.0, 50.0], n_intervals)
        n_vals = s_cpu.num_values()
        params = flex.double(rng.uniform(0.8, 1.2, size=n_vals))
        x_pts = flex.double(rng.uniform(0.0, 50.0, size=200))

        v_cpu, w_cpu, sw_cpu = s_cpu.multi_value_weight(x_pts, params)
        dv_cpu = np.asarray(
            flumpy.to_numpy(w_cpu.as_dense_matrix()), dtype=np.float64
        ).reshape(200, n_vals) / np.asarray(flumpy.to_numpy(sw_cpu))[:, None]

        x_t = torch.from_numpy(np.asarray(flumpy.to_numpy(x_pts))).to(
            dev, torch.float64
        )
        p_t = torch.from_numpy(np.asarray(flumpy.to_numpy(params))).to(
            dev, torch.float64
        )
        v_gpu, dv_gpu = eval_smoother_1d_gpu(
            s_gpu, x_t, p_t, first_fixed=False, compute_derivatives=True
        )
        np.testing.assert_allclose(
            v_gpu.cpu().numpy(), flumpy.to_numpy(v_cpu), rtol=1e-13, atol=1e-13
        )
        np.testing.assert_allclose(
            dv_gpu.cpu().numpy(), dv_cpu, rtol=1e-13, atol=1e-13
        )

        # first_fixed=True
        v_ff_cpu, w_ff_cpu, sw_ff_cpu = s_cpu.multi_value_weight_first_fixed(
            x_pts, params
        )
        dv_ff_cpu = np.asarray(
            flumpy.to_numpy(w_ff_cpu.as_dense_matrix()), dtype=np.float64
        ).reshape(200, n_vals - 1) / np.asarray(flumpy.to_numpy(sw_ff_cpu))[:, None]
        v_ff_gpu, dv_ff_gpu = eval_smoother_1d_gpu(
            s_gpu, x_t, p_t, first_fixed=True, compute_derivatives=True
        )
        np.testing.assert_allclose(
            v_ff_gpu.cpu().numpy(), flumpy.to_numpy(v_ff_cpu), rtol=1e-13, atol=1e-13
        )
        np.testing.assert_allclose(
            dv_ff_gpu.cpu().numpy(), dv_ff_cpu, rtol=1e-13, atol=1e-13
        )

    # 2D Smoother
    s2d = GaussianSmoother2DGPU([0.0, 50.0], 3, [0.0, 100.0], 4)
    n2d = s2d.num_x_values() * s2d.num_y_values()
    p2d = flex.double(rng.uniform(0.8, 1.2, size=n2d))
    x2 = flex.double(rng.uniform(0.0, 50.0, size=150))
    y2 = flex.double(rng.uniform(0.0, 100.0, size=150))
    v2_cpu, w2_cpu, sw2_cpu = s2d.multi_value_weight(x2, y2, p2d)
    dv2_cpu = np.asarray(
        flumpy.to_numpy(w2_cpu.as_dense_matrix()), dtype=np.float64
    ).reshape(150, n2d) / np.asarray(flumpy.to_numpy(sw2_cpu))[:, None]
    v2_gpu, dv2_gpu = eval_smoother_2d_gpu(
        s2d,
        torch.from_numpy(np.asarray(flumpy.to_numpy(x2))).to(dev, torch.float64),
        torch.from_numpy(np.asarray(flumpy.to_numpy(y2))).to(dev, torch.float64),
        torch.from_numpy(np.asarray(flumpy.to_numpy(p2d))).to(dev, torch.float64),
    )
    np.testing.assert_allclose(
        v2_gpu.cpu().numpy(), flumpy.to_numpy(v2_cpu), rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(dv2_gpu.cpu().numpy(), dv2_cpu, rtol=1e-13, atol=1e-13)

    # 3D Smoother
    s3d = GaussianSmoother3DGPU([0.0, 50.0], 2, [0.0, 80.0], 3, [0.0, 40.0], 4)
    n3d = s3d.num_x_values() * s3d.num_y_values() * s3d.num_z_values()
    p3d = flex.double(rng.uniform(0.8, 1.2, size=n3d))
    x3 = flex.double(rng.uniform(0.0, 50.0, size=100))
    y3 = flex.double(rng.uniform(0.0, 80.0, size=100))
    z3 = flex.double(rng.uniform(0.0, 40.0, size=100))
    v3_cpu, w3_cpu, sw3_cpu = s3d.multi_value_weight(x3, y3, z3, p3d)
    dv3_cpu = np.asarray(
        flumpy.to_numpy(w3_cpu.as_dense_matrix()), dtype=np.float64
    ).reshape(100, n3d) / np.asarray(flumpy.to_numpy(sw3_cpu))[:, None]
    v3_gpu, dv3_gpu = eval_smoother_3d_gpu(
        s3d,
        torch.from_numpy(np.asarray(flumpy.to_numpy(x3))).to(dev, torch.float64),
        torch.from_numpy(np.asarray(flumpy.to_numpy(y3))).to(dev, torch.float64),
        torch.from_numpy(np.asarray(flumpy.to_numpy(z3))).to(dev, torch.float64),
        torch.from_numpy(np.asarray(flumpy.to_numpy(p3d))).to(dev, torch.float64),
    )
    np.testing.assert_allclose(
        v3_gpu.cpu().numpy(), flumpy.to_numpy(v3_cpu), rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(dv3_gpu.cpu().numpy(), dv3_cpu, rtol=1e-13, atol=1e-13)


def test_all_scale_components_and_refiner_calculator_gpu():
    exp = _make_synthetic_experiment()
    refl = _make_synthetic_scaling_reflections(n_groups=60, mult=3)
    refl = calc_crystal_frame_vectors_gpu(refl, exp)
    n = len(refl)
    x_coord = refl["xyzobs.px.value"].parts()[2]
    y_coord = refl["xyzobs.px.value"].parts()[0]
    z_coord = refl["xyzobs.px.value"].parts()[1]
    d_col = refl["d"]

    components = {}

    c_single = SingleScaleFactor(flex.double([1.25]))
    c_single.data = {"id": refl["id"]}
    c_single.update_reflection_data()
    components["single"] = c_single

    c_b = SingleBScaleFactor(flex.double([1.8]))
    c_b.data = {"d": d_col}
    c_b.update_reflection_data()
    components["b"] = c_b

    c_ld = LinearDoseDecay(flex.double([0.05]))
    c_ld.data = {"d": d_col, "x": x_coord}
    c_ld.update_reflection_data()
    components["lin_dose"] = c_ld

    c_qd = QuadraticDoseDecay(flex.double([0.08]))
    c_qd.data = {"d": d_col, "x": x_coord}
    c_qd.update_reflection_data()
    components["quad_dose"] = c_qd

    c_s1d = SmoothScaleComponent1D(flex.double([1.0, 1.1, 0.9, 1.05, 0.95]))
    c_s1d.data = {"x": x_coord}
    c_s1d.update_reflection_data()
    components["s1d"] = c_s1d

    c_sb1d = SmoothBScaleComponent1D(flex.double([0.0, 0.5, -0.3, 0.2, 0.1]))
    c_sb1d.data = {"x": x_coord, "d": d_col}
    c_sb1d.update_reflection_data()
    components["sb1d"] = c_sb1d

    c_s2d = SmoothScaleComponent2D(flex.double([1.0 + 0.02 * i for i in range(15)]), (3, 5))
    c_s2d.data = {"x": x_coord, "y": y_coord}
    c_s2d.update_reflection_data()
    components["s2d"] = c_s2d

    c_s3d = SmoothScaleComponent3D(
        flex.double([1.0 + 0.01 * i for i in range(30)]), (2, 3, 5)
    )
    c_s3d.data = {"x": x_coord, "y": y_coord, "z": z_coord}
    c_s3d.update_reflection_data()
    components["s3d"] = c_s3d

    # SHScaleComponent in speed mode
    lmax = 2
    n_sh = lmax * (lmax + 2)
    c_sh_speed = SHScaleComponent(flex.double([0.02 * (i + 1) for i in range(n_sh)]))
    c_sh_speed.data = {"sph_harm_table": sph_harm_table(refl, lmax)}
    c_sh_speed.update_reflection_data()
    components["sh_speed"] = c_sh_speed

    # SHScaleComponent in memory mode
    c_sh_mem = SHScaleComponent(flex.double([0.02 * (i + 1) for i in range(n_sh)]))
    SHScaleComponent.coefficients_list = create_sph_harm_lookup_table(
        lmax, points_per_degree=2
    )
    tp0 = calc_theta_phi_gpu(refl["s0c"])
    tp1 = calc_theta_phi_gpu(refl["s1c"])
    c_sh_mem.data = {
        "s0_lookup": calc_lookup_index_gpu(tp0, 2),
        "s1_lookup": calc_lookup_index_gpu(tp1, 2),
    }
    c_sh_mem.update_reflection_data()
    components["sh_mem"] = c_sh_mem

    c_anal = AnalyticalComponent(flex.double([]))
    c_anal.data = {"correction": flex.double(n, 0.85)}
    c_anal.update_reflection_data()
    components["analytical"] = c_anal

    for name, comp in components.items():
        s_cpu, d_cpu = comp.calculate_scales_and_derivatives(block_id=0)
        s_gpu, d_gpu = evaluate_component_gpu(comp, block_id=0, compute_derivatives=True)
        np.testing.assert_allclose(
            s_gpu.cpu().numpy(),
            flumpy.to_numpy(s_cpu),
            rtol=1e-13,
            atol=1e-13,
            err_msg=f"Scale mismatch for {name}",
        )
        if d_cpu.n_cols > 0:
            d_cpu_np = np.asarray(
                flumpy.to_numpy(d_cpu.as_dense_matrix()), dtype=np.float64
            ).reshape(d_cpu.n_rows, d_cpu.n_cols)
            np.testing.assert_allclose(
                d_gpu.cpu().numpy(),
                d_cpu_np,
                rtol=1e-13,
                atol=1e-13,
                err_msg=f"Derivative mismatch for {name}",
            )

    # Test RefinerCalculatorGPU with a subset of active components (so constant_g_values is exercised)
    apm = scaling_active_parameter_manager(
        components, ["s1d", "sb1d", "sh_speed"]
    )
    s_rc_cpu, d_rc_cpu = RefinerCalculator.calculate_scales_and_derivatives(apm, 0)
    s_rc_gpu, d_rc_gpu = RefinerCalculatorGPU.calculate_scales_and_derivatives_gpu(
        apm, 0
    )
    np.testing.assert_allclose(
        s_rc_gpu.cpu().numpy(), flumpy.to_numpy(s_rc_cpu), rtol=1e-13, atol=1e-13
    )
    d_rc_cpu_np = np.asarray(
        flumpy.to_numpy(d_rc_cpu.as_dense_matrix()), dtype=np.float64
    ).reshape(d_rc_cpu.n_rows, d_rc_cpu.n_cols)
    np.testing.assert_allclose(
        d_rc_gpu.cpu().numpy(), d_rc_cpu_np, rtol=1e-13, atol=1e-13
    )


def test_geometry_and_lookup_utilities_gpu():
    exp = _make_synthetic_experiment()
    refl_cpu = _make_synthetic_scaling_reflections(n_groups=40, mult=3)
    refl_gpu = copy.deepcopy(refl_cpu)

    refl_cpu = calc_crystal_frame_vectors(refl_cpu, exp)
    refl_gpu = calc_crystal_frame_vectors_gpu(refl_gpu, exp)
    np.testing.assert_allclose(
        flumpy.to_numpy(refl_gpu["s0c"]),
        flumpy.to_numpy(refl_cpu["s0c"]),
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        flumpy.to_numpy(refl_gpu["s1c"]),
        flumpy.to_numpy(refl_cpu["s1c"]),
        rtol=1e-13,
        atol=1e-13,
    )

    axis = (1.0, 0.0, 0.0)
    s0c_cpu = align_axis_along_z(axis, refl_cpu["s0c"])
    s0c_gpu = align_axis_along_z_gpu(axis, refl_gpu["s0c"])
    np.testing.assert_allclose(
        flumpy.to_numpy(s0c_gpu), flumpy.to_numpy(s0c_cpu), rtol=1e-13, atol=1e-13
    )

    tp_cpu = calc_theta_phi(s0c_cpu)
    tp_gpu = calc_theta_phi_gpu(s0c_gpu)
    np.testing.assert_allclose(
        flumpy.to_numpy(tp_gpu), flumpy.to_numpy(tp_cpu), rtol=1e-13, atol=1e-13
    )

    li_cpu = calc_lookup_index(tp_cpu, 2.0)
    li_gpu = calc_lookup_index_gpu(tp_gpu, 2.0)
    np.testing.assert_array_equal(flumpy.to_numpy(li_gpu), flumpy.to_numpy(li_cpu))


def test_ih_table_and_target_function_gpu():
    exp = _make_synthetic_experiment()
    refl = _make_synthetic_scaling_reflections(n_groups=50, mult=4)
    sg = exp.crystal.get_space_group()
    ih_table = IhTable([refl], sg, nblocks=1)
    block = ih_table.blocked_data_list[0]

    # Set non-trivial scale factors and derivatives
    rng = np.random.default_rng(7)
    g_vals = rng.uniform(0.7, 1.4, size=block.size)
    block.inverse_scale_factors = g_vals
    block.calc_Ih()
    ih_cpu = block.Ih_values.copy()

    ih_gpu, sumgsq_gpu, _ = calc_Ih_gpu(block, sync_to_cpu=False)
    np.testing.assert_allclose(ih_gpu.cpu().numpy(), ih_cpu, rtol=1e-13, atol=1e-13)

    # Set a random derivative matrix (n_refl, 6)
    n_params = 6
    d_np = rng.normal(size=(block.size, n_params))
    d_sparse = sparse.matrix(block.size, n_params)
    b = flumpy.from_numpy(np.ascontiguousarray(d_np, dtype=np.float64))
    b.reshape(flex.grid(block.size, n_params))
    d_sparse.assign_block(b, 0, 0)

    # Compare ScalingTarget vs ScalingTargetGPU
    block.derivatives = d_sparse
    f_cpu, g_cpu = ScalingTarget.compute_functional_gradients(block)

    block.derivatives = d_sparse
    f_gpu, g_gpu = ScalingTargetGPU.compute_functional_gradients(block)
    assert abs(f_gpu - f_cpu) / max(abs(f_cpu), 1.0) < 1e-13
    np.testing.assert_allclose(
        flumpy.to_numpy(g_gpu), flumpy.to_numpy(g_cpu), rtol=1e-12, atol=1e-10
    )

    # Compare Jacobians
    block.derivatives = d_sparse
    r_cpu, j_cpu, w_cpu = ScalingTarget.compute_residuals_and_gradients(block)
    block.derivatives = d_sparse
    r_gpu, j_gpu, w_gpu = ScalingTargetGPU.compute_residuals_and_dense_jacobian_gpu(
        block
    )
    j_cpu_np = np.asarray(
        flumpy.to_numpy(j_cpu.as_dense_matrix()), dtype=np.float64
    ).reshape(block.size, n_params)
    np.testing.assert_allclose(
        r_gpu.cpu().numpy(), flumpy.to_numpy(r_cpu), rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(
        j_gpu.cpu().numpy(), j_cpu_np, rtol=1e-12, atol=1e-12
    )

    # Compare ScalingTargetFixedIH vs ScalingTargetFixedIHGPU
    block.derivatives = d_sparse
    f_fix_cpu, g_fix_cpu = ScalingTargetFixedIH.compute_functional_gradients(block)
    block.derivatives = d_sparse
    f_fix_gpu, g_fix_gpu = ScalingTargetFixedIHGPU.compute_functional_gradients(block)
    assert abs(f_fix_gpu - f_fix_cpu) / max(abs(f_fix_cpu), 1.0) < 1e-13
    np.testing.assert_allclose(
        flumpy.to_numpy(g_fix_gpu), flumpy.to_numpy(g_fix_cpu), rtol=1e-12, atol=1e-10
    )


def test_outlier_rejection_gpu_vs_cpu():
    exp = _make_synthetic_experiment()
    refl = _make_synthetic_scaling_reflections(n_groups=60, mult=5, seed=99)
    # Inject several strong outliers
    refl["intensity"][5] *= 8.0
    refl["intensity"][22] *= 10.0
    refl["intensity"][47] *= 0.01
    refl["intensity"][103] *= 12.0

    for method in ("simple", "standard"):
        r_cpu = reject_outliers(copy.deepcopy(refl), exp, method=method, zmax=6.0)
        r_gpu = reject_outliers_gpu(copy.deepcopy(refl), exp, method=method, zmax=6.0)
        flags_cpu = flumpy.to_numpy(
            r_cpu.get_flags(r_cpu.flags.outlier_in_scaling)
        )
        flags_gpu = flumpy.to_numpy(
            r_gpu.get_flags(r_gpu.flags.outlier_in_scaling)
        )
        np.testing.assert_array_equal(flags_gpu, flags_cpu)
        assert np.count_nonzero(flags_gpu) >= 3

    # Targeted outlier rejection
    target_refl = _make_synthetic_scaling_reflections(n_groups=60, mult=5, seed=99)
    ih_unscaled = IhTable([refl], exp.crystal.get_space_group(), nblocks=1)
    ih_target = IhTable([target_refl], exp.crystal.get_space_group(), nblocks=1)
    out_cpu = determine_outlier_index_arrays(
        ih_unscaled, method="target", zmax=6.0, target=ih_target
    )[0]
    out_gpu = determine_outlier_index_arrays_gpu(
        ih_unscaled, method="target", zmax=6.0, target=ih_target
    )[0]
    np.testing.assert_array_equal(out_gpu, out_cpu)


def test_combine_intensities_and_error_model_gpu_vs_cpu():
    exp = _make_synthetic_experiment()
    refl = _make_synthetic_scaling_reflections(n_groups=100, mult=4, seed=17)

    for imid in (0, 1, 250.0):
        i_cpu, v_cpu = combine_intensities(copy.deepcopy(refl), imid)
        i_gpu, v_gpu = combine_intensities_gpu(copy.deepcopy(refl), imid)
        np.testing.assert_allclose(
            flumpy.to_numpy(i_gpu), flumpy.to_numpy(i_cpu), rtol=1e-13, atol=1e-13
        )
        np.testing.assert_allclose(
            flumpy.to_numpy(v_gpu), flumpy.to_numpy(v_cpu), rtol=1e-13, atol=1e-13
        )

    # Error model refinement (both individual and regression)
    ih_table = IhTable(
        [refl],
        exp.crystal.get_space_group(),
        nblocks=1,
        additional_cols=["partiality"],
    )
    for min_mode in ("individual", "regression"):
        em_params = em_phil_scope.extract().basic
        em_params.minimisation = min_mode
        em_params.min_Ih = 10.0
        m_cpu = BasicErrorModel(basic_params=em_params)
        m_gpu = BasicErrorModel(basic_params=em_params)
        run_error_model_refinement(m_cpu, copy.deepcopy(ih_table))
        run_error_model_refinement_gpu(m_gpu, copy.deepcopy(ih_table))
        np.testing.assert_allclose(
            m_gpu.parameters, m_cpu.parameters, rtol=1e-6, atol=1e-6
        )


@pytest.mark.skipif(not os.path.isdir(I19_DIR), reason="i19 test data not found")
def test_end_to_end_scale_i19_models_and_engines(tmp_path):
    """Test dials_gpu.scale vs dials.scale on real i19 data across models and engines."""
    from dials.command_line.dials_import import run as import_run
    from dials_gpu.command_line.find_spots import run as find_spots_run
    from dials_gpu.command_line.index import run as index_run
    from dials_gpu.command_line.integrate import run as integrate_run

    nxs_files = sorted(glob.glob(os.path.join(I19_DIR, "*.nxs")))
    assert len(nxs_files) > 0
    nxs = nxs_files[0]

    imp_expt = str(tmp_path / "imported.expt")
    strong_refl = str(tmp_path / "strong.refl")
    idx_expt = str(tmp_path / "indexed.expt")
    idx_refl = str(tmp_path / "indexed.refl")
    int_expt = str(tmp_path / "integrated.expt")
    int_refl = str(tmp_path / "integrated.refl")

    import_run([nxs, f"output.experiments={imp_expt}"])
    find_spots_run([imp_expt, f"output.reflections={strong_refl}", "nproc=1"])
    index_run(
        [
            imp_expt,
            strong_refl,
            f"output.experiments={idx_expt}",
            f"output.reflections={idx_refl}",
        ]
    )
    integrate_run(
        [
            idx_expt,
            idx_refl,
            f"output.experiments={int_expt}",
            f"output.reflections={int_refl}",
            "profile.fitting=False",
        ]
    )

    from dxtbx.model import ExperimentList
    from dxtbx.serialize import load
    from cctbx import sgtbx

    expts_orig = load.experiment_list(int_expt, check_format=False)
    # Assign P222 symmetry so a single 25-image sweep has symmetry-equivalent groups
    expts_orig[0].crystal.set_space_group(sgtbx.space_group_info("P222").group())
    refls_orig = [flex.reflection_table.from_file(int_refl)]

    # 1. Single-dataset models and engines
    for model_type in ("physical", "dose_decay"):
        for engine in ("SimpleLBFGS", "GaussNewton", "LevMar"):
            if model_type == "dose_decay" and engine != "LevMar":
                continue
            params = scale_phil_scope.extract()
            params.model = model_type
            params.scaling_refinery.engine = engine
            params.reflection_selection.method = "use_all"
            params.scaling_options.nproc = 1
            params.output.html = None

            expt_cpu, refl_cpu = cpu_run_scaling(
                copy.deepcopy(params),
                copy.deepcopy(expts_orig),
                [copy.deepcopy(refls_orig[0])],
            )
            expt_gpu, refl_gpu = gpu_run_scaling(
                copy.deepcopy(params),
                copy.deepcopy(expts_orig),
                [copy.deepcopy(refls_orig[0])],
            )

            sf_cpu = flumpy.to_numpy(refl_cpu["inverse_scale_factor"])
            sf_gpu = flumpy.to_numpy(refl_gpu["inverse_scale_factor"])
            tol = 2e-3 if engine == "SimpleLBFGS" else 1e-4
            np.testing.assert_allclose(
                sf_gpu,
                sf_cpu,
                rtol=tol,
                atol=tol,
                err_msg=f"Single-dataset mismatch for model={model_type}, engine={engine}",
            )

    # 2. Multi-dataset models (physical, KB, dose_decay) and engines
    e1 = copy.deepcopy(expts_orig[0])
    e2 = copy.deepcopy(expts_orig[0])
    e1.identifier = ""
    e2.identifier = ""
    expts_multi = ExperimentList([e1, e2])
    r1 = copy.deepcopy(refls_orig[0])
    r2 = copy.deepcopy(refls_orig[0])
    for k in list(r1.experiment_identifiers().keys()):
        del r1.experiment_identifiers()[k]
    for k in list(r2.experiment_identifiers().keys()):
        del r2.experiment_identifiers()[k]
    r2["intensity.sum.value"] = r2["intensity.sum.value"] * 1.35
    r2["intensity.sum.variance"] = r2["intensity.sum.variance"] * (1.35**2)
    refls_multi = [r1, r2]

    for model_type in ("physical", "KB", "dose_decay"):
        for engine in ("SimpleLBFGS", "LevMar"):
            if model_type != "physical" and engine != "SimpleLBFGS":
                continue
            params = scale_phil_scope.extract()
            params.model = model_type
            params.scaling_refinery.engine = engine
            params.reflection_selection.method = "use_all"
            params.scaling_options.nproc = 1
            params.output.html = None

            expt_cpu, refl_cpu = cpu_run_scaling(
                copy.deepcopy(params),
                copy.deepcopy(expts_multi),
                [copy.deepcopy(t) for t in refls_multi],
            )
            expt_gpu, refl_gpu = gpu_run_scaling(
                copy.deepcopy(params),
                copy.deepcopy(expts_multi),
                [copy.deepcopy(t) for t in refls_multi],
            )

            sf_cpu = flumpy.to_numpy(refl_cpu["inverse_scale_factor"])
            sf_gpu = flumpy.to_numpy(refl_gpu["inverse_scale_factor"])
            n_single = refls_orig[0].size()
            valid = (sf_cpu[:n_single] > 0) & (sf_cpu[n_single:] > 0)
            ratio_cpu = sf_cpu[n_single:][valid] / sf_cpu[:n_single][valid]
            ratio_gpu = sf_gpu[n_single:][valid] / sf_gpu[:n_single][valid]
            np.testing.assert_allclose(
                ratio_gpu,
                ratio_cpu,
                rtol=5e-3,
                atol=5e-3,
                err_msg=f"Multi-dataset relative ratio mismatch for model={model_type}, engine={engine}",
            )
            np.testing.assert_allclose(
                sf_gpu,
                sf_cpu,
                rtol=0.2,
                atol=0.2,
                err_msg=f"Multi-dataset mismatch for model={model_type}, engine={engine}",
            )

    # Test CLI invocation of dials_gpu.scale
    scaled_expt = str(tmp_path / "scaled.expt")
    scaled_refl = str(tmp_path / "scaled.refl")
    cli_bin = os.path.join(os.path.dirname(sys.executable), "dials_gpu.scale")
    if not os.path.isfile(cli_bin):
        cli_bin = "/home/davidm/diffraction/dials/conda_base/bin/dials_gpu.scale"
    res = subprocess.run(
        [
            cli_bin,
            int_expt,
            int_refl,
            f"output.experiments={scaled_expt}",
            f"output.reflections={scaled_refl}",
            "output.html=None",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.returncode == 0
    assert os.path.isfile(scaled_expt)
    assert os.path.isfile(scaled_refl)
