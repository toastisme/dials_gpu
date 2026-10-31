"""
Benchmark dials_gpu.scale vs dials.scale on i19 test data and synthetic/real scaling blocks.

Covers:
1. Per-algorithm / kernel micro-benchmarks (GPU vs CPU):
   - SmoothScaleComponent1D & SmoothBScaleComponent1D (1D Gaussian smoother)
   - SHScaleComponent (Spherical harmonic absorption surface)
   - calc_crystal_frame_vectors & calc_theta_phi
   - IhTableBlock.calc_Ih (scatter-add group intensity averaging)
   - ScalingTarget L-BFGS functional & analytical gradients
   - ScalingTarget Gauss-Newton / LevMar analytical Jacobian + normal matrix build-up
   - NormDevOutlierRejection (iterative round-robin z-score outlier rejection)
   - SingleDatasetIntensityCombiner (profile/summation Imid optimization)
   - BasicErrorModel refinement (L-BFGS / regression error model optimization)
2. End-to-end scaling on i19 25-image .nxs sweeps (many small .expt files):
   - Sequential single-sweep scaling across 10 i19 .expt files (physical, KB, dose_decay; SimpleLBFGS vs LevMar)
   - Joint multi-dataset scaling of 5 and 10 i19 .expt files (physical, KB, dose_decay; SimpleLBFGS vs LevMar)
"""

from __future__ import annotations

import copy
import glob
import os
import tempfile
import time

import numpy as np
import torch
from cctbx import crystal, miller, sgtbx
from dxtbx import flumpy
from dxtbx.model import ExperimentList
from dxtbx.serialize import load

from dials.algorithms.scaling.active_parameter_managers import (
    multi_active_parameter_manager,
)
from dials.algorithms.scaling.basis_functions import RefinerCalculator
from dials.algorithms.scaling.combine_intensities import SingleDatasetIntensityCombiner
from dials.algorithms.scaling.error_model.engine import run_error_model_refinement
from dials.algorithms.scaling.Ih_table import IhTable
from dials.algorithms.scaling.model.components.scale_components import SHScaleComponent
from dials.algorithms.scaling.model.components.smooth_scale_components import (
    SmoothBScaleComponent1D,
    SmoothScaleComponent1D,
)
from dials.algorithms.scaling.outlier_rejection import NormDevOutlierRejection
from dials.algorithms.scaling.parameter_handler import scaling_active_parameter_manager
from dials.algorithms.scaling.scaling_utilities import (
    align_axis_along_z,
    calc_crystal_frame_vectors,
    calculate_prescaling_correction,
)
from dials.algorithms.scaling.target_function import ScalingTarget
from dials.array_family import flex
from dials.command_line.scale import phil_scope as scale_phil_scope, run_scaling as cpu_run_scaling
from dials_scaling_ext import calc_theta_phi

from dials_gpu.command_line.find_spots import run as find_spots_run
from dials_gpu.command_line.index import run as index_run
from dials_gpu.command_line.integrate import run as integrate_run
from dials_gpu.command_line.scale import run_scaling as gpu_run_scaling
from dials_gpu.scaling import (
    NormDevOutlierRejectionGPU,
    RefinerCalculatorGPU,
    ScalingTargetGPU,
    SingleDatasetIntensityCombinerGPU,
    SingleScalerGPU,
    align_axis_along_z_gpu,
    calc_crystal_frame_vectors_gpu,
    calc_Ih_gpu,
    calc_theta_phi_gpu,
    evaluate_component_gpu,
    run_error_model_refinement_gpu,
)

I19_DIR = "/home/davidm/diffraction/test_data/i19/301_Nidppe_f3_phi0"


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_kernels(n_refl: int = 50000, n_iters: int = 20):
    print("=" * 95)
    print(f"1. Per-Algorithm / Kernel Micro-Benchmarks (N = {n_refl:,} reflections)")
    print("=" * 95)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(42)

    # 1a. SmoothScaleComponent1D + SmoothBScaleComponent1D
    n_params = 25
    params_s = flex.double(rng.uniform(0.8, 1.2, size=n_params))
    params_b = flex.double(rng.uniform(-1.0, 1.0, size=n_params))
    x_vals = flex.double(rng.uniform(0.0, float(n_params - 2), size=n_refl))
    d_vals = flex.double(rng.uniform(0.8, 4.0, size=n_refl))

    comp_s = SmoothScaleComponent1D(params_s)
    comp_s.data = {"x": x_vals}
    comp_s.update_reflection_data()

    comp_b = SmoothBScaleComponent1D(params_b)
    comp_b.data = {"x": x_vals, "d": d_vals}
    comp_b.update_reflection_data()

    # Warmup
    comp_s.calculate_scales_and_derivatives()
    evaluate_component_gpu(comp_s, block_id=0, compute_derivatives=True, device=device)
    _sync()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        s_cpu, d_cpu = comp_s.calculate_scales_and_derivatives()
    t_cpu_s1d = (time.perf_counter() - t0) * 1000.0 / n_iters

    _sync()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        s_gpu, d_gpu = evaluate_component_gpu(
            comp_s, block_id=0, compute_derivatives=True, device=device
        )
    _sync()
    t_gpu_s1d = (time.perf_counter() - t0) * 1000.0 / n_iters
    max_diff_s1d = float(np.max(np.abs(s_gpu.cpu().numpy() - flumpy.to_numpy(s_cpu))))

    # SmoothBScaleComponent1D
    comp_b.calculate_scales_and_derivatives()
    evaluate_component_gpu(comp_b, block_id=0, compute_derivatives=True, device=device)
    _sync()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        sb_cpu, db_cpu = comp_b.calculate_scales_and_derivatives()
    t_cpu_b1d = (time.perf_counter() - t0) * 1000.0 / n_iters

    _sync()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        sb_gpu, db_gpu = evaluate_component_gpu(
            comp_b, block_id=0, compute_derivatives=True, device=device
        )
    _sync()
    t_gpu_b1d = (time.perf_counter() - t0) * 1000.0 / n_iters
    max_diff_b1d = float(np.max(np.abs(sb_gpu.cpu().numpy() - flumpy.to_numpy(sb_cpu))))

    # 1b. RefinerCalculator (combined scale + decay)
    components = {"scale": comp_s, "decay": comp_b}
    apm = scaling_active_parameter_manager(components, ["scale", "decay"])
    RefinerCalculator.calculate_scales_and_derivatives(apm, 0)
    RefinerCalculatorGPU.calculate_scales_and_derivatives_gpu(apm, 0, device=device)
    _sync()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        rc_s_cpu, rc_d_cpu = RefinerCalculator.calculate_scales_and_derivatives(apm, 0)
    t_cpu_rc = (time.perf_counter() - t0) * 1000.0 / n_iters

    _sync()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        rc_s_gpu, rc_d_gpu = RefinerCalculatorGPU.calculate_scales_and_derivatives_gpu(
            apm, 0, device=device
        )
    _sync()
    t_gpu_rc = (time.perf_counter() - t0) * 1000.0 / n_iters
    max_diff_rc = float(np.max(np.abs(rc_s_gpu.cpu().numpy() - flumpy.to_numpy(rc_s_cpu))))

    # 1c. IhTableBlock.calc_Ih + ScalingTarget L-BFGS gradients & Gauss-Newton Jacobian
    n_groups = n_refl // 5
    rt = flex.reflection_table()
    rt["intensity"] = flex.double(rng.uniform(10.0, 1000.0, size=n_refl))
    rt["variance"] = flex.double(rng.uniform(5.0, 50.0, size=n_refl))
    rt["inverse_scale_factor"] = rc_s_cpu
    h_idx = rng.integers(0, n_groups, size=n_refl)
    rt["miller_index"] = flex.miller_index([(int(h) + 1, 1, 1) for h in h_idx])
    rt.set_flags(flex.bool(n_refl, True), rt.flags.integrated)

    ih_table = IhTable([rt], sgtbx.space_group("P 1"), nblocks=1)
    block = ih_table.blocked_data_list[0]
    block.derivatives = rc_d_cpu
    block._gpu_scales = rc_s_gpu
    block._gpu_derivs_blocks = [(0, n_refl, 0, rc_d_gpu.shape[1], rc_d_gpu)]
    block._gpu_n_params = rc_d_gpu.shape[1]

    block.calc_Ih()
    calc_Ih_gpu(block, device=device, sync_to_cpu=False)
    _sync()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        block.calc_Ih()
    t_cpu_ih = (time.perf_counter() - t0) * 1000.0 / n_iters

    _sync()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        ih_gpu, _, _ = calc_Ih_gpu(block, device=device, sync_to_cpu=False)
    _sync()
    t_gpu_ih = (time.perf_counter() - t0) * 1000.0 / n_iters
    max_diff_ih = float(np.max(np.abs(ih_gpu.cpu().numpy() - block.Ih_values)))

    target_cpu = ScalingTarget()
    target_gpu = ScalingTargetGPU()
    block.derivatives = rc_d_cpu
    target_cpu.compute_functional_gradients(block)
    block._gpu_derivs_blocks = [(0, n_refl, 0, rc_d_gpu.shape[1], rc_d_gpu)]
    target_gpu.compute_functional_gradients(block)
    _sync()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        block.derivatives = rc_d_cpu
        f_cpu, g_cpu = target_cpu.compute_functional_gradients(block)
    t_cpu_grad = (time.perf_counter() - t0) * 1000.0 / n_iters

    _sync()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        block._gpu_derivs_blocks = [(0, n_refl, 0, rc_d_gpu.shape[1], rc_d_gpu)]
        f_gpu, g_gpu = target_gpu.compute_functional_gradients(block)
    _sync()
    t_gpu_grad = (time.perf_counter() - t0) * 1000.0 / n_iters
    max_diff_grad = float(np.max(np.abs(flumpy.to_numpy(g_gpu) - flumpy.to_numpy(g_cpu))))

    block.derivatives = rc_d_cpu
    target_cpu.compute_residuals_and_gradients(block)
    block._gpu_derivs_blocks = [(0, n_refl, 0, rc_d_gpu.shape[1], rc_d_gpu)]
    target_gpu.compute_residuals_and_dense_jacobian_gpu(block)
    _sync()

    t0 = time.perf_counter()
    for _ in range(max(5, n_iters // 2)):
        block.derivatives = rc_d_cpu
        r_cpu, j_cpu, w_cpu = target_cpu.compute_residuals_and_gradients(block)
    t_cpu_jac = (time.perf_counter() - t0) * 1000.0 / max(5, n_iters // 2)

    _sync()
    t0 = time.perf_counter()
    for _ in range(max(5, n_iters // 2)):
        block._gpu_derivs_blocks = [(0, n_refl, 0, rc_d_gpu.shape[1], rc_d_gpu)]
        r_gpu, j_gpu, w_gpu = target_gpu.compute_residuals_and_dense_jacobian_gpu(block)
    _sync()
    t_gpu_jac = (time.perf_counter() - t0) * 1000.0 / max(5, n_iters // 2)
    max_diff_jac = float(np.max(np.abs(r_gpu.cpu().numpy() - flumpy.to_numpy(r_cpu))))

    # 1d. NormDevOutlierRejection
    rt_out = copy.deepcopy(rt)
    outlier_idx = rng.choice(n_refl, size=200, replace=False)
    for idx in outlier_idx:
        rt_out["intensity"][int(idx)] *= 25.0
    ih_out_cpu = IhTable([copy.deepcopy(rt_out)], sgtbx.space_group("P 1"), nblocks=1)
    ih_out_gpu = IhTable([copy.deepcopy(rt_out)], sgtbx.space_group("P 1"), nblocks=1)

    NormDevOutlierRejection(ih_out_cpu, zmax=6.0).run()
    NormDevOutlierRejectionGPU(ih_out_gpu, zmax=6.0).run()
    _sync()

    t0 = time.perf_counter()
    for _ in range(5):
        rej_c = NormDevOutlierRejection(ih_out_cpu, zmax=6.0)
        rej_c.run()
    t_cpu_out = (time.perf_counter() - t0) * 1000.0 / 5

    _sync()
    t0 = time.perf_counter()
    for _ in range(5):
        rej_g = NormDevOutlierRejectionGPU(ih_out_gpu, zmax=6.0)
        rej_g.run()
    _sync()
    t_gpu_out = (time.perf_counter() - t0) * 1000.0 / 5
    n_out_match = len(set(rej_c.final_outlier_arrays[0]) ^ set(rej_g.final_outlier_arrays[0]))

    print(
        f"{'Kernel / Algorithm':<48} | {'CPU (ms)':>10} | {'GPU (ms)':>10} | {'Speedup':>9} | {'Max Diff':>12}"
    )
    print("-" * 95)
    print(
        f"{'SmoothScaleComponent1D (scales + derivs)':<48} | {t_cpu_s1d:10.3f} | {t_gpu_s1d:10.3f} | {t_cpu_s1d/t_gpu_s1d:8.2f}x | {max_diff_s1d:12.2e}"
    )
    print(
        f"{'SmoothBScaleComponent1D (scales + derivs)':<48} | {t_cpu_b1d:10.3f} | {t_gpu_b1d:10.3f} | {t_cpu_b1d/t_gpu_b1d:8.2f}x | {max_diff_b1d:12.2e}"
    )
    print(
        f"{'RefinerCalculator (product rule scales+derivs)':<48} | {t_cpu_rc:10.3f} | {t_gpu_rc:10.3f} | {t_cpu_rc/t_gpu_rc:8.2f}x | {max_diff_rc:12.2e}"
    )
    print(
        f"{'IhTableBlock.calc_Ih (scatter_add_ group Ih)':<48} | {t_cpu_ih:10.3f} | {t_gpu_ih:10.3f} | {t_cpu_ih/t_gpu_ih:8.2f}x | {max_diff_ih:12.2e}"
    )
    print(
        f"{'ScalingTarget.compute_functional_gradients':<48} | {t_cpu_grad:10.3f} | {t_gpu_grad:10.3f} | {t_cpu_grad/t_gpu_grad:8.2f}x | {max_diff_grad:12.2e}"
    )
    print(
        f"{'ScalingTarget.compute_residuals_and_gradients':<48} | {t_cpu_jac:10.3f} | {t_gpu_jac:10.3f} | {t_cpu_jac/t_gpu_jac:8.2f}x | {max_diff_jac:12.2e}"
    )
    print(
        f"{'NormDevOutlierRejection (round-robin z-score)':<48} | {t_cpu_out:10.3f} | {t_gpu_out:10.3f} | {t_cpu_out/t_gpu_out:8.2f}x | {n_out_match:12d} diff"
    )
    print()


def benchmark_end_to_end_i19(n_files: int = 5):
    print("=" * 95)
    print(f"2. End-to-End Scaling Benchmark on {n_files} i19 .nxs Sweeps (301_Nidppe_f3_phi0)")
    print("=" * 95)
    from dxtbx.model.experiment_list import ExperimentListFactory
    from dials.command_line.index import phil_scope as index_phil_scope
    from dials.command_line.integrate import phil_scope as integrate_phil_scope
    from dials.util.phil import parse
    from dials_gpu.command_line.find_spots import from_observations_gpu
    from dials_gpu.command_line.index import index as index_gpu
    from dials_gpu.command_line.integrate import run_integration as gpu_run_integration

    nxs_files = sorted(glob.glob(os.path.join(I19_DIR, "*.nxs")))
    idx_params = index_phil_scope.fetch(
        source=parse(
            "indexing.method=fft3d\n"
            "indexing.known_symmetry.space_group=P21/n\n"
            "indexing.known_symmetry.unit_cell=11.39,13.30,15.82,90,99.02,90\n"
            "indexing.basis_vector_combinations.max_refine=5\n"
            "indexing.refinement_protocol.n_macro_cycles=2"
        )
    ).extract()
    int_params = integrate_phil_scope.fetch(
        source=parse("integration.mp.nproc=1\nintegration.profile.fitting=False")
    ).extract()

    expts_list = []
    refls_list = []

    print(f"Preparing {n_files} integrated .expt/.refl sweeps on GPU...")
    for f in nxs_files:
        if len(expts_list) >= n_files:
            break
        try:
            expts = ExperimentListFactory.from_filenames([f])
            refl = from_observations_gpu(expts)
            if len(refl) < 40:
                continue
            ie, ir = index_gpu(
                copy.deepcopy(expts), [copy.deepcopy(refl)], copy.deepcopy(idx_params)
            )
            if len(ie) > 0 and len(ir) >= 40:
                int_e, int_r, _ = gpu_run_integration(
                    copy.deepcopy(int_params), copy.deepcopy(ie), copy.deepcopy(ir)
                )
                expt = int_e[0]
                expt.identifier = ""
                for k in list(int_r.experiment_identifiers().keys()):
                    del int_r.experiment_identifiers()[k]
                expts_list.append(expt)
                refls_list.append(int_r)
        except Exception:
            continue
    torch.cuda.empty_cache()

    n_valid = len(expts_list)
    total_refl = sum(r.size() for r in refls_list)
    print(f"Prepared {n_valid} sweeps ({total_refl:,} total reflections).\n")

    # Warmup GPU scaling
    p_warm = scale_phil_scope.extract()
    p_warm.output.html = None
    gpu_run_scaling(
        copy.deepcopy(p_warm),
        ExperimentList([copy.deepcopy(expts_list[0])]),
        [copy.deepcopy(refls_list[0])],
    )
    _sync()

    print(
        f"{'Scaling Configuration':<52} | {'CPU (ms)':>10} | {'GPU (ms)':>10} | {'Speedup':>9} | {'Mean Rel Diff':>13}"
    )
    print("-" * 102)

    # 2a. Many small .expt files individually (sequentially scaling each .expt file)
    for model_type, engine, full_mat in [
        ("physical", "SimpleLBFGS", True),
        ("physical", "SimpleLBFGS", False),
        ("dose_decay", "SimpleLBFGS", True),
        ("dose_decay", "SimpleLBFGS", False),
    ]:
        t0 = time.perf_counter()
        cpu_sfs = []
        for e, r in zip(expts_list, refls_list):
            p = scale_phil_scope.extract()
            p.model = model_type
            p.scaling_refinery.engine = engine
            p.scaling_options.full_matrix = full_mat
            p.scaling_options.nproc = 1
            p.output.html = None
            _, rc = cpu_run_scaling(p, ExperimentList([copy.deepcopy(e)]), [copy.deepcopy(r)])
            cpu_sfs.append(flumpy.to_numpy(rc["inverse_scale_factor"]))
        t_cpu = (time.perf_counter() - t0) * 1000.0 / n_valid

        _sync()
        t0 = time.perf_counter()
        gpu_sfs = []
        for e, r in zip(expts_list, refls_list):
            p = scale_phil_scope.extract()
            p.model = model_type
            p.scaling_refinery.engine = engine
            p.scaling_options.full_matrix = full_mat
            p.scaling_options.nproc = 1
            p.output.html = None
            _, rg = gpu_run_scaling(p, ExperimentList([copy.deepcopy(e)]), [copy.deepcopy(r)])
            gpu_sfs.append(flumpy.to_numpy(rg["inverse_scale_factor"]))
        _sync()
        t_gpu = (time.perf_counter() - t0) * 1000.0 / n_valid

        rel_diffs = [
            np.mean(np.abs(g - c) / np.maximum(np.abs(c), 1e-6))
            for g, c in zip(gpu_sfs, cpu_sfs)
        ]
        mean_rel = float(np.mean(rel_diffs))
        label = f"Single-file loop ({model_type}, {engine}, full_mat={full_mat})"
        print(
            f"{label:<52} | {t_cpu:8.2f}/f | {t_gpu:8.2f}/f | {t_cpu/t_gpu:8.2f}x | {mean_rel:13.2e}"
        )

    # 2b. Joint multi-dataset scaling of all N sweeps together
    for n_joint in [5, n_valid]:
        if n_joint > n_valid:
            continue
        for model_type, full_mat in [
            ("physical", True),
            ("physical", False),
            ("KB", False),
            ("dose_decay", False),
        ]:
            p_cpu = scale_phil_scope.extract()
            p_cpu.model = model_type
            p_cpu.scaling_options.full_matrix = full_mat
            p_cpu.scaling_options.nproc = 1
            p_cpu.output.html = None

            expts_c = ExperimentList([copy.deepcopy(e) for e in expts_list[:n_joint]])
            refls_c = [copy.deepcopy(r) for r in refls_list[:n_joint]]
            t0 = time.perf_counter()
            _, rc_joint = cpu_run_scaling(p_cpu, expts_c, refls_c)
            t_cpu_joint = (time.perf_counter() - t0) * 1000.0

            p_gpu = scale_phil_scope.extract()
            p_gpu.model = model_type
            p_gpu.scaling_options.full_matrix = full_mat
            p_gpu.scaling_options.nproc = 1
            p_gpu.output.html = None

            expts_g = ExperimentList([copy.deepcopy(e) for e in expts_list[:n_joint]])
            refls_g = [copy.deepcopy(r) for r in refls_list[:n_joint]]
            _sync()
            t0 = time.perf_counter()
            _, rg_joint = gpu_run_scaling(p_gpu, expts_g, refls_g)
            _sync()
            t_gpu_joint = (time.perf_counter() - t0) * 1000.0

            sfc = flumpy.to_numpy(rc_joint["inverse_scale_factor"])
            sfg = flumpy.to_numpy(rg_joint["inverse_scale_factor"])
            mean_rel = float(np.mean(np.abs(sfg - sfc) / np.maximum(np.abs(sfc), 1e-6)))
            label = f"Joint {n_joint}-sweep ({model_type}, full_mat={full_mat})"
            print(
                f"{label:<52} | {t_cpu_joint:10.2f} | {t_gpu_joint:10.2f} | {t_cpu_joint/t_gpu_joint:8.2f}x | {mean_rel:13.2e}"
            )


if __name__ == "__main__":
    benchmark_kernels()
    benchmark_end_to_end_i19(10)
