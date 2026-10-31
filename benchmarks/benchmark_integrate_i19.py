from __future__ import annotations

import copy
import glob
import pathlib
import time

import numpy as np
import torch
from dxtbx.model.experiment_list import ExperimentListFactory

from dials.algorithms.integration.fit import ProfileFitter as CPUProfileFitter
from dials.algorithms.integration.sum import integrate_by_summation as cpu_sum
from dials.array_family import flex
from dials.command_line.index import phil_scope as index_phil_scope
from dials.command_line.integrate import (
    phil_scope as integrate_phil_scope,
    run_integration as cpu_run_integration,
)
from dials.extensions import Background, Centroid
from dials.util.phil import parse
from dials_gpu.command_line.find_spots import from_observations_gpu
from dials_gpu.command_line.index import index as index_gpu
from dials_gpu.command_line.integrate import run_integration as gpu_run_integration
from dials_gpu.integration.background import GLMBackgroundGPU, SimpleBackgroundGPU
from dials_gpu.integration.centroid import SimpleCentroidGPU, compute_centroid_gpu
from dials_gpu.integration.fit import ProfileFitterGPU, compute_corrections_gpu
from dials_gpu.integration.profile import ProfileModelFactoryGPU
from dials_gpu.integration.summation import compute_summed_intensity_gpu
from dials.algorithms.profile_model.factory import ProfileModelFactory as CPUProfileModelFactory


I19_DIR = pathlib.Path("/home/davidm/diffraction/test_data/i19/301_Nidppe_f3_phi0")


def prepare_i19_inputs(n_files: int = 5):
    nxs_files = sorted(glob.glob(str(I19_DIR / "*.nxs")))
    selected = []
    idx_params = index_phil_scope.fetch(source=parse("indexing.method=fft3d")).extract()
    for f in nxs_files:
        if len(selected) >= n_files:
            break
        try:
            expts = ExperimentListFactory.from_filenames([f])
            refl = from_observations_gpu(expts)
            if len(refl) < 40:
                continue
            ie, ir = index_gpu(copy.deepcopy(expts), [copy.deepcopy(refl)], copy.deepcopy(idx_params))
            if len(ie) > 0 and len(ir) >= 40:
                selected.append((pathlib.Path(f).name, ie, ir))
        except Exception:
            continue
    torch.cuda.empty_cache()
    return selected


def benchmark_kernels(expts, refl):
    print("\n" + "=" * 80)
    print("1. Per-Algorithm / Kernel Micro-Benchmarks on i19 Shoeboxes")
    print("=" * 80)

    params = integrate_phil_scope.fetch(
        source=parse("integration.mp.nproc=1\nintegration.debug.output=True\nintegration.debug.separate_files=False")
    ).extract()

    # Run once with debug.output=True to obtain populated shoeboxes on all predicted reflections
    expts_with_prof, refl_with_sbox, _ = gpu_run_integration(
        copy.deepcopy(params), copy.deepcopy(expts), copy.deepcopy(refl)
    )
    print(f"Extracted {len(refl_with_sbox)} predicted reflection shoeboxes from {expts[0].imageset.get_path(0)}")

    # 1a. ProfileModelFactory (sigma_b, sigma_m, zeta, partiality)
    from dials.command_line.integrate import process_reference, filter_reference_pixels
    prof_ref, _ = process_reference(copy.deepcopy(refl))
    prof_ref = filter_reference_pixels(prof_ref, copy.deepcopy(expts))
    pred = flex.reflection_table.from_predictions_multi(copy.deepcopy(expts))
    _, prof_ref, _ = pred.match_with_reference(prof_ref)

    t0 = time.perf_counter()
    for _ in range(10):
        CPUProfileModelFactory.create(params, copy.deepcopy(expts), copy.deepcopy(prof_ref))
    t_prof_cpu = (time.perf_counter() - t0) / 10.0 * 1000.0

    # Warmup GPU
    ProfileModelFactoryGPU.create(params, copy.deepcopy(expts), copy.deepcopy(prof_ref))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        ProfileModelFactoryGPU.create(params, copy.deepcopy(expts), copy.deepcopy(prof_ref))
    torch.cuda.synchronize()
    t_prof_gpu = (time.perf_counter() - t0) / 10.0 * 1000.0
    print(f"ProfileModelFactory.create : CPU = {t_prof_cpu:7.2f} ms | GPU = {t_prof_gpu:7.2f} ms | Speedup = {t_prof_cpu / t_prof_gpu:5.2f}x")

    # Replicate reflection table to ~3,000 shoeboxes to measure steady-state kernel throughput
    big_refl = flex.reflection_table()
    for _ in range(4):
        big_refl.extend(copy.deepcopy(refl_with_sbox))
    print(f"Benchmarking shoebox kernels on N = {len(big_refl)} 3D shoeboxes:")

    # 1b. GLMBackground (constant3d)
    glm_params = integrate_phil_scope.fetch(
        source=parse("integration.background.algorithm=glm\nintegration.background.glm.model.algorithm=constant3d")
    ).extract()
    cpu_glm = Background.load("glm")(glm_params, expts_with_prof)
    gpu_glm = GLMBackgroundGPU(glm_params, expts_with_prof)

    r_cpu = copy.deepcopy(big_refl)
    t0 = time.perf_counter()
    cpu_glm.compute_background(r_cpu)
    t_glm_cpu = (time.perf_counter() - t0) * 1000.0

    r_gpu = copy.deepcopy(big_refl)
    gpu_glm.compute_background(r_gpu)
    torch.cuda.synchronize()
    r_gpu = copy.deepcopy(big_refl)
    t0 = time.perf_counter()
    gpu_glm.compute_background(r_gpu)
    torch.cuda.synchronize()
    t_glm_gpu = (time.perf_counter() - t0) * 1000.0
    max_bg_diff = np.max(np.abs(r_cpu["background.mean"].as_numpy_array() - r_gpu["background.mean"].as_numpy_array()))
    print(f"GLMBackground (constant3d) : CPU = {t_glm_cpu:7.2f} ms | GPU = {t_glm_gpu:7.2f} ms | Speedup = {t_glm_cpu / t_glm_gpu:5.2f}x | max diff = {max_bg_diff:.2e}")

    # 1c. SimpleBackground (tukey + constant3d)
    simp_params = integrate_phil_scope.fetch(
        source=parse(
            "integration.background.algorithm=simple\n"
            "integration.background.simple.outlier.algorithm=tukey\n"
            "integration.background.simple.model.algorithm=constant3d"
        )
    ).extract()
    cpu_simp = Background.load("simple")(simp_params, expts_with_prof)
    gpu_simp = SimpleBackgroundGPU(simp_params, expts_with_prof)

    r_cpu = copy.deepcopy(big_refl)
    t0 = time.perf_counter()
    cpu_simp.compute_background(r_cpu)
    t_simp_cpu = (time.perf_counter() - t0) * 1000.0

    r_gpu = copy.deepcopy(big_refl)
    gpu_simp.compute_background(r_gpu)
    torch.cuda.synchronize()
    r_gpu = copy.deepcopy(big_refl)
    t0 = time.perf_counter()
    gpu_simp.compute_background(r_gpu)
    torch.cuda.synchronize()
    t_simp_gpu = (time.perf_counter() - t0) * 1000.0
    max_simp_diff = np.max(np.abs(r_cpu["background.mean"].as_numpy_array() - r_gpu["background.mean"].as_numpy_array()))
    print(f"SimpleBackground (tukey3d) : CPU = {t_simp_cpu:7.2f} ms | GPU = {t_simp_gpu:7.2f} ms | Speedup = {t_simp_cpu / t_simp_gpu:5.2f}x | max diff = {max_simp_diff:.2e}")

    # 1d. SimpleCentroid
    cpu_cent = Centroid.load("simple")(params, expts_with_prof)
    r_cpu = copy.deepcopy(big_refl)
    t0 = time.perf_counter()
    cpu_cent.compute_centroid(r_cpu)
    t_cent_cpu = (time.perf_counter() - t0) * 1000.0

    r_gpu = copy.deepcopy(big_refl)
    compute_centroid_gpu(r_gpu, expts_with_prof)
    torch.cuda.synchronize()
    r_gpu = copy.deepcopy(big_refl)
    t0 = time.perf_counter()
    compute_centroid_gpu(r_gpu, expts_with_prof)
    torch.cuda.synchronize()
    t_cent_gpu = (time.perf_counter() - t0) * 1000.0
    max_cent_diff = np.max(np.abs(r_cpu["xyzobs.px.value"].as_numpy_array() - r_gpu["xyzobs.px.value"].as_numpy_array()))
    print(f"SimpleCentroid (3D shoebox): CPU = {t_cent_cpu:7.2f} ms | GPU = {t_cent_gpu:7.2f} ms | Speedup = {t_cent_cpu / t_cent_gpu:5.2f}x | max diff = {max_cent_diff:.2e}")

    # 1e. Summation integration
    r_cpu = copy.deepcopy(big_refl)
    t0 = time.perf_counter()
    r_cpu.compute_summed_intensity()
    t_sum_cpu = (time.perf_counter() - t0) * 1000.0

    r_gpu = copy.deepcopy(big_refl)
    compute_summed_intensity_gpu(r_gpu)
    torch.cuda.synchronize()
    r_gpu = copy.deepcopy(big_refl)
    t0 = time.perf_counter()
    compute_summed_intensity_gpu(r_gpu)
    torch.cuda.synchronize()
    t_sum_gpu = (time.perf_counter() - t0) * 1000.0
    max_sum_diff = np.max(np.abs(r_cpu["intensity.sum.value"].as_numpy_array() - r_gpu["intensity.sum.value"].as_numpy_array()))
    print(f"Summation Integration      : CPU = {t_sum_cpu:7.2f} ms | GPU = {t_sum_gpu:7.2f} ms | Speedup = {t_sum_cpu / t_sum_gpu:5.2f}x | max diff = {max_sum_diff:.2e}")

    # 1f. LP + QE Corrections
    r_cpu = copy.deepcopy(big_refl)
    t0 = time.perf_counter()
    for _ in range(10):
        r_cpu.compute_corrections(expts_with_prof)
    t_corr_cpu = (time.perf_counter() - t0) / 10.0 * 1000.0

    r_gpu = copy.deepcopy(big_refl)
    compute_corrections_gpu(r_gpu, expts_with_prof)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        compute_corrections_gpu(r_gpu, expts_with_prof)
    torch.cuda.synchronize()
    t_corr_gpu = (time.perf_counter() - t0) / 10.0 * 1000.0
    max_lp_diff = np.max(np.abs(r_cpu["lp"].as_numpy_array() - r_gpu["lp"].as_numpy_array()))
    print(f"LP + QE Corrections        : CPU = {t_corr_cpu:7.2f} ms | GPU = {t_corr_gpu:7.2f} ms | Speedup = {t_corr_cpu / t_corr_gpu:5.2f}x | max diff = {max_lp_diff:.2e}")


def benchmark_end_to_end(datasets):
    print("\n" + "=" * 80)
    print(f"2. End-to-End Integration across {len(datasets)} i19 25-image .nxs Datasets")
    print("=" * 80)

    configs = [
        ("Default (3D + GLM + Profile Fitting, nproc=4 vs GPU)", "integration.mp.nproc=4", "integration.mp.nproc=1"),
        ("Default (3D + GLM + Profile Fitting, nproc=1 vs GPU)", "integration.mp.nproc=1", "integration.mp.nproc=1"),
        ("Summation Only (3D + GLM, profile.fitting=False, nproc=4 vs GPU)", "integration.mp.nproc=4\nintegration.profile.fitting=False", "integration.mp.nproc=1\nintegration.profile.fitting=False"),
        ("Summation Only (3D + GLM, profile.fitting=False, nproc=1 vs GPU)", "integration.mp.nproc=1\nintegration.profile.fitting=False", "integration.mp.nproc=1\nintegration.profile.fitting=False"),
    ]

    for label, cpu_phil_str, gpu_phil_str in configs:
        cpu_params = integrate_phil_scope.fetch(source=parse(cpu_phil_str)).extract()
        gpu_params = integrate_phil_scope.fetch(source=parse(gpu_phil_str)).extract()

        # Warmup GPU on first dataset
        gpu_run_integration(copy.deepcopy(gpu_params), copy.deepcopy(datasets[0][1]), copy.deepcopy(datasets[0][2]))
        torch.cuda.synchronize()

        cpu_times = []
        gpu_times = []
        n_cpu_tot = 0
        n_gpu_tot = 0

        for name, expts, refl in datasets:
            t0 = time.perf_counter()
            _, r_cpu, _ = cpu_run_integration(
                copy.deepcopy(cpu_params), copy.deepcopy(expts), copy.deepcopy(refl)
            )
            cpu_times.append((time.perf_counter() - t0) * 1000.0)
            n_cpu_tot += len(r_cpu)

            t0 = time.perf_counter()
            _, r_gpu, _ = gpu_run_integration(
                copy.deepcopy(gpu_params), copy.deepcopy(expts), copy.deepcopy(refl)
            )
            torch.cuda.synchronize()
            gpu_times.append((time.perf_counter() - t0) * 1000.0)
            n_gpu_tot += len(r_gpu)

        mean_cpu = np.mean(cpu_times)
        mean_gpu = np.mean(gpu_times)
        print(
            f"{label:64s}: CPU = {mean_cpu:7.1f} ms/file | GPU = {mean_gpu:7.1f} ms/file | "
            f"Speedup = {mean_cpu / mean_gpu:5.2f}x | Refls = {n_cpu_tot}/{n_gpu_tot}"
        )


if __name__ == "__main__":
    print("Preparing indexed i19 datasets...")
    datasets = prepare_i19_inputs(n_files=5)
    print(f"Prepared {len(datasets)} indexed i19 datasets: {[d[0] for d in datasets]}")
    benchmark_kernels(datasets[0][1], datasets[0][2])
    benchmark_end_to_end(datasets)
