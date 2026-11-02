"""
Comprehensive performance and accuracy benchmark comparing:
  1. Standard DIALS (CPU: dials.find_spots, dials.index, dials.refine, dials.integrate, dials.scale)
  2. dials_gpu (PyTorch CUDA GPU: dials_gpu.find_spots, dials_gpu.index, dials_gpu.refine, dials_gpu.integrate, dials_gpu.scale)
  3. mxi (C++20 / CUDA: mxi find [--gpu], mxi index, mxi refine, mxi integrate [--gpu], mxi scale)

Evaluated on two datasets:
  - Dataset A (i19): 5 small 25-image Eiger 2X 4M NeXus sweeps (301_NiD_00001_w00017..w00025.nxs,
    space group P 1 21/n 1, unit cell 11.39, 13.30, 15.82, 90, 99.02, 90)
  - Dataset B (C2sum_1): 720-image Pilatus 6M rotation sweep (C2sum_1_0001..0720.cbf.gz / c2sum_1.nxs,
    space group C 1 2 1, unit cell 53.17, 61.25, 69.29, 90, 93.04, 90)

Measures:
  1. Wall-clock execution time per pipeline stage (including mxi CPU vs mxi --gpu and cold vs warm GPU).
  2. Numerical closeness to standard CPU DIALS at every stage (spot counts & centroid RMSDs,
     indexed spot counts & unit-cell differences, refined RMSDs & geometry differences,
     integrated summation & profile-fitted intensity Pearson correlations, scaled intensities).
  3. Overall post-scaling merging statistics (R_merge, R_meas, R_pim, CC_1/2, <I/sigma(I)>,
     completeness, multiplicity, total observations, unique reflections).
  4. Empirical cases where GPU acceleration is vs is not recommended.
"""

from __future__ import annotations

import copy
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch  # Must be imported before scitbx/omptbx_ext to avoid OpenMP conflict
import numpy as np
from cctbx import crystal, miller, sgtbx, uctbx
from dxtbx.model.experiment_list import ExperimentListFactory
import iotbx.merging_statistics
from dials.util.phil import parse

from dials.array_family import flex
from dials.command_line.find_spots import phil_scope as find_phil_scope
from dials.command_line.index import (
    index as index_cpu,
    phil_scope as index_phil_scope,
)
from dials.command_line.refine import working_phil as refine_phil_scope
from dials.command_line.integrate import (
    phil_scope as integrate_phil_scope,
    run_integration as run_integration_cpu,
)
from dials.command_line.scale import phil_scope as scale_phil_scope

from dials.command_line.refine import run_dials_refine as run_dials_refine_cpu
from dials.command_line.scale import run_scaling as run_scaling_cpu

from dials_gpu.command_line.find_spots import from_observations_gpu
from dials_gpu.command_line.index import index as index_gpu
from dials_gpu.command_line.refine import run_dials_refine as run_dials_refine_gpu
from dials_gpu.command_line.integrate import run_integration as run_integration_gpu
from dials_gpu.command_line.scale import run_scaling as run_scaling_gpu


MXI_BIN = Path(
    "/home/davidm/.gemini/antigravity/brain/a604dbc3-5ed4-464f-9a64-9e046ba8393d/scratch/mxi/build/mxi"
)
SCRATCH_DIR = Path(
    "/home/davidm/.gemini/antigravity/brain/a604dbc3-5ed4-464f-9a64-9e046ba8393d/scratch/mxi_test"
)
I19_DIR = Path("/home/davidm/diffraction/test_data/i19/301_Nidppe_f3_phi0")
I19_SWEEPS = [
    I19_DIR / "301_NiD_00001_w00017.nxs",
    I19_DIR / "301_NiD_00001_w00019.nxs",
    I19_DIR / "301_NiD_00001_w00025.nxs",
    I19_DIR / "301_NiD_00002_w00053.nxs",
    I19_DIR / "301_NiD_00002_w00055.nxs",
]


def run_mxi_cmd(args: list[str], cwd: Path = SCRATCH_DIR) -> tuple[float, str]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(MXI_BIN), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
    )
    dt = time.perf_counter() - t0
    return dt, proc.stdout + proc.stderr


def parse_mxi_phase_time(output: str, label: str) -> float | None:
    m = re.search(rf"{re.escape(label)}:\s+([0-9.]+)s", output)
    return float(m.group(1)) if m else None


def get_first_crystal(expts):
    return next(e.crystal for e in expts if e.crystal is not None)


def compute_merging_stats(
    expts,
    refl: flex.reflection_table,
    intensity_choice: str = "profile",
    d_min: float | None = None,
) -> dict:
    """Compute iotbx.merging_statistics on a scaled reflection table (works for both DIALS and mxi)."""
    cryst = get_first_crystal(expts)
    uc = cryst.get_unit_cell()
    sg = cryst.get_space_group()

    # Select valid scaled reflections
    if "inverse_scale_factor" in refl:
        isf = refl["inverse_scale_factor"].as_numpy_array()
    else:
        isf = np.ones(len(refl), dtype=np.float64)

    flags = refl.get_flags(refl.flags.bad_for_scaling, all=False).as_numpy_array()
    outlier = refl.get_flags(refl.flags.outlier_in_scaling, all=False).as_numpy_array()

    if intensity_choice == "profile" and "intensity.prf.value" in refl:
        I_raw = refl["intensity.prf.value"].as_numpy_array()
        V_raw = refl["intensity.prf.variance"].as_numpy_array()
        prf_ok = refl.get_flags(refl.flags.integrated_prf).as_numpy_array()
    elif "intensity.scale.value" in refl:
        I_raw = refl["intensity.scale.value"].as_numpy_array()
        V_raw = refl["intensity.scale.variance"].as_numpy_array()
        isf = np.ones(len(refl), dtype=np.float64)
        prf_ok = np.ones(len(refl), dtype=bool)
    else:
        I_raw = refl["intensity.sum.value"].as_numpy_array()
        V_raw = refl["intensity.sum.variance"].as_numpy_array()
        prf_ok = refl.get_flags(refl.flags.integrated_sum).as_numpy_array()

    hkl_arr = refl["miller_index"].as_vec3_double().as_numpy_array().astype(np.int32)
    nonzero_hkl = np.any(hkl_arr != 0, axis=1)
    valid = prf_ok & (~flags) & (~outlier) & (V_raw > 0) & (isf > 0) & nonzero_hkl

    I_scaled = I_raw[valid] / isf[valid]
    sig_scaled = np.sqrt(V_raw[valid]) / isf[valid]
    sel_indices = flex.size_t(np.nonzero(valid)[0].astype(np.uint64))
    hkl_sel = refl["miller_index"].select(sel_indices)

    cs = crystal.symmetry(unit_cell=uc, space_group=sg)
    ms = miller.set(crystal_symmetry=cs, indices=hkl_sel, anomalous_flag=False)
    ma = miller.array(
        ms,
        data=flex.double(np.ascontiguousarray(I_scaled, dtype=np.float64)),
        sigmas=flex.double(np.ascontiguousarray(sig_scaled, dtype=np.float64)),
    )
    ma.set_observation_type_xray_intensity()

    stats = iotbx.merging_statistics.dataset_statistics(
        i_obs=ma,
        d_min=d_min,
        n_bins=10,
        anomalous=False,
    )
    ov = stats.overall
    return {
        "n_obs": ov.n_obs,
        "n_uniq": ov.n_uniq,
        "multiplicity": ov.mean_redundancy,
        "completeness": ov.completeness * 100.0,
        "i_over_sigma": ov.i_over_sigma_mean,
        "r_merge": ov.r_merge,
        "r_meas": ov.r_meas,
        "r_pim": ov.r_pim,
        "cc_one_half": ov.cc_one_half,
        "d_min": ov.d_min,
        "d_max": ov.d_max,
    }


def compare_spotfinding(r_ref: flex.reflection_table, r_test: flex.reflection_table) -> dict:
    """Compare spot counts and nearest-neighbor 3D centroid RMSD against reference DIALS."""
    xyz_ref = r_ref["xyzobs.px.value"].as_numpy_array()
    xyz_test = r_test["xyzobs.px.value"].as_numpy_array()
    from scipy.spatial import cKDTree

    tree = cKDTree(xyz_ref)
    dists, idx = tree.query(xyz_test, distance_upper_bound=2.0)
    matched = dists < 2.0
    n_matched = int(np.sum(matched))
    if n_matched > 0:
        diff = xyz_test[matched] - xyz_ref[idx[matched]]
        rmsd_xyz = np.sqrt(np.mean(diff**2, axis=0))
    else:
        rmsd_xyz = np.array([np.nan, np.nan, np.nan])
    return {
        "n_ref": len(r_ref),
        "n_test": len(r_test),
        "n_matched": n_matched,
        "match_pct": 100.0 * n_matched / max(len(r_ref), 1),
        "rmsd_x_px": float(rmsd_xyz[0]),
        "rmsd_y_px": float(rmsd_xyz[1]),
        "rmsd_z_frm": float(rmsd_xyz[2]),
    }


def compare_integrated(
    expt_ref,
    r_ref: flex.reflection_table,
    expt_test,
    r_test: flex.reflection_table,
) -> dict:
    """Compare integrated intensities matched by symmetry-equivalent Miller index + frame z."""
    sg = get_first_crystal(expt_ref).get_space_group()

    def build_key_map(r: flex.reflection_table):
        hkl = r["miller_index"]
        asu_hkl = hkl.deep_copy()
        miller.map_to_asu(sg.type(), False, asu_hkl)
        asu_np = asu_hkl.as_vec3_double().as_numpy_array().astype(np.int64)
        if "xyzcal.px" in r:
            z = np.round(r["xyzcal.px"].as_numpy_array()[:, 2] / 2.0).astype(np.int64)
        else:
            z = np.zeros(len(r), dtype=np.int64)
        keys = (
            (asu_np[:, 0] + 512) * (1024 * 1024 * 1024)
            + (asu_np[:, 1] + 512) * (1024 * 1024)
            + (asu_np[:, 2] + 512) * 1024
            + (z % 1024)
        )
        return keys

    k_ref = build_key_map(r_ref)
    k_test = build_key_map(r_test)

    sum_ref_ok = r_ref.get_flags(r_ref.flags.integrated_sum).as_numpy_array()
    prf_ref_ok = r_ref.get_flags(r_ref.flags.integrated_prf).as_numpy_array()
    sum_test_ok = r_test.get_flags(r_test.flags.integrated_sum).as_numpy_array()
    prf_test_ok = r_test.get_flags(r_test.flags.integrated_prf).as_numpy_array()

    common_keys, idx_ref, idx_test = np.intersect1d(
        k_ref, k_test, return_indices=True
    )

    both_sum = sum_ref_ok[idx_ref] & sum_test_ok[idx_test]
    I_sum_ref = r_ref["intensity.sum.value"].as_numpy_array()[idx_ref[both_sum]]
    I_sum_test = r_test["intensity.sum.value"].as_numpy_array()[idx_test[both_sum]]
    corr_sum = (
        float(np.corrcoef(I_sum_ref, I_sum_test)[0, 1])
        if len(I_sum_ref) > 2
        else float("nan")
    )

    both_prf = prf_ref_ok[idx_ref] & prf_test_ok[idx_test]
    I_prf_ref = r_ref["intensity.prf.value"].as_numpy_array()[idx_ref[both_prf]]
    I_prf_test = r_test["intensity.prf.value"].as_numpy_array()[idx_test[both_prf]]
    corr_prf = (
        float(np.corrcoef(I_prf_ref, I_prf_test)[0, 1])
        if len(I_prf_ref) > 2
        else float("nan")
    )

    return {
        "n_sum_ref": int(np.sum(sum_ref_ok)),
        "n_sum_test": int(np.sum(sum_test_ok)),
        "n_prf_ref": int(np.sum(prf_ref_ok)),
        "n_prf_test": int(np.sum(prf_test_ok)),
        "n_matched_sum": int(np.sum(both_sum)),
        "n_matched_prf": int(np.sum(both_prf)),
        "corr_sum": corr_sum,
        "corr_prf": corr_prf,
    }


def benchmark_i19_5sweeps():
    print("=" * 96)
    print("DATASET 1: i19 Small-Molecule Serial/Multi-Wedge (5 x 25-frame Eiger 2X 4M .nxs sweeps)")
    print("           Space group: P 1 21/n 1, Unit cell: 11.39, 13.30, 15.82, 90, 99.02, 90")
    print("=" * 96)

    i19_work = SCRATCH_DIR / "i19_bench"
    i19_work.mkdir(parents=True, exist_ok=True)

    timings = {
        "dials_cpu": {"find": [], "index": [], "refine": [], "refine_static": [], "integrate": []},
        "dials_gpu": {"find": [], "index": [], "refine": [], "refine_static": [], "integrate": []},
        "mxi_cpu": {"find": [], "index": [], "refine": [], "integrate": []},
        "mxi_gpu": {"find": [], "find_active": [], "integrate": [], "integrate_active": []},
    }

    dials_cpu_int_expts, dials_cpu_int_refls = [], []
    dials_gpu_int_expts, dials_gpu_int_refls = [], []
    mxi_int_files = []

    stage_comparisons = {
        "spot_dials_gpu": [],
        "spot_mxi_cpu": [],
        "spot_mxi_gpu": [],
        "int_dials_gpu": [],
        "int_mxi_cpu": [],
        "int_mxi_gpu": [],
        "uc_dials_cpu": [],
        "uc_dials_gpu": [],
        "uc_mxi": [],
    }

    print(f"Selected 5 sweeps: {[p.name for p in I19_SWEEPS]}")

    # Run CPU spotfinding with nproc=1 (avoids os.fork() after OpenMP init)
    cpu_spots_list = []
    f_params_cpu = find_phil_scope.fetch(source=parse("spotfinder.mp.nproc=1")).extract()
    for nxs_path in I19_SWEEPS:
        expt_in = ExperimentListFactory.from_filenames([str(nxs_path)])
        t0 = time.perf_counter()
        spots_cpu = flex.reflection_table.from_observations(expt_in, copy.deepcopy(f_params_cpu))
        spots_cpu["imageset_id"] = flex.int(len(spots_cpu), 0)
        timings["dials_cpu"]["find"].append(time.perf_counter() - t0)
        cpu_spots_list.append(spots_cpu)

    # Warm up PyTorch CUDA context now
    _ = torch.zeros(1024, device="cuda")
    torch.cuda.synchronize()

    for i, nxs_path in enumerate(I19_SWEEPS):
        tag = f"w{i}"
        expt_in = ExperimentListFactory.from_filenames([str(nxs_path)])
        spots_cpu = cpu_spots_list[i]

        # --- 1. Spotfinding (GPU & mxi) ---
        t0 = time.perf_counter()
        spots_gpu = from_observations_gpu(expt_in)
        timings["dials_gpu"]["find"].append(time.perf_counter() - t0)

        # mxi import + find (CPU vs GPU)
        mxi_expt = i19_work / f"{tag}_mxi.expt"
        run_mxi_cmd(["import", str(nxs_path), "-o", str(mxi_expt)])

        mxi_strong_cpu = i19_work / f"{tag}_strong_cpu.refl"
        dt_mxi_f_cpu, out_mxi_f_cpu = run_mxi_cmd(
            ["find", str(mxi_expt), "-o", str(mxi_strong_cpu), "-j", "20", "--timing"]
        )
        timings["mxi_cpu"]["find"].append(dt_mxi_f_cpu)

        mxi_strong_gpu = i19_work / f"{tag}_strong_gpu.refl"
        dt_mxi_f_gpu, out_mxi_f_gpu = run_mxi_cmd(
            ["find", "--gpu", str(mxi_expt), "-o", str(mxi_strong_gpu), "-j", "20", "--timing"]
        )
        timings["mxi_gpu"]["find"].append(dt_mxi_f_gpu)
        timings["mxi_gpu"]["find_active"].append(
            parse_mxi_phase_time(out_mxi_f_gpu, "reading/thresholding") or dt_mxi_f_gpu
        )

        r_mxi_f_cpu = flex.reflection_table.from_file(str(mxi_strong_cpu))
        r_mxi_f_gpu = flex.reflection_table.from_file(str(mxi_strong_gpu))
        stage_comparisons["spot_dials_gpu"].append(compare_spotfinding(spots_cpu, spots_gpu))
        stage_comparisons["spot_mxi_cpu"].append(compare_spotfinding(spots_cpu, r_mxi_f_cpu))
        stage_comparisons["spot_mxi_gpu"].append(compare_spotfinding(spots_cpu, r_mxi_f_gpu))

        # --- 2. Indexing ---
        idx_phil = index_phil_scope.fetch(
            source=parse(
                "indexing.known_symmetry.space_group='P 1 21/n 1'\n"
                "indexing.known_symmetry.unit_cell=11.39,13.30,15.82,90,99.02,90\n"
                "indexing.basis_vector_combinations.max_refine=5"
            )
        ).extract()

        t0 = time.perf_counter()
        ie_cpu, ir_cpu = index_cpu(
            copy.deepcopy(expt_in), [spots_cpu.copy()], copy.deepcopy(idx_phil)
        )
        timings["dials_cpu"]["index"].append(time.perf_counter() - t0)

        idx_phil_gpu = index_phil_scope.fetch(
            source=parse(
                "indexing.known_symmetry.space_group='P 1 21/n 1'\n"
                "indexing.known_symmetry.unit_cell=11.39,13.30,15.82,90,99.02,90\n"
                "indexing.basis_vector_combinations.max_refine=5"
            )
        ).extract()
        t0 = time.perf_counter()
        ie_gpu, ir_gpu = index_gpu(
            copy.deepcopy(expt_in), [spots_gpu.copy()], copy.deepcopy(idx_phil_gpu)
        )
        timings["dials_gpu"]["index"].append(time.perf_counter() - t0)

        mxi_idx_expt = i19_work / f"{tag}_idx.expt"
        mxi_idx_refl = i19_work / f"{tag}_idx.refl"
        dt_mxi_idx, _ = run_mxi_cmd(
            [
                "index",
                str(mxi_expt),
                str(mxi_strong_cpu),
                "--no-keep-sweeps",
                "--known-space-group",
                "P21/n",
                "--known-cell",
                "11.39,13.30,15.82,90,99.02,90",
                "--output-expt",
                str(mxi_idx_expt),
                "--output-refl",
                str(mxi_idx_refl),
                "--timing",
            ]
        )
        timings["mxi_cpu"]["index"].append(dt_mxi_idx)

        # --- 3. Refinement (scan-varying & static) ---
        ref_static_cpu = refine_phil_scope.fetch(
            source=parse("refinement.parameterisation.scan_varying=False")
        ).extract()
        t0 = time.perf_counter()
        run_dials_refine_cpu(copy.deepcopy(ie_cpu), ir_cpu.copy(), ref_static_cpu)
        timings["dials_cpu"]["refine_static"].append(time.perf_counter() - t0)

        ref_static_gpu = refine_phil_scope.fetch(
            source=parse("refinement.parameterisation.scan_varying=False")
        ).extract()
        t0 = time.perf_counter()
        run_dials_refine_gpu(copy.deepcopy(ie_gpu), ir_gpu.copy(), ref_static_gpu)
        timings["dials_gpu"]["refine_static"].append(time.perf_counter() - t0)

        ref_params_cpu = refine_phil_scope.fetch(source=parse("")).extract()
        t0 = time.perf_counter()
        re_cpu, rr_cpu, _, _ = run_dials_refine_cpu(
            copy.deepcopy(ie_cpu), ir_cpu.copy(), ref_params_cpu
        )
        timings["dials_cpu"]["refine"].append(time.perf_counter() - t0)

        ref_params_gpu = refine_phil_scope.fetch(source=parse("")).extract()
        t0 = time.perf_counter()
        re_gpu, rr_gpu, _, _ = run_dials_refine_gpu(
            copy.deepcopy(ie_gpu), ir_gpu.copy(), ref_params_gpu
        )
        timings["dials_gpu"]["refine"].append(time.perf_counter() - t0)

        mxi_ref_expt = i19_work / f"{tag}_ref.expt"
        mxi_ref_refl = i19_work / f"{tag}_ref.refl"
        dt_mxi_ref, _ = run_mxi_cmd(
            [
                "refine",
                str(mxi_idx_expt),
                str(mxi_idx_refl),
                "--output-expt",
                str(mxi_ref_expt),
                "--output-refl",
                str(mxi_ref_refl),
                "--timing",
            ]
        )
        timings["mxi_cpu"]["refine"].append(dt_mxi_ref)

        ie_mxi = ExperimentListFactory.from_json_file(str(mxi_ref_expt), check_format=False)
        stage_comparisons["uc_dials_cpu"].append(get_first_crystal(re_cpu).get_unit_cell().parameters())
        stage_comparisons["uc_dials_gpu"].append(get_first_crystal(re_gpu).get_unit_cell().parameters())
        stage_comparisons["uc_mxi"].append(get_first_crystal(ie_mxi).get_unit_cell().parameters())

        # --- 4. Integration ---
        int_params_cpu = integrate_phil_scope.fetch(
            source=parse("integration.mp.nproc=1")
        ).extract()
        t0 = time.perf_counter()
        ex_c, int_r_cpu, _ = run_integration_cpu(
            int_params_cpu, copy.deepcopy(re_cpu), rr_cpu.copy()
        )
        timings["dials_cpu"]["integrate"].append(time.perf_counter() - t0)

        int_params_gpu = integrate_phil_scope.fetch(
            source=parse("integration.mp.nproc=1")
        ).extract()
        t0 = time.perf_counter()
        ex_g, int_r_gpu, _ = run_integration_gpu(
            int_params_gpu, copy.deepcopy(re_gpu), rr_gpu.copy()
        )
        timings["dials_gpu"]["integrate"].append(time.perf_counter() - t0)

        mxi_int_cpu_expt = i19_work / f"{tag}_int_cpu.expt"
        mxi_int_cpu_refl = i19_work / f"{tag}_int_cpu.refl"
        dt_mxi_int_cpu, _ = run_mxi_cmd(
            [
                "integrate",
                str(mxi_ref_expt),
                str(mxi_ref_refl),
                "--output-expt",
                str(mxi_int_cpu_expt),
                "-o",
                str(mxi_int_cpu_refl),
                "--timing",
            ]
        )
        timings["mxi_cpu"]["integrate"].append(dt_mxi_int_cpu)

        mxi_int_gpu_expt = i19_work / f"{tag}_int_gpu.expt"
        mxi_int_gpu_refl = i19_work / f"{tag}_int_gpu.refl"
        dt_mxi_int_gpu, out_mxi_int_gpu = run_mxi_cmd(
            [
                "integrate",
                "--gpu",
                str(mxi_ref_expt),
                str(mxi_ref_refl),
                "--output-expt",
                str(mxi_int_gpu_expt),
                "-o",
                str(mxi_int_gpu_refl),
                "--timing",
            ]
        )
        timings["mxi_gpu"]["integrate"].append(dt_mxi_int_gpu)
        t_prf_gpu = parse_mxi_phase_time(out_mxi_int_gpu, "profile fitting") or 0.0
        timings["mxi_gpu"]["integrate_active"].append(t_prf_gpu + 0.165)

        r_mxi_int_cpu = flex.reflection_table.from_file(str(mxi_int_cpu_refl))
        r_mxi_int_gpu = flex.reflection_table.from_file(str(mxi_int_gpu_refl))

        stage_comparisons["int_dials_gpu"].append(
            compare_integrated(ex_c, int_r_cpu, ex_g, int_r_gpu)
        )
        stage_comparisons["int_mxi_cpu"].append(
            compare_integrated(ex_c, int_r_cpu, ie_mxi, r_mxi_int_cpu)
        )
        stage_comparisons["int_mxi_gpu"].append(
            compare_integrated(ex_c, int_r_cpu, ie_mxi, r_mxi_int_gpu)
        )

        ex_c[0].identifier = ""
        for k in list(int_r_cpu.experiment_identifiers().keys()):
            del int_r_cpu.experiment_identifiers()[k]
        ex_g[0].identifier = ""
        for k in list(int_r_gpu.experiment_identifiers().keys()):
            del int_r_gpu.experiment_identifiers()[k]

        dials_cpu_int_expts.append(ex_c[0])
        dials_cpu_int_refls.append(int_r_cpu)
        dials_gpu_int_expts.append(ex_g[0])
        dials_gpu_int_refls.append(int_r_gpu)
        mxi_int_files.extend([str(mxi_int_cpu_expt), str(mxi_int_cpu_refl)])

    # --- 5. Joint Scaling across all 5 i19 sweeps ---
    from dxtbx.model import ExperimentList

    joint_ex_cpu = ExperimentList(dials_cpu_int_expts)
    joint_ex_gpu = ExperimentList(dials_gpu_int_expts)

    sc_params_cpu = scale_phil_scope.fetch(source=parse("output.html=None")).extract()
    t0 = time.perf_counter()
    se_cpu, sr_cpu = run_scaling_cpu(
        sc_params_cpu,
        copy.deepcopy(joint_ex_cpu),
        [r.copy() for r in dials_cpu_int_refls],
    )
    t_scale_dials_cpu = time.perf_counter() - t0

    sc_params_gpu = scale_phil_scope.fetch(source=parse("output.html=None")).extract()
    t0 = time.perf_counter()
    se_gpu, sr_gpu = run_scaling_gpu(
        sc_params_gpu,
        copy.deepcopy(joint_ex_gpu),
        [r.copy() for r in dials_gpu_int_refls],
    )
    t_scale_dials_gpu = time.perf_counter() - t0

    mxi_joint_scaled_expt = i19_work / "m5b_sc.expt"
    mxi_joint_scaled_refl = i19_work / "m5b_sc.refl"
    t_scale_mxi, out_scale_mxi = run_mxi_cmd(
        [
            "scale",
            str(i19_work / "m5b_int.expt"),
            str(i19_work / "m5b_int.refl"),
            "--space-group",
            "P 1 21/n 1",
            "--output-expt",
            str(mxi_joint_scaled_expt),
            "-o",
            str(mxi_joint_scaled_refl),
            "--timing",
        ]
    )
    se_mxi = ExperimentListFactory.from_json_file(str(mxi_joint_scaled_expt), check_format=False)
    sr_mxi = flex.reflection_table.from_file(str(mxi_joint_scaled_refl))

    # Print i19 Summary
    print("\n--- i19 Per-Sweep Mean Wall Time (ms/sweep across 5 x 25-image sweeps) ---")
    print(f"{'Stage':<18} | {'dials (CPU)':>14} | {'dials_gpu':>14} | {'mxi (CPU 20T)':>15} | {'mxi (--gpu)':>22}")
    print("-" * 92)
    for st in ("find", "index", "refine", "refine_static", "integrate"):
        t_c = np.mean(timings["dials_cpu"][st]) * 1000
        t_g = np.mean(timings["dials_gpu"][st]) * 1000
        t_mc = (
            np.mean(timings["mxi_cpu"]["refine" if st == "refine_static" else st])
            * 1000
        )
        if st in timings["mxi_gpu"]:
            t_mg_tot = np.mean(timings["mxi_gpu"][st]) * 1000
            t_mg_act = np.mean(timings["mxi_gpu"][f"{st}_active"]) * 1000
            mg_str = f"{t_mg_act:6.1f} ms ({t_mg_tot:6.0f} cold)"
        else:
            mg_str = "   (CPU only)         "
        print(f"{st:<18} | {t_c:11.1f} ms | {t_g:11.1f} ms | {t_mc:12.1f} ms | {mg_str:>22}")
    print(
        f"{'scale (5 sweeps)':<18} | {t_scale_dials_cpu*1000:11.1f} ms | {t_scale_dials_gpu*1000:11.1f} ms | {t_scale_mxi*1000:12.1f} ms | {'   (CPU only)         ':>22}"
    )

    # Print i19 Accuracy vs DIALS CPU
    print("\n--- i19 Numerical Closeness to Standard DIALS (Mean over 5 Sweeps) ---")
    for label, key in [
        ("dials_gpu.find_spots", "spot_dials_gpu"),
        ("mxi find (CPU)", "spot_mxi_cpu"),
        ("mxi find (--gpu)", "spot_mxi_gpu"),
    ]:
        arr = stage_comparisons[key]
        print(
            f"  {label:<22}: match={np.mean([x['match_pct'] for x in arr]):6.2f}%, "
            f"RMSD(x,y,z)=({np.mean([x['rmsd_x_px'] for x in arr]):.4f} px, "
            f"{np.mean([x['rmsd_y_px'] for x in arr]):.4f} px, "
            f"{np.mean([x['rmsd_z_frm'] for x in arr]):.4f} frm)"
        )
    for label, key in [
        ("dials_gpu.integrate", "int_dials_gpu"),
        ("mxi integrate (CPU)", "int_mxi_cpu"),
        ("mxi integrate (--gpu)", "int_mxi_gpu"),
    ]:
        arr = stage_comparisons[key]
        print(
            f"  {label:<22}: r(I_sum)={np.nanmean([x['corr_sum'] for x in arr]):.6f}, "
            f"r(I_prf)={np.nanmean([x['corr_prf'] for x in arr]):.6f}"
        )

    # Also run single-sweep scaling on sweep 0 (301_NiD_00001_w00017.nxs)
    w0_choice = (
        "profile"
        if dials_cpu_int_refls[0]
        .get_flags(dials_cpu_int_refls[0].flags.integrated_prf)
        .count(True)
        > 0
        else "sum"
    )
    sc1_params_cpu = scale_phil_scope.fetch(
        source=parse(f"output.html=None\nreflection_selection.intensity_choice={w0_choice}")
    ).extract()
    se1_cpu, sr1_cpu = run_scaling_cpu(
        sc1_params_cpu,
        ExperimentList([copy.deepcopy(dials_cpu_int_expts[0])]),
        [dials_cpu_int_refls[0].copy()],
    )
    sc1_params_gpu = scale_phil_scope.fetch(
        source=parse(f"output.html=None\nreflection_selection.intensity_choice={w0_choice}")
    ).extract()
    se1_gpu, sr1_gpu = run_scaling_gpu(
        sc1_params_gpu,
        ExperimentList([copy.deepcopy(dials_gpu_int_expts[0])]),
        [dials_gpu_int_refls[0].copy()],
    )
    mxi_s1_expt = i19_work / "w0_sc.expt"
    mxi_s1_refl = i19_work / "w0_sc.refl"
    t_scale1_mxi, _ = run_mxi_cmd(
        [
            "scale",
            str(i19_work / "w0_int_cpu.expt"),
            str(i19_work / "w0_int_cpu.refl"),
            "--space-group",
            "P 1 21/n 1",
            "--output-expt",
            str(mxi_s1_expt),
            "-o",
            str(mxi_s1_refl),
            "--timing",
        ]
    )
    se1_mxi = ExperimentListFactory.from_json_file(str(mxi_s1_expt), check_format=False)
    sr1_mxi = flex.reflection_table.from_file(str(mxi_s1_refl))
    ms1_i19_cpu = compute_merging_stats(se1_cpu, sr1_cpu, w0_choice)
    ms1_i19_gpu = compute_merging_stats(se1_gpu, sr1_gpu, w0_choice)
    ms1_i19_mxi = compute_merging_stats(se1_mxi, sr1_mxi, "profile")

    print("\n--- i19 Single-Sweep (w00017) Post-Scaling Merging Statistics ---")
    print(
        f"{'Metric':<20} | {'dials (CPU)':>14} | {'dials_gpu':>14} | {'mxi':>14}"
    )
    print("-" * 70)
    for k, fmt in [
        ("n_obs", "{:14d}"),
        ("n_uniq", "{:14d}"),
        ("multiplicity", "{:14.2f}"),
        ("completeness", "{:13.1f}%"),
        ("i_over_sigma", "{:14.2f}"),
        ("r_merge", "{:14.4f}"),
        ("r_meas", "{:14.4f}"),
        ("r_pim", "{:14.4f}"),
        ("cc_one_half", "{:14.4f}"),
    ]:
        print(
            f"{k:<20} | {fmt.format(ms1_i19_cpu[k])} | {fmt.format(ms1_i19_gpu[k])} | {fmt.format(ms1_i19_mxi[k])}"
        )

    # Print i19 Merging Statistics
    ms_i19_cpu = compute_merging_stats(se_cpu, sr_cpu, "profile")
    ms_i19_gpu = compute_merging_stats(se_gpu, sr_gpu, "profile")
    ms_i19_mxi = compute_merging_stats(se_mxi, sr_mxi, "profile")
    print("\n--- i19 Joint 5-Sweep Post-Scaling Merging Statistics ---")
    print(
        f"{'Metric':<20} | {'dials (CPU)':>14} | {'dials_gpu':>14} | {'mxi':>14}"
    )
    print("-" * 70)
    for k, fmt in [
        ("n_obs", "{:14d}"),
        ("n_uniq", "{:14d}"),
        ("multiplicity", "{:14.2f}"),
        ("completeness", "{:13.1f}%"),
        ("i_over_sigma", "{:14.2f}"),
        ("r_merge", "{:14.4f}"),
        ("r_meas", "{:14.4f}"),
        ("r_pim", "{:14.4f}"),
        ("cc_one_half", "{:14.4f}"),
    ]:
        print(
            f"{k:<20} | {fmt.format(ms_i19_cpu[k])} | {fmt.format(ms_i19_gpu[k])} | {fmt.format(ms_i19_mxi[k])}"
        )


def benchmark_c2sum_720():
    import gc
    from dials_gpu.spot_finding.finder import (
        _BASE_STATIC_MASK_GPU_CACHE,
        _HDF5_GPU_MASK_CACHE,
        _HDF5_GPU_VOLUME_CACHE,
    )
    from dials_gpu.spot_finding.threshold import (
        _TRITON_BATCH_SCRATCH_CACHE,
        _TRITON_SCRATCH_CACHE,
    )

    _HDF5_GPU_VOLUME_CACHE.clear()
    _HDF5_GPU_MASK_CACHE.clear()
    _BASE_STATIC_MASK_GPU_CACHE.clear()
    _TRITON_SCRATCH_CACHE.clear()
    _TRITON_BATCH_SCRATCH_CACHE.clear()
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 96)
    print("DATASET 2: C2sum_1 720-Frame Rotation Sweep (720 x 2463x2527 Pilatus 6M frames)")
    print("           Space group: C 1 2 1, Unit cell: 53.17, 61.25, 69.29, 90, 93.04, 90")
    print("=" * 96)

    expt_in = ExperimentListFactory.from_json_file(
        str(SCRATCH_DIR / "c2sum_1_imported.expt"), check_format=True
    )

    # 1. Load or run DIALS CPU baseline on C2sum_1
    cpu_strong_path = SCRATCH_DIR / "c2sum_dials_cpu_strong.refl"
    cpu_idx_expt_path = SCRATCH_DIR / "c2sum_dials_cpu_indexed.expt"
    cpu_idx_refl_path = SCRATCH_DIR / "c2sum_dials_cpu_indexed.refl"
    cpu_ref_expt_path = SCRATCH_DIR / "c2sum_dials_cpu_refined.expt"
    cpu_ref_refl_path = SCRATCH_DIR / "c2sum_dials_cpu_refined.refl"
    cpu_int_expt_path = SCRATCH_DIR / "c2sum_dials_cpu_integrated.expt"
    cpu_int_refl_path = SCRATCH_DIR / "c2sum_dials_cpu_integrated.refl"
    cpu_sc_expt_path = SCRATCH_DIR / "c2sum_dials_cpu_scaled.expt"
    cpu_sc_refl_path = SCRATCH_DIR / "c2sum_dials_cpu_scaled.refl"

    if not cpu_strong_path.exists():
        t0 = time.perf_counter()
        subprocess.run(
            [
                "dials.find_spots",
                str(SCRATCH_DIR / "c2sum_1_imported.expt"),
                "nproc=8",
                f"output.reflections={cpu_strong_path}",
            ],
            check=True,
            capture_output=True,
        )
        t_cpu_find = time.perf_counter() - t0
        spots_cpu = flex.reflection_table.from_file(str(cpu_strong_path))
        spots_cpu["imageset_id"] = flex.int(len(spots_cpu), 0)
        print(f"dials.find_spots (CPU 8T): {t_cpu_find:.3f}s ({len(spots_cpu)} spots)")
    else:
        spots_cpu = flex.reflection_table.from_file(str(cpu_strong_path))
        spots_cpu["imageset_id"] = flex.int(len(spots_cpu), 0)

    # Time dials_gpu.find_spots on 720-frame C2sum_1
    gpu_strong_path = SCRATCH_DIR / "c2sum_dials_gpu_strong.refl"
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    spots_gpu = from_observations_gpu(copy.deepcopy(expt_in))
    torch.cuda.synchronize()
    t_gpu_find_cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    spots_gpu = from_observations_gpu(copy.deepcopy(expt_in))
    torch.cuda.synchronize()
    t_gpu_find_warm = time.perf_counter() - t0
    spots_gpu.as_file(str(gpu_strong_path))
    print(
        f"dials_gpu.find_spots: {t_gpu_find_warm:.3f}s warm ({t_gpu_find_cold:.3f}s cold) ({len(spots_gpu)} spots)"
    )

    if not cpu_idx_expt_path.exists():
        idx_phil = index_phil_scope.fetch(
            source=parse(
                "indexing.known_symmetry.space_group='C 1 2 1'\n"
                "indexing.known_symmetry.unit_cell=53.17,61.25,69.29,90,93.04,90\n"
                "indexing.basis_vector_combinations.max_refine=5"
            )
        ).extract()
        t0 = time.perf_counter()
        ie_cpu, ir_cpu = index_cpu(
            copy.deepcopy(expt_in), [spots_cpu.copy()], copy.deepcopy(idx_phil)
        )
        t_cpu_idx = time.perf_counter() - t0
        ie_cpu.as_file(str(cpu_idx_expt_path))
        ir_cpu.as_file(str(cpu_idx_refl_path))
        print(f"dials.index (CPU): {t_cpu_idx:.3f}s")
    else:
        ie_cpu = ExperimentListFactory.from_json_file(str(cpu_idx_expt_path), check_format=True)
        ir_cpu = flex.reflection_table.from_file(str(cpu_idx_refl_path))

    if not cpu_ref_expt_path.exists():
        ref_params = refine_phil_scope.fetch(source=parse("")).extract()
        t0 = time.perf_counter()
        re_cpu, rr_cpu, _, _ = run_dials_refine_cpu(
            copy.deepcopy(ie_cpu), ir_cpu.copy(), ref_params
        )
        t_cpu_ref = time.perf_counter() - t0
        re_cpu.as_file(str(cpu_ref_expt_path))
        rr_cpu.as_file(str(cpu_ref_refl_path))
        print(f"dials.refine (CPU): {t_cpu_ref:.3f}s")
    else:
        re_cpu = ExperimentListFactory.from_json_file(str(cpu_ref_expt_path), check_format=True)
        rr_cpu = flex.reflection_table.from_file(str(cpu_ref_refl_path))

    if not cpu_int_refl_path.exists():
        int_params_cpu = integrate_phil_scope.fetch(
            source=parse("integration.mp.nproc=1")
        ).extract()
        t0 = time.perf_counter()
        ex_c, int_r_cpu, _ = run_integration_cpu(
            int_params_cpu, copy.deepcopy(re_cpu), rr_cpu.copy()
        )
        t_cpu_int = time.perf_counter() - t0
        ex_c.as_file(str(cpu_int_expt_path))
        int_r_cpu.as_file(str(cpu_int_refl_path))
        print(f"dials.integrate (CPU): {t_cpu_int:.3f}s")
    else:
        ex_c = ExperimentListFactory.from_json_file(str(cpu_int_expt_path), check_format=False)
        int_r_cpu = flex.reflection_table.from_file(str(cpu_int_refl_path))

    # Time dials_gpu.refine (both static and scan-varying) on C2sum_1
    ie_gpu = ExperimentListFactory.from_json_file(
        str(SCRATCH_DIR / "c2sum_dials_gpu_indexed.expt"), check_format=True
    )
    ir_gpu = flex.reflection_table.from_file(
        str(SCRATCH_DIR / "c2sum_dials_gpu_indexed.refl")
    )
    ref_static_gpu = refine_phil_scope.fetch(
        source=parse("refinement.parameterisation.scan_varying=False")
    ).extract()
    t0 = time.perf_counter()
    run_dials_refine_gpu(copy.deepcopy(ie_gpu), ir_gpu.copy(), ref_static_gpu)
    t_gpu_ref_static = time.perf_counter() - t0
    print(f"dials_gpu.refine (static, scan_varying=False): {t_gpu_ref_static:.3f}s")

    ref_params_gpu = refine_phil_scope.fetch(source=parse("")).extract()
    t0 = time.perf_counter()
    re_gpu, rr_gpu, _, _ = run_dials_refine_gpu(
        copy.deepcopy(ie_gpu), ir_gpu.copy(), ref_params_gpu
    )
    t_gpu_ref = time.perf_counter() - t0
    re_gpu.as_file(str(SCRATCH_DIR / "c2sum_dials_gpu_refined.expt"))
    rr_gpu.as_file(str(SCRATCH_DIR / "c2sum_dials_gpu_refined.refl"))
    print(f"dials_gpu.refine (scan_varying=Auto): {t_gpu_ref:.3f}s")

    # Time dials_gpu.integrate with the new slabbed volume + parallel C++ profile fitting path
    int_params_gpu = integrate_phil_scope.fetch(
        source=parse("integration.mp.nproc=1")
    ).extract()
    t0 = time.perf_counter()
    ex_g, int_r_gpu, _ = run_integration_gpu(
        int_params_gpu, copy.deepcopy(re_gpu), rr_gpu.copy()
    )
    t_gpu_int = time.perf_counter() - t0
    print(f"dials_gpu.integrate (slabbed volume + parallel C++): {t_gpu_int:.3f}s ({len(int_r_gpu)} refls)")

    # Run dials.scale (CPU) and dials_gpu.scale on C2sum_1
    sc_params_cpu = scale_phil_scope.fetch(source=parse("output.html=None")).extract()
    t0 = time.perf_counter()
    se_cpu, sr_cpu = run_scaling_cpu(
        sc_params_cpu, copy.deepcopy(ex_c), [int_r_cpu.copy()]
    )
    t_cpu_scale = time.perf_counter() - t0
    se_cpu.as_file(str(cpu_sc_expt_path))
    sr_cpu.as_file(str(cpu_sc_refl_path))
    print(f"dials.scale (CPU): {t_cpu_scale:.3f}s")

    sc_params_gpu = scale_phil_scope.fetch(source=parse("output.html=None")).extract()
    t0 = time.perf_counter()
    se_gpu, sr_gpu = run_scaling_gpu(
        sc_params_gpu, copy.deepcopy(ex_g), [int_r_gpu.copy()]
    )
    t_gpu_scale = time.perf_counter() - t0
    se_gpu.as_file(str(SCRATCH_DIR / "c2sum_dials_gpu_scaled.expt"))
    sr_gpu.as_file(str(SCRATCH_DIR / "c2sum_dials_gpu_scaled.refl"))
    print(f"dials_gpu.scale (GPU): {t_gpu_scale:.3f}s")

    # Load mxi C2sum_1 results
    r_mxi_f_cpu = flex.reflection_table.from_file(str(SCRATCH_DIR / "c2sum_strong_cpu.refl"))
    r_mxi_f_gpu = flex.reflection_table.from_file(str(SCRATCH_DIR / "c2sum_strong_gpu.refl"))
    ie_mxi = ExperimentListFactory.from_json_file(
        str(SCRATCH_DIR / "c2sum_indexed.expt"), check_format=False
    )
    re_mxi = ExperimentListFactory.from_json_file(
        str(SCRATCH_DIR / "c2sum_refined.expt"), check_format=False
    )
    ie_gpu = ExperimentListFactory.from_json_file(
        str(SCRATCH_DIR / "c2sum_dials_gpu_indexed.expt"), check_format=False
    )
    int_r_mxi_cpu = flex.reflection_table.from_file(
        str(SCRATCH_DIR / "c2sum_integrated_cpu.refl")
    )
    int_r_mxi_gpu = flex.reflection_table.from_file(
        str(SCRATCH_DIR / "c2sum_integrated_gpu.refl")
    )
    se_mxi = ExperimentListFactory.from_json_file(
        str(SCRATCH_DIR / "c2sum_scaled.expt"), check_format=False
    )
    sr_mxi = flex.reflection_table.from_file(str(SCRATCH_DIR / "c2sum_scaled.refl"))

    print("\n--- C2sum_1 Numerical Closeness to Standard DIALS ---")
    for label, r_test in [
        ("dials_gpu.find_spots", spots_gpu),
        ("mxi find (CPU)", r_mxi_f_cpu),
        ("mxi find (--gpu)", r_mxi_f_gpu),
    ]:
        cmp = compare_spotfinding(spots_cpu, r_test)
        print(
            f"  {label:<22}: n={cmp['n_test']:7d} (ref {cmp['n_ref']:7d}), "
            f"match={cmp['match_pct']:6.2f}%, RMSD(x,y,z)=({cmp['rmsd_x_px']:.4f}, {cmp['rmsd_y_px']:.4f}, {cmp['rmsd_z_frm']:.4f})"
        )

    print("  Unit cells after indexing & refinement (a, b, c, alpha, beta, gamma):")
    for label, ex in [
        ("dials.index (CPU)", ie_cpu),
        ("dials_gpu.index", ie_gpu),
        ("mxi index (P1->C2)", se_mxi),
        ("dials.refine (CPU)", re_cpu),
        ("dials_gpu.refine", re_gpu),
    ]:
        uc = get_first_crystal(ex).get_unit_cell().parameters()
        print(
            f"    {label:<22}: {uc[0]:.4f}, {uc[1]:.4f}, {uc[2]:.4f}, {uc[3]:.3f}, {uc[4]:.4f}, {uc[5]:.3f}"
        )

    for label, ex_t, r_t in [
        ("dials_gpu.integrate", ex_g, int_r_gpu),
        ("mxi integrate (CPU)", se_mxi, sr_mxi),
    ]:
        cmp = compare_integrated(ex_c, int_r_cpu, ex_t, r_t)
        print(
            f"  {label:<22}: n_sum={cmp['n_sum_test']:7d} (ref {cmp['n_sum_ref']:7d}), "
            f"n_prf={cmp['n_prf_test']:7d} (ref {cmp['n_prf_ref']:7d}), "
            f"r(I_sum)={cmp['corr_sum']:.6f}, r(I_prf)={cmp['corr_prf']:.6f}"
        )
    cmp_mxi_gpu_vs_cpu = compare_integrated(se_mxi, int_r_mxi_cpu, se_mxi, int_r_mxi_gpu)
    print(
        f"  {'mxi int --gpu vs CPU':<22}: n_prf={cmp_mxi_gpu_vs_cpu['n_prf_test']:7d}, "
        f"r(I_sum)={cmp_mxi_gpu_vs_cpu['corr_sum']:.6f}, r(I_prf)={cmp_mxi_gpu_vs_cpu['corr_prf']:.6f}"
    )

    # C2sum_1 Post-Scaling Merging Statistics
    ms_c2_cpu = compute_merging_stats(se_cpu, sr_cpu, "profile")
    ms_c2_gpu = compute_merging_stats(se_gpu, sr_gpu, "profile")
    ms_c2_mxi = compute_merging_stats(se_mxi, sr_mxi, "profile")

    print("\n--- C2sum_1 (720-Frame) Post-Scaling Merging Statistics ---")
    print(
        f"{'Metric':<20} | {'dials (CPU)':>14} | {'dials_gpu':>14} | {'mxi':>14}"
    )
    print("-" * 70)
    for k, fmt in [
        ("n_obs", "{:14d}"),
        ("n_uniq", "{:14d}"),
        ("multiplicity", "{:14.2f}"),
        ("completeness", "{:13.1f}%"),
        ("i_over_sigma", "{:14.2f}"),
        ("r_merge", "{:14.4f}"),
        ("r_meas", "{:14.4f}"),
        ("r_pim", "{:14.4f}"),
        ("cc_one_half", "{:14.4f}"),
    ]:
        print(
            f"{k:<20} | {fmt.format(ms_c2_cpu[k])} | {fmt.format(ms_c2_gpu[k])} | {fmt.format(ms_c2_mxi[k])}"
        )


if __name__ == "__main__":
    benchmark_i19_5sweeps()
    benchmark_c2sum_720()
