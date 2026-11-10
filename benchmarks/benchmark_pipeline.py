"""
General end-to-end benchmark: DIALS (CPU) vs dials_gpu on any number of datasets.

For each dataset, this script runs
    dials.import  (shared by both implementations)
    find_spots -> index -> refine -> integrate -> scale
once as ``dials.<stage>`` and once as ``dials_gpu.<stage>``. Each chain feeds on its
own outputs. Every stage runs as a separate subprocess, so the timings are the real
command-line wall-clock times (including Python/CUDA start-up), and a crash, hang or
"exit 0 but wrote nothing" in one stage cannot take down the benchmark.

Failures are first-class results, never just a short time:
  * FAILED   - non-zero exit code (the log tail is recorded)
  * TIMEOUT  - killed after --timeout seconds
  * INVALID  - exit 0, but expected output files are missing/unreadable, or contain
               no usable data (0 spots, 0 indexed, 0 integrated, 0 scaled)
  * SKIPPED  - not run because an earlier stage in the same chain did not succeed
A timing is only reported for an OK stage, a speed-up only when both are OK, and a
pipeline total only when every stage of that chain is OK.

Closeness of dials_gpu to dials is measured at each stage (when both are OK):
  find_spots  spot counts, % of DIALS spots matched within 2 px, centroid RMSD
  index       number indexed, Niggli-reduced unit-cell differences
  refine      positional RMSDs of each chain, unit-cell differences
  integrate   counts and Pearson r of summation/profile intensities, matched by
              predicted position (so it does not depend on the indexing setting)
  scale       Pearson r of scaled intensities, and merging statistics of each chain
Comparisons outside the tolerances in TOLERANCES are flagged DIVERGENT.

Usage (run from an environment where dials and dials_gpu are installed):

    # Quick: one dataset per --dataset, NAME=PATH[,PATH...] (globs allowed)
    libtbx.python benchmarks/benchmark_pipeline.py \\
        --dataset "c2sum=/data/C2sum_1/C2sum_1_*.cbf.gz" \\
        --dataset "i19=/data/i19/w00017.nxs,/data/i19/w00019.nxs" \\
        --workdir /scratch/bench

    # Full control: JSON config (see benchmarks/datasets.example.json)
    libtbx.python benchmarks/benchmark_pipeline.py --config datasets.json --workdir /scratch/bench

Results: <workdir>/results.json (everything), <workdir>/report.md (summary tables),
and per-stage outputs and logs under <workdir>/<dataset>/{import,dials,dials_gpu}/.
The exit code is 1 if any stage failed, timed out, was invalid or was skipped.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

STAGES = ["find_spots", "index", "refine", "integrate", "scale"]
IMPLS = {"dials": "dials", "dials_gpu": "dials_gpu"}  # name -> command prefix

# Each stage: (input files from the previous stage, output files it must write).
# The output names are passed explicitly so neither implementation can pick its own.
STAGE_IO = {
    "find_spots": (["imported.expt"], ["strong.refl"]),
    "index": (["imported.expt", "strong.refl"], ["indexed.expt", "indexed.refl"]),
    "refine": (["indexed.expt", "indexed.refl"], ["refined.expt", "refined.refl"]),
    "integrate": (["refined.expt", "refined.refl"], ["integrated.expt", "integrated.refl"]),
    "scale": (["integrated.expt", "integrated.refl"], ["scaled.expt", "scaled.refl"]),
}

# Stage arguments both implementations always get. The html report is turned off
# because dials.scale writes one by default and dials_gpu.scale does not, which
# would make the scaling timings unfair.
BASE_ARGS = {
    "find_spots": ["output.reflections=strong.refl"],
    "index": ["output.experiments=indexed.expt", "output.reflections=indexed.refl"],
    "refine": ["output.experiments=refined.expt", "output.reflections=refined.refl"],
    "integrate": [
        "output.experiments=integrated.expt",
        "output.reflections=integrated.refl",
    ],
    "scale": [
        "output.experiments=scaled.expt",
        "output.reflections=scaled.refl",
        "output.html=None",
    ],
}

# Closeness thresholds. A comparison outside these is flagged DIVERGENT.
TOLERANCES = {
    "spot_match_pct_min": 95.0,  # % of DIALS strong spots matched by dials_gpu
    "spot_count_rel_diff_max": 0.05,  # |n_gpu - n_cpu| / n_cpu
    "indexed_rel_diff_max": 0.05,
    "cell_length_rel_diff_max": 0.005,  # 0.5 % on Niggli cell lengths
    "cell_angle_diff_max_deg": 0.5,
    "rmsd_xy_rel_diff_max": 0.20,  # refined positional RMSD vs DIALS
    "integrated_rel_diff_max": 0.05,
    "intensity_corr_min": 0.99,
    "matched_frac_min": 0.90,  # fraction of DIALS reflections matched for r
    "cc_half_diff_max": 0.02,
    "i_over_sigma_rel_diff_max": 0.10,
}

OK, FAILED, TIMEOUT, INVALID, SKIPPED = "OK", "FAILED", "TIMEOUT", "INVALID", "SKIPPED"


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass
class Dataset:
    name: str
    images: list[str]
    # Extra arguments per stage ("import" too), applied to both implementations.
    args: dict[str, list[str]] = field(default_factory=dict)
    # Extra arguments for one implementation only: {"dials": {"integrate": [...]}}.
    impl_args: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    timeout: float | None = None


def _as_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return shlex.split(v)
    return [str(x) for x in v]


def expand_images(patterns: list[str], base: Path) -> list[str]:
    files: list[str] = []
    for pat in patterns:
        p = os.path.expanduser(pat)
        if not os.path.isabs(p):
            p = str(base / p)
        matches = sorted(glob.glob(p))
        if not matches:
            raise SystemExit(f"error: no files match {pat!r}")
        files.extend(matches)
    return files


def load_datasets(args) -> list[Dataset]:
    datasets: list[Dataset] = []
    if args.config:
        cfg_path = Path(args.config).resolve()
        cfg = json.loads(cfg_path.read_text())
        defaults = cfg.get("defaults", {})
        for entry in cfg["datasets"]:
            stage_args = {}
            for stage in ["import", *STAGES]:
                merged = _as_list(defaults.get(stage)) + _as_list(entry.get(stage))
                if merged:
                    stage_args[stage] = merged
            impl_args = {}
            for impl in IMPLS:
                src = {**defaults.get(impl, {}), **entry.get(impl, {})}
                impl_args[impl] = {k: _as_list(v) for k, v in src.items()}
            datasets.append(
                Dataset(
                    name=entry["name"],
                    images=expand_images(_as_list(entry["images"]), cfg_path.parent),
                    args=stage_args,
                    impl_args=impl_args,
                    timeout=entry.get("timeout", defaults.get("timeout")),
                )
            )
    for spec in args.dataset or []:
        if "=" not in spec:
            raise SystemExit(f"error: --dataset must be NAME=PATH[,PATH...], got {spec!r}")
        name, paths = spec.split("=", 1)
        datasets.append(
            Dataset(name=name, images=expand_images(paths.split(","), Path.cwd()))
        )
    if not datasets:
        raise SystemExit("error: give at least one --dataset or a --config")
    names = [d.name for d in datasets]
    if len(set(names)) != len(names):
        raise SystemExit(f"error: duplicate dataset names in {names}")
    return datasets


# --------------------------------------------------------------------------------------
# Running stages
# --------------------------------------------------------------------------------------


@dataclass
class StageResult:
    status: str
    command: list[str] = field(default_factory=list)
    times: list[float] = field(default_factory=list)  # one per repeat (OK runs only)
    returncode: int | None = None
    reason: str = ""
    log_tail: str = ""
    warnings: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)  # e.g. n_spots, n_indexed

    @property
    def time(self) -> float | None:
        if self.status != OK or not self.times:
            return None
        return statistics.median(self.times)


def tail(path: Path, n: int = 25) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def run_stage(
    cmd: list[str], cwd: Path, log_name: str, timeout: float, repeats: int
) -> StageResult:
    """Run one stage `repeats` times. Any bad run makes the whole stage bad."""
    res = StageResult(status=OK, command=cmd)
    log_path = cwd / f"{log_name}.out"
    for rep in range(repeats):
        with open(log_path, "w") as log:
            log.write(f"# cwd: {cwd}\n# cmd: {' '.join(cmd)}\n# repeat {rep + 1}/{repeats}\n")
            log.flush()
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout
                )
            except subprocess.TimeoutExpired:
                res.status = TIMEOUT
                res.reason = f"killed after {timeout:.0f} s (repeat {rep + 1})"
                res.log_tail = tail(log_path)
                return res
            except OSError as e:
                res.status = FAILED
                res.reason = f"could not start: {e}"
                return res
            dt = time.perf_counter() - t0
        res.returncode = proc.returncode
        if proc.returncode != 0:
            res.status = FAILED
            sig = f" (signal {-proc.returncode})" if proc.returncode < 0 else ""
            res.reason = f"exit code {proc.returncode}{sig} after {dt:.1f} s (repeat {rep + 1})"
            res.log_tail = tail(log_path)
            return res
        res.times.append(dt)

    # Exit 0 is not enough: tracebacks/"Sorry" in the output are worth a warning.
    text = log_path.read_text(errors="replace")
    for marker in ("Traceback (most recent call last)", "Sorry:", "CUDA out of memory"):
        if marker in text:
            res.warnings.append(f"output contains {marker!r} despite exit code 0")
    return res


def validate_outputs(stage: str, cwd: Path, res: StageResult) -> None:
    """Load the stage outputs and check they contain usable data; mark INVALID if not."""
    from dials.array_family import flex
    from dxtbx.model.experiment_list import ExperimentList

    outputs = STAGE_IO[stage][1]
    missing = [f for f in outputs if not (cwd / f).is_file() or (cwd / f).stat().st_size == 0]
    if missing:
        res.status = INVALID
        res.reason = f"exit 0 but output file(s) missing or empty: {', '.join(missing)}"
        res.log_tail = tail(cwd / f"{stage}.out")
        return
    try:
        refl = flex.reflection_table.from_file(str(cwd / outputs[-1]))
        expts = None
        if outputs[0].endswith(".expt"):
            expts = ExperimentList.from_file(str(cwd / outputs[0]), check_format=False)
    except Exception as e:  # noqa: BLE001
        res.status = INVALID
        res.reason = f"could not read outputs: {type(e).__name__}: {e}"
        return

    s = res.summary
    s["n_reflections"] = len(refl)
    if expts is not None:
        s["n_experiments"] = len(expts)
        s["n_crystals"] = sum(1 for e in expts if e.crystal is not None)
    if stage == "find_spots":
        s["n_spots"] = len(refl)
        if len(refl) == 0:
            res.status, res.reason = INVALID, "no strong spots found"
    elif stage in ("index", "refine"):
        n = int(refl.get_flags(refl.flags.indexed).count(True))
        s["n_indexed"] = n
        if stage == "refine":
            s["n_used_in_refinement"] = int(
                refl.get_flags(refl.flags.used_in_refinement).count(True)
            )
        if n == 0 or s.get("n_crystals", 0) == 0:
            res.status, res.reason = INVALID, "no reflections indexed / no crystal model"
        elif stage == "refine" and s["n_used_in_refinement"] == 0:
            res.status, res.reason = INVALID, "no reflections used in refinement"
    elif stage == "integrate":
        s["n_integrated_sum"] = int(refl.get_flags(refl.flags.integrated_sum).count(True))
        s["n_integrated_prf"] = int(refl.get_flags(refl.flags.integrated_prf).count(True))
        if s["n_integrated_sum"] == 0 and s["n_integrated_prf"] == 0:
            res.status, res.reason = INVALID, "no reflections integrated"
    elif stage == "scale":
        good = ~refl.get_flags(refl.flags.bad_for_scaling, all=False)
        if "inverse_scale_factor" not in refl:
            res.status, res.reason = INVALID, "no inverse_scale_factor column"
            return
        s["n_scaled"] = int(good.count(True))
        if s["n_scaled"] == 0:
            res.status, res.reason = INVALID, "no reflections scaled"


def stage_command(impl: str, stage: str, ds: Dataset, nproc: int | None) -> list[str]:
    cmd = [f"{IMPLS[impl]}.{stage}", *STAGE_IO[stage][0], *BASE_ARGS[stage]]
    if impl == "dials" and nproc is not None:
        if stage == "find_spots":
            cmd.append(f"spotfinder.mp.nproc={nproc}")
        elif stage == "integrate":
            cmd.append(f"integration.mp.nproc={nproc}")
    cmd += ds.args.get(stage, [])
    cmd += ds.impl_args.get(impl, {}).get(stage, [])
    return cmd


# --------------------------------------------------------------------------------------
# Comparisons (dials_gpu vs dials, both stages OK)
# --------------------------------------------------------------------------------------


def _load(d: Path, stage: str):
    from dials.array_family import flex
    from dxtbx.model.experiment_list import ExperimentList

    outs = STAGE_IO[stage][1]
    refl = flex.reflection_table.from_file(str(d / outs[-1]))
    expts = (
        ExperimentList.from_file(str(d / outs[0]), check_format=False)
        if outs[0].endswith(".expt")
        else None
    )
    return expts, refl


def _rel(a, b):
    if a is None or b is None or a == 0:
        return None
    return (b - a) / abs(a)


def _niggli(cryst):
    """Reduced (Niggli) cell of the crystal's *primitive* lattice, so that the same lattice
    indexed in a centred setting (e.g. C2) and in a primitive one (P1) compare equal."""
    from cctbx import crystal

    cs = crystal.symmetry(unit_cell=cryst.get_unit_cell(), space_group=cryst.get_space_group())
    return cs.niggli_cell().unit_cell().parameters()


def compare_cells(e_ref, e_test) -> dict:
    """Compare unit cells of crystals indexed from the same data (sweep) in each chain.

    Experiments are paired by the files/image range they came from, not by position,
    since one chain may have failed to index a sweep the other indexed.
    """

    def by_data(expts):
        out = {}
        for e in expts:
            if e.crystal is not None:
                out.setdefault(_imageset_key(e), e.crystal)
        return out

    a_map, b_map = by_data(e_ref), by_data(e_test)
    common = [k for k in a_map if k in b_map]
    out = {
        "n_sweeps_indexed_dials": len(a_map),
        "n_sweeps_indexed_dials_gpu": len(b_map),
        "n_sweeps_compared": len(common),
    }
    if not common:
        out["max_cell_length_rel_diff"] = None
        out["max_cell_angle_diff_deg"] = None
        return out
    len_rel = ang = 0.0
    for k in common:
        a, b = _niggli(a_map[k]), _niggli(b_map[k])
        len_rel = max(len_rel, *(abs(b[i] - a[i]) / a[i] for i in range(3)))
        # A cell with angles near 90 deg can reduce to the all-acute or the all-obtuse
        # Niggli form (e.g. beta 99 deg vs 81 deg for the same lattice), so compare each
        # angle with the other cell's angle and with its supplement.
        ang = max(ang, *(min(abs(b[i] - a[i]), abs(180.0 - a[i] - b[i])) for i in range(3, 6)))
    out.update(
        cell_dials=[round(x, 4) for x in _niggli(a_map[common[0]])],
        cell_dials_gpu=[round(x, 4) for x in _niggli(b_map[common[0]])],
        max_cell_length_rel_diff=len_rel,
        max_cell_angle_diff_deg=ang,
    )
    return out


def _imageset_key(expt) -> tuple:
    """Identify an experiment's data (files + image range), independent of its id."""
    iset = expt.imageset
    try:
        paths = tuple(iset.paths())
    except Exception:  # noqa: BLE001
        paths = (str(iset),)
    rng = tuple(expt.scan.get_image_range()) if expt.scan is not None else None
    return paths, rng


def _groups(refl, expts) -> dict[tuple, np.ndarray]:
    """Row indices of `refl` grouped by (data identity, panel).

    Experiment ids are not comparable between the two chains (e.g. if one indexer
    drops a sweep, the remaining sweep becomes id 0), so when an ExperimentList is
    given, each id is mapped to the files and image range it came from. Without one
    (strong spots, whose ids both come from the shared imported.expt), the id is used.
    """
    ids = refl["id"].as_numpy_array() if "id" in refl else np.zeros(len(refl), int)
    panel = (
        refl["panel"].as_numpy_array().astype(np.int64)
        if "panel" in refl
        else np.zeros(len(refl), np.int64)
    )
    out: dict[tuple, np.ndarray] = {}
    for i in np.unique(ids):
        if i < 0:
            continue  # unindexed
        data = _imageset_key(expts[int(i)]) if expts is not None else int(i)
        for p in np.unique(panel[ids == i]):
            key = (data, int(p))
            sel = np.nonzero((ids == i) & (panel == p))[0]
            out[key] = np.concatenate([out[key], sel]) if key in out else sel
    return out


def match_by_position(r_ref, r_test, column: str, tol: float = 2.0, e_ref=None, e_test=None):
    """Nearest-neighbour match of r_test onto r_ref by a .px xyz column, within the
    same data (sweep) and panel.

    Returns (idx_ref, idx_test) of one-to-one matches within `tol` (pixels/frames).
    """
    from scipy.spatial import cKDTree

    xyz_ref = r_ref[column].as_numpy_array()
    xyz_test = r_test[column].as_numpy_array()
    g_ref = _groups(r_ref, e_ref)
    g_test = _groups(r_test, e_test)
    out_ref, out_test = [], []
    for g in g_ref.keys() & g_test.keys():
        i_ref = g_ref[g]
        i_test = g_test[g]
        tree = cKDTree(xyz_ref[i_ref])
        d, j = tree.query(xyz_test[i_test], distance_upper_bound=tol)
        ok = np.isfinite(d)
        # keep one-to-one: for each ref spot, the closest test spot only
        best: dict[int, tuple[float, int]] = {}
        for k_test, k_ref, dist in zip(i_test[ok], i_ref[j[ok]], d[ok]):
            if k_ref not in best or dist < best[k_ref][0]:
                best[k_ref] = (dist, k_test)
        for k_ref, (_, k_test) in best.items():
            out_ref.append(k_ref)
            out_test.append(k_test)
    return np.asarray(out_ref, dtype=np.int64), np.asarray(out_test, dtype=np.int64)


def _corr(a, b) -> float | None:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def compare_find_spots(d_ref: Path, d_test: Path) -> dict:
    _, r_ref = _load(d_ref, "find_spots")
    _, r_test = _load(d_test, "find_spots")
    i_ref, i_test = match_by_position(r_ref, r_test, "xyzobs.px.value")
    out = {
        "n_spots_dials": len(r_ref),
        "n_spots_dials_gpu": len(r_test),
        "spot_count_rel_diff": _rel(len(r_ref), len(r_test)),
        "n_matched": len(i_ref),
        "match_pct": 100.0 * len(i_ref) / max(len(r_ref), 1),
    }
    if len(i_ref):
        diff = (
            r_test["xyzobs.px.value"].as_numpy_array()[i_test]
            - r_ref["xyzobs.px.value"].as_numpy_array()[i_ref]
        )
        rms = np.sqrt(np.mean(diff**2, axis=0))
        out.update(
            centroid_rmsd_x_px=float(rms[0]),
            centroid_rmsd_y_px=float(rms[1]),
            centroid_rmsd_z_img=float(rms[2]),
        )
    return out


def compare_index(d_ref: Path, d_test: Path) -> dict:
    e_ref, r_ref = _load(d_ref, "index")
    e_test, r_test = _load(d_test, "index")
    n_ref = int(r_ref.get_flags(r_ref.flags.indexed).count(True))
    n_test = int(r_test.get_flags(r_test.flags.indexed).count(True))
    out = {
        "n_indexed_dials": n_ref,
        "n_indexed_dials_gpu": n_test,
        "indexed_rel_diff": _rel(n_ref, n_test),
        "n_crystals_dials": sum(1 for e in e_ref if e.crystal is not None),
        "n_crystals_dials_gpu": sum(1 for e in e_test if e.crystal is not None),
    }
    out.update(compare_cells(e_ref, e_test))
    return out


def refined_rmsds(refl) -> dict:
    sel = refl.get_flags(refl.flags.used_in_refinement)
    r = refl.select(sel)
    if len(r) == 0:
        return {}
    obs = r["xyzobs.px.value"].as_numpy_array()
    cal = r["xyzcal.px"].as_numpy_array()
    obs_mm = r["xyzobs.mm.value"].as_numpy_array()
    cal_mm = r["xyzcal.mm"].as_numpy_array()
    return {
        "rmsd_x_px": float(np.sqrt(np.mean((obs[:, 0] - cal[:, 0]) ** 2))),
        "rmsd_y_px": float(np.sqrt(np.mean((obs[:, 1] - cal[:, 1]) ** 2))),
        "rmsd_phi_deg": float(np.degrees(np.sqrt(np.mean((obs_mm[:, 2] - cal_mm[:, 2]) ** 2)))),
        "n_used": len(r),
    }


def compare_refine(d_ref: Path, d_test: Path) -> dict:
    e_ref, r_ref = _load(d_ref, "refine")
    e_test, r_test = _load(d_test, "refine")
    a, b = refined_rmsds(r_ref), refined_rmsds(r_test)
    out = {f"{k}_dials": v for k, v in a.items()}
    out.update({f"{k}_dials_gpu": v for k, v in b.items()})
    if a and b:
        out["rmsd_xy_rel_diff"] = max(
            abs(_rel(a["rmsd_x_px"], b["rmsd_x_px"]) or 0),
            abs(_rel(a["rmsd_y_px"], b["rmsd_y_px"]) or 0),
        )
    out.update(compare_cells(e_ref, e_test))
    return out


def compare_integrate(d_ref: Path, d_test: Path) -> dict:
    e_ref, r_ref = _load(d_ref, "integrate")
    e_test, r_test = _load(d_test, "integrate")
    out = {}
    for kind, flag in (("sum", r_ref.flags.integrated_sum), ("prf", r_ref.flags.integrated_prf)):
        col = f"intensity.{kind}.value"
        a = r_ref.select(r_ref.get_flags(flag))
        b = r_test.select(r_test.get_flags(flag))
        out[f"n_{kind}_dials"] = len(a)
        out[f"n_{kind}_dials_gpu"] = len(b)
        out[f"{kind}_count_rel_diff"] = _rel(len(a), len(b))
        if len(a) == 0 or len(b) == 0 or col not in a or col not in b:
            continue
        i_a, i_b = match_by_position(a, b, "xyzcal.px", tol=1.0, e_ref=e_ref, e_test=e_test)
        out[f"n_{kind}_matched"] = len(i_a)
        out[f"{kind}_matched_frac"] = len(i_a) / len(a)
        out[f"corr_{kind}"] = _corr(
            a[col].as_numpy_array()[i_a], b[col].as_numpy_array()[i_b]
        )
    return out


def merging_stats(expts, refl) -> dict:
    import iotbx.merging_statistics
    from dials.algorithms.scaling.scaling_library import scaled_data_as_miller_array

    # Only experiments with a crystal: some programs (e.g. mxi) keep a crystal-less
    # experiment per sweep alongside the crystal's.
    from dxtbx.model.experiment_list import ExperimentList

    expts = ExperimentList([e for e in expts if e.crystal is not None])
    arr = scaled_data_as_miller_array([refl], expts, anomalous_flag=False)
    stats = iotbx.merging_statistics.dataset_statistics(
        i_obs=arr,
        n_bins=10,
        anomalous=False,
        use_internal_variance=False,
        eliminate_sys_absent=False,
        assert_is_not_unique_set_under_symmetry=False,
    )
    ov = stats.overall
    return {
        "space_group": str(arr.space_group_info()),
        "d_min": ov.d_min,
        "n_obs": ov.n_obs,
        "n_uniq": ov.n_uniq,
        "multiplicity": ov.mean_redundancy,
        "completeness_pct": ov.completeness * 100.0,
        "i_over_sigma": ov.i_over_sigma_mean,
        "r_merge": ov.r_merge,
        "r_pim": ov.r_pim,
        "cc_half": ov.cc_one_half,
    }


def compare_scale(d_ref: Path, d_test: Path) -> dict:
    e_ref, r_ref = _load(d_ref, "scale")
    e_test, r_test = _load(d_test, "scale")
    out = {}
    try:
        out["merging_dials"] = merging_stats(e_ref, r_ref)
    except Exception as e:  # noqa: BLE001
        out["merging_dials"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        out["merging_dials_gpu"] = merging_stats(e_test, r_test)
    except Exception as e:  # noqa: BLE001
        out["merging_dials_gpu"] = {"error": f"{type(e).__name__}: {e}"}
    m_a, m_b = out["merging_dials"], out["merging_dials_gpu"]
    if "cc_half" in m_a and "cc_half" in m_b:
        out["cc_half_diff"] = m_b["cc_half"] - m_a["cc_half"]
        out["i_over_sigma_rel_diff"] = _rel(m_a["i_over_sigma"], m_b["i_over_sigma"])

    def good(r):
        sel = ~r.get_flags(r.flags.bad_for_scaling, all=False)
        sel &= ~r.get_flags(r.flags.outlier_in_scaling)
        sel &= r["inverse_scale_factor"] > 0
        return r.select(sel)

    a, b = good(r_ref), good(r_test)
    if len(a) and len(b):
        i_a, i_b = match_by_position(a, b, "xyzcal.px", tol=1.0, e_ref=e_ref, e_test=e_test)
        out["n_scaled_matched"] = len(i_a)
        out["scaled_matched_frac"] = len(i_a) / len(a)
        ia = (a["intensity.scale.value"] / a["inverse_scale_factor"]).as_numpy_array()
        ib = (b["intensity.scale.value"] / b["inverse_scale_factor"]).as_numpy_array()
        out["corr_scaled"] = _corr(ia[i_a], ib[i_b])
    return out


COMPARATORS = {
    "find_spots": compare_find_spots,
    "index": compare_index,
    "refine": compare_refine,
    "integrate": compare_integrate,
    "scale": compare_scale,
}


def divergence_flags(stage: str, c: dict) -> list[str]:
    t = TOLERANCES
    flags = []

    def over(key, limit, label, absval=True):
        v = c.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return
        if (abs(v) if absval else v) > limit:
            flags.append(f"{label} = {v:.4g} (limit {limit:g})")

    def under(key, limit, label):
        v = c.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            if key.startswith("corr_") and key.replace("corr_", "n_") + "_dials" in c:
                flags.append(f"{label} could not be computed")
            return
        if v < limit:
            flags.append(f"{label} = {v:.4g} (limit {limit:g})")

    if stage == "find_spots":
        under("match_pct", t["spot_match_pct_min"], "spot match %")
        over("spot_count_rel_diff", t["spot_count_rel_diff_max"], "spot count rel. diff")
    if stage in ("index", "refine"):
        over("max_cell_length_rel_diff", t["cell_length_rel_diff_max"], "cell length rel. diff")
        over("max_cell_angle_diff_deg", t["cell_angle_diff_max_deg"], "cell angle diff (deg)")
        if c.get("n_sweeps_indexed_dials") != c.get("n_sweeps_indexed_dials_gpu"):
            flags.append(
                f"number of sweeps indexed differs ({c.get('n_sweeps_indexed_dials')} vs "
                f"{c.get('n_sweeps_indexed_dials_gpu')})"
            )
    if stage == "index":
        over("indexed_rel_diff", t["indexed_rel_diff_max"], "n indexed rel. diff")
        if c.get("n_crystals_dials") != c.get("n_crystals_dials_gpu"):
            flags.append(
                f"number of crystals differs ({c.get('n_crystals_dials')} vs "
                f"{c.get('n_crystals_dials_gpu')})"
            )
    if stage == "refine":
        over("rmsd_xy_rel_diff", t["rmsd_xy_rel_diff_max"], "refined RMSD rel. diff")
    if stage == "integrate":
        for k in ("sum", "prf"):
            over(f"{k}_count_rel_diff", t["integrated_rel_diff_max"], f"n integrated ({k}) rel. diff")
            under(f"{k}_matched_frac", t["matched_frac_min"], f"matched fraction ({k})")
            under(f"corr_{k}", t["intensity_corr_min"], f"r(I_{k})")
    if stage == "scale":
        under("corr_scaled", t["intensity_corr_min"], "r(I_scaled)")
        under("scaled_matched_frac", t["matched_frac_min"], "matched fraction (scaled)")
        over("cc_half_diff", t["cc_half_diff_max"], "CC1/2 diff")
        over("i_over_sigma_rel_diff", t["i_over_sigma_rel_diff_max"], "<I/sig> rel. diff")
        for side in ("merging_dials", "merging_dials_gpu"):
            if "error" in c.get(side, {}):
                flags.append(f"{side} failed: {c[side]['error']}")
    return flags


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def run_dataset(ds: Dataset, workdir: Path, opts) -> dict:
    root = workdir / ds.name
    timeout = ds.timeout or opts.timeout
    result: dict = {
        "name": ds.name,
        "n_images_files": len(ds.images),
        "images": ds.images if len(ds.images) <= 20 else ds.images[:3] + ["..."] + ds.images[-3:],
        "import": None,
        "stages": {impl: {} for impl in opts.implementations},
        "comparisons": {},
    }
    print(f"\n=== {ds.name}: {len(ds.images)} file(s) ===", flush=True)

    imp_dir = root / "import"
    imp_dir.mkdir(parents=True, exist_ok=True)
    imp_cmd = ["dials.import", *ds.images, "output.experiments=imported.expt", *ds.args.get("import", [])]
    imp = run_stage(imp_cmd, imp_dir, "import", timeout, 1)
    if imp.status == OK and not (imp_dir / "imported.expt").is_file():
        imp.status, imp.reason = INVALID, "imported.expt was not written"
    if imp.status == OK:
        from dxtbx.model.experiment_list import ExperimentList

        try:
            el = ExperimentList.from_file(str(imp_dir / "imported.expt"), check_format=False)
            imp.summary = {
                "n_experiments": len(el),
                "n_images": sum(len(e.imageset) for e in el),
            }
        except Exception as e:  # noqa: BLE001
            imp.status, imp.reason = INVALID, f"could not read imported.expt: {e}"
    result["import"] = asdict(imp)
    print(f"  import       {_fmt_status(imp)}", flush=True)

    for impl in opts.implementations:
        d = root / impl
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        if imp.status == OK:
            shutil.copy(imp_dir / "imported.expt", d / "imported.expt")
        upstream_ok = imp.status == OK
        for stage in STAGES:
            if not upstream_ok:
                res = StageResult(status=SKIPPED, reason="earlier stage did not succeed")
            else:
                cmd = stage_command(impl, stage, ds, opts.nproc)
                res = run_stage(cmd, d, stage, timeout, opts.repeats)
                if res.status == OK:
                    validate_outputs(stage, d, res)
            upstream_ok = res.status == OK
            result["stages"][impl][stage] = {**asdict(res), "time": res.time}
            print(f"  {impl:<10} {stage:<11} {_fmt_status(res)}", flush=True)

    if set(IMPLS) <= set(opts.implementations):
        for stage in STAGES:
            a = result["stages"]["dials"][stage]
            b = result["stages"]["dials_gpu"][stage]
            if a["status"] != OK or b["status"] != OK:
                result["comparisons"][stage] = {"skipped": "not OK in both implementations"}
                continue
            try:
                c = COMPARATORS[stage](root / "dials", root / "dials_gpu")
                c["divergent"] = divergence_flags(stage, c)
            except Exception as e:  # noqa: BLE001
                c = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(),
                     "divergent": [f"comparison failed: {type(e).__name__}: {e}"]}
            c["speedup"] = a["time"] / b["time"] if b["time"] else None
            result["comparisons"][stage] = c
    return result


def _fmt_status(res: StageResult) -> str:
    if res.status == OK:
        t = res.time
        extra = ", ".join(f"{k}={v}" for k, v in res.summary.items() if k.startswith("n_"))
        warn = f"  [WARNING: {'; '.join(res.warnings)}]" if res.warnings else ""
        return f"OK      {t:8.2f} s  {extra}{warn}"
    return f"{res.status:<7} {res.reason}"


def _num(v, fmt="{:.3f}"):
    if v is None:
        return "–"
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    return fmt.format(v)


def _cell(stage_res: dict) -> str:
    if stage_res["status"] == OK:
        return f"{stage_res['time']:.2f} s"
    return f"**{stage_res['status']}**"


def write_report(results: dict, path: Path) -> list[str]:
    """Write a markdown report; return a list of problems for the console summary."""
    impls = results["implementations"]
    both = set(IMPLS) <= set(impls)
    lines = [
        "# dials vs dials_gpu pipeline benchmark",
        "",
        f"- Date: {results['date']}",
        f"- Host: {results['system']['host']}, {results['system']['cpu_count']} CPUs, "
        f"GPU: {results['system'].get('gpu') or 'unknown'}",
        f"- DIALS: {results['system'].get('dials_version')}, dials_gpu commit "
        f"{results['system'].get('dials_gpu_commit')}",
        f"- Repeats per stage: {results['options']['repeats']} (median shown), "
        f"timeout {results['options']['timeout']:.0f} s, DIALS nproc: "
        f"{results['options']['nproc'] or 'DIALS default'}",
        "",
        "Times are command-line wall-clock times, so each dials_gpu stage includes "
        "Python start-up and CUDA initialisation (about 1.5–2 s). A time is shown only "
        "for an OK stage; FAILED, TIMEOUT, INVALID and SKIPPED stages are never timed.",
        "",
        "## Timings",
        "",
    ]
    cols = ["Dataset", "Stage", *impls] + (["Speed-up"] if both else [])
    lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    problems: list[str] = []
    for ds in results["datasets"]:
        imp = ds["import"]
        lines.append(f"| {ds['name']} | import (shared) | {_cell({**imp, 'time': (imp['times'] or [None])[0]})} |"
                     + " |" * (len(impls) - 1) + (" |" if both else ""))
        if imp["status"] != OK:
            problems.append(f"{ds['name']}: import {imp['status']}: {imp['reason']}")
        totals = {}
        for stage in STAGES:
            row = f"| {ds['name']} | {stage} | "
            row += " | ".join(_cell(ds["stages"][i][stage]) for i in impls)
            if both:
                sp = ds["comparisons"].get(stage, {}).get("speedup")
                row += f" | {_num(sp, '{:.2f}×')}"
            lines.append(row + " |")
            for i in impls:
                r = ds["stages"][i][stage]
                if r["status"] in (FAILED, TIMEOUT, INVALID):
                    problems.append(f"{ds['name']}: {i}.{stage} {r['status']}: {r['reason']}")
                for w in r.get("warnings", []):
                    problems.append(f"{ds['name']}: {i}.{stage} warning: {w}")
        for i in impls:
            st = [ds["stages"][i][s] for s in STAGES]
            totals[i] = sum(s["time"] for s in st) if all(s["status"] == OK for s in st) else None
        row = f"| {ds['name']} | **total** | " + " | ".join(
            f"**{totals[i]:.2f} s**" if totals[i] is not None else "**incomplete**" for i in impls
        )
        if both:
            sp = (totals["dials"] / totals["dials_gpu"]) if totals["dials"] and totals["dials_gpu"] else None
            row += f" | **{_num(sp, '{:.2f}×')}**"
        lines.append(row + " |")

    if both:
        lines += ["", "## Closeness of dials_gpu to dials", ""]
        lines += ["| Dataset | Stage | Key metrics | Verdict |", "|---|---|---|---|"]
        for ds in results["datasets"]:
            for stage in STAGES:
                c = ds["comparisons"].get(stage, {})
                if "skipped" in c:
                    lines.append(f"| {ds['name']} | {stage} | – | not compared (stage not OK in both) |")
                    continue
                metrics = _key_metrics(stage, c)
                if c.get("divergent"):
                    verdict = "**DIVERGENT**: " + "; ".join(c["divergent"])
                    problems.append(f"{ds['name']}: {stage} DIVERGENT: " + "; ".join(c["divergent"]))
                else:
                    verdict = "within tolerance"
                lines.append(f"| {ds['name']} | {stage} | {metrics} | {verdict} |")

        lines += ["", "## Merging statistics after scaling", ""]
        lines += ["| Dataset | Impl | Space group | d_min | Compl. % | Mult. | <I/σ> | R_merge | R_pim | CC½ |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for ds in results["datasets"]:
            c = ds["comparisons"].get("scale", {})
            for impl in IMPLS:
                m = c.get(f"merging_{impl}")
                if not m or "error" in m:
                    continue
                lines.append(
                    f"| {ds['name']} | {impl} | {m['space_group']} | {_num(m['d_min'], '{:.2f}')} | "
                    f"{_num(m['completeness_pct'], '{:.1f}')} | {_num(m['multiplicity'], '{:.2f}')} | "
                    f"{_num(m['i_over_sigma'], '{:.2f}')} | {_num(m['r_merge'])} | {_num(m['r_pim'])} | "
                    f"{_num(m['cc_half'], '{:.4f}')} |"
                )

    if problems:
        lines += ["", "## Problems", ""] + [f"- {p}" for p in problems]
        lines += ["", "### Failure details", ""]
        for ds in results["datasets"]:
            entries = [("import", ds["import"])] + [
                (f"{i}.{s}", ds["stages"][i][s]) for i in impls for s in STAGES
            ]
            for label, r in entries:
                if r["status"] in (FAILED, TIMEOUT, INVALID):
                    lines += [f"**{ds['name']} / {label}: {r['status']}** — {r['reason']}",
                              f"`{' '.join(r['command'])}`", "", "```", r["log_tail"] or "(no output)", "```", ""]
    path.write_text("\n".join(lines) + "\n")
    return problems


def _cell_diff_str(c: dict) -> str:
    sweeps = (f"sweeps indexed {c.get('n_sweeps_indexed_dials')} vs "
              f"{c.get('n_sweeps_indexed_dials_gpu')}")
    if c.get("max_cell_length_rel_diff") is None:
        return f"{sweeps}; no sweep indexed by both, cells not compared"
    return (f"{sweeps}; Δcell {100 * c['max_cell_length_rel_diff']:.3f}% / "
            f"{c['max_cell_angle_diff_deg']:.3f}° over {c['n_sweeps_compared']} common sweep(s)")


def _key_metrics(stage: str, c: dict) -> str:
    if "error" in c:
        return f"comparison error: {c['error']}"
    if stage == "find_spots":
        return (f"spots {c['n_spots_dials']} vs {c['n_spots_dials_gpu']}, "
                f"matched {c['match_pct']:.1f}%, centroid RMSD x/y/z "
                f"{_num(c.get('centroid_rmsd_x_px'))}/{_num(c.get('centroid_rmsd_y_px'))}/"
                f"{_num(c.get('centroid_rmsd_z_img'))}")
    if stage == "index":
        return (f"indexed {c['n_indexed_dials']} vs {c['n_indexed_dials_gpu']}, "
                + _cell_diff_str(c))
    if stage == "refine":
        return (f"RMSD x,y,φ {_num(c.get('rmsd_x_px_dials'))},{_num(c.get('rmsd_y_px_dials'))},"
                f"{_num(c.get('rmsd_phi_deg_dials'))} vs {_num(c.get('rmsd_x_px_dials_gpu'))},"
                f"{_num(c.get('rmsd_y_px_dials_gpu'))},{_num(c.get('rmsd_phi_deg_dials_gpu'))}; "
                + _cell_diff_str(c))
    if stage == "integrate":
        return (f"sum {c['n_sum_dials']} vs {c['n_sum_dials_gpu']} r={_num(c.get('corr_sum'), '{:.5f}')}; "
                f"prf {c['n_prf_dials']} vs {c['n_prf_dials_gpu']} r={_num(c.get('corr_prf'), '{:.5f}')}")
    if stage == "scale":
        return (f"r(I_scaled)={_num(c.get('corr_scaled'), '{:.5f}')} on {c.get('n_scaled_matched', 0)} matched, "
                f"ΔCC½={_num(c.get('cc_half_diff'), '{:+.4f}')}")
    return ""


def system_info() -> dict:
    info = {"host": platform.node(), "cpu_count": os.cpu_count(), "python": sys.version.split()[0]}
    try:
        info["gpu"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        info["gpu"] = None
    try:
        out = subprocess.run(["dials.version"], capture_output=True, text=True, timeout=60).stdout
        info["dials_version"] = out.splitlines()[0] if out else None
    except (OSError, subprocess.SubprocessError):
        info["dials_version"] = None
    try:
        info["dials_gpu_commit"] = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "describe", "--always", "--dirty"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        info["dials_gpu_commit"] = None
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="JSON file describing datasets (see datasets.example.json)")
    p.add_argument("--dataset", action="append", metavar="NAME=PATH[,PATH...]",
                   help="add a dataset; paths may be globs (quote them). Repeatable.")
    p.add_argument("--workdir", default="pipeline_bench", help="output directory (default: %(default)s)")
    p.add_argument("--timeout", type=float, default=3600, help="per-stage timeout in seconds (default: %(default)s)")
    p.add_argument("--repeats", type=int, default=1, help="run each stage N times; median time reported")
    p.add_argument("--nproc", type=int, default=None,
                   help="nproc for dials.find_spots/dials.integrate (default: DIALS default)")
    p.add_argument("--implementations", nargs="+", choices=list(IMPLS), default=list(IMPLS),
                   help="which implementations to run (default: both)")
    p.add_argument("--only", nargs="+", metavar="NAME", help="run only these datasets from the config")
    opts = p.parse_args(argv)

    datasets = load_datasets(opts)
    if opts.only:
        unknown = set(opts.only) - {d.name for d in datasets}
        if unknown:
            raise SystemExit(f"error: unknown dataset(s): {', '.join(sorted(unknown))}")
        datasets = [d for d in datasets if d.name in opts.only]

    needed = ["dials.import"] + [f"{IMPLS[i]}.{s}" for i in opts.implementations for s in STAGES]
    missing = [c for c in needed if shutil.which(c) is None]
    if missing:
        raise SystemExit(f"error: commands not on PATH: {', '.join(missing)}. "
                         "Activate the DIALS environment and run build.sh first.")

    workdir = Path(opts.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    results = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "system": system_info(),
        "options": {"timeout": opts.timeout, "repeats": opts.repeats, "nproc": opts.nproc},
        "implementations": opts.implementations,
        "tolerances": TOLERANCES,
        "datasets": [],
    }
    for ds in datasets:
        results["datasets"].append(run_dataset(ds, workdir, opts))
        # Save after every dataset so a long run is never lost.
        (workdir / "results.json").write_text(json.dumps(results, indent=2, default=str))
        problems = write_report(results, workdir / "report.md")

    print("\n" + (workdir / "report.md").read_text())
    print(f"Results: {workdir / 'results.json'}\nReport:  {workdir / 'report.md'}")
    if problems:
        print(f"\n{len(problems)} problem(s) found; see the Problems section above.")
    stage_bad = any(
        ds["import"]["status"] != OK
        or any(ds["stages"][i][s]["status"] != OK for i in opts.implementations for s in STAGES)
        for ds in results["datasets"]
    )
    return 1 if stage_bad else 0


if __name__ == "__main__":
    sys.exit(main())
