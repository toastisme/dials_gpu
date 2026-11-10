"""
Compare dials (CPU), dials_gpu and mxi, stage by stage, on one or more imported.expt files.

For each .expt file the same pipeline runs with each implementation:

    find_spots -> index -> refine -> integrate -> scale

plus timed variants that are not fed forward (static refinement; dials integration with
nproc=1; mxi spot finding and integration with --gpu). Each implementation's stages use
its own previous outputs.

  * dials, dials_gpu: in-process (Python API), each in its own worker process. This keeps
    CUDA out of the DIALS process (DIALS forks workers) and means a crash in one cannot
    take the others down. Times exclude Python/CUDA process start-up.
  * mxi (https://github.com/APS-24ID/mxi): one `mxi <step>` command per stage, reading
    and writing DIALS .expt/.refl files. Its main time is the `total` mxi reports with
    --timing (its own work, including reading and writing those files); the command's
    wall-clock time, which adds process start-up and, for --gpu, CUDA initialisation,
    is shown beside it. mxi reads NXmx (HDF5) images only: for CBF data it fails at
    spot finding, and that is reported as such.

Repeats: dials runs each stage once (--cpu-repeats), dials_gpu --gpu-repeats times in
one process (default 2: the first call, then "warm" calls that can reuse in-process
caches), mxi --mxi-repeats times (default 2, each a fresh process).

Failures are never timed. A stage that raises or exits non-zero is FAILED (with the
error or log), a worker or command killed by a signal is CRASHED, one over --timeout is
TIMEOUT, and later stages are SKIPPED. Every main stage's output is also checked for
real content -- strong spots, indexed reflections and a crystal, reflections used in
refinement, integrated reflections, and scale factors that are not all exactly 1 -- and
a stage that exits cleanly without it is INVALID. Results are saved as they come, so a
crash keeps everything measured before it.

Closeness to dials is checked at every stage (spot matching, cell, RMSDs, intensity
correlations, merging statistics) with the comparisons and tolerances of
benchmarks/benchmark_pipeline.py. For dials_gpu, a result outside them is DIVERGENT. mxi
is an independent implementation, so its comparison is shown for information only.

Usage (inside the DIALS environment, with dials_gpu installed, e.g. via ./build.sh, and
mxi built, e.g. cmake -DMXI_CUDA=ON; see the README):

    libtbx.python benchmarks/compare_stages.py /data/i19/imported.expt --mxi /path/to/mxi
    libtbx.python benchmarks/compare_stages.py a/imported.expt b/imported.expt \\
        --workdir /scratch/compare --cpu-integrate-nproc1 \\
        --space-group C2 --unit-cell 53.17,61.25,69.29,90,93.04,90 --index-phil "max_refine=5"

--space-group / --unit-cell give all three implementations the same known symmetry
(dials: space_group= unit_cell=; mxi: index --known-cell/--known-space-group, scale
--space-group). Without them everything is processed in P1; mxi's scaling then usually
has no symmetry equivalents to fit and is reported INVALID.

--cpu-integrate-nproc1 is recommended: DIALS with nproc>1 splits integration into jobs
and its results differ slightly from nproc=1, which is what dials_gpu reproduces, so the
nproc=1 output is the like-for-like reference for the integration comparisons.

PHIL options take the same short or full parameter names as the dials commands; an
unknown or ambiguous name stops the run before any work starts. --mxi-<step>-args are
passed to that mxi command as they are.

Outputs: <workdir>/results.json, <workdir>/report.md, and per-dataset stage outputs
under <workdir>/<dataset>/{dials,dials_gpu,mxi}/. Exit code 1 if any stage did not
succeed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMPLS = ("dials", "dials_gpu", "mxi")
OK, FAILED, CRASHED, TIMEOUT, SKIPPED, INVALID = "OK", "FAILED", "CRASHED", "TIMEOUT", "SKIPPED", "INVALID"
BAD = (FAILED, CRASHED, TIMEOUT, INVALID)

# (stage key, label, chained?) in run order. Chained stages feed the next stage and are
# compared between implementations; the others are extra timed variants.
STAGES = [
    ("find_spots", "Spot finding", True),
    ("index", "Indexing", True),
    ("refine_static", "Refinement (static)", False),
    ("refine", "Refinement (default: scan-varying where the scan allows)", True),
    ("integrate_nproc1", "Integration (dials nproc=1)", False),
    ("integrate", "Integration", True),
    ("scale", "Scaling", True),
]
CHAIN = [s for s, _, chained in STAGES if chained]
MXI_GPU_STAGES = ("find_spots_gpu", "integrate_gpu")

# File names match benchmarks/benchmark_pipeline.py so its comparisons can be reused.
OUTPUTS = {
    "find_spots": (None, "strong.refl"),
    "index": ("indexed.expt", "indexed.refl"),
    "refine": ("refined.expt", "refined.refl"),
    "integrate": ("integrated.expt", "integrated.refl"),
    "scale": ("scaled.expt", "scaled.refl"),
}

# PHIL scope for each --<stage>-phil option, as used by the matching dials command.
PHIL_SCOPES = {
    "find": ("dials.command_line.find_spots", "phil_scope"),
    "index": ("dials.command_line.index", "phil_scope"),
    "refine": ("dials.command_line.refine", "working_phil"),
    "integrate": ("dials.command_line.integrate", "phil_scope"),
    "scale": ("dials.command_line.scale", "phil_scope"),
}
MXI_STEPS = ("find", "index", "refine", "integrate", "scale")


def extract_phil(scope, *items):
    """Extract params from assignments such as "space_group=C2" or full paths.

    Uses the command-line argument interpreter, like the dials tools, so short names
    resolve; an unknown or ambiguous name raises instead of being silently ignored.
    Each item is a string (one assignment) or a list of them; empty items are skipped.
    """
    args = []
    for it in items:
        args.extend([it] if isinstance(it, str) else list(it))
    args = [a for a in args if a]
    sources = scope.command_line_argument_interpreter().process_args(args) if args else []
    return scope.fetch(sources=sources).extract()


def check_phil(cfg: dict) -> None:
    """Fail early (before any worker starts) if a --<stage>-phil option does not parse."""
    import importlib

    for stage, (module, attr) in PHIL_SCOPES.items():
        if cfg[f"{stage}_phil"]:
            scope = getattr(importlib.import_module(module), attr)
            try:
                extract_phil(scope, cfg[f"{stage}_phil"])
            except Exception as e:  # noqa: BLE001
                raise SystemExit(f"error: --{stage}-phil: {e}")


# --------------------------------------------------------------------------------------
# dials / dials_gpu worker: runs every stage for one implementation in this process
# --------------------------------------------------------------------------------------


def worker(impl: str, expt_path: str, outdir: Path, cfg: dict) -> None:
    gpu = impl == "dials_gpu"
    if gpu:
        import torch  # import before scitbx, as elsewhere in dials_gpu

        if not torch.cuda.is_available():
            raise SystemExit("dials_gpu worker: CUDA is not available")

    from dials.array_family import flex
    from dials.command_line.find_spots import phil_scope as find_phil
    from dials.command_line.index import phil_scope as index_phil
    from dials.command_line.integrate import phil_scope as integrate_phil
    from dials.command_line.refine import working_phil as refine_phil
    from dials.command_line.scale import phil_scope as scale_phil
    from dxtbx.model.experiment_list import ExperimentListFactory

    if gpu:
        from dials_gpu.command_line.find_spots import from_observations_gpu
        from dials_gpu.command_line.index import index as index_fn
        from dials_gpu.command_line.integrate import run_integration as integrate_fn
        from dials_gpu.command_line.refine import run_dials_refine as refine_fn
        from dials_gpu.command_line.scale import run_scaling as scale_fn

        def sync():
            torch.cuda.synchronize()

        def find_fn(expts, params):
            return from_observations_gpu(expts, params)
    else:
        from dials.command_line.index import index as index_fn
        from dials.command_line.integrate import run_integration as integrate_fn
        from dials.command_line.refine import run_dials_refine as refine_fn
        from dials.command_line.scale import run_scaling as scale_fn

        def sync():
            pass

        def find_fn(expts, params):
            return flex.reflection_table.from_observations(expts, params)

    repeats = cfg["gpu_repeats"] if gpu else cfg["cpu_repeats"]
    nproc = cfg["nproc"]
    state = {"stages": {}}
    state_path = outdir / "worker_results.json"

    def save():
        state_path.write_text(json.dumps(state, indent=1))

    phil = extract_phil

    def stage(key, fn, n=repeats):
        """Run fn n times; record times, or the error. Returns the last result or None."""
        rec = {"status": OK, "times": []}
        state["stages"][key] = {**rec, "status": "RUNNING"}
        save()
        out = None
        for _ in range(n):
            sync()
            t0 = time.perf_counter()
            try:
                out = fn()
                sync()
            except Exception as e:  # noqa: BLE001
                rec.update(status=FAILED, error=f"{type(e).__name__}: {e}",
                           traceback=traceback.format_exc(limit=8))
                rec["times"] = []
                break
            rec["times"].append(time.perf_counter() - t0)
        state["stages"][key] = rec
        save()
        msg = "  ".join(f"{t:8.3f} s" for t in rec["times"]) if rec["status"] == OK else rec["error"]
        print(f"[{impl}] {key:<17} {rec['status']:<7} {msg}", flush=True)
        return out if rec["status"] == OK else None

    def skip(key):
        state["stages"][key] = {"status": SKIPPED, "times": [], "error": "earlier stage did not succeed"}
        save()

    expts = ExperimentListFactory.from_json_file(expt_path, check_format=True)
    state["dynamic_mask"] = any(e.imageset.has_dynamic_mask() for e in expts)
    state["n_experiments"] = len(expts)
    state["n_images"] = sum(len(e.imageset) for e in expts if e.imageset is not None)
    save()

    # 1. Spot finding
    fp = phil(find_phil, f"spotfinder.mp.nproc={nproc}" if not gpu else "", cfg["find_phil"])
    strong = stage("find_spots", lambda: find_fn(copy.deepcopy(expts), copy.deepcopy(fp)))
    if strong is None:
        for k, _, _ in STAGES[1:]:
            skip(k)
        return
    strong.as_file(str(outdir / "strong.refl"))

    # 2. Indexing (known symmetry from --space-group / --unit-cell first, then --index-phil)
    ip = phil(index_phil, cfg["dials_symmetry_phil"], cfg["index_phil"])
    res = stage("index", lambda: index_fn(copy.deepcopy(expts), [strong.copy()], copy.deepcopy(ip)))
    if res is None:
        for k, _, _ in STAGES[2:]:
            skip(k)
        return
    ie, ir = res
    ie.as_file(str(outdir / "indexed.expt"))
    ir.as_file(str(outdir / "indexed.refl"))

    # 3. Refinement: static variant (timed only), then the default which feeds integration
    rps = phil(refine_phil, "refinement.parameterisation.scan_varying=False", cfg["refine_phil"])
    stage("refine_static", lambda: refine_fn(copy.deepcopy(ie), ir.copy(), copy.deepcopy(rps)))
    rpa = phil(refine_phil, cfg["refine_phil"])
    res = stage("refine", lambda: refine_fn(copy.deepcopy(ie), ir.copy(), copy.deepcopy(rpa)))
    if res is None:
        for k in ("integrate_nproc1", "integrate", "scale"):
            skip(k)
        return
    re_, rr = res[0], res[1]
    re_.as_file(str(outdir / "refined.expt"))
    rr.as_file(str(outdir / "refined.refl"))

    # 4. Integration. dials_gpu.integrate always runs in-process (nproc=1 is its normal
    # mode), so for dials_gpu the chained stage *is* the nproc=1 run. For dials the
    # chained stage uses all --nproc processes; nproc=1 is an optional extra run.
    if gpu:
        state["stages"]["integrate_nproc1"] = {"status": SKIPPED, "times": [],
                                               "error": "same as 'integrate' for dials_gpu"}
        inp = phil(integrate_phil, "integration.mp.nproc=1", cfg["integrate_phil"])
    else:
        if cfg["cpu_integrate_nproc1"]:
            inp1 = phil(integrate_phil, "integration.mp.nproc=1", cfg["integrate_phil"])
            res1 = stage("integrate_nproc1", lambda: integrate_fn(copy.deepcopy(inp1), copy.deepcopy(re_), rr.copy()))
            if res1 is not None:
                # Saved for the closeness checks: DIALS with nproc>1 splits integration into
                # jobs and gives slightly different results; dials_gpu matches nproc=1.
                (outdir / "nproc1").mkdir(exist_ok=True)
                res1[0].as_file(str(outdir / "nproc1" / "integrated.expt"))
                res1[1].as_file(str(outdir / "nproc1" / "integrated.refl"))
                del res1
        else:
            state["stages"]["integrate_nproc1"] = {"status": SKIPPED, "times": [],
                                                   "error": "not requested (--cpu-integrate-nproc1)"}
        inp = phil(integrate_phil, f"integration.mp.nproc={nproc}", cfg["integrate_phil"])
    res = stage("integrate", lambda: integrate_fn(copy.deepcopy(inp), copy.deepcopy(re_), rr.copy()))
    if res is None:
        skip("scale")
        return
    xe, xr = res[0], res[1]
    xe.as_file(str(outdir / "integrated.expt"))
    xr.as_file(str(outdir / "integrated.refl"))

    # 5. Scaling (no html report: dials.scale writes one by default, dials_gpu does not)
    sp = phil(scale_phil, "output.html=None", cfg["scale_phil"])
    res = stage("scale", lambda: scale_fn(copy.deepcopy(sp), copy.deepcopy(xe), [xr.copy()]))
    if res is None:
        return
    se, sr = res
    se.as_file(str(outdir / "scaled.expt"))
    sr.as_file(str(outdir / "scaled.refl"))


def _clear(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.glob("*"):
        if f.is_file() or f.is_symlink():
            f.unlink()
        elif f.is_dir():
            shutil.rmtree(f)


def run_worker(impl: str, expt: Path, outdir: Path, cfg: dict, timeout: float) -> dict:
    _clear(outdir)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", impl, str(expt),
           "--worker-out", str(outdir), "--worker-config", json.dumps(cfg)]
    log_path = outdir / "worker.log"
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        try:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            rc, timed_out = proc.returncode, False
        except subprocess.TimeoutExpired:
            rc, timed_out = None, True
    wall = time.perf_counter() - t0

    state_path = outdir / "worker_results.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"stages": {}}
    stages = state["stages"]
    if timed_out or rc != 0:
        why = (f"worker killed after --timeout {timeout:.0f} s" if timed_out
               else f"worker exited with code {rc}" + (f" (signal {-rc})" if rc < 0 else ""))
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-15:])
        died = False
        for key, _, _ in STAGES:
            s = stages.get(key)
            if s is None or s.get("status") == "RUNNING":
                if not died:  # the stage the worker died in gets the status and log tail
                    stages[key] = {"status": TIMEOUT if timed_out else CRASHED, "times": [],
                                   "error": why, "log_tail": tail}
                    died = True
                else:
                    stages[key] = {"status": SKIPPED, "times": [],
                                   "error": "earlier stage did not succeed"}
    state["worker_wall_s"] = wall
    state["worker_returncode"] = rc
    return state


# --------------------------------------------------------------------------------------
# mxi: one command per stage
# --------------------------------------------------------------------------------------

_MXI_TOTAL = re.compile(r"^\s*total\s+([0-9.]+)\s*s\b", re.M)


def mxi_info(binary: str | None) -> dict:
    """Locate mxi and read its version; 'gpu' is True for a CUDA (or Metal) build."""
    path = shutil.which(binary) if binary else (shutil.which("mxi") or (os.environ.get("MXI") and shutil.which(os.environ["MXI"])))
    if not path:
        return {"path": None}
    try:
        out = subprocess.run([path, "find", "--version"], capture_output=True, text=True, timeout=60)
        version = (out.stdout or out.stderr).strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return {"path": path, "version": None, "gpu": False}
    return {"path": path, "version": version, "gpu": "CUDA" in version or "Metal" in version}


def run_mxi(expt: Path, outdir: Path, cfg: dict, timeout: float) -> dict:
    """Run mxi's chain as separate commands; returns a state like the dials workers'."""
    _clear(outdir)
    mxi = cfg["mxi"]["path"]
    threads = ["-j", str(cfg["mxi_threads"])]
    args = cfg["mxi_args"]
    deadline = time.monotonic() + timeout
    state = {"stages": {}, "mxi_version": cfg["mxi"].get("version")}
    shutil.copy(expt, outdir / "imported.expt")

    def run(key, cmd, cwd=outdir):
        """Run cmd --mxi-repeats times; record internal totals and wall times, or the failure."""
        rec = {"status": OK, "times": [], "wall": [], "command": " ".join(shlex.quote(c) for c in cmd)}
        for rep in range(cfg["mxi_repeats"]):
            left = deadline - time.monotonic()
            if left <= 0:
                rec.update(status=TIMEOUT, times=[], error=f"--timeout {timeout:.0f} s reached")
                break
            log_path = cwd / f"{key}.out"
            t0 = time.perf_counter()
            try:
                with open(log_path, "w") as log:
                    proc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, timeout=left)
            except subprocess.TimeoutExpired:
                rec.update(status=TIMEOUT, times=[], error=f"killed after --timeout {timeout:.0f} s",
                           log_tail=_tail(log_path))
                break
            except OSError as e:
                rec.update(status=FAILED, times=[], error=f"could not run mxi: {e}")
                break
            wall = time.perf_counter() - t0
            if proc.returncode != 0:
                rc = proc.returncode
                rec.update(status=CRASHED if rc < 0 else FAILED, times=[],
                           error=f"exit code {rc}" + (f" ({signal.Signals(-rc).name})" if rc < 0 else "")
                           + f" after {wall:.1f} s", log_tail=_tail(log_path))
                break
            m = _MXI_TOTAL.findall(log_path.read_text(errors="replace"))
            rec["times"].append(float(m[-1]) if m else wall)
            rec["wall"].append(wall)
            rec["time_source"] = "mxi --timing total" if m else "wall clock (no --timing total found)"
        state["stages"][key] = rec
        msg = "  ".join(f"{t:.3f} s" for t in rec["times"]) if rec["status"] == OK else rec.get("error")
        print(f"[mxi] {key:<17} {rec['status']:<7} {msg}", flush=True)
        return rec["status"] == OK

    def skip(keys, why="earlier stage did not succeed"):
        for k in keys:
            state["stages"].setdefault(k, {"status": SKIPPED, "times": [], "error": why})

    def checked(key):
        """After a chained stage: check its outputs hold real content (INVALID if not)."""
        if state["stages"][key]["status"] != OK:
            return False
        ok, why = validate(key, outdir)
        if not ok:
            state["stages"][key].update(status=INVALID, invalid_times=state["stages"][key]["times"],
                                        times=[], error=why)
            print(f"[mxi] {key:<17} INVALID {why}", flush=True)
        return ok

    sym = cfg["symmetry"]
    index_sym = (["--known-cell", sym["unit_cell"]] + (["--known-space-group", sym["space_group"]] if sym["space_group"] else [])
                 if sym["unit_cell"] else [])
    rest = [k for k, _, _ in STAGES] + list(MXI_GPU_STAGES)
    gpu_dir = outdir / "gpu"
    gpu_dir.mkdir()
    static_dir = outdir / "static"
    static_dir.mkdir()

    # 1. Spot finding (and the --gpu variant, timed only)
    run("find_spots", [mxi, "find", "imported.expt", "-o", "strong.refl", "--timing", *threads, *args["find"]])
    if not checked("find_spots"):
        skip(rest)
        return state
    if cfg["mxi_gpu"]:
        if run("find_spots_gpu", [mxi, "find", str(outdir / "imported.expt"), "-o", "strong.refl",
                                  "--gpu", "--timing", *threads, *args["find"]], cwd=gpu_dir):
            ok, why = validate("find_spots", gpu_dir)
            if not ok:
                state["stages"]["find_spots_gpu"].update(status=INVALID, times=[], error=why)
    else:
        skip(["find_spots_gpu"], "mxi build has no GPU")

    # 2. Indexing
    run("index", [mxi, "index", "imported.expt", "strong.refl", "--output-expt", "indexed.expt",
                  "--output-refl", "indexed.refl", "--timing", *threads, *index_sym, *args["index"]])
    if not checked("index"):
        skip(rest)
        return state

    # 3. Refinement: static (timed only), then mxi's default, which feeds integration
    run("refine_static", [mxi, "refine", str(outdir / "indexed.expt"), str(outdir / "indexed.refl"), "--static",
                          "--output-expt", "refined.expt", "--output-refl", "refined.refl", "--timing",
                          *threads, *args["refine"]], cwd=static_dir)
    run("refine", [mxi, "refine", "indexed.expt", "indexed.refl", "--output-expt", "refined.expt",
                   "--output-refl", "refined.refl", "--timing", *threads, *args["refine"]])
    if not checked("refine"):
        skip(rest)
        return state

    # 4. Integration (and the --gpu variant, timed only)
    skip(["integrate_nproc1"], "dials only")
    run("integrate", [mxi, "integrate", "refined.expt", "refined.refl", "-o", "integrated.refl",
                      "--output-expt", "integrated.expt", "--timing", *threads, *args["integrate"]])
    if cfg["mxi_gpu"]:
        if run("integrate_gpu", [mxi, "integrate", str(outdir / "refined.expt"), str(outdir / "refined.refl"),
                                 "-o", "integrated.refl", "--output-expt", "integrated.expt", "--gpu",
                                 "--timing", *threads, *args["integrate"]], cwd=gpu_dir):
            ok, why = validate("integrate", gpu_dir)
            if not ok:
                state["stages"]["integrate_gpu"].update(status=INVALID, times=[], error=why)
    else:
        skip(["integrate_gpu"], "mxi build has no GPU")
    if not checked("integrate"):
        skip(rest)
        return state

    # 5. Scaling, in the known space group (mxi scale works in P1 unless told otherwise)
    scale_sym = ["--space-group", sym["space_group"]] if sym["space_group"] else []
    run("scale", [mxi, "scale", "integrated.expt", "integrated.refl", "-o", "scaled.refl",
                  "--output-expt", "scaled.expt", "--timing", *threads, *scale_sym, *args["scale"]])
    checked("scale")
    return state


def _tail(path: Path, n: int = 15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


# --------------------------------------------------------------------------------------
# Output validation (all implementations): "exit 0" is not enough
# --------------------------------------------------------------------------------------


def validate(stage: str, d: Path) -> tuple[bool, str]:
    """Check a chained stage's outputs in d hold real content. Returns (ok, reason)."""
    from dials.array_family import flex
    from dxtbx.model.experiment_list import ExperimentListFactory

    expt_name, refl_name = OUTPUTS[stage]
    for f in (expt_name, refl_name):
        if f and (not (d / f).is_file() or (d / f).stat().st_size == 0):
            return False, f"output {f} missing or empty"
    try:
        r = flex.reflection_table.from_file(str(d / refl_name))
        el = ExperimentListFactory.from_json_file(str(d / expt_name), check_format=False) if expt_name else None
    except Exception as e:  # noqa: BLE001
        return False, f"could not read outputs: {type(e).__name__}: {e}"
    if stage == "find_spots":
        return (len(r) > 0, "no strong spots")
    if stage in ("index", "refine") and not any(e.crystal is not None for e in el):
        return False, "no crystal model in the experiments"
    if stage == "index":
        return (r.get_flags(r.flags.indexed).count(True) > 0, "no reflections indexed")
    if stage == "refine":
        return (r.get_flags(r.flags.used_in_refinement).count(True) > 0, "no reflections used in refinement")
    if stage == "integrate":
        n = r.get_flags(r.flags.integrated_sum).count(True) + r.get_flags(r.flags.integrated_prf).count(True)
        return (n > 0, "no reflections integrated")
    if stage == "scale":
        if "inverse_scale_factor" not in r:
            return False, "no inverse_scale_factor column"
        good = r.select(~r.get_flags(r.flags.bad_for_scaling, all=False))
        if len(good) == 0:
            return False, "no reflections scaled"
        isf = good["inverse_scale_factor"].as_numpy_array()
        import numpy as np

        n_bad = int((~np.isfinite(isf)).sum())
        if n_bad:
            return False, f"non-finite inverse scale factors on {n_bad} of {len(isf)} scaled reflections"
        if np.allclose(isf, 1.0, rtol=0.0, atol=1e-6):
            return False, ("no scaling applied: every inverse_scale_factor is 1 "
                           "(e.g. no symmetry equivalents to fit; give --space-group)")
    return True, ""


def post_validate(state: dict, d: Path) -> None:
    """Mark dials/dials_gpu chained stages INVALID when their outputs hold nothing."""
    for key in CHAIN:
        s = state["stages"].get(key)
        if s and s.get("status") == OK:
            ok, why = validate(key, d)
            if not ok:
                s.update(status=INVALID, invalid_times=s.get("times"), times=[], error=why)


# --------------------------------------------------------------------------------------
# Comparisons
# --------------------------------------------------------------------------------------


def _cmp(bp, key, ref: Path, test: Path) -> dict:
    try:
        c = bp.COMPARATORS[key](ref, test)
        c["divergent"] = bp.divergence_flags(key, c)
        c["summary"] = bp._key_metrics(key, c)
    except Exception as e:  # noqa: BLE001
        c = {"error": f"{type(e).__name__}: {e}", "divergent": [f"comparison failed: {e}"],
             "summary": f"comparison error: {e}"}
    return c


def compare(ds_dir: Path, results: dict, test_impl: str) -> dict:
    """Compare test_impl's chained stages with dials'.

    Integration: DIALS with nproc>1 splits the sweep into jobs and handles reflections near
    job boundaries differently from nproc=1, which dials_gpu reproduces. When dials'
    nproc=1 output is available it is the main reference; the nproc=N comparison, and
    DIALS's own nproc=N vs nproc=1 difference, are kept for information.
    """
    sys.path.insert(0, str(HERE))
    import benchmark_pipeline as bp

    ok = {i: {k: (results.get(i) or {}).get("stages", {}).get(k, {}).get("status") == OK for k in CHAIN}
          for i in ("dials", test_impl)}
    comps = {}
    for key in CHAIN:
        if not (ok["dials"][key] and ok[test_impl][key]):
            comps[key] = {"skipped": "not OK in both implementations"}
            continue
        comps[key] = _cmp(bp, key, ds_dir / "dials", ds_dir / test_impl)
    ref1 = ds_dir / "dials" / "nproc1"
    if (ref1 / "integrated.refl").is_file() and ok[test_impl]["integrate"]:
        comps["integrate_vs_nproc1"] = _cmp(bp, "integrate", ref1, ds_dir / test_impl)
        if "integrate" in comps and "skipped" not in comps["integrate"]:
            comps["integrate"]["informational"] = True
        if ok["dials"]["integrate"]:
            comps["integrate_dials_self"] = _cmp(bp, "integrate", ref1, ds_dir / "dials")
            comps["integrate_dials_self"]["informational"] = True
    return comps


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def _times(s: dict | None) -> list[float]:
    return (s.get("times") or []) if s and s.get("status") == OK else []


def _status_cell(s: dict | None) -> str | None:
    if not s:
        return "–"
    if s.get("status") == SKIPPED:
        return "–"
    if s.get("status") != OK:
        return f"**{s['status']}**"
    return None


def _cell_gpu(s: dict | None) -> str:
    """dials_gpu: best warm time, with the first call in brackets."""
    c = _status_cell(s)
    if c:
        return c
    t = _times(s)
    return f"{min(t[1:]):.2f} s ({t[0]:.2f} s first)" if len(t) > 1 else f"{t[0]:.2f} s"


def _cell_best(s: dict | None, wall: bool = False) -> str:
    c = _status_cell(s)
    if c:
        return c
    t = _times(s)
    txt = f"{min(t):.2f} s"
    if wall and s.get("wall"):
        txt += f" ({min(s['wall']):.2f} s wall)"
    return txt


def _best(s: dict | None, warm: bool = False) -> float | None:
    t = _times(s)
    if not t:
        return None
    return min(t[1:]) if warm and len(t) > 1 else min(t)


def write_report(all_results: dict, path: Path) -> list[str]:
    sysinfo = all_results["system"]
    cfg = all_results["config"]
    impls = all_results["implementations"]
    mxi = cfg.get("mxi") or {}
    sym = cfg["symmetry"]
    lines = [
        "# dials vs dials_gpu vs mxi: stage-by-stage comparison", "",
        f"- Date: {all_results['date']}",
        f"- Host: {sysinfo['host']}, {sysinfo['cpu_count']} CPUs ({sysinfo.get('cpu_model')})",
        f"- GPU: {sysinfo.get('gpu')}",
        f"- DIALS: {sysinfo.get('dials_version')}; dials_gpu {sysinfo.get('dials_gpu_commit')}; "
        f"torch {sysinfo.get('torch_version')}",
        f"- mxi: {mxi.get('version') or 'not run'}" + (f" (`{mxi.get('path')}`)" if mxi.get("path") else ""),
        f"- Implementations: {', '.join(impls)}",
        f"- dials nproc: {cfg['nproc']}; mxi threads: {cfg['mxi_threads']}; repeats: dials {cfg['cpu_repeats']}, "
        f"dials_gpu {cfg['gpu_repeats']}, mxi {cfg['mxi_repeats']}",
        f"- Known symmetry: space group `{sym['space_group'] or '-'}`, unit cell `{sym['unit_cell'] or '-'}`",
        "- Extra dials PHIL: " + ", ".join(f"{s} `{' '.join(cfg[f'{s}_phil']) or '-'}`" for s in PHIL_SCOPES),
        "- Extra mxi arguments: " + ", ".join(f"{s} `{' '.join(cfg['mxi_args'][s]) or '-'}`" for s in MXI_STEPS),
        "",
        "dials and dials_gpu times are in-process API calls (CUDA synchronised), without process start-up. "
        "dials_gpu shows its best warm call and, in brackets, its first call; warm calls repeat the same data "
        "in one process and can reuse in-process caches, so for one new dataset the first call is the "
        "conservative figure. mxi times are the `total` each mxi command reports with `--timing` (its own "
        "work, including reading and writing the .expt/.refl files), best of the repeats; the command's "
        "wall-clock time, adding process start-up and for `--gpu` CUDA initialisation, is in brackets. "
        "Speed-ups are against the best dials time; for mxi, against its faster of CPU and GPU. "
        "FAILED, CRASHED, TIMEOUT, INVALID and SKIPPED stages are never timed.",
        "",
    ]
    if any(("space_group" in a or "unit_cell" in a) for a in cfg["index_phil"]) and "mxi" in impls and not sym["space_group"]:
        lines += ["> **Note:** symmetry was given to dials through `--index-phil` only; mxi received none. "
                  "Use `--space-group` / `--unit-cell` so all implementations use the same symmetry.", ""]
    problems = []
    for ds in all_results["datasets"]:
        name = ds["name"]
        res = ds["results"]
        lines += [f"## {name}", "", f"`{ds['expt']}`", ""]
        info = next((res[i] for i in ("dials_gpu", "dials") if res.get(i, {}).get("n_images") is not None), {})
        if info:
            lines.append(f"{info.get('n_experiments')} experiment(s), {info.get('n_images')} images.")
        if any((res.get(i) or {}).get("dynamic_mask") for i in ("dials", "dials_gpu")):
            msg = (f"{name}: the imageset has a dynamic (goniometer shadow) mask. dials_gpu's fast "
                   "slabbed integration path (single-panel sweeps of more than 64 images) does not "
                   "support it, so integration falls back to a much slower path; re-import with "
                   "dynamic_shadowing=False if shadowing is not needed.")
            lines += ["", f"> **Note:** {msg.split(': ', 1)[1]}"]
            problems.append(msg)

        st = {i: (res.get(i) or {}).get("stages", {}) for i in IMPLS}
        header = ["Stage"]
        header += ["dials"] if "dials" in impls else []
        header += ["dials_gpu warm (first)"] if "dials_gpu" in impls else []
        header += ["mxi CPU", "mxi GPU"] if "mxi" in impls else []
        header += ["dials_gpu speed-up"] if {"dials", "dials_gpu"} <= set(impls) else []
        header += ["mxi speed-up"] if {"dials", "mxi"} <= set(impls) else []
        lines += ["", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for key, label, _ in STAGES:
            a = st["dials"].get(key)
            g = st["dials_gpu"].get(key)
            m = st["mxi"].get(key)
            mg = {"find_spots": st["mxi"].get("find_spots_gpu"),
                  "integrate": st["mxi"].get("integrate_gpu"),
                  "integrate_nproc1": st["mxi"].get("integrate_gpu")}.get(key)
            if key == "integrate_nproc1":
                if (a or {}).get("status") in (None, SKIPPED):
                    continue
                g, m = st["dials_gpu"].get("integrate"), st["mxi"].get("integrate")
                label = "Integration: dials with nproc=1"
            if key == "integrate":
                label = f"Integration: dials with nproc={cfg['nproc']}"
            if key == "refine_static":
                label = "Refinement, static"
            if key == "refine":
                label = "Refinement, default"
            row = [label]
            row += [_cell_best(a)] if "dials" in impls else []
            row += [_cell_gpu(g)] if "dials_gpu" in impls else []
            if "mxi" in impls:
                row += [_cell_best(m, wall=True), _cell_best(mg, wall=True) if mg is not None else "–"]
            ta = _best(a)
            if {"dials", "dials_gpu"} <= set(impls):
                tg = _best(g, warm=True)
                row.append(f"{ta / tg:.2f}×" if ta and tg else "–")
            if {"dials", "mxi"} <= set(impls):
                tm = [t for t in (_best(m), _best(mg)) if t]
                row.append(f"{ta / min(tm):.2f}×" if ta and tm else "–")
            lines.append("| " + " | ".join(row) + " |")
        for impl in impls:
            for key, s in st[impl].items():
                if s.get("status") in BAD:
                    problems.append(f"{name}: {impl} {key} {s['status']}: {s.get('error')}")

        for test in ("dials_gpu", "mxi"):
            comps = ds["comparisons"].get(test)
            if not comps:
                continue
            informational = test == "mxi"
            title = ("Closeness of mxi to dials (independent implementation: for information)"
                     if informational else "Closeness of dials_gpu to dials")
            lines += ["", f"| Stage | {title} | Verdict |", "|---|---|---|"]
            rows = []
            for key in CHAIN:
                if key == "integrate":
                    rows += [("integrate_vs_nproc1", "integrate (vs dials nproc=1)"),
                             ("integrate", f"integrate (vs dials nproc={cfg['nproc']})"),
                             ("integrate_dials_self", f"*dials nproc={cfg['nproc']} vs dials nproc=1 (DIALS's own variation)*")]
                else:
                    rows.append((key, key))
            for key, label in rows:
                if key not in comps:
                    continue
                c = comps[key]
                if "skipped" in c:
                    lines.append(f"| {label} | – | not compared (stage not OK in both) |")
                    continue
                if informational or c.get("informational"):
                    verdict = ("for reference; outside dials_gpu tolerances: " + "; ".join(c["divergent"])
                               if c.get("divergent") else "for reference; within tolerance")
                elif c.get("divergent"):
                    verdict = "**DIVERGENT**: " + "; ".join(c["divergent"])
                    problems.append(f"{name}: dials_gpu {label} DIVERGENT: " + "; ".join(c["divergent"]))
                else:
                    verdict = "within tolerance"
                lines.append(f"| {label} | {c.get('summary', '')} | {verdict} |")
            if "integrate_vs_nproc1" not in comps and comps.get("integrate", {}).get("summary"):
                lines += ["", f"Integration is compared with dials run with nproc={cfg['nproc']}, which splits "
                          "the sweep into jobs and differs slightly from nproc=1 (what dials_gpu reproduces). "
                          "Use --cpu-integrate-nproc1 for a like-for-like integration comparison."]

        merging = []
        for impl in impls:
            src = ds["comparisons"].get("dials_gpu" if impl != "mxi" else "mxi", {}).get("scale", {})
            m = src.get("merging_dials" if impl == "dials" else "merging_dials_gpu")
            if impl == "dials" and not m:
                m = ds["comparisons"].get("mxi", {}).get("scale", {}).get("merging_dials")
            if m and "error" not in m:
                merging.append((impl, m))
        if merging:
            lines += ["", "| Merging | Space group | CC½ | ⟨I/σ⟩ | R_merge | Completeness | Multiplicity | n_obs |",
                      "|---|---|---|---|---|---|---|---|"]
            for impl, m in merging:
                lines.append(f"| {impl} | {m['space_group']} | {m['cc_half']:.4f} | {m['i_over_sigma']:.2f} | "
                             f"{m['r_merge']:.4f} | {m['completeness_pct']:.1f}% | {m['multiplicity']:.2f} | {m['n_obs']} |")

        for impl in impls:
            for key, s in st[impl].items():
                if s.get("status") in BAD:
                    detail = (s.get("traceback") or s.get("log_tail") or "").strip()
                    lines += ["", f"**{impl} {key}: {s['status']}** — {s.get('error')}"]
                    if s.get("command"):
                        lines += ["", f"`{s['command']}`"]
                    if detail:
                        lines += ["", "```", detail, "```"]
        lines.append("")
    if problems:
        lines += ["## Problems", ""] + [f"- {p}" for p in problems] + [""]
    path.write_text("\n".join(lines))
    return problems


def system_info() -> dict:
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    cpu_model = None
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    dials_v = run(["dials.version"])
    return {
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "cpu_model": cpu_model,
        "gpu": run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "dials_version": dials_v.splitlines()[0] if dials_v else None,
        "dials_gpu_commit": run(["git", "-C", str(HERE), "describe", "--always", "--dirty"]),
        "torch_version": run([sys.executable, "-c", "import torch; print(torch.__version__)"]),
        "python": sys.version.split()[0],
    }


def dataset_names(paths: list[Path]) -> list[str]:
    """Readable unique names: the .expt stem, prefixed by its folder when it is generic."""
    names = []
    for p in paths:
        stem = p.stem
        name = f"{p.parent.name}_{stem}" if stem in ("imported", "experiments") and p.parent.name else stem
        base, k = name, 2
        while name in names:
            name, k = f"{base}_{k}", k + 1
        names.append(name)
    return names


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("expts", nargs="*", help="imported .expt file(s)")
    p.add_argument("--workdir", default="compare_stages_out", help="output directory (default: %(default)s)")
    p.add_argument("--implementations", nargs="+", choices=IMPLS, metavar="IMPL",
                   help="which to run: dials dials_gpu mxi (default: all; mxi only if found)")
    p.add_argument("--space-group", default="", help="known space group for all implementations, e.g. C2")
    p.add_argument("--unit-cell", default="", help="known unit cell a,b,c,alpha,beta,gamma for all implementations")
    p.add_argument("--nproc", type=int, default=os.cpu_count(),
                   help="processes for dials spot finding and integration (default: all CPUs, %(default)s)")
    p.add_argument("--cpu-integrate-nproc1", action="store_true",
                   help="also run dials integration with nproc=1: timed, and the reference for integration closeness (recommended)")
    p.add_argument("--gpu-repeats", type=int, default=2, help="dials_gpu runs per stage (default: %(default)s)")
    p.add_argument("--cpu-repeats", type=int, default=1, help="dials runs per stage (default: %(default)s)")
    p.add_argument("--timeout", type=float, default=4 * 3600,
                   help="time limit in s for each implementation on each dataset (default: %(default)s)")
    for stage in PHIL_SCOPES:
        p.add_argument(f"--{stage}-phil", default="", metavar="PHIL",
                       help=f"extra PHIL for dials and dials_gpu {stage}, e.g. \"a=1 b=2\"")
    p.add_argument("--mxi", default=None, metavar="PATH", help="mxi program (default: $MXI, or mxi on the PATH)")
    p.add_argument("--mxi-threads", type=int, default=os.cpu_count(), help="mxi -j (default: all CPUs, %(default)s)")
    p.add_argument("--mxi-repeats", type=int, default=2, help="mxi runs per stage (default: %(default)s)")
    p.add_argument("--no-mxi-gpu", action="store_true", help="do not time mxi find/integrate --gpu")
    for step in MXI_STEPS:
        p.add_argument(f"--mxi-{step}-args", default="", metavar="ARGS", help=f"extra arguments for mxi {step}")
    p.add_argument("--only", choices=IMPLS[:2], help=argparse.SUPPRESS)  # kept for old command lines
    p.add_argument("--worker", choices=IMPLS[:2], help=argparse.SUPPRESS)
    p.add_argument("--worker-out", help=argparse.SUPPRESS)
    p.add_argument("--worker-config", help=argparse.SUPPRESS)
    opts = p.parse_args(argv)

    if opts.worker:
        worker(opts.worker, opts.expts[0], Path(opts.worker_out), json.loads(opts.worker_config))
        return 0

    if not opts.expts:
        p.error("give at least one .expt file")
    paths = [Path(e).resolve() for e in opts.expts]
    missing = [str(x) for x in paths if not x.is_file()]
    if missing:
        p.error(f"not found: {', '.join(missing)}")

    requested = opts.implementations or ([opts.only] if opts.only else None)
    impls = list(requested or IMPLS)
    mxi = mxi_info(opts.mxi) if "mxi" in impls else {"path": None}
    if "mxi" in impls and not mxi["path"]:
        if requested:
            p.error("mxi not found: give --mxi PATH, set $MXI, or put mxi on the PATH")
        print("note: mxi not found (--mxi PATH, $MXI or PATH); comparing dials and dials_gpu only", flush=True)
        impls.remove("mxi")

    sg, uc = opts.space_group.strip(), opts.unit_cell.strip().replace(" ", ",")
    cfg = {
        "nproc": opts.nproc,
        "cpu_integrate_nproc1": opts.cpu_integrate_nproc1,
        "gpu_repeats": max(1, opts.gpu_repeats),
        "cpu_repeats": max(1, opts.cpu_repeats),
        "symmetry": {"space_group": sg, "unit_cell": uc},
        # known symmetry for dials/dials_gpu indexing, applied before --index-phil
        "dials_symmetry_phil": ([f"space_group={sg}"] if sg else []) + ([f"unit_cell={uc}"] if uc else []),
        # PHIL given as "a=1 b=2" on the command line -> list of assignments
        **{f"{s}_phil": shlex.split(getattr(opts, f"{s}_phil")) for s in PHIL_SCOPES},
        "mxi": mxi,
        "mxi_threads": opts.mxi_threads,
        "mxi_repeats": max(1, opts.mxi_repeats),
        "mxi_gpu": bool(mxi.get("gpu")) and not opts.no_mxi_gpu,
        "mxi_args": {s: shlex.split(getattr(opts, f"mxi_{s}_args")) for s in MXI_STEPS},
    }
    if sg and not uc and "mxi" in impls:
        print("note: mxi index needs --unit-cell to use a known space group; mxi will index without one "
              "and scale in --space-group", flush=True)
    check_phil({**cfg, "index_phil": cfg["dials_symmetry_phil"] + cfg["index_phil"]})
    workdir = Path(opts.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    all_results = {"date": datetime.now().isoformat(timespec="seconds"), "system": system_info(),
                   "config": cfg, "implementations": impls, "datasets": []}

    problems: list[str] = []
    for name, expt in zip(dataset_names(paths), paths):
        print(f"\n=== {name}: {expt}", flush=True)
        ds = {"name": name, "expt": str(expt), "results": {}, "comparisons": {}}
        for impl in IMPLS:
            if impl not in impls:
                ds["results"][impl] = {"stages": {}}
                continue
            d = workdir / name / impl
            if impl == "mxi":
                print(f"--- mxi (outputs and logs: {d})", flush=True)
                ds["results"][impl] = run_mxi(expt, d, cfg, opts.timeout)
                keys = [k for k, _, _ in STAGES] + list(MXI_GPU_STAGES)
            else:
                print(f"--- {impl} (log: {d / 'worker.log'})", flush=True)
                ds["results"][impl] = run_worker(impl, expt, d, cfg, opts.timeout)
                post_validate(ds["results"][impl], d)
                keys = [k for k, _, _ in STAGES]
            for key in keys:
                s = ds["results"][impl]["stages"].get(key, {})
                t = "  ".join(f"{x:.2f} s" for x in _times(s))
                print(f"    {key:<17} {s.get('status', '?'):<8} {t or s.get('error', '')}", flush=True)
        if "dials" in impls:
            for test in ("dials_gpu", "mxi"):
                if test in impls:
                    ds["comparisons"][test] = compare(workdir / name, ds["results"], test)
        all_results["datasets"].append(ds)
        (workdir / "results.json").write_text(json.dumps(all_results, indent=1, default=str))
        problems = write_report(all_results, workdir / "report.md")

    print("\n" + (workdir / "report.md").read_text())
    print(f"Results: {workdir / 'results.json'}\nReport:  {workdir / 'report.md'}")
    bad = any(s.get("status") in BAD
              or (s.get("status") == SKIPPED and "earlier stage" in s.get("error", ""))
              for ds in all_results["datasets"] for i in impls
              for s in ds["results"][i]["stages"].values())
    if problems:
        print(f"\n{len(problems)} problem(s); see the Problems section of the report.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
