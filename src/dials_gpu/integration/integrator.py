from __future__ import annotations

from collections import OrderedDict
import ctypes
import functools
import logging
import math
import os
import pickle

import dxtbx.flumpy as flumpy
import libtbx
import numpy as np
import torch
import dials.algorithms.integration.integrator as _dials_integrator
import dials.algorithms.integration.processor as _dials_processor
from dials_gpu.integration import _shoebox_pack_ext
from dials_gpu.spot_finding.finder import (
    _HDF5_GPU_VOLUME_CACHE,
    _get_cbf_batch_reader,
    _try_load_hdf5_raw_volume_gpu,
)
from dials_gpu.spot_finding.threshold import get_default_device
from dials.algorithms.integration.integrator import (
    Executor,
    FrameSlicedIntegratorExecutor,
    Integrator,
    Integrator2D,
    Integrator3D,
    Integrator3DThreaded,
    IntegratorExecutor,
    IntegratorFlat3D,
    IntegratorSingle2D,
    IntegratorStills,
    MaskCode,
    Parameters,
    ProfileModellerExecutor,
    ProfileValidatorExecutor,
    _finalize,
    _finalize_stills,
    _initialize_rotation,
    _initialize_stills,
    generate_phil_scope,
    phil_scope,
)
from dials.algorithms.integration.processor import (
    Processor2D,
    Processor3D,
    ProcessorFlat3D,
    ProcessorSingle2D,
    ProcessorStills,
    build_processor,
)

# Ensure compatibility if C++ extension binary in conda_base was built with 2-arg shoebox_memory / 4-arg max_memory_needed
_orig_max_memory_needed = _dials_integrator.max_memory_needed
if not getattr(_orig_max_memory_needed, "_is_compat_wrapped", False):

    def _compat_max_memory_needed(
        reflections, frame0, frame1, flatten, bytes_per_reflection=0
    ):
        try:
            return _orig_max_memory_needed(
                reflections, frame0, frame1, flatten, bytes_per_reflection
            )
        except Exception:
            return (
                _orig_max_memory_needed(reflections, frame0, frame1, flatten)
                + len(reflections) * bytes_per_reflection
            )

    _compat_max_memory_needed._is_compat_wrapped = True
    _dials_integrator.max_memory_needed = _compat_max_memory_needed

_orig_shoebox_memory = _dials_processor.JobList.shoebox_memory
if not getattr(_orig_shoebox_memory, "_is_compat_wrapped", False):

    def _compat_shoebox_memory(
        self, reflections, flatten=False, bytes_per_reflection=0
    ):
        try:
            return _orig_shoebox_memory(
                self, reflections, flatten, bytes_per_reflection
            )
        except Exception:
            return _orig_shoebox_memory(self, reflections, flatten)

    _compat_shoebox_memory._is_compat_wrapped = True
    _dials_processor.JobList.shoebox_memory = _compat_shoebox_memory
from dials.array_family import flex
from dials.util import Sorry

from dials_gpu.integration.background import (
    AutoBackgroundGPU,
    GLMBackgroundGPU,
    NullBackgroundGPU,
    SimpleBackgroundGPU,
)
from dials_gpu.integration.centroid import SimpleCentroidGPU, compute_centroid_gpu
from dials_gpu.integration.fit import compute_corrections_gpu
from dials_gpu.integration.profile import (
    compute_d_gpu,
    compute_zeta_gpu,
)
from dials_gpu.integration.summation import compute_summed_intensity_gpu

logger = logging.getLogger(__name__)


def _get_background_cls_gpu(alg: str):
    if alg == "glm":
        return GLMBackgroundGPU
    if alg == "simple":
        return SimpleBackgroundGPU
    if alg == "null":
        return NullBackgroundGPU
    if alg in ("auto", "Auto", libtbx.Auto):
        return AutoBackgroundGPU
    from dials.extensions import Background

    return Background.load(alg)


def _get_centroid_cls_gpu(alg: str):
    if alg == "simple":
        return SimpleCentroidGPU
    from dials.extensions import Centroid

    return Centroid.load(alg)


def _parallel_fit_reflections(fitter, reflections):
    """
    OpenMP parallel GaussianRS profile fitting on single-experiment reflection tables,
    passing individual flex columns to avoid flex_table Boost ABI differences.
    """
    if len(reflections) == 0:
        return flex.bool()
    try:
        if len(fitter) == 1:
            n_r = len(reflections)
            val = flex.double(n_r, 0.0)
            var = flex.double(n_r, -1.0)
            cor = flex.double(n_r, 0.0)
            flags = reflections["flags"]
            success = _shoebox_pack_ext.parallel_fit_multi(
                fitter,
                reflections["shoebox"],
                reflections["s1"],
                reflections["xyzcal.px"],
                reflections["xyzcal.mm"],
                flags,
                val,
                var,
                cor,
            )
            reflections["intensity.prf.value"] = val
            reflections["intensity.prf.variance"] = var
            reflections["profile.correlation"] = cor
            reflections["flags"] = flags
            reflections.set_flags(
                ~success, reflections.flags.failed_during_profile_fitting
            )
            return success
    except Exception:
        pass
    return reflections.compute_fitted_intensity(fitter)


def _parallel_model_reflections(fitter, reflections):
    """
    OpenMP parallel GaussianRS profile modelling on single-experiment reflection tables.
    """
    if len(reflections) == 0:
        return
    try:
        if len(fitter) == 1:
            flags = reflections["flags"]
            _shoebox_pack_ext.parallel_model_multi(
                fitter,
                reflections["shoebox"],
                reflections["partiality"],
                reflections["s1"],
                reflections["xyzcal.px"],
                reflections["xyzcal.mm"],
                flags,
            )
            reflections["flags"] = flags
            return
    except Exception:
        pass
    fitter.model(reflections)


def _check_overloads_fast(reflections, experiments):
    if len(reflections) == 0:
        return
    try:
        if len(experiments) == 1 and len(experiments[0].detector) == 1:
            sat = float(experiments[0].detector[0].get_trusted_range()[1])
            flags = reflections["flags"]
            _shoebox_pack_ext.parallel_check_overloads_single_panel(
                reflections["shoebox"], flags, sat
            )
            reflections["flags"] = flags
            return
    except Exception:
        pass
    reflections.is_overloaded(experiments)


def _compute_mask_fast(reflections, experiments):
    if len(reflections) == 0:
        return
    try:
        if len(experiments) == 1:
            expt = experiments[0]
            prof = expt.profile
            if (
                hasattr(prof, "_n_sigma")
                and hasattr(prof, "_sigma_b")
                and hasattr(prof, "_sigma_m")
            ):
                if (
                    isinstance(prof._sigma_b, (int, float))
                    and isinstance(prof._sigma_m, (int, float))
                    and expt.goniometer is not None
                    and expt.scan is not None
                    and not expt.scan.is_still()
                ):
                    delta_b = float(prof._n_sigma * prof._sigma_b)
                    _shoebox_pack_ext.parallel_compute_mask_3d_fast(
                        expt.beam,
                        expt.detector,
                        expt.goniometer,
                        expt.scan,
                        delta_b,
                        reflections["shoebox"],
                        reflections["s1"],
                        reflections["xyzcal.px"],
                        reflections["panel"],
                    )
                    return
                from dials.algorithms.profile_model.gaussian_rs import MaskCalculator

                delta_b = prof._n_sigma * prof._sigma_b
                delta_m = prof._n_sigma * prof._sigma_m
                calc = MaskCalculator(
                    expt.crystal,
                    expt.beam,
                    expt.detector,
                    expt.goniometer,
                    expt.scan,
                    delta_b,
                    delta_m,
                )
                _shoebox_pack_ext.parallel_compute_mask_vec3(
                    calc,
                    reflections["shoebox"],
                    reflections["s1"],
                    reflections["xyzcal.px"],
                    reflections["panel"],
                )
                return
    except Exception:
        pass
    reflections.compute_mask(experiments)


def _compute_bbox_fast(reflections, experiments, sigma_b_multiplier=2.0):
    if len(reflections) == 0:
        reflections["bbox"] = flex.int6()
        return reflections["bbox"]
    try:
        if len(experiments) == 1:
            expt = experiments[0]
            prof = expt.profile
            if (
                hasattr(prof, "_n_sigma")
                and hasattr(prof, "_sigma_b")
                and hasattr(prof, "_sigma_m")
                and isinstance(prof._sigma_b, (int, float))
                and isinstance(prof._sigma_m, (int, float))
                and expt.goniometer is not None
                and expt.scan is not None
                and not expt.scan.is_still()
            ):
                delta_b = float(prof._n_sigma * prof._sigma_b * sigma_b_multiplier)
                delta_m = float(prof._n_sigma * prof._sigma_m)
                reflections["bbox"] = _shoebox_pack_ext.parallel_compute_bbox_3d_fast(
                    expt.beam,
                    expt.detector,
                    expt.goniometer,
                    expt.scan,
                    delta_b,
                    delta_m,
                    reflections["s1"],
                    reflections["xyzcal.px"],
                    reflections["panel"],
                )
                return reflections["bbox"]
            reflections["bbox"] = prof.compute_bbox(
                reflections,
                expt.crystal,
                expt.beam,
                expt.detector,
                expt.goniometer,
                expt.scan,
                sigma_b_multiplier=sigma_b_multiplier,
            )
            return reflections["bbox"]
    except Exception:
        pass
    return reflections.compute_bbox(
        experiments, sigma_b_multiplier=sigma_b_multiplier
    )


class ProfileModellerExecutorGPU(ProfileModellerExecutor):
    """
    GPU-accelerated ProfileModellerExecutor that uses GPU background
    and summation integration before accumulating reference profiles.
    """

    def process(self, frame, reflections):
        _check_overloads_fast(reflections, self.experiments)
        _compute_mask_fast(reflections, self.experiments)
        reflections.compute_background(self.experiments)
        compute_summed_intensity_gpu(reflections)
        if hasattr(self.profile_fitter, "modellers"):
            if "profile.index" not in reflections:
                _parallel_model_reflections(
                    self.profile_fitter.modellers[0], reflections
                )
            else:
                for i, modeller in enumerate(self.profile_fitter.modellers):
                    mask = reflections["profile.index"] != i
                    indices = flex.size_t(range(len(mask))).select(mask)
                    if len(indices) > 0:
                        subsample = reflections.select(indices)
                        _parallel_model_reflections(modeller, subsample)
                        reflections.set_selected(indices, subsample)
        else:
            _parallel_model_reflections(self.profile_fitter, reflections)


class ProfileValidatorExecutorGPU(ProfileValidatorExecutor):
    """
    GPU-accelerated ProfileValidatorExecutor that uses GPU background,
    centroid, and summation integration before validating reference profiles.
    """

    def process(self, frame, reflections):
        _check_overloads_fast(reflections, self.experiments)
        _compute_mask_fast(reflections, self.experiments)
        reflections.compute_background(self.experiments)
        compute_centroid_gpu(reflections, self.experiments)
        compute_summed_intensity_gpu(reflections)
        if hasattr(self.profile_fitter, "modellers"):
            results = []
            for i, modeller in enumerate(self.profile_fitter.modellers):
                mask = reflections["profile.index"] != i
                indices = flex.size_t(range(len(mask))).select(mask)
                if len(indices) > 0:
                    subsample = reflections.select(indices)
                    _parallel_fit_reflections(modeller, subsample)
                    reflections.set_selected(indices, subsample)
                    results.append(flex.mean(subsample["profile.correlation"]))
                else:
                    results.append(None)
            self.results = results
        else:
            self.results = self.profile_fitter.validate(reflections)


class IntegratorExecutorGPU(IntegratorExecutor):
    """
    GPU-accelerated IntegratorExecutor that executes background modelling,
    centroiding, summation integration, and profile fitting on the GPU.
    """

    def process(self, frame, reflections):
        _check_overloads_fast(reflections, self.experiments)
        _compute_mask_fast(reflections, self.experiments)

        sbox = reflections["shoebox"]
        n_r = len(reflections)
        nvalfg = flex.int(n_r, 0)
        flags = reflections["flags"]
        _shoebox_pack_ext.parallel_pre_background_mask_flags(
            sbox, flags, float(self.valid_foreground_threshold), nvalfg
        )
        reflections["flags"] = flags

        # Compute background on GPU (via flex.reflection_table.background_algorithm)
        reflections.compute_background(self.experiments)
        # Compute centroid on GPU
        compute_centroid_gpu(reflections, self.experiments)
        # Compute summed intensity on GPU
        compute_summed_intensity_gpu(reflections)

        if self.profile_fitter:
            _parallel_fit_reflections(self.profile_fitter, reflections)

        nvalid = flex.int(n_r, 0)
        nbg = flex.int(n_r, 0)
        nbg_used = flex.int(n_r, 0)
        _shoebox_pack_ext.parallel_post_mask_counts(sbox, nvalid, nbg, nbg_used)
        reflections["num_pixels.valid"] = nvalid
        reflections["num_pixels.background"] = nbg
        reflections["num_pixels.background_used"] = nbg_used
        reflections["num_pixels.foreground"] = nvalfg


class FrameSlicedIntegratorExecutorGPU(FrameSlicedIntegratorExecutor):
    """
    GPU-accelerated FrameSlicedIntegratorExecutor that runs GPU integration
    and attaches per-frame shoebox results (`FrameSlicedShoebox`).
    """

    def process(self, frame, reflections):
        IntegratorExecutorGPU.process(self, frame, reflections)

        ids = reflections["id"]
        columns = (
            reflections["shoebox"],
            reflections["miller_index"],
            reflections["xyzcal.mm"].parts()[2],
            reflections["zeta"],
            reflections["xyzcal.px"].parts()[2],
        )
        first_id, last_id = flex.min(ids), flex.max(ids)
        if first_id == last_id:
            sliced = self._slice_shoeboxes(*columns, first_id)
        else:
            sliced = flex.frame_sliced_shoebox(len(reflections))
            for expt_id in range(first_id, last_id + 1):
                sel = ids == expt_id
                if sel.count(True) == 0:
                    continue
                sliced.set_selected(
                    sel,
                    self._slice_shoeboxes(
                        *(column.select(sel) for column in columns), expt_id
                    ),
                )
        reflections["frame_sliced_shoebox"] = sliced


def _initialize_rotation_gpu(experiments, params, reflections):
    """
    GPU-accelerated initialization for oscillation data.
    Matches `_initialize_rotation` in `dials.algorithms.integration.integrator`.
    """
    if len(experiments) == 1:
        reflections["zeta"] = compute_zeta_gpu(experiments[0], reflections["s1"])
    else:
        reflections.compute_zeta_multi(experiments)
    reflections["d"] = compute_d_gpu(experiments, reflections)
    if (
        "bbox" not in reflections
        or getattr(reflections, "_bbox_precomputed_sigma_b_mult", None)
        != float(params.profile.sigma_b_multiplier)
    ):
        _compute_bbox_fast(
            reflections,
            experiments,
            sigma_b_multiplier=params.profile.sigma_b_multiplier,
        )

    mask = flex.abs(reflections["zeta"]) < params.filter.min_zeta
    reflections.set_flags(mask, reflections.flags.dont_integrate)

    if params.filter.powder_filter is not None:
        mask = params.filter.powder_filter(reflections["d"])
        reflections.set_flags(mask, reflections.flags.in_powder_ring)


def _initialize_stills_gpu(experiments, params, reflections):
    """
    GPU-accelerated initialization for stills data.
    Matches `_initialize_stills` in `dials.algorithms.integration.integrator`.
    """
    reflections["d"] = compute_d_gpu(experiments, reflections)
    if (
        "bbox" not in reflections
        or getattr(reflections, "_bbox_precomputed_sigma_b_mult", None)
        != float(params.profile.sigma_b_multiplier)
    ):
        _compute_bbox_fast(
            reflections,
            experiments,
            sigma_b_multiplier=params.profile.sigma_b_multiplier,
        )

    z0, z1 = reflections["bbox"].parts()[4:6]
    assert (z1 - z0).all_eq(1), "bbox is invalid"

    if params.filter.powder_filter is not None:
        mask = params.filter.powder_filter(reflections["d"])
        reflections.set_flags(mask, reflections.flags.in_powder_ring)


def _finalize_rotation_gpu(reflections, experiments, params):
    """
    GPU-accelerated finalization for oscillation data.
    Matches `_finalize_rotation` in `dials.algorithms.integration.integrator`,
    computing LP/QE corrections on GPU.
    """
    reflections, experiments = _finalize(reflections, experiments, params)
    compute_corrections_gpu(reflections, experiments)
    return reflections, experiments


_HOST_VOL_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_HOST_STATIC_MASK_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()


_CBF_NUM_THREADS = min(20, max(2, os.cpu_count() or 8))


def _read_cbf_batch_fast(cbf_fn, paths, dst_ptr: int, n_elems: int) -> bool:
    if cbf_fn is None:
        return False
    B = len(paths)
    if B == 0:
        return True
    try:
        paths_b = [p.encode("utf-8") if isinstance(p, str) else p for p in paths]
        c_paths = (ctypes.c_char_p * B)(*paths_b)
        rc = cbf_fn(c_paths, B, n_elems, dst_ptr, _CBF_NUM_THREADS)
        return rc == 0
    except Exception:
        return False


def _get_static_mask_and_trusted_range_cpu(imageset, path, lookup_mask):
    detector = imageset.get_detector()
    panel = detector[0]
    W, H = panel.get_image_size()
    ext_mask_data = imageset.external_lookup.mask.data
    has_ext_mask = not ext_mask_data.empty()
    mask_key = (path, H, W, has_ext_mask)
    base_mask = _HOST_STATIC_MASK_CACHE.get(mask_key)
    if base_mask is not None:
        _HOST_STATIC_MASK_CACHE.move_to_end(mask_key)
    else:
        base_mask = np.ascontiguousarray(
            flumpy.to_numpy(panel.get_untrusted_rectangle_mask()).view(np.uint8)
        )
        if has_ext_mask:
            em = np.ascontiguousarray(
                flumpy.to_numpy(ext_mask_data[0].data()).view(np.uint8)
            )
            base_mask = base_mask & em
        if len(_HOST_STATIC_MASK_CACHE) >= 16:
            _HOST_STATIC_MASK_CACHE.popitem(last=False)
        _HOST_STATIC_MASK_CACHE[mask_key] = base_mask

    if lookup_mask is not None:
        assert len(lookup_mask) == 1, f"Mask/Image are incorrect size 1 {len(lookup_mask)}"
        lm = np.ascontiguousarray(flumpy.to_numpy(lookup_mask[0]).view(np.uint8))
        static_mask = np.ascontiguousarray(base_mask & lm)
    else:
        static_mask = base_mask

    tr0, tr1 = panel.get_trusted_range()
    tr_min = int(math.ceil(tr0))
    tr_max = int(math.floor(tr1))
    return static_mask, tr_min, tr_max, H, W


def _try_get_hdf5_volume_and_mask_cpu(imageset, lookup_mask, cache_host_vol: bool = True):
    """
    Fast path for single-panel HDF5/NeXus or CBF sweeps:
    returns `(vol_np, static_mask_np, tr_min, tr_max, H, W)` where `vol_np` is a
    contiguous `(B, H, W)` int32 numpy array and `static_mask_np` is a contiguous
    `(H, W)` uint8 numpy array.
    """
    detector = imageset.get_detector()
    if len(detector) != 1 or imageset.has_dynamic_mask():
        return None
    if (
        not imageset.external_lookup.pedestal.data.empty()
        or not imageset.external_lookup.gain.data.empty()
    ):
        return None
    try:
        path = imageset.get_path(0)
        indices = tuple(imageset.indices())
    except Exception:
        return None
    if not path or len(indices) == 0 or len(indices) > 64:
        return None
    is_hdf5 = path.endswith((".nxs", ".h5", ".hdf5"))
    is_cbf = path.endswith((".cbf", ".cbf.gz"))
    if not (is_hdf5 or is_cbf):
        return None
    B = len(indices)
    z0 = int(indices[0])
    vol_key = (path, z0, B)
    vol_np = _HOST_VOL_CACHE.get(vol_key)
    if vol_np is not None:
        _HOST_VOL_CACHE.move_to_end(vol_key)
    elif is_hdf5:
        # Check if already resident in GPU volume cache from dials_gpu.find_spots
        dev = get_default_device()
        dev_key = (dev.type, dev.index if dev.index is not None else 0)
        gpu_key = (path, z0, B, dev_key)
        if gpu_key in _HDF5_GPU_VOLUME_CACHE:
            vol_t = _HDF5_GPU_VOLUME_CACHE[gpu_key]
            vol_np = np.ascontiguousarray(vol_t.cpu().numpy())
        else:
            vol_t = _try_load_hdf5_raw_volume_gpu(imageset, torch.device("cpu"))
            if vol_t is None or vol_t.dtype != torch.int32:
                return None
            vol_np = np.ascontiguousarray(vol_t.numpy())
        if cache_host_vol:
            if len(_HOST_VOL_CACHE) >= 8:
                _HOST_VOL_CACHE.popitem(last=False)
            _HOST_VOL_CACHE[vol_key] = vol_np
    else:
        try:
            cbf_ext = _get_cbf_batch_reader()
            panel = detector[0]
            W, H = panel.get_image_size()
            paths = [imageset.get_path(i) for i in range(B)]
            if cbf_ext is not None and all(p and p.endswith((".cbf", ".cbf.gz")) for p in paths):
                vol_np = np.empty((B, H, W), dtype=np.int32)
                ok = _read_cbf_batch_fast(cbf_ext, paths, int(vol_np.ctypes.data), H * W)
                if not ok:
                    return None
            else:
                frames = [flumpy.to_numpy(imageset.get_raw_data(i)[0]) for i in range(B)]
                if frames[0].dtype != np.int32:
                    return None
                vol_np = np.stack(frames, axis=0)
            if cache_host_vol:
                if len(_HOST_VOL_CACHE) >= 8:
                    _HOST_VOL_CACHE.popitem(last=False)
                _HOST_VOL_CACHE[vol_key] = vol_np
        except Exception:
            return None

    static_mask, tr_min, tr_max, H, W = _get_static_mask_and_trusted_range_cpu(
        imageset, path, lookup_mask
    )
    return vol_np, static_mask, tr_min, tr_max, H, W


def _try_run_slabbed_volume_integration(task_self, imageset, frame0, frame1, time_fn):
    """
    Fast multi-threaded slabbed volume extraction + GPU integration for large sweeps
    (> 64 frames) on single-panel static-mask HDF5/NeXus or CBF/CBF.GZ detectors.
    Streams frames in overlapping slabs so each frame in [frame0, frame1) is decoded
    from disk only ONCE per pass using OpenMP/multi-threaded C++ readers, extracts
    shoeboxes in parallel in C++, and batches GPU background + summation + profile
    fitting per slab.
    """
    if task_self.params.debug.output:
        return None
    detector = imageset.get_detector()
    if len(detector) != 1 or imageset.has_dynamic_mask():
        return None
    if (
        not imageset.external_lookup.pedestal.data.empty()
        or not imageset.external_lookup.gain.data.empty()
    ):
        return None
    try:
        path0 = imageset.get_path(0)
        n_frames = len(imageset)
    except Exception:
        return None
    if not path0 or n_frames <= 64 or (frame1 - frame0) != n_frames:
        return None
    is_hdf5 = path0.endswith((".nxs", ".h5", ".hdf5"))
    is_cbf = path0.endswith((".cbf", ".cbf.gz"))
    if not (is_hdf5 or is_cbf):
        return None

    cbf_ext = _get_cbf_batch_reader() if is_cbf else None
    cbf_paths = None
    if is_cbf:
        try:
            cbf_paths = [imageset.get_path(i) for i in range(n_frames)]
            if not all(p and p.endswith((".cbf", ".cbf.gz")) for p in cbf_paths):
                cbf_ext = None
        except Exception:
            cbf_ext = None

    static_mask_np, tr_min, tr_max, H, W = _get_static_mask_and_trusted_range_cpu(
        imageset, path0, task_self.params.lookup.mask
    )
    rejected_arr = np.array(
        [1 if imageset.is_marked_for_rejection(i) else 0 for i in range(n_frames)],
        dtype=np.uint8,
    )

    refls = task_self.reflections
    n_total = len(refls)
    if n_total == 0:
        return 0.0, 0.0, 0.0

    _, _, _, _, z0_flex, z1_flex = refls["bbox"].parts()
    z0_arr = z0_flex.as_numpy_array()
    z1_arr = z1_flex.as_numpy_array()
    dz_arr = z1_arr - z0_arr

    in_bounds = (z0_arr >= frame0) & (z1_arr <= frame1) & (dz_arr > 0)
    slab_step = 64
    norm_mask = in_bounds & (dz_arr <= slab_step)
    wide_mask = in_bounds & (dz_arr > slab_step)

    has_wide = bool(np.any(wide_mask))
    wide_ind_np = np.nonzero(wide_mask)[0] if has_wide else None
    wide_refl = None
    wide_valid_flags = None
    if has_wide:
        wide_ind = flex.size_t(np.ascontiguousarray(wide_ind_np, dtype=np.uintp))
        wide_refl = refls.select(wide_ind)
        wide_refl["shoebox"] = flex.shoebox(
            wide_refl["panel"],
            wide_refl["bbox"],
            allocate=True,
            flatten=task_self.params.shoebox.flatten,
        )
        wide_valid_flags = np.zeros(len(wide_refl), dtype=np.uint8)

    max_norm_dz = int(dz_arr[norm_mask].max()) if np.any(norm_mask) else 1
    overlap = max(0, max_norm_dz - 1)
    max_buf_frames = min(n_frames, slab_step + overlap)
    slab_buf = np.empty((max_buf_frames, H, W), dtype=np.int32)

    def _read_chunk_into(f_start, f_end, dst_slice):
        count = f_end - f_start
        if count <= 0:
            return True
        idx0 = f_start - frame0
        idx1 = f_end - frame0
        if is_cbf and cbf_ext is not None:
            return _read_cbf_batch_fast(
                cbf_ext, cbf_paths[idx0:idx1], int(dst_slice.ctypes.data), H * W
            )
        elif is_hdf5:
            sub_iset = imageset[idx0:idx1]
            vol_t = _try_load_hdf5_raw_volume_gpu(sub_iset, torch.device("cpu"))
            if vol_t is None or vol_t.dtype != torch.int32:
                return False
            np.copyto(dst_slice, vol_t.numpy())
            return True
        else:
            for k in range(count):
                arr = flumpy.to_numpy(imageset.get_raw_data(idx0 + k)[0])
                if arr.dtype != np.int32:
                    return False
                np.copyto(dst_slice[k], arr)
            return True

    read_time = 0.0
    extract_time = 0.0
    process_time = 0.0

    prev_f0 = frame0
    prev_f1 = frame0

    for z0_start in range(frame0, frame1, slab_step):
        z0_end = min(z0_start + slab_step, frame1)
        slab_sel = norm_mask & (z0_arr >= z0_start) & (z0_arr < z0_end)
        has_norm = bool(np.any(slab_sel))
        if has_norm:
            req_f1 = int(z1_arr[slab_sel].max())
        else:
            req_f1 = z0_end
        if z0_end == frame1:
            req_f1 = frame1
        slab_f1 = max(req_f1, prev_f1)

        if not has_norm and slab_f1 == prev_f1:
            continue

        st_r = time_fn()
        n_ov = max(0, prev_f1 - z0_start)
        if n_ov > 0:
            ov_offset = z0_start - prev_f0
            slab_buf[:n_ov] = slab_buf[ov_offset : ov_offset + n_ov]
        new_f0 = max(z0_start, prev_f1)
        n_new = slab_f1 - new_f0
        if n_new > 0:
            ok = _read_chunk_into(new_f0, slab_f1, slab_buf[n_ov : n_ov + n_new])
            if not ok:
                return None
        read_time += time_fn() - st_r

        if has_wide and n_new > 0:
            st_e = time_fn()
            new_view = slab_buf[n_ov : n_ov + n_new]
            rej_new = np.ascontiguousarray(
                rejected_arr[new_f0 - frame0 : slab_f1 - frame0]
            )
            _shoebox_pack_ext.extract_wide_shoeboxes_slice_from_volume(
                wide_refl["shoebox"],
                int(new_view.ctypes.data),
                int(static_mask_np.ctypes.data),
                int(rej_new.ctypes.data),
                tr_min,
                tr_max,
                new_f0,
                slab_f1,
                H,
                W,
                int(wide_valid_flags.ctypes.data),
            )
            extract_time += time_fn() - st_e

        prev_f0 = z0_start
        prev_f1 = slab_f1

        if has_norm:
            slab_ind_np = np.nonzero(slab_sel)[0]
            slab_ind = flex.size_t(np.ascontiguousarray(slab_ind_np, dtype=np.uintp))
            sub_refl = refls.select(slab_ind)
            sub_refl["shoebox"] = flex.shoebox(
                sub_refl["panel"],
                sub_refl["bbox"],
                allocate=False,
                flatten=task_self.params.shoebox.flatten,
            )

            st_e = time_fn()
            n_sub = len(sub_refl)
            valid_flags = np.empty(n_sub, dtype=np.uint8)
            slab_len = slab_f1 - z0_start
            slab_view = slab_buf[:slab_len]
            rej_view = np.ascontiguousarray(
                rejected_arr[z0_start - frame0 : slab_f1 - frame0]
            )
            n_valid = _shoebox_pack_ext.extract_shoeboxes_from_volume(
                sub_refl["shoebox"],
                int(slab_view.ctypes.data),
                int(static_mask_np.ctypes.data),
                int(rej_view.ctypes.data),
                tr_min,
                tr_max,
                z0_start,
                slab_f1,
                H,
                W,
                int(valid_flags.ctypes.data),
            )
            extract_time += time_fn() - st_e

            st_p = time_fn()
            if n_valid == n_sub and n_sub > 0:
                task_self.executor.process(slab_f1 - 1, sub_refl)
            elif n_valid > 0:
                v_ind = flex.size_t(
                    np.ascontiguousarray(np.nonzero(valid_flags)[0], dtype=np.uintp)
                )
                v_refl = sub_refl.select(v_ind)
                task_self.executor.process(slab_f1 - 1, v_refl)
                sub_refl.set_selected(v_ind, v_refl)
            del sub_refl["shoebox"]
            refls.set_selected(slab_ind, sub_refl)
            process_time += time_fn() - st_p

    if has_wide:
        if prev_f1 < frame1:
            st_r = time_fn()
            n_rem = frame1 - prev_f1
            rem_buf = slab_buf[:n_rem] if n_rem <= max_buf_frames else np.empty((n_rem, H, W), dtype=np.int32)
            ok = _read_chunk_into(prev_f1, frame1, rem_buf)
            if not ok:
                return None
            read_time += time_fn() - st_r

            st_e = time_fn()
            rej_rem = np.ascontiguousarray(
                rejected_arr[prev_f1 - frame0 : frame1 - frame0]
            )
            _shoebox_pack_ext.extract_wide_shoeboxes_slice_from_volume(
                wide_refl["shoebox"],
                int(rem_buf.ctypes.data),
                int(static_mask_np.ctypes.data),
                int(rej_rem.ctypes.data),
                tr_min,
                tr_max,
                prev_f1,
                frame1,
                H,
                W,
                int(wide_valid_flags.ctypes.data),
            )
            extract_time += time_fn() - st_e

        st_p = time_fn()
        n_wide = len(wide_refl)
        n_w_valid = int(np.count_nonzero(wide_valid_flags))
        if n_w_valid == n_wide and n_wide > 0:
            task_self.executor.process(frame1 - 1, wide_refl)
        elif n_w_valid > 0:
            wv_ind = flex.size_t(
                np.ascontiguousarray(np.nonzero(wide_valid_flags)[0], dtype=np.uintp)
            )
            wv_refl = wide_refl.select(wv_ind)
            task_self.executor.process(frame1 - 1, wv_refl)
            wide_refl.set_selected(wv_ind, wv_refl)
        del wide_refl["shoebox"]
        refls.set_selected(wide_ind, wide_refl)
        process_time += time_fn() - st_p

    return read_time, extract_time, process_time


class IntegratorGPU(Integrator):
    """
    Base GPU Integrator class using `IntegratorExecutorGPU` and `FrameSlicedIntegratorExecutorGPU`.
    Caches `make_image(image, mask)` across the profile-modelling, validation, and integration
    passes when running single-process on short sweeps (<= 100 frames) so HDF5/CBF frames are
    only decompressed from disk once, and uses OpenMP C++ direct volume extraction + single-batch
    GPU execution for single-panel HDF5/NeXus and CBF sweeps.
    """

    ExecutorClass = IntegratorExecutorGPU

    def integrate(self):
        if torch.cuda.is_available() and torch.get_num_threads() > 1:
            torch.set_num_threads(1)
        orig_modeller_exec = _dials_integrator.ProfileModellerExecutor
        orig_validator_exec = _dials_integrator.ProfileValidatorExecutor
        _dials_integrator.ProfileModellerExecutor = ProfileModellerExecutorGPU
        _dials_integrator.ProfileValidatorExecutor = ProfileValidatorExecutorGPU

        # Enable fast task execution when nproc == 1; only cache full decoded frames in RAM
        # for short sweeps (<= 100 frames). For longer HDF5 sweeps, _HDF5_COMPRESSED_CACHE
        # caches compressed bitshuffle-LZ4 chunks across passes and extracts shoeboxes in C++ OpenMP.
        use_frame_cache = (
            self.params.integration.mp.nproc == 1
            and self.params.integration.mp.njobs == 1
        )
        can_cache_frames = (
            sum(len(e.imageset) for e in self.experiments if e.imageset is not None) <= 100
        )
        frame_cache = {}
        orig_task_call = _dials_processor.Task.__call__

        if use_frame_cache:
            make_image = _dials_processor.make_image
            ShoeboxProcessor = _dials_processor.ShoeboxProcessor
            job_global = _dials_processor.job
            time_fn = _dials_processor.time

            def _cached_task_call(task_self):
                start_time = time_fn()
                job_global.index = task_self.index

                exp_id = list(set(task_self.reflections["id"]))
                root_imageset = task_self.experiments[exp_id[0]].imageset
                for i in exp_id[1:]:
                    assert task_self.experiments[i].imageset == root_imageset, (
                        "Task can only handle 1 imageset"
                    )

                frame0, frame1 = task_self.job
                try:
                    allowed_range = root_imageset.get_array_range()
                except Exception:
                    allowed_range = 0, len(root_imageset)

                try:
                    assert frame0 < frame1
                    assert allowed_range[1] > allowed_range[0]
                    assert frame0 >= allowed_range[0]
                    assert frame1 <= allowed_range[1]
                    assert (frame1 - frame0) <= len(root_imageset)
                    imageset = root_imageset
                    if len(imageset) > 1:
                        index0 = frame0 - allowed_range[0]
                        index1 = frame1 - allowed_range[0]
                        imageset = imageset[index0:index1]
                except Exception as e:
                    raise RuntimeError(f"Programmer Error: bad array range: {e}")

                try:
                    frame0, frame1 = imageset.get_array_range()
                except Exception:
                    frame0, frame1 = (0, len(imageset))

                task_self.executor.initialize(frame0, frame1, task_self.reflections)

                slab_res = _try_run_slabbed_volume_integration(
                    task_self, imageset, frame0, frame1, time_fn
                )
                if slab_res is not None:
                    read_time, extract_time, process_time = slab_res
                    task_self.executor.finalize()
                    return _dials_processor.dials.algorithms.integration.Result(
                        index=task_self.index,
                        reflections=task_self.reflections,
                        data=task_self.executor.data(),
                        read_time=read_time,
                        extract_time=extract_time,
                        process_time=process_time,
                        total_time=time_fn() - start_time,
                    )

                task_self.reflections["shoebox"] = flex.shoebox(
                    task_self.reflections["panel"],
                    task_self.reflections["bbox"],
                    allocate=False,
                    flatten=task_self.params.shoebox.flatten,
                )

                st_read = time_fn()
                fast_vol = _try_get_hdf5_volume_and_mask_cpu(
                    imageset, task_self.params.lookup.mask, cache_host_vol=can_cache_frames
                )
                if fast_vol is not None:
                    vol_np, static_mask_np, tr_min, tr_max, H, W = fast_vol
                    rejected_arr = np.array(
                        [
                            1 if imageset.is_marked_for_rejection(i) else 0
                            for i in range(len(imageset))
                        ],
                        dtype=np.uint8,
                    )
                    read_time = time_fn() - st_read

                    st_ext = time_fn()
                    n_refl = len(task_self.reflections)
                    valid_flags = np.empty(n_refl, dtype=np.uint8)
                    n_valid = _shoebox_pack_ext.extract_shoeboxes_from_volume(
                        task_self.reflections["shoebox"],
                        int(vol_np.ctypes.data),
                        int(static_mask_np.ctypes.data),
                        int(rejected_arr.ctypes.data),
                        tr_min,
                        tr_max,
                        frame0,
                        frame1,
                        H,
                        W,
                        int(valid_flags.ctypes.data),
                    )
                    extract_time = time_fn() - st_ext

                    st_proc = time_fn()
                    if n_valid == n_refl and n_refl > 0:
                        task_self.executor.process(frame1 - 1, task_self.reflections)
                    elif n_valid > 0:
                        ind = flex.size_t(
                            np.ascontiguousarray(
                                np.nonzero(valid_flags)[0], dtype=np.uintp
                            )
                        )
                        sub_refl = task_self.reflections.select(ind)
                        task_self.executor.process(frame1 - 1, sub_refl)
                        task_self.reflections.set_selected(ind, sub_refl)
                    process_time = time_fn() - st_proc
                else:
                    processor = ShoeboxProcessor(
                        task_self.reflections,
                        len(imageset.get_detector()),
                        frame0,
                        frame1,
                        task_self.params.debug.output,
                    )

                    read_time = 0.0
                    root_key = id(root_imageset)
                    lookup_key = id(task_self.params.lookup.mask)
                    for i in range(len(imageset)):
                        st = time_fn()
                        rejected = imageset.is_marked_for_rejection(i)
                        cache_key = (root_key, frame0 + i, lookup_key, rejected)
                        c_img = frame_cache.get(cache_key)
                        if c_img is None:
                            image = imageset.get_corrected_data(i)
                            if rejected:
                                mask = tuple(
                                    flex.bool(im.accessor(), False) for im in image
                                )
                            else:
                                mask = imageset.get_mask(i)
                                if task_self.params.lookup.mask is not None:
                                    assert len(mask) == len(
                                        task_self.params.lookup.mask
                                    ), (
                                        "Mask/Image are incorrect size %d %d"
                                        % (
                                            len(mask),
                                            len(task_self.params.lookup.mask),
                                        )
                                    )
                                    mask = tuple(
                                        m1 & m2
                                        for m1, m2 in zip(
                                            task_self.params.lookup.mask, mask
                                        )
                                    )
                            c_img = make_image(image, mask)
                            if can_cache_frames:
                                frame_cache[cache_key] = c_img
                            del image
                            del mask
                        read_time += time_fn() - st
                        processor.next(c_img, task_self.executor)
                    assert processor.finished(), "Data processor is not finished"
                    extract_time = processor.extract_time()
                    process_time = processor.process_time()

                if task_self.params.debug.output and task_self.params.debug.separate_files:
                    output = task_self.reflections
                    if task_self.params.debug.select is not None:
                        output = output.select(task_self.params.debug.select(output))
                    if task_self.params.debug.split_experiments:
                        output = output.split_by_experiment_id()
                        for table in output:
                            i = table["id"][0]
                            table.as_file("shoeboxes_%d_%d.refl" % (task_self.index, i))
                    else:
                        output.as_file("shoeboxes_%d.refl" % task_self.index)

                if task_self.params.debug.separate_files or not task_self.params.debug.output:
                    del task_self.reflections["shoebox"]

                task_self.executor.finalize()

                return _dials_processor.dials.algorithms.integration.Result(
                    index=task_self.index,
                    reflections=task_self.reflections,
                    data=task_self.executor.data(),
                    read_time=read_time,
                    extract_time=extract_time,
                    process_time=process_time,
                    total_time=time_fn() - start_time,
                )

            _dials_processor.Task.__call__ = _cached_task_call

        try:
            if self.params.frame_slice_shoeboxes:
                self.ExecutorClass = FrameSlicedIntegratorExecutorGPU
                self.params.frame_slice_shoeboxes = False
                try:
                    return super().integrate()
                finally:
                    self.params.frame_slice_shoeboxes = True
                    self.ExecutorClass = IntegratorExecutorGPU
            return super().integrate()
        finally:
            if use_frame_cache:
                _dials_processor.Task.__call__ = orig_task_call
                frame_cache.clear()
            _dials_integrator.ProfileModellerExecutor = orig_modeller_exec
            _dials_integrator.ProfileValidatorExecutor = orig_validator_exec


class Integrator3DGPU(IntegratorGPU):
    """GPU-accelerated 3D integrator."""

    initialize_reflections = staticmethod(_initialize_rotation_gpu)
    ProcessorClass = Processor3D
    finalize_reflections = staticmethod(_finalize_rotation_gpu)


class IntegratorFlat3DGPU(IntegratorGPU):
    """GPU-accelerated Flat3D integrator."""

    initialize_reflections = staticmethod(_initialize_rotation_gpu)
    ProcessorClass = ProcessorFlat3D
    finalize_reflections = staticmethod(_finalize_rotation_gpu)


class Integrator2DGPU(IntegratorGPU):
    """GPU-accelerated 2D integrator."""

    initialize_reflections = staticmethod(_initialize_rotation_gpu)
    ProcessorClass = Processor2D
    finalize_reflections = staticmethod(_finalize_rotation_gpu)


class IntegratorSingle2DGPU(IntegratorGPU):
    """GPU-accelerated Single2D integrator."""

    initialize_reflections = staticmethod(_initialize_rotation_gpu)
    ProcessorClass = ProcessorSingle2D
    finalize_reflections = staticmethod(_finalize_rotation_gpu)


class IntegratorStillsGPU(IntegratorGPU):
    """GPU-accelerated Stills integrator."""

    initialize_reflections = staticmethod(_initialize_stills_gpu)
    ProcessorClass = ProcessorStills
    finalize_reflections = staticmethod(_finalize_stills)


class Integrator3DThreadedGPU(Integrator3DThreaded):
    """GPU-compatible 3D Threaded integrator."""

    def initialise(self):
        _initialize_rotation_gpu(self.experiments, self.params.integration, self.reflections)

    def finalise(self):
        self.reflections, self.experiments = _finalize_rotation_gpu(
            self.reflections, self.experiments, self.params
        )


def create_integrator_gpu(params, experiments, reflections):
    """
    Create a GPU integrator object with a given configuration.
    Matches `dials.algorithms.integration.integrator.create_integrator`.
    """
    for exp in experiments:
        if exp.imageset is None:
            raise Sorry(
                """
      One or more experiment does not contain an imageset. Access to the
      image data is crucial for integration.
    """
            )

    if params.integration.lookup.mask and isinstance(
        params.integration.lookup.mask, str
    ):
        with open(params.integration.lookup.mask, "rb") as infile:
            params.integration.lookup.mask = pickle.load(infile, encoding="bytes")

    if params.integration.mp.nproc is libtbx.Auto:
        params.integration.mp.nproc = 1

    BackgroundAlgorithm = _get_background_cls_gpu(
        params.integration.background.algorithm
    )
    flex.reflection_table.background_algorithm = staticmethod(
        functools.partial(BackgroundAlgorithm, params)
    )
    CentroidAlgorithm = _get_centroid_cls_gpu(
        params.integration.centroid.algorithm
    )
    flex.reflection_table.centroid_algorithm = staticmethod(
        functools.partial(CentroidAlgorithm, params)
    )

    if params.integration.integrator in ("auto", "Auto", libtbx.Auto):
        if experiments.all_stills():
            params.integration.integrator = "stills"
        else:
            params.integration.integrator = "3d"

    IntegratorClass = {
        "3d": Integrator3DGPU,
        "flat3d": IntegratorFlat3DGPU,
        "2d": Integrator2DGPU,
        "single2d": IntegratorSingle2DGPU,
        "stills": IntegratorStillsGPU,
        "3d_threaded": Integrator3DThreadedGPU,
    }.get(params.integration.integrator)
    if not IntegratorClass:
        raise ValueError(f"Unknown integration type {params.integration.integrator}")

    if params.integration.frame_slice_shoeboxes and any(
        experiment.is_still() for experiment in experiments
    ):
        raise Sorry(
            """
      frame_slice_shoeboxes=True cannot be used with still experiments, as
      there is no rotation scan from which to calculate per-frame data.
    """
        )

    if (
        params.integration.frame_slice_shoeboxes
        and IntegratorClass is IntegratorFlat3DGPU
    ):
        raise Sorry(
            """
      frame_slice_shoeboxes=True cannot be used with integrator=flat3d, as
      flattened shoeboxes have had their frames summed together.
    """
        )

    if experiments.all_stills():
        for experiment in experiments:
            experiment.scan = None

    return IntegratorClass(experiments, reflections, params)
