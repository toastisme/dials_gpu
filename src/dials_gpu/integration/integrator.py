from __future__ import annotations

from collections import OrderedDict
import functools
import logging
import math
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


class ProfileModellerExecutorGPU(ProfileModellerExecutor):
    """
    GPU-accelerated ProfileModellerExecutor that uses GPU background,
    centroid, and summation integration before accumulating reference profiles.
    """

    def process(self, frame, reflections):
        reflections.is_overloaded(self.experiments)
        reflections.compute_mask(self.experiments)
        reflections.compute_background(self.experiments)
        compute_centroid_gpu(reflections, self.experiments)
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
        reflections.is_overloaded(self.experiments)
        reflections.compute_mask(self.experiments)
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
        reflections.is_overloaded(self.experiments)
        reflections.compute_mask(self.experiments)
        reflections.contains_invalid_pixels()

        sbox = reflections["shoebox"]
        nvalfg = sbox.count_mask_values(MaskCode.Valid | MaskCode.Foreground)
        nforeg = sbox.count_mask_values(MaskCode.Foreground)
        fraction_valid = nvalfg.as_double() / nforeg.as_double()
        selection = fraction_valid < self.valid_foreground_threshold
        reflections.set_flags(selection, reflections.flags.dont_integrate)

        # Compute background on GPU (via flex.reflection_table.background_algorithm)
        reflections.compute_background(self.experiments)
        # Compute centroid on GPU
        compute_centroid_gpu(reflections, self.experiments)
        # Compute summed intensity on GPU
        compute_summed_intensity_gpu(reflections)

        if self.profile_fitter:
            _parallel_fit_reflections(self.profile_fitter, reflections)

        reflections["num_pixels.valid"] = sbox.count_mask_values(MaskCode.Valid)
        reflections["num_pixels.background"] = sbox.count_mask_values(
            MaskCode.Valid | MaskCode.Background
        )
        reflections["num_pixels.background_used"] = sbox.count_mask_values(
            MaskCode.Valid | MaskCode.Background | MaskCode.BackgroundUsed
        )
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
    reflections.compute_bbox(
        experiments, sigma_b_multiplier=params.profile.sigma_b_multiplier
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
    reflections.compute_bbox(
        experiments, sigma_b_multiplier=params.profile.sigma_b_multiplier
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


def _try_get_hdf5_volume_and_mask_cpu(imageset, lookup_mask):
    """
    Fast path for single-panel HDF5/NeXus sweeps (e.g. i19 25-image .nxs files):
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
    if not path or not path.endswith((".nxs", ".h5", ".hdf5")) or len(indices) == 0:
        return None
    B = len(indices)
    z0 = int(indices[0])
    vol_key = (path, z0, B)
    vol_np = _HOST_VOL_CACHE.get(vol_key)
    if vol_np is not None:
        _HOST_VOL_CACHE.move_to_end(vol_key)
    else:
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
        if len(_HOST_VOL_CACHE) >= 8:
            _HOST_VOL_CACHE.popitem(last=False)
        _HOST_VOL_CACHE[vol_key] = vol_np

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
    return vol_np, static_mask, tr_min, tr_max, H, W


class IntegratorGPU(Integrator):
    """
    Base GPU Integrator class using `IntegratorExecutorGPU` and `FrameSlicedIntegratorExecutorGPU`.
    Caches `make_image(image, mask)` across the profile-modelling, validation, and integration
    passes when running single-process on short sweeps (<= 100 frames) so HDF5/CBF frames are
    only decompressed from disk once, and uses OpenMP C++ direct volume extraction + single-batch
    GPU execution for single-panel HDF5/NeXus sweeps.
    """

    ExecutorClass = IntegratorExecutorGPU

    def integrate(self):
        orig_modeller_exec = _dials_integrator.ProfileModellerExecutor
        orig_validator_exec = _dials_integrator.ProfileValidatorExecutor
        _dials_integrator.ProfileModellerExecutor = ProfileModellerExecutorGPU
        _dials_integrator.ProfileValidatorExecutor = ProfileValidatorExecutorGPU

        # Enable in-memory frame cache for short sweeps (e.g. 25-image i19 Nexus files)
        # when nproc == 1 so ProfileModellerExecutor and IntegratorExecutor share decoded frames.
        use_frame_cache = (
            self.params.integration.mp.nproc == 1
            and self.params.integration.mp.njobs == 1
            and sum(len(e.imageset) for e in self.experiments if e.imageset is not None) <= 100
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

                task_self.reflections["shoebox"] = flex.shoebox(
                    task_self.reflections["panel"],
                    task_self.reflections["bbox"],
                    allocate=False,
                    flatten=task_self.params.shoebox.flatten,
                )

                st_read = time_fn()
                fast_vol = _try_get_hdf5_volume_and_mask_cpu(
                    imageset, task_self.params.lookup.mask
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
