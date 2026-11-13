from __future__ import annotations

import atexit
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import ctypes
import functools
import logging
import math
import multiprocessing
from multiprocessing import shared_memory
import os
import pickle
import struct
import threading
import libtbx
from libtbx.introspection import number_of_processors
import numpy as np
import torch

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    hdf5plugin = None

try:
    import h5py
except ImportError:
    h5py = None

CPU_COUNT = number_of_processors(return_value_if_unknown=1)
from dxtbx import flumpy
from dxtbx.format.image import ImageBool
from dxtbx.imageset import ImageSequence, ImageSet
from dxtbx.model import ExperimentList
from iotbx.phil import parse

import dials.extensions
import dials.util.masking
from dials.algorithms.spot_finding.factory import (
    BackgroundGradientFilter,
    SpotFinderFactory,
    phil_scope,
)
from dials.algorithms.spot_finding.finder import (
    LaueSpotFinder,
    SpotFinder,
    TOFSpotFinder,
)
from dials.array_family import flex
from dials.util import Sorry
from dials_gpu.spot_finding.connected_components import (
    GPUFramePixelList,
    PixelListLabellerGPU,
    build_shoeboxes_from_gpu_spots,
    extract_spots_from_labeller_gpu,
)
from dials_gpu.spot_finding.filters import (
    FilterRunnerGPU,
    PeakCentroidDistanceFilterGPU,
    SpotDensityFilterGPU,
)
from dials_gpu.spot_finding.threshold import (
    DispersionExtendedSpotFinderThresholdGPU,
    DispersionSpotFinderThresholdGPU,
    RadialProfileSpotFinderThresholdGPU,
    estimate_global_threshold,
    flex_to_tensor,
    get_default_device,
    tensor_to_flex_bool,
)

logger = logging.getLogger("dials.algorithms.spot_finding.finder")
ext_logger = logging.getLogger("dials.extensions.dispersion_spotfinder_threshold_ext")


_BSHUF_LIB = None
_BSHUF_BATCH_EXT = None
_CBF_READ_FN = None
_BSHUF_POOL: ThreadPoolExecutor | None = None
_HDF5_GPU_VOLUME_CACHE: OrderedDict[tuple, torch.Tensor] = OrderedDict()
_HDF5_GPU_MASK_CACHE: OrderedDict[tuple, torch.Tensor] = OrderedDict()
_HDF5_COMPRESSED_CACHE: OrderedDict[tuple, tuple[list[bytes], int, int, np.dtype]] = OrderedDict()
_BASE_STATIC_MASK_GPU_CACHE: OrderedDict[tuple, list[torch.Tensor]] = OrderedDict()
_PINNED_HOST_VOL_CACHE: dict[tuple, torch.Tensor] = {}
_PINNED_HOST_VOL_TAG: dict[tuple, tuple] = {}
_PIPELINE_PINNED_CACHE: dict[tuple, list[torch.Tensor]] = {}


@atexit.register
def _cleanup_spotfinding_caches():
    _PIPELINE_PINNED_CACHE.clear()
    _PINNED_HOST_VOL_CACHE.clear()
    _PINNED_HOST_VOL_TAG.clear()
    _HDF5_GPU_VOLUME_CACHE.clear()
    _HDF5_GPU_MASK_CACHE.clear()
    _BASE_STATIC_MASK_GPU_CACHE.clear()


def _get_cbf_batch_reader():
    global _CBF_READ_FN
    if _CBF_READ_FN is not None:
        return _CBF_READ_FN
    try:
        so_path = os.path.join(os.path.dirname(__file__), "_cbf_read_ext.so")
        if not os.path.exists(so_path):
            return None
        lib = ctypes.CDLL(so_path)
        fn = lib.read_cbf_batch_omp
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        fn.restype = ctypes.c_int
        _CBF_READ_FN = fn
        return _CBF_READ_FN
    except Exception:
        return None


def _get_bshuf_decompress():
    global _BSHUF_LIB
    if _BSHUF_LIB is not None:
        return _BSHUF_LIB
    if hdf5plugin is None:
        return None
    try:
        plugin_dir = os.path.join(os.path.dirname(hdf5plugin.__file__), "plugins")
        lib_path = os.path.join(plugin_dir, "libh5bshuf.so")
        if not os.path.exists(lib_path):
            return None
        lib = ctypes.CDLL(lib_path)
        fn = lib.bshuf_decompress_lz4
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        fn.restype = ctypes.c_int64
        _BSHUF_LIB = fn
        return _BSHUF_LIB
    except Exception:
        return None


def _get_bshuf_batch_ext():
    global _BSHUF_BATCH_EXT
    if _BSHUF_BATCH_EXT is not None:
        return _BSHUF_BATCH_EXT
    bshuf_fn = _get_bshuf_decompress()
    if bshuf_fn is None:
        return None
    try:
        so_path = os.path.join(os.path.dirname(__file__), "_cbf_read_ext.so")
        if not os.path.exists(so_path):
            return None
        lib = ctypes.CDLL(so_path)
        fn_i32 = lib.bshuf_decode_batch_i32
        fn_i32.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        fn_i32.restype = ctypes.c_int

        fn_pread = lib.bshuf_pread_direct_batch
        fn_pread.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        fn_pread.restype = ctypes.c_int

        fn_direct = lib.bshuf_decode_direct_batch
        fn_direct.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        fn_direct.restype = ctypes.c_int

        bshuf_ptr = ctypes.cast(bshuf_fn, ctypes.c_void_p).value
        _BSHUF_BATCH_EXT = (bshuf_ptr, fn_i32, fn_pread, fn_direct)
        return _BSHUF_BATCH_EXT
    except Exception:
        return None


def _dtype_to_bshuf_code(dtype: np.dtype) -> int:
    if dtype == np.uint16:
        return 1
    if dtype == np.int16:
        return 2
    if dtype == np.uint32:
        return 3
    if dtype == np.int32:
        return 4
    return 0


def _get_bshuf_pool() -> ThreadPoolExecutor:
    global _BSHUF_POOL
    if _BSHUF_POOL is None:
        # One thread per core: decoding (decompress, widen 16-bit data to int32, map
        # invalid pixels) is CPU-bound and dominates integration's read time for large
        # 16-bit detectors, so a lower cap leaves cores idle on big machines.
        _BSHUF_POOL = ThreadPoolExecutor(max_workers=max(2, CPU_COUNT))
    return _BSHUF_POOL


_HDF5_LAYOUT_CACHE: OrderedDict[str, tuple | None] = OrderedDict()
_HDF5_OPEN_FILES: OrderedDict[str, "h5py.File"] = OrderedDict()


def _hdf5_resolve_path(master_path: str, file_name: str) -> str:
    if file_name == ".":
        return master_path
    if os.path.isabs(file_name):
        return file_name
    return os.path.join(os.path.dirname(os.path.abspath(master_path)), file_name)


def _hdf5_open(file_path: str):
    """Keep a few image files open: a sweep is read in many small batches."""
    fh = _HDF5_OPEN_FILES.get(file_path)
    if fh is not None and fh.id.valid:
        _HDF5_OPEN_FILES.move_to_end(file_path)
        return fh
    # dxtbx opens NXmx files with swmr=True, and HDF5 refuses to open a file that is
    # already open in this process with a different SWMR flag, so match it.
    try:
        fh = h5py.File(file_path, "r", swmr=True)
    except Exception:
        fh = h5py.File(file_path, "r")
    _HDF5_OPEN_FILES[file_path] = fh
    while len(_HDF5_OPEN_FILES) > 8:
        _, old = _HDF5_OPEN_FILES.popitem(last=False)
        try:
            old.close()
        except Exception:
            pass
    return fh


def _hdf5_bit_depth(f) -> int | None:
    """The bit depth dxtbx uses to flag invalid pixels (see FormatNXmx and
    FormatNXmxEigerFilewriter.get_raw_data)."""
    det = "/entry/instrument/detector"
    names = ["bit_depth_readout"]
    if f"{det}/detectorSpecific/eiger_fw_version" in f:
        names.insert(0, "bit_depth_image")
    for name in names:
        try:
            return int(f[f"{det}/{name}"][()])
        except Exception:
            continue
    return None


def _hdf5_frame_layout(path: str):
    """
    Describe where each frame of an NXmx file is stored, as
    `(H, W, dtype, invalid_top, segments)`, with `segments` a list of
    `(first_frame, n_frames, file_path, dataset_name, first_source_frame)`.
    Handles a plain `/entry/data/data`, a virtual `/entry/data/data` with any number of
    sources (in the same or in external files), and DECTRIS filewriter master files whose
    `/entry/data/data_NNNNNN` entries link to the data files. Returns None for anything
    else. `invalid_top` is 2**bit_depth for 8/16/32-bit data with a bit depth recorded:
    dxtbx maps `invalid_top - 1` to -1 and `invalid_top - 2` to -2, so the fast path
    must too.
    """
    if path in _HDF5_LAYOUT_CACHE:
        _HDF5_LAYOUT_CACHE.move_to_end(path)
        return _HDF5_LAYOUT_CACHE[path]
    layout = None
    try:
        f = _hdf5_open(path)
        bit_depth = _hdf5_bit_depth(f)
        segments = []
        if "/entry/data/data" in f:
            dset = f["/entry/data/data"]
            if dset.ndim != 3:
                raise ValueError
            n, H, W = dset.shape
            if not dset.is_virtual:
                segments.append((0, n, path, dset.name, 0))
            else:
                for v in dset.virtual_sources():
                    if "%" in v.file_name:
                        raise ValueError  # printf-style source names
                    v_start, _, v_count, v_block = v.vspace.get_regular_hyperslab()
                    if v.src_space.get_select_type() == h5py.h5s.SEL_ALL:
                        # The whole source dataset (its dataspace extent is not stored)
                        s_start, s_count, s_block = (0, 0, 0), (1, 1, 1), v_block
                    else:
                        s_start, _, s_count, s_block = v.src_space.get_regular_hyperslab()
                    if (
                        tuple(v_count) != (1, 1, 1)
                        or tuple(s_count) != (1, 1, 1)
                        or tuple(v_block) != tuple(s_block)
                        or tuple(v_start[1:]) != (0, 0)
                        or tuple(s_start[1:]) != (0, 0)
                        or tuple(v_block[1:]) != (H, W)
                    ):
                        raise ValueError
                    segments.append(
                        (
                            int(v_start[0]),
                            int(v_block[0]),
                            _hdf5_resolve_path(path, v.file_name),
                            v.dset_name,
                            int(s_start[0]),
                        )
                    )
            dtype = dset.dtype
        else:
            grp = f["/entry/data"]
            names = sorted(
                k for k in grp if k.startswith("data_") and k[5:].isdigit() and len(k) == 11
            )
            if not names:
                raise ValueError
            dtype = None
            H = W = None
            z = 0
            for k in names:
                link = grp.get(k, getlink=True)
                if isinstance(link, h5py.ExternalLink):
                    fpath = _hdf5_resolve_path(path, link.filename)
                    dname = link.path
                elif isinstance(link, h5py.HardLink):
                    fpath, dname = path, grp[k].name
                else:
                    raise ValueError
                # Only shapes are needed here: use a short-lived handle rather than the
                # open-file cache, which would evict the master file.
                if fpath == path:
                    shape, dt = f[dname].shape, f[dname].dtype
                else:
                    with h5py.File(fpath, "r", swmr=True) as df:
                        shape, dt = df[dname].shape, df[dname].dtype
                if len(shape) != 3 or (H is not None and shape[1:] != (H, W)):
                    raise ValueError
                if dtype is not None and dt != dtype:
                    raise ValueError
                _, H, W = shape
                dtype = dt
                segments.append((z, shape[0], fpath, dname, 0))
                z += shape[0]
        if dtype not in (np.uint32, np.int32, np.uint16, np.int16):
            raise ValueError
        segments.sort()
        top = 2**31 if bit_depth == 32 else (2**bit_depth if bit_depth in (8, 16) else None)
        layout = (int(H), int(W), np.dtype(dtype), top, segments)
    except Exception:
        layout = None
    _HDF5_LAYOUT_CACHE[path] = layout
    while len(_HDF5_LAYOUT_CACHE) > 128:
        _HDF5_LAYOUT_CACHE.popitem(last=False)
    return layout


def _hdf5_pieces(layout, z0: int, B: int):
    """Split frames [z0, z0 + B) into `(out_offset, dset, src_frame, n)` reads, or None
    if some frame is not covered."""
    _, _, _, _, segments = layout
    pieces = []
    z = z0
    end = z0 + B
    for v0, n, fpath, dname, s0 in segments:
        if z >= end:
            break
        if v0 + n <= z:
            continue
        if v0 > z:
            return None
        take = min(end, v0 + n) - z
        pieces.append((z - z0, _hdf5_open(fpath)[dname], s0 + (z - v0), take))
        z += take
    return pieces if z == end else None


def _map_invalid_pixels(frame_i32: np.ndarray, top: int | None) -> None:
    """Map `top - 1` to -1 and `top - 2` to -2, as dxtbx does, in place."""
    if top is None:
        return
    if top <= 2**16:
        # Values never exceed top - 1 here, so subtracting `top` from the two highest
        # values is the mapping: one compare and a masked subtract, no fancy indexing.
        np.subtract(frame_i32, top, out=frame_i32, where=frame_i32 >= top - 2)
    elif frame_i32.max() >= top - 2:  # 32-bit: these values are rare, check first
        frame_i32[frame_i32 == top - 1] = -1
        frame_i32[frame_i32 == top - 2] = -2


def map_invalid_pixels_gpu(vol: torch.Tensor, top: int | None) -> None:
    """GPU version of `_map_invalid_pixels` for an int32 tensor, in place. Works a frame at
    a time so the temporaries stay small."""
    if top is None:
        return
    for f in vol if vol.dim() == 3 else (vol,):
        torch.where(f >= top - 2, (f - (top - 1)) - 1, f, out=f)


def _hdf5_slice(imageset, slice_start: int, slice_len: int | None):
    """`(path, z0, B, n_total)` for a contiguous slice of a single-panel HDF5 sequence of at
    most 64 frames, or None."""
    if h5py is None or not isinstance(imageset, ImageSequence):
        return None
    if len(imageset.get_detector()) != 1:
        return None
    try:
        path = imageset.get_path(0)
    except Exception:
        return None
    if not path or not path.endswith((".nxs", ".h5", ".hdf5")):
        return None
    all_indices = tuple(imageset.indices())
    if slice_len is None:
        indices = all_indices[slice_start:]
    else:
        indices = all_indices[slice_start : slice_start + slice_len]
    B = len(indices)
    if B == 0 or B > 64:
        return None
    z0 = int(indices[0])
    if indices != tuple(range(z0, z0 + B)):
        return None
    return path, z0, B, len(all_indices)


_RAW_FDS: OrderedDict[str, int] = OrderedDict()
_RAW_FDS_LOCK = threading.Lock()


def _raw_fd(file_path: str) -> int:
    """A cached read-only OS file descriptor for os.pread of raw chunk bytes."""
    with _RAW_FDS_LOCK:
        fd = _RAW_FDS.get(file_path)
        if fd is not None:
            _RAW_FDS.move_to_end(file_path)
            return fd
        fd = os.open(file_path, os.O_RDONLY)
        _RAW_FDS[file_path] = fd
        while len(_RAW_FDS) > 16:
            _, old = _RAW_FDS.popitem(last=False)
            try:
                os.close(old)
            except OSError:
                pass
        return fd


def _hdf5_read_chunks_parallel(pieces) -> list[bytes] | None:
    """
    Read the compressed chunks of `pieces` (one chunk per frame) concurrently: look up
    each chunk's file offset and size through HDF5, then os.pread them on the decode
    thread pool (pread releases the GIL; h5py serialises all its calls). On network
    filesystems the serial read_direct_chunk loop left the decode threads idle while one
    thread waited on each chunk in turn. Returns None if any chunk cannot be located this
    way (e.g. a partial filter mask), so the caller can use read_direct_chunk instead.
    """
    try:
        locs = []
        for _, d, s0, n in pieces:
            fpath = d.file.filename
            did = d.id
            for i in range(n):
                info = did.get_chunk_info_by_coord((s0 + i, 0, 0))
                if info.byte_offset is None or info.filter_mask != 0 or not info.size:
                    return None
                locs.append((fpath, int(info.byte_offset), int(info.size)))

        def _read(loc):
            fpath, off, size = loc
            fd = _raw_fd(fpath)
            buf = os.pread(fd, size, off)
            while len(buf) < size:  # short read (rare; e.g. signals on NFS)
                more = os.pread(fd, size - len(buf), off + len(buf))
                if not more:
                    raise OSError(f"short read of chunk at {off} in {fpath}")
                buf += more
            return buf

        return list(_get_bshuf_pool().map(_read, locs))
    except Exception:
        return None


def _hdf5_get_chunks(path: str, layout, z0: int, B: int):
    """
    `(chunks, pieces)` for frames [z0, z0 + B): `chunks` is the list of compressed
    bitshuffle-LZ4 frames (from the host cache or read directly), or None when the data
    are stored any other way, in which case `pieces` (see `_hdf5_pieces`) is set for
    h5py reads. Raises if the frames cannot be located.
    """
    H, W, dtype, _, _ = layout
    comp_key = (path, z0, B)
    bshuf_fn = _get_bshuf_decompress()
    comp_entry = _HDF5_COMPRESSED_CACHE.get(comp_key)
    if comp_entry is not None and bshuf_fn is not None:
        _HDF5_COMPRESSED_CACHE.move_to_end(comp_key)
        return comp_entry[0], None
    pieces = _hdf5_pieces(layout, z0, B)
    if pieces is None:
        raise ValueError("frames not found")
    if bshuf_fn is not None and all(
        d.chunks == (1, H, W)
        and d.id.get_create_plist().get_nfilters() == 1
        and d.id.get_create_plist().get_filter(0)[0] == 32008
        for _, d, _, _ in pieces
    ):
        chunks = _hdf5_read_chunks_parallel(pieces)
        if chunks is None:
            chunks = []
            for _, d, s0, n in pieces:
                did = d.id
                chunks.extend(did.read_direct_chunk((s0 + i, 0, 0))[1] for i in range(n))
        if len(_HDF5_COMPRESSED_CACHE) >= 100:
            _HDF5_COMPRESSED_CACHE.popitem(last=False)
        _HDF5_COMPRESSED_CACHE[comp_key] = (chunks, H, W, dtype)
        return chunks, None
    return None, pieces


def _bshuf_decompress_one(chunk: bytes, dst_ptr: int, n_elems: int, elem_size: int) -> None:
    _, block_size = struct.unpack(">QI", chunk[:12])
    in_ptr = ctypes.cast(ctypes.c_char_p(chunk), ctypes.c_void_p).value + 12
    res = _get_bshuf_decompress()(
        in_ptr, dst_ptr, n_elems, elem_size, block_size // elem_size
    )
    if res < 0:
        raise RuntimeError(f"bshuf_decompress_lz4 failed with code {res}")


def try_read_hdf5_raw_frames(imageset, slice_start: int, slice_len: int, dst: torch.Tensor):
    """
    Decompress frames of a bitshuffle-LZ4 HDF5 sequence into `dst` *without* converting
    them: `dst` is a flat pinned tensor whose storage receives `B * H * W` values of the
    file's own 2- or 4-byte integer type. Returns `(dtype, invalid_top)` (the caller widens
    to int32 and applies `map_invalid_pixels_gpu`), or None if the data are not stored as
    one bitshuffle-LZ4 chunk per frame, in which case nothing was written.
    Used by the spot-finding pipeline to move the conversion off the CPU: for 16-bit
    data this also halves the host-to-device transfer.
    """
    sl = _hdf5_slice(imageset, slice_start, slice_len)
    if sl is None:
        return None
    path, z0, B, _ = sl
    layout = _hdf5_frame_layout(path)
    if layout is None:
        return None
    H, W, dtype, top, _ = layout
    if dtype.itemsize not in (2, 4) or dst.numel() * dst.element_size() < B * H * W * dtype.itemsize:
        return None
    try:
        chunks, _ = _hdf5_get_chunks(path, layout, z0, B)
    except Exception:
        return None
    if chunks is None:
        return None
    n_elems = H * W
    elem_size = dtype.itemsize
    base = dst.data_ptr()
    batch_ext = _get_bshuf_batch_ext()
    if batch_ext is not None:
        bshuf_ptr, _, _, fn_direct = batch_ext
        c_ptrs = (ctypes.c_void_p * B)(
            *(ctypes.cast(ctypes.c_char_p(c), ctypes.c_void_p).value for c in chunks)
        )
        c_lens = (ctypes.c_size_t * B)(*(len(c) for c in chunks))
        rc = fn_direct(
            bshuf_ptr,
            c_ptrs,
            c_lens,
            B,
            n_elems,
            elem_size,
            base,
            max(2, min(32, CPU_COUNT)),
        )
        if rc == 0:
            return dtype, top
    list(
        _get_bshuf_pool().map(
            lambda i: _bshuf_decompress_one(chunks[i], base + i * n_elems * elem_size, n_elems, elem_size),
            range(B),
        )
    )
    return dtype, top


def _try_read_hdf5_slab_frames_into(
    path: str,
    z0: int,
    count: int,
    dst_slice: np.ndarray,
    cache_compressed: bool = False,
) -> bool:
    """
    Fast batch HDF5 frame reader for slabbed volume integration.
    When `dst_slice.dtype == layout_dtype` (e.g. uint16 for 16-bit Eiger/Pilatus HDF5) and
    `not cache_compressed`, uses `bshuf_pread_direct_batch` in C++ to `pread` and decompress
    bitshuffle-LZ4 chunks directly into `dst_slice` without Python chunk allocations or
    full-frame int32 widening passes. Also supports int32 `dst_slice` via `bshuf_decode_batch_i32`.
    """
    if count <= 0:
        return True
    layout = _hdf5_frame_layout(path)
    if layout is None:
        return False
    H, W, dtype, top, segments = layout
    if dst_slice.shape != (count, H, W):
        return False
    n_elems = H * W
    elem_size = int(dtype.itemsize)
    dtype_code = _dtype_to_bshuf_code(dtype)
    if dtype_code == 0:
        return False
    batch_ext = _get_bshuf_batch_ext()
    n_thr = max(2, min(32, CPU_COUNT))

    # Fast path 1: C++ pread + direct bshuf into native-dtype dst_slice (no int32 widening)
    if (
        not cache_compressed
        and dst_slice.dtype == dtype
        and batch_ext is not None
    ):
        bshuf_ptr, _, fn_pread, _ = batch_ext
        try:
            for p0 in range(0, count, 64):
                B = min(64, count - p0)
                cur_z0 = z0 + p0
                cur_end = cur_z0 + B
                fds = (ctypes.c_int * B)()
                offs = (ctypes.c_int64 * B)()
                szs = (ctypes.c_size_t * B)()
                k = 0
                z_cursor = cur_z0
                for v0, seg_n, fpath, dname, s0 in segments:
                    if z_cursor >= cur_end:
                        break
                    if v0 + seg_n <= z_cursor:
                        continue
                    if v0 > z_cursor:
                        raise ValueError("gap in segments")
                    take = min(cur_end, v0 + seg_n) - z_cursor
                    dset = _hdf5_open(fpath)[dname]
                    if (
                        dset.chunks != (1, H, W)
                        or dset.id.get_create_plist().get_nfilters() != 1
                        or dset.id.get_create_plist().get_filter(0)[0] != 32008
                    ):
                        raise ValueError("not bshuf")
                    fd = _raw_fd(fpath)
                    did = dset.id
                    src_start = s0 + (z_cursor - v0)
                    for i in range(take):
                        info = did.get_chunk_info_by_coord((src_start + i, 0, 0))
                        if info.byte_offset is None or info.filter_mask != 0 or not info.size:
                            raise ValueError("chunk info unavailable")
                        fds[k] = fd
                        offs[k] = int(info.byte_offset)
                        szs[k] = int(info.size)
                        k += 1
                    z_cursor += take
                if k != B:
                    raise ValueError("incomplete chunk list")
                out_ptr = dst_slice[p0 : p0 + B].ctypes.data
                rc = fn_pread(
                    bshuf_ptr, fds, offs, szs, B, n_elems, elem_size, out_ptr, n_thr
                )
                if rc != 0:
                    raise RuntimeError(f"bshuf_pread_direct_batch failed ({rc})")
            return True
        except Exception:
            pass

    # Fast path 2: read in 64-frame batches via _hdf5_get_chunks
    try:
        for p0 in range(0, count, 64):
            B = min(64, count - p0)
            chunks, pieces = _hdf5_get_chunks(path, layout, z0 + p0, B)
            sub_dst = dst_slice[p0 : p0 + B]
            if chunks is not None:
                if dst_slice.dtype == dtype and batch_ext is not None:
                    bshuf_ptr, _, _, fn_direct = batch_ext
                    c_ptrs = (ctypes.c_void_p * B)(
                        *(ctypes.cast(ctypes.c_char_p(c), ctypes.c_void_p).value for c in chunks)
                    )
                    c_lens = (ctypes.c_size_t * B)(*(len(c) for c in chunks))
                    rc = fn_direct(
                        bshuf_ptr,
                        c_ptrs,
                        c_lens,
                        B,
                        n_elems,
                        elem_size,
                        sub_dst.ctypes.data,
                        n_thr,
                    )
                    if rc == 0:
                        continue
                if dst_slice.dtype == np.int32 and batch_ext is not None:
                    bshuf_ptr, fn_i32, _, _ = batch_ext
                    c_ptrs = (ctypes.c_void_p * B)(
                        *(ctypes.cast(ctypes.c_char_p(c), ctypes.c_void_p).value for c in chunks)
                    )
                    c_lens = (ctypes.c_size_t * B)(*(len(c) for c in chunks))
                    rc = fn_i32(
                        bshuf_ptr,
                        c_ptrs,
                        c_lens,
                        B,
                        n_elems,
                        dtype_code,
                        int(top or 0),
                        sub_dst.ctypes.data,
                        n_thr,
                    )
                    if rc == 0:
                        continue
                if dst_slice.dtype != np.int32:
                    return False
                direct = elem_size == 4
                out_base = sub_dst.ctypes.data

                def _decomp_one(i: int):
                    if direct:
                        dst_ptr = out_base + i * n_elems * 4
                        tmp = None
                    else:
                        tmp = np.empty(n_elems, dtype=dtype)
                        dst_ptr = tmp.ctypes.data
                    _bshuf_decompress_one(chunks[i], dst_ptr, n_elems, elem_size)
                    frame = sub_dst[i]
                    if tmp is not None:
                        np.copyto(frame.reshape(-1), tmp, casting="unsafe")
                    _map_invalid_pixels(frame, top)

                list(_get_bshuf_pool().map(_decomp_one, range(B)))
            else:
                for off, d, s, n in pieces:
                    arr = d[s : s + n]
                    dst = sub_dst[off : off + n]
                    if dst_slice.dtype == dtype:
                        np.copyto(dst, arr)
                    elif dst_slice.dtype == np.int32:
                        if arr.dtype == np.uint32:
                            np.copyto(dst, arr.view(np.int32))
                        else:
                            np.copyto(dst, arr, casting="unsafe")
                        for k in range(n):
                            _map_invalid_pixels(dst[k], top)
                    else:
                        return False
        return True
    except Exception:
        return False


def _try_load_hdf5_raw_volume_gpu(
    imageset,
    device: torch.device,
    slice_start: int = 0,
    slice_len: int | None = None,
    pinned_out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """
    Fast path for single-panel HDF5/NeXus ImageSequences, returning frames as an int32
    `(B, H, W)` tensor with the same values dxtbx's `get_raw_data` gives:
    - Tier-1 LRU cache: keeps up to 8 decoded volumes on GPU VRAM for short sweeps
      (zero-overhead RL parameter sweeps).
    - Tier-2 LRU cache: keeps up to 100 compressed bitshuffle-LZ4 chunk lists in host RAM.
    - Direct HDF5 chunk read + multithreaded bshuf_decompress_lz4 into contiguous memory,
      for a plain or virtual `/entry/data/data` (single or multiple source files) and for
      DECTRIS master files with `data_NNNNNN` links. Other compressions fall back to
      h5py slice reads.
    """
    sl = _hdf5_slice(imageset, slice_start, slice_len)
    if sl is None:
        return None
    path, z0, B, n_total = sl

    use_vram_cache = n_total <= 64
    dev_key = (device.type, device.index if device.index is not None else 0)
    gpu_key = (path, z0, B, dev_key)
    if use_vram_cache and gpu_key in _HDF5_GPU_VOLUME_CACHE:
        _HDF5_GPU_VOLUME_CACHE.move_to_end(gpu_key)
        return _HDF5_GPU_VOLUME_CACHE[gpu_key]

    layout = _hdf5_frame_layout(path)
    if layout is None:
        return None
    H, W, dtype, top, _ = layout

    comp_key = (path, z0, B)
    try:
        chunks, pieces = _hdf5_get_chunks(path, layout, z0, B)
    except Exception:
        return None

    # Pick the int32 destination: the caller's pinned buffer, a reusable pinned buffer,
    # or a fresh array.
    pin_key = (B, H, W, np.int32)
    pinned_t = None
    if pinned_out is not None:
        out_np = pinned_out[:B].numpy()
    else:
        pinned_t = _PINNED_HOST_VOL_CACHE.get(pin_key)
        if pinned_t is not None and _PINNED_HOST_VOL_TAG.get(pin_key) == comp_key:
            if device.type == "cpu":
                return pinned_t
            vol_t = pinned_t.to(device=device, non_blocking=True)
            if use_vram_cache and device.type == "cuda":
                if len(_HDF5_GPU_VOLUME_CACHE) >= 8:
                    _HDF5_GPU_VOLUME_CACHE.popitem(last=False)
                _HDF5_GPU_VOLUME_CACHE[gpu_key] = vol_t
            return vol_t
        if pinned_t is None:
            if len(_PINNED_HOST_VOL_CACHE) >= 2:
                _PINNED_HOST_VOL_CACHE.clear()
                _PINNED_HOST_VOL_TAG.clear()
            pinned_t = torch.empty(
                (B, H, W), dtype=torch.int32, pin_memory=torch.cuda.is_available()
            )
            _PINNED_HOST_VOL_CACHE[pin_key] = pinned_t
        # The buffer is about to be overwritten: it no longer holds any cached frames.
        _PINNED_HOST_VOL_TAG.pop(pin_key, None)
        out_np = pinned_t.numpy()

    try:
        if chunks is not None:
            elem_size = int(dtype.itemsize)
            n_elems = H * W
            dtype_code = _dtype_to_bshuf_code(dtype)
            batch_ext = _get_bshuf_batch_ext() if dtype_code != 0 else None
            decoded_fast = False
            if batch_ext is not None:
                bshuf_ptr, fn_i32, _, _ = batch_ext
                c_ptrs = (ctypes.c_void_p * B)(
                    *(ctypes.cast(ctypes.c_char_p(c), ctypes.c_void_p).value for c in chunks)
                )
                c_lens = (ctypes.c_size_t * B)(*(len(c) for c in chunks))
                rc = fn_i32(
                    bshuf_ptr,
                    c_ptrs,
                    c_lens,
                    B,
                    n_elems,
                    dtype_code,
                    int(top or 0),
                    out_np.ctypes.data,
                    max(2, min(32, CPU_COUNT)),
                )
                if rc == 0:
                    decoded_fast = True
            if not decoded_fast:
                direct = elem_size == 4  # int32 / uint32 decompress straight into int32
                out_base = out_np.ctypes.data

                def _decomp_one(i: int):
                    if direct:
                        dst_ptr = out_base + i * n_elems * 4
                        tmp = None
                    else:
                        tmp = np.empty(n_elems, dtype=dtype)
                        dst_ptr = tmp.ctypes.data
                    _bshuf_decompress_one(chunks[i], dst_ptr, n_elems, elem_size)
                    frame = out_np[i]
                    if tmp is not None:
                        np.copyto(frame.reshape(-1), tmp, casting="unsafe")
                    _map_invalid_pixels(frame, top)

                list(_get_bshuf_pool().map(_decomp_one, range(B)))
        else:
            for off, d, s, n in pieces:
                arr = d[s : s + n]
                dst = out_np[off : off + n]
                if arr.dtype == np.uint32:
                    np.copyto(dst, arr.view(np.int32))
                else:
                    np.copyto(dst, arr, casting="unsafe")
                for k in range(n):
                    _map_invalid_pixels(dst[k], top)
    except Exception:
        return None

    if pinned_out is not None:
        return pinned_out[:B]

    if device.type == "cpu":
        # Only the CPU result aliases the reusable buffer; tag it so a repeat call for
        # the same frames can return it without decoding again.
        _PINNED_HOST_VOL_TAG[pin_key] = comp_key
        return pinned_t
    vol_t = pinned_t.to(device=device, non_blocking=True)
    _PINNED_HOST_VOL_TAG[pin_key] = comp_key
    if use_vram_cache and device.type == "cuda":
        if len(_HDF5_GPU_VOLUME_CACHE) >= 8:
            _HDF5_GPU_VOLUME_CACHE.popitem(last=False)
        _HDF5_GPU_VOLUME_CACHE[gpu_key] = vol_t
    return vol_t


class DispersionThresholdExtGPU:
    """
    GPU replacement for DispersionSpotFinderThresholdExt and DispersionExtendedSpotFinderThresholdExt.
    """

    def __init__(self, params, extended: bool = True, device: torch.device | None = None):
        self.params = params
        self.extended = extended
        self.device = device or get_default_device()

    def _get_strategy(self, image, mask):
        params = self.params
        if params.spotfinder.threshold.dispersion.global_threshold is libtbx.Auto:
            params.spotfinder.threshold.dispersion.global_threshold = int(
                estimate_global_threshold(image, mask, device=self.device)
            )
            ext_logger.info(
                "Setting global_threshold: %i",
                params.spotfinder.threshold.dispersion.global_threshold,
            )

        cls = (
            DispersionExtendedSpotFinderThresholdGPU
            if self.extended
            else DispersionSpotFinderThresholdGPU
        )
        return cls(
            kernel_size=params.spotfinder.threshold.dispersion.kernel_size,
            gain=params.spotfinder.threshold.dispersion.gain,
            mask=params.spotfinder.lookup.mask,
            n_sigma_b=params.spotfinder.threshold.dispersion.sigma_background,
            n_sigma_s=params.spotfinder.threshold.dispersion.sigma_strong,
            min_count=params.spotfinder.threshold.dispersion.min_local,
            global_threshold=params.spotfinder.threshold.dispersion.global_threshold,
            device=self.device,
        )

    def compute_threshold(self, image, mask, **kwargs):
        strategy = self._get_strategy(image, mask)
        return strategy.compute_threshold(image, mask, gain=kwargs.get("gain"))

    def compute_threshold_tensor(self, img_t: torch.Tensor, mask_t: torch.Tensor, gain=None, **kwargs) -> torch.Tensor:
        strategy = self._get_strategy(img_t, mask_t)
        return strategy.compute_threshold_tensor(img_t, mask_t, gain=gain)

    def compute_threshold_batch_tensor(
        self, img_vol: torch.Tensor, mask_vol: torch.Tensor, gain=None, **kwargs
    ) -> torch.Tensor:
        strategy = self._get_strategy(img_vol[0], mask_vol[0])
        return strategy.compute_threshold_batch_tensor(img_vol, mask_vol, gain=gain)


def build_reflection_table_from_gpu_results(
    panel_t: torch.Tensor,
    bbox_t: torch.Tensor,
    xyzobs_px_val_t: torch.Tensor,
    xyzobs_px_var_t: torch.Tensor,
    intensity_val_t: torch.Tensor,
    intensity_var_t: torch.Tensor,
    shoeboxes: flex.shoebox | None = None,
    n_signal_t: torch.Tensor | None = None,
    include_id_flags: bool = True,
) -> flex.reflection_table:
    """
    Construct a dials.array_family.flex.reflection_table from filtered GPU tensors.
    """
    K = int(panel_t.shape[0])
    rlist = flex.reflection_table()
    if K == 0:
        if include_id_flags:
            rlist["id"] = flex.int()
            rlist["flags"] = flex.size_t()
        rlist["panel"] = flex.size_t()
        rlist["xyzobs.px.value"] = flex.vec3_double()
        rlist["xyzobs.px.variance"] = flex.vec3_double()
        rlist["intensity.sum.value"] = flex.double()
        rlist["intensity.sum.variance"] = flex.double()
        rlist["bbox"] = flex.int6()
        if shoeboxes is not None:
            rlist["shoebox"] = shoeboxes
        if n_signal_t is not None:
            rlist["n_signal"] = flex.int()
        return rlist

    panel_np = np.ascontiguousarray(panel_t.detach().cpu().numpy(), dtype=np.uint64)
    bbox_np = np.ascontiguousarray(bbox_t.detach().cpu().numpy(), dtype=np.int32)
    xyz_val_np = np.ascontiguousarray(xyzobs_px_val_t.detach().cpu().numpy(), dtype=np.float64)
    xyz_var_np = np.ascontiguousarray(xyzobs_px_var_t.detach().cpu().numpy(), dtype=np.float64)
    int_val_np = np.ascontiguousarray(intensity_val_t.detach().cpu().numpy(), dtype=np.float64)
    int_var_np = np.ascontiguousarray(intensity_var_t.detach().cpu().numpy(), dtype=np.float64)

    if include_id_flags:
        rlist["id"] = flex.int(K, 0)
        rlist["flags"] = flex.size_t(K, 0)

    rlist["panel"] = flumpy.from_numpy(panel_np)
    rlist["xyzobs.px.value"] = flumpy.vec_from_numpy(xyz_val_np)
    rlist["xyzobs.px.variance"] = flumpy.vec_from_numpy(xyz_var_np)
    rlist["intensity.sum.value"] = flumpy.from_numpy(int_val_np)
    rlist["intensity.sum.variance"] = flumpy.from_numpy(int_var_np)
    rlist["bbox"] = flex.int6(
        flumpy.from_numpy(np.ascontiguousarray(bbox_np[:, 0])),
        flumpy.from_numpy(np.ascontiguousarray(bbox_np[:, 1])),
        flumpy.from_numpy(np.ascontiguousarray(bbox_np[:, 2])),
        flumpy.from_numpy(np.ascontiguousarray(bbox_np[:, 3])),
        flumpy.from_numpy(np.ascontiguousarray(bbox_np[:, 4])),
        flumpy.from_numpy(np.ascontiguousarray(bbox_np[:, 5])),
    )
    if shoeboxes is not None:
        rlist["shoebox"] = shoeboxes
    if n_signal_t is not None:
        n_sig_np = np.ascontiguousarray(n_signal_t.detach().cpu().numpy(), dtype=np.int32)
        rlist["n_signal"] = flumpy.from_numpy(n_sig_np)

    return rlist


class ExtractPixelsFromImageGPU:
    """
    GPU-accelerated replacement for ExtractPixelsFromImage.
    Thresholds each panel on the GPU and returns GPUFramePixelList objects.
    """

    def __init__(
        self,
        imageset,
        threshold_function,
        mask,
        region_of_interest,
        max_strong_pixel_fraction: float,
        compute_mean_background: bool,
        device: torch.device | None = None,
    ):
        self.threshold_function = threshold_function
        self.imageset = imageset
        self.mask = mask
        self.region_of_interest = region_of_interest
        self.max_strong_pixel_fraction = max_strong_pixel_fraction
        self.compute_mean_background = compute_mean_background
        self.device = device or get_default_device()

        if self.mask is not None:
            detector = self.imageset.get_detector()
            assert len(self.mask) == len(detector)
            self._static_mask_tensors = [
                flex_to_tensor(m, device=self.device, dtype=torch.bool) for m in self.mask
            ]
        else:
            self._static_mask_tensors = None

        self._masker = (
            self.imageset.masker()
            if hasattr(self.imageset, "masker")
            else None
        )
        self._can_use_fast_gpu_raw = (
            isinstance(self.imageset, ImageSequence)
            and (not self.imageset.has_dynamic_mask() or self._masker is not None)
            and hasattr(self.threshold_function, "compute_threshold_tensor")
        )
        if self._can_use_fast_gpu_raw:
            self._detector = self.imageset.get_detector()
            self._scan = self.imageset.get_scan()
            self._image_range_start = self._scan.get_image_range()[0]
            ext_mask_data = self.imageset.external_lookup.mask.data
            has_ext_mask = not ext_mask_data.empty()
            self._full_static_mask_tensors = []
            self._trusted_ranges = []

            try:
                path_key = self.imageset.get_path(0)
            except Exception:
                path_key = None
            dev_key = (self.device.type, self.device.index if self.device.index is not None else 0)
            cache_key = (
                path_key,
                len(self._detector),
                has_ext_mask,
                tuple(tuple(p.get_image_size()) for p in self._detector),
                dev_key,
            )
            base_masks = _BASE_STATIC_MASK_GPU_CACHE.get(cache_key) if path_key else None
            if base_masks is not None:
                _BASE_STATIC_MASK_GPU_CACHE.move_to_end(cache_key)
            else:
                base_masks = []
                for i_panel, panel in enumerate(self._detector):
                    sm = flex_to_tensor(
                        panel.get_untrusted_rectangle_mask(),
                        device=self.device,
                        dtype=torch.bool,
                    )
                    if has_ext_mask:
                        em = flex_to_tensor(
                            ext_mask_data[i_panel].data(),
                            device=self.device,
                            dtype=torch.bool,
                        )
                        sm = sm & em
                    base_masks.append(sm)
                if path_key:
                    if len(_BASE_STATIC_MASK_GPU_CACHE) >= 32:
                        _BASE_STATIC_MASK_GPU_CACHE.popitem(last=False)
                    _BASE_STATIC_MASK_GPU_CACHE[cache_key] = base_masks

            self._base_static_mask_tensors = base_masks
            self._path_key = path_key
            for i_panel, panel in enumerate(self._detector):
                sm = base_masks[i_panel]
                if self._static_mask_tensors is not None:
                    sm = sm & self._static_mask_tensors[i_panel]
                self._full_static_mask_tensors.append(sm)
                self._trusted_ranges.append(panel.get_trusted_range())

            if self.imageset.external_lookup.gain.data.empty():
                panel_gains = [float(p.get_gain()) for p in self._detector]
                if all(g > 0.0 for g in panel_gains) and any(
                    abs(g - 1.0) > 1e-7 for g in panel_gains
                ):
                    self._gain_tensors = [
                        torch.tensor(g, device=self.device, dtype=torch.float64)
                        for g in panel_gains
                    ]
                else:
                    self._gain_tensors = None
            else:
                gain_img = self.imageset.get_gain(0)
                if gain_img:
                    self._gain_tensors = [
                        flex_to_tensor(
                            gain_img[i], device=self.device, dtype=torch.float64
                        )
                        for i in range(len(self._detector))
                    ]
                else:
                    self._gain_tensors = None

            if self.imageset.external_lookup.pedestal.data.empty():
                panel_peds = [float(p.get_pedestal()) for p in self._detector]
                if any(abs(p) > 1e-7 for p in panel_peds):
                    self._pedestal_tensors = [
                        torch.tensor(p, device=self.device, dtype=torch.float64)
                        for p in panel_peds
                    ]
                else:
                    self._pedestal_tensors = None
            else:
                ped_img = self.imageset.get_pedestal(0)
                if ped_img:
                    self._pedestal_tensors = [
                        flex_to_tensor(
                            ped_img[i], device=self.device, dtype=torch.float64
                        )
                        for i in range(len(self._detector))
                    ]
                else:
                    self._pedestal_tensors = None

    def process_raw_tensors(
        self,
        index: int,
        raw_tensors: list[torch.Tensor] | None,
        emit_log: bool = True,
    ) -> tuple[list[GPUFramePixelList], list[str]]:
        frame = self.imageset.get_array_range()[0] + index
        rejected = raw_tensors is None or self.imageset.is_marked_for_rejection(index)
        num_panels = len(self._full_static_mask_tensors)

        dyn_mask_tuple = None
        if not rejected and self._masker is not None:
            angle = self._scan.get_angle_from_image_index(
                index + self._image_range_start, deg=True
            )
            ext = self._masker.project_extrema(self._detector, angle)
            if any(len(e) > 0 for e in ext):
                dyn_mask_tuple = self._masker.get_mask(self._detector, angle)

        result = []
        num_strong = 0
        average_background = 0.0

        for i_panel in range(num_panels):
            static_mask_t = self._full_static_mask_tensors[i_panel]
            H, W = static_mask_t.shape

            if rejected:
                img_t = torch.empty((0, 0), dtype=torch.float64, device=self.device)
                mask_t = static_mask_t
                threshold_mask = torch.zeros_like(static_mask_t)
            else:
                raw_t = raw_tensors[i_panel]
                tr0, tr1 = self._trusted_ranges[i_panel]
                mask_t = static_mask_t & (raw_t >= tr0) & (raw_t <= tr1)
                if dyn_mask_tuple is not None and len(dyn_mask_tuple[i_panel]) > 0:
                    dm_t = flex_to_tensor(
                        dyn_mask_tuple[i_panel], device=self.device, dtype=torch.bool
                    )
                    mask_t = mask_t & dm_t
                if (
                    self._pedestal_tensors is None
                    and self._gain_tensors is None
                    and self.region_of_interest is None
                    and not self.compute_mean_background
                    and isinstance(self.threshold_function, DispersionThresholdExtGPU)
                ):
                    img_t = raw_t
                else:
                    img_t = raw_t.to(torch.float64)
                    if self._pedestal_tensors is not None:
                        img_t = img_t - self._pedestal_tensors[i_panel]
                    if self._gain_tensors is not None:
                        img_t = img_t / self._gain_tensors[i_panel]

                if self.region_of_interest is not None:
                    x0, x1, y0, y1 = self.region_of_interest
                    assert x0 < x1 and y0 < y1, "Invalid region of interest"
                    assert x0 >= 0 and y0 >= 0 and x1 <= W and y1 <= H, "ROI out of bounds"
                    thresh_roi = self.threshold_function.compute_threshold_tensor(
                        img_t[y0:y1, x0:x1],
                        mask_t[y0:y1, x0:x1],
                        imageset=self.imageset,
                        i_panel=i_panel,
                        region_of_interest=self.region_of_interest,
                    )
                    threshold_mask = torch.zeros_like(mask_t)
                    threshold_mask[y0:y1, x0:x1] = thresh_roi
                else:
                    threshold_mask = self.threshold_function.compute_threshold_tensor(
                        img_t,
                        mask_t,
                        imageset=self.imageset,
                        i_panel=i_panel,
                        region_of_interest=self.region_of_interest,
                    )

            yx = torch.nonzero(threshold_mask, as_tuple=False)
            n_strong_panel = int(yx.shape[0])
            if n_strong_panel > 0:
                vals = img_t[threshold_mask].to(torch.float64)
                z = torch.full((n_strong_panel, 1), frame, dtype=torch.int64, device=self.device)
                coords = torch.cat([z, yx.to(torch.int64)], dim=1)
            else:
                coords = torch.empty((0, 3), dtype=torch.int64, device=self.device)
                vals = torch.empty((0,), dtype=torch.float64, device=self.device)

            plist = GPUFramePixelList(frame=frame, size=(H, W), coords=coords, values=vals)
            result.append(plist)

            if self.compute_mean_background and not rejected:
                bg_sel = mask_t & (~threshold_mask)
                bg_vals = img_t[bg_sel]
                if bg_vals.numel() > 0:
                    average_background += float(bg_vals.mean().item())

            num_strong += n_strong_panel

            if self.max_strong_pixel_fraction < 1:
                num_image = H * W
                max_strong = int(math.ceil(self.max_strong_pixel_fraction * num_image))
                if n_strong_panel > max_strong:
                    raise RuntimeError(
                        f"\n  The number of strong pixels found ({n_strong_panel}) is greater than the\n"
                        f"  maximum allowed ({max_strong}). Try changing spot finding parameters\n"
                    )

        average_background /= num_panels
        frame_log = frame + 1
        log_msgs = [f"Found {num_strong} strong pixels on image {frame_log}"]
        if self.compute_mean_background:
            log_msgs.append(f"Average background on image {frame_log} is {average_background}")
        if emit_log:
            for msg in log_msgs:
                logger.info(msg)
        return result, log_msgs

    def __call__(self, index: int) -> list[GPUFramePixelList]:
        if self._can_use_fast_gpu_raw:
            if self.imageset.is_marked_for_rejection(index):
                raw_tensors = None
            else:
                raw_image = self.imageset.get_raw_data(index)
                raw_tensors = [
                    flex_to_tensor(raw_image[i], device=self.device)
                    for i in range(len(raw_image))
                ]
            res, _ = self.process_raw_tensors(index, raw_tensors, emit_log=True)
            return res

        if isinstance(self.imageset, ImageSequence):
            frame = self.imageset.get_array_range()[0] + index
        else:
            ind = self.imageset.indices()
            if len(ind) > 1:
                assert all(i1 + 1 == i2 for i1, i2 in zip(ind[:-1], ind[1:-1]))
            frame = ind[index]

        image = self.imageset.get_corrected_data(index)
        mask = self.imageset.get_mask(index)

        result = []
        num_strong = 0
        average_background = 0.0
        rejected = self.imageset.is_marked_for_rejection(index)

        for i_panel in range(len(image)):
            im_panel = image[i_panel]
            mk_panel = mask[i_panel]
            img_t = flex_to_tensor(im_panel, device=self.device, dtype=torch.float64)
            mask_t = flex_to_tensor(mk_panel, device=self.device, dtype=torch.bool)
            if self._static_mask_tensors is not None:
                mask_t = mask_t & self._static_mask_tensors[i_panel]

            H, W = img_t.shape

            if rejected:
                threshold_mask = torch.zeros_like(mask_t)
            elif self.region_of_interest is not None:
                x0, x1, y0, y1 = self.region_of_interest
                assert x0 < x1 and y0 < y1, "Invalid region of interest"
                assert x0 >= 0 and y0 >= 0 and x1 <= W and y1 <= H, "ROI out of bounds"
                img_roi = img_t[y0:y1, x0:x1]
                mask_roi = mask_t[y0:y1, x0:x1]
                if hasattr(self.threshold_function, "compute_threshold_tensor"):
                    thresh_roi = self.threshold_function.compute_threshold_tensor(
                        img_roi,
                        mask_roi,
                        imageset=self.imageset,
                        i_panel=i_panel,
                        region_of_interest=self.region_of_interest,
                    )
                else:
                    im_flex = im_panel[y0:y1, x0:x1]
                    mk_flex = tensor_to_flex_bool(mask_roi)
                    tr_flex = self.threshold_function.compute_threshold(
                        im_flex,
                        mk_flex,
                        imageset=self.imageset,
                        i_panel=i_panel,
                        region_of_interest=self.region_of_interest,
                    )
                    thresh_roi = flex_to_tensor(tr_flex, device=self.device, dtype=torch.bool)
                threshold_mask = torch.zeros_like(mask_t)
                threshold_mask[y0:y1, x0:x1] = thresh_roi
            else:
                if hasattr(self.threshold_function, "compute_threshold_tensor"):
                    threshold_mask = self.threshold_function.compute_threshold_tensor(
                        img_t,
                        mask_t,
                        imageset=self.imageset,
                        i_panel=i_panel,
                        region_of_interest=self.region_of_interest,
                    )
                else:
                    mk_flex = tensor_to_flex_bool(mask_t)
                    tr_flex = self.threshold_function.compute_threshold(
                        im_panel,
                        mk_flex,
                        imageset=self.imageset,
                        i_panel=i_panel,
                    )
                    threshold_mask = flex_to_tensor(tr_flex, device=self.device, dtype=torch.bool)

            yx = torch.nonzero(threshold_mask, as_tuple=False)
            n_strong_panel = int(yx.shape[0])
            if n_strong_panel > 0:
                vals = img_t[threshold_mask]
                z = torch.full((n_strong_panel, 1), frame, dtype=torch.int64, device=self.device)
                coords = torch.cat([z, yx.to(torch.int64)], dim=1)
            else:
                coords = torch.empty((0, 3), dtype=torch.int64, device=self.device)
                vals = torch.empty((0,), dtype=torch.float64, device=self.device)

            plist = GPUFramePixelList(frame=frame, size=(H, W), coords=coords, values=vals)
            result.append(plist)

            if self.compute_mean_background:
                bg_sel = mask_t & (~threshold_mask)
                bg_vals = img_t[bg_sel]
                if bg_vals.numel() > 0:
                    average_background += float(bg_vals.mean().item())

            num_strong += n_strong_panel

            if self.max_strong_pixel_fraction < 1:
                num_image = H * W
                max_strong = int(math.ceil(self.max_strong_pixel_fraction * num_image))
                if n_strong_panel > max_strong:
                    raise RuntimeError(
                        f"\n  The number of strong pixels found ({n_strong_panel}) is greater than the\n"
                        f"  maximum allowed ({max_strong}). Try changing spot finding parameters\n"
                    )

        average_background /= len(image)

        if isinstance(self.imageset, ImageSequence):
            frame_log = frame + 1
        else:
            frame_log = index + 1
        logger.info(f"Found {num_strong} strong pixels on image {frame_log}")
        if self.compute_mean_background:
            logger.info(f"Average background on image {frame_log} is {average_background}")

        return result


class ExtractPixelsFromImage2DNoShoeboxesGPU(ExtractPixelsFromImageGPU):
    """
    GPU-accelerated replacement for ExtractPixelsFromImage2DNoShoeboxes.
    """

    def __init__(
        self,
        imageset,
        threshold_function,
        mask,
        region_of_interest,
        max_strong_pixel_fraction: float,
        compute_mean_background: bool,
        min_spot_size: int,
        max_spot_size: int,
        filter_spots,
        device: torch.device | None = None,
    ):
        super().__init__(
            imageset=imageset,
            threshold_function=threshold_function,
            mask=mask,
            region_of_interest=region_of_interest,
            max_strong_pixel_fraction=max_strong_pixel_fraction,
            compute_mean_background=compute_mean_background,
            device=device,
        )
        self.min_spot_size = min_spot_size
        self.max_spot_size = max_spot_size
        self.filter_spots = filter_spots

    def __call__(self, index: int) -> list[flex.reflection_table]:
        pixel_labeller = [
            PixelListLabellerGPU(device=self.device)
            for _ in range(len(self.imageset.get_detector()))
        ]
        result = super().__call__(index)
        for plabeller, plist in zip(pixel_labeller, result):
            plabeller.add(plist)

        rlist, _ = pixel_list_to_reflection_table_gpu(
            self.imageset,
            pixel_labeller,
            filter_spots=self.filter_spots,
            min_spot_size=self.min_spot_size,
            max_spot_size=self.max_spot_size,
            write_hot_pixel_mask=False,
            include_shoeboxes=True,
            include_n_signal=False,
            force_2d=True,
            device=self.device,
        )
        return [rlist]


def pixel_list_to_reflection_table_gpu(
    imageset,
    pixel_labeller: list[PixelListLabellerGPU],
    filter_spots,
    min_spot_size: int,
    max_spot_size: int,
    write_hot_pixel_mask: bool,
    include_shoeboxes: bool = True,
    include_n_signal: bool = False,
    force_2d: bool = False,
    device: torch.device | None = None,
) -> tuple[flex.reflection_table, tuple]:
    """
    GPU replacement for dials.algorithms.spot_finding.finder.pixel_list_to_reflection_table.
    """
    device = device or get_default_device()
    if isinstance(imageset, ImageSequence) and not force_2d:
        twod = imageset.get_scan().is_still()
    else:
        twod = True

    hot_pixels = tuple(flex.size_t() for _ in range(len(imageset.get_detector())))

    num_total = 0
    num_below_min = 0
    num_above_max = 0
    num_valid = 0

    panel_list = []
    bbox_list = []
    xyz_val_list = []
    xyz_var_list = []
    int_val_list = []
    int_var_list = []
    n_signal_list = []
    peak_coords_list = []
    panel_spot_results = []

    for i, labeller in enumerate(pixel_labeller):
        if labeller.num_pixels() > 0:
            res = extract_spots_from_labeller_gpu(
                labeller=labeller,
                panel_id=i,
                zstart=0,
                twod=twod,
                min_pixels=min_spot_size,
                max_pixels=max_spot_size,
                find_hot_pixels=write_hot_pixel_mask,
                compute_shoeboxes=include_shoeboxes,
            )
            num_total += res["num_total_spots"]
            num_below_min += res["num_below_min"]
            num_above_max += res["num_above_max"]
            num_valid += res["num_valid_spots"]
            if write_hot_pixel_mask and len(res["hot_pixels"]) > 0:
                hot_pixels[i].extend(res["hot_pixels"])
            if res["num_valid_spots"] > 0:
                panel_list.append(res["panel"])
                bbox_list.append(res["bbox"])
                xyz_val_list.append(res["xyzobs_px_value"])
                xyz_var_list.append(res["xyzobs_px_variance"])
                int_val_list.append(res["intensity_sum_value"])
                int_var_list.append(res["intensity_sum_variance"])
                n_signal_list.append(res["n_signal"])
                peak_coords_list.append(res["peak_coords"])
                panel_spot_results.append(res)

    logger.info(f"\nExtracted {num_total} spots")
    logger.info(f"Removed {num_below_min} spots with size < {min_spot_size} pixels")
    logger.info(f"Removed {num_above_max} spots with size > {max_spot_size} pixels")

    if num_valid == 0:
        empty_rlist = build_reflection_table_from_gpu_results(
            panel_t=torch.empty((0,), dtype=torch.int64, device=device),
            bbox_t=torch.empty((0, 6), dtype=torch.int32, device=device),
            xyzobs_px_val_t=torch.empty((0, 3), dtype=torch.float64, device=device),
            xyzobs_px_var_t=torch.empty((0, 3), dtype=torch.float64, device=device),
            intensity_val_t=torch.empty((0,), dtype=torch.float64, device=device),
            intensity_var_t=torch.empty((0,), dtype=torch.float64, device=device),
            shoeboxes=flex.shoebox() if include_shoeboxes else None,
            n_signal_t=torch.empty((0,), dtype=torch.int32, device=device) if include_n_signal else None,
            include_id_flags=True,
        )
        return empty_rlist, hot_pixels

    panel_t = torch.cat(panel_list, dim=0)
    bbox_t = torch.cat(bbox_list, dim=0)
    xyz_val_t = torch.cat(xyz_val_list, dim=0)
    xyz_var_t = torch.cat(xyz_var_list, dim=0)
    int_val_t = torch.cat(int_val_list, dim=0)
    int_var_t = torch.cat(int_var_list, dim=0)
    n_sig_t = torch.cat(n_signal_list, dim=0)
    peak_t = torch.cat(peak_coords_list, dim=0)

    if include_shoeboxes:
        logger.info(f"Calculated {num_valid} spot centroids")
        logger.info(f"Calculated {num_valid} spot intensities")

    # Check if filter needs shoeboxes prior to filtering (e.g. BackgroundGradientFilter)
    needs_pre_shoeboxes = getattr(filter_spots, "needs_shoeboxes_for_filtering", False)
    pre_shoeboxes = None
    if needs_pre_shoeboxes:
        sbox_all = flex.shoebox()
        for res in panel_spot_results:
            sb = build_shoeboxes_from_gpu_spots(
                panel_t=res["panel"],
                bbox_t=res["bbox"],
                pixel_coords=res["pixel_coords"],
                pixel_values_f32=res["pixel_values"],
                pixel_spot_idx=res["pixel_spot_idx"],
                zstart=res["zstart"],
                selected_spot_mask=None,
            )
            sbox_all.extend(sb)
        pre_shoeboxes = sbox_all

    # Run spot filters on GPU
    if filter_spots is not None:
        init_flags = torch.ones((num_valid,), dtype=torch.bool, device=device)
        flags_t = filter_spots(
            init_flags,
            peak_coords=peak_t,
            centroids=xyz_val_t,
            sequence=imageset,
            shoeboxes=pre_shoeboxes,
        )
    else:
        flags_t = torch.ones((num_valid,), dtype=torch.bool, device=device)

    # Select filtered spots
    panel_sel = panel_t[flags_t]
    bbox_sel = bbox_t[flags_t]
    xyz_val_sel = xyz_val_t[flags_t]
    xyz_var_sel = xyz_var_t[flags_t]
    int_val_sel = int_val_t[flags_t]
    int_var_sel = int_var_t[flags_t]
    n_sig_sel = n_sig_t[flags_t] if include_n_signal else None

    shoeboxes_sel = None
    if include_shoeboxes:
        if pre_shoeboxes is not None:
            flags_np = np.ascontiguousarray(flags_t.detach().cpu().numpy(), dtype=bool)
            shoeboxes_sel = pre_shoeboxes.select(flumpy.from_numpy(flags_np))
        else:
            shoeboxes_sel = flex.shoebox()
            offset = 0
            for res in panel_spot_results:
                k_panel = res["num_valid_spots"]
                panel_mask = flags_t[offset : offset + k_panel]
                offset += k_panel
                sb = build_shoeboxes_from_gpu_spots(
                    panel_t=res["panel"],
                    bbox_t=res["bbox"],
                    pixel_coords=res["pixel_coords"],
                    pixel_values_f32=res["pixel_values"],
                    pixel_spot_idx=res["pixel_spot_idx"],
                    zstart=res["zstart"],
                    selected_spot_mask=panel_mask,
                )
                shoeboxes_sel.extend(sb)

    rlist = build_reflection_table_from_gpu_results(
        panel_t=panel_sel,
        bbox_t=bbox_sel,
        xyzobs_px_val_t=xyz_val_sel,
        xyzobs_px_var_t=xyz_var_sel,
        intensity_val_t=int_val_sel,
        intensity_var_t=int_var_sel,
        shoeboxes=shoeboxes_sel,
        n_signal_t=n_sig_sel,
        include_id_flags=True,
    )

    return rlist, hot_pixels


class ExtractSpotsGPU:
    """
    GPU-accelerated replacement for ExtractSpots in dials/algorithms/spot_finding/finder.py.
    """

    def __init__(
        self,
        threshold_function=None,
        mask=None,
        region_of_interest=None,
        max_strong_pixel_fraction: float = 0.1,
        compute_mean_background: bool = False,
        mp_method=None,
        mp_nproc: int = 1,
        mp_njobs: int = 1,
        mp_chunksize: int = 1,
        min_spot_size: int = 1,
        max_spot_size: int = 20,
        filter_spots=None,
        no_shoeboxes_2d: bool = False,
        min_chunksize: int = 50,
        write_hot_pixel_mask: bool = False,
        force_2d: bool = False,
        output_shoeboxes: bool = True,
        device: torch.device | None = None,
    ):
        self.threshold_function = threshold_function
        self.mask = mask
        self.mp_method = mp_method
        self.mp_chunksize = mp_chunksize
        self.mp_nproc = mp_nproc
        self.mp_njobs = mp_njobs
        self.max_strong_pixel_fraction = max_strong_pixel_fraction
        self.compute_mean_background = compute_mean_background
        self.region_of_interest = region_of_interest
        self.min_spot_size = min_spot_size
        self.max_spot_size = max_spot_size
        self.filter_spots = filter_spots
        self.no_shoeboxes_2d = no_shoeboxes_2d
        self.min_chunksize = min_chunksize
        self.write_hot_pixel_mask = write_hot_pixel_mask
        self.force_2d = force_2d
        self.output_shoeboxes = output_shoeboxes
        self.device = device or get_default_device()

    def __call__(self, imageset):
        if self.no_shoeboxes_2d:
            return self._find_spots_2d_no_shoeboxes(imageset)
        return self._find_spots(imageset)

    def _find_spots(self, imageset):
        function = ExtractPixelsFromImageGPU(
            imageset=imageset,
            threshold_function=self.threshold_function,
            mask=self.mask,
            max_strong_pixel_fraction=self.max_strong_pixel_fraction,
            compute_mean_background=self.compute_mean_background,
            region_of_interest=self.region_of_interest,
            device=self.device,
        )

        indices = list(range(len(imageset)))
        num_panels = len(imageset.get_detector())
        pixel_labeller = [
            PixelListLabellerGPU(device=self.device) for _ in range(num_panels)
        ]

        mp_nproc = self.mp_nproc
        if mp_nproc is libtbx.Auto:
            mp_nproc = CPU_COUNT

        logger.info("Extracting strong pixels from images on GPU")
        vol_t = None
        is_hdf5 = False
        try:
            p0 = imageset.get_path(0)
            if p0 and p0.endswith((".nxs", ".h5", ".hdf5")):
                is_hdf5 = True
        except Exception:
            pass

        if (
            function._can_use_fast_gpu_raw
            and num_panels == 1
            and self.device.type == "cuda"
            and len(indices) <= 64
        ):
            vol_t = _try_load_hdf5_raw_volume_gpu(imageset, self.device)

        if vol_t is not None and hasattr(
            self.threshold_function, "compute_threshold_batch_tensor"
        ):
            self._run_batch_volume_gpu(
                imageset=imageset,
                function=function,
                vol_t=vol_t,
                pixel_labeller=pixel_labeller,
            )
        elif vol_t is not None:
            for task in indices:
                res, _ = function.process_raw_tensors(
                    task, [vol_t[task]], emit_log=True
                )
                for plabeller, plist in zip(pixel_labeller, res):
                    plabeller.add(plist)
                res.clear()
        elif (
            function._can_use_fast_gpu_raw
            and num_panels == 1
            and self.device.type == "cuda"
            and hasattr(self.threshold_function, "compute_threshold_batch_tensor")
            and len(indices) > 1
        ):
            self._run_pipelined_batch_volume_gpu(
                imageset=imageset,
                function=function,
                indices=indices,
                pixel_labeller=pixel_labeller,
                is_hdf5=is_hdf5,
            )
        elif (
            function._can_use_fast_gpu_raw
            and not is_hdf5
            and len(indices) > 1
            and isinstance(mp_nproc, int)
            and mp_nproc > 1
        ):
            self._run_parallel_raw_prefetch(
                imageset=imageset,
                function=function,
                indices=indices,
                pixel_labeller=pixel_labeller,
                nproc=min(mp_nproc, len(indices) - 1, 16),
            )
        else:
            for task in indices:
                result = function(task)
                assert len(pixel_labeller) == len(result), "Inconsistent size"
                for plabeller, plist in zip(pixel_labeller, result):
                    plabeller.add(plist)
                result.clear()

        return pixel_list_to_reflection_table_gpu(
            imageset,
            pixel_labeller,
            filter_spots=self.filter_spots,
            min_spot_size=self.min_spot_size,
            max_spot_size=self.max_spot_size,
            write_hot_pixel_mask=self.write_hot_pixel_mask,
            include_shoeboxes=self.output_shoeboxes,
            include_n_signal=not self.output_shoeboxes,
            force_2d=self.force_2d,
            device=self.device,
        )

    def _run_pipelined_batch_volume_gpu(
        self,
        imageset,
        function: ExtractPixelsFromImageGPU,
        indices: list[int],
        pixel_labeller: list[PixelListLabellerGPU],
        is_hdf5: bool = False,
    ):
        """
        3-stage overlapped CPU decompression + PCIe H2D DMA + GPU batch thresholding pipeline
        for single-panel sweeps (both multi-file .cbf.gz/.cbf and >64-frame HDF5 .nxs).
        """
        H, W = function._base_static_mask_tensors[0].shape
        n_elems = H * W
        if len(indices) > 64:
            _HDF5_GPU_VOLUME_CACHE.clear()
            _HDF5_GPU_MASK_CACHE.clear()

        pin_key = (32, H, W)
        pinned_bufs = _PIPELINE_PINNED_CACHE.get(pin_key)
        if pinned_bufs is None:
            _PIPELINE_PINNED_CACHE.clear()
            pinned_bufs = [
                torch.empty((32, H, W), dtype=torch.int32, pin_memory=True),
                torch.empty((32, H, W), dtype=torch.int32, pin_memory=True),
            ]
            _PIPELINE_PINNED_CACHE[pin_key] = pinned_bufs

        gpu_bufs = [
            torch.empty((32, H, W), dtype=torch.int32, device=self.device),
            torch.empty((32, H, W), dtype=torch.int32, device=self.device),
        ]
        copy_stream = torch.cuda.Stream(device=self.device)
        h2d_events = [torch.cuda.Event(), torch.cuda.Event()]
        compute_done_events = [torch.cuda.Event(), torch.cuda.Event()]

        cbf_fn = _get_cbf_batch_reader() if not is_hdf5 else None
        n_threads = min(20, max(2, CPU_COUNT))
        # Device staging for 16-bit frames: 8 frames, shared by both slots (all its uses are
        # ordered on copy_stream), allocated on first use.
        stage_frames = 8
        raw16_gpu = [None]

        def _stage_raw_hdf5(slot: int, s: int, B: int) -> bool:
            """Decompress without conversion, copy at the file's bit depth, and widen to
            int32 and flag invalid pixels on the GPU (cheap there, and on the CPU these
            passes cost several times the decompression on large detectors)."""
            flat = pinned_bufs[slot].view(-1)
            raw = try_read_hdf5_raw_frames(imageset, s, B, flat)
            if raw is None:
                return False
            dtype, top = raw
            n = B * n_elems
            dst = gpu_bufs[slot][:B]
            with torch.cuda.stream(copy_stream):
                if dtype.itemsize == 2:
                    if raw16_gpu[0] is None:
                        raw16_gpu[0] = torch.empty(
                            (stage_frames, n_elems), dtype=torch.int16, device=self.device
                        )
                    host16 = flat.view(torch.int16)[:n].view(B, n_elems)
                    dst_flat = dst.view(B, n_elems)
                    for b0 in range(0, B, stage_frames):
                        nb = min(stage_frames, B - b0)
                        g16 = raw16_gpu[0][:nb]
                        g16.copy_(host16[b0 : b0 + nb], non_blocking=True)
                        for k in range(nb):
                            f = dst_flat[b0 + k]
                            f.copy_(g16[k])  # sign-extends
                            if dtype.kind != "u":
                                map_invalid_pixels_gpu(f, top)
                            elif top == 2**16:
                                # After sign extension 65535 and 65534 are already -1 and
                                # -2; only the other values >= 32768 need restoring.
                                torch.where(f < -2, f + 65536, f, out=f)
                            else:
                                f.bitwise_and_(0xFFFF)
                                map_invalid_pixels_gpu(f, top)
                else:
                    # int32, or uint32 reinterpreted as int32 like dxtbx's astype
                    dst.copy_(pinned_bufs[slot][:B], non_blocking=True)
                    map_invalid_pixels_gpu(dst, top)
                h2d_events[slot].record(copy_stream)
            return True

        def _load_and_stage(slot: int, s: int, B: int):
            compute_done_events[slot].synchronize()
            loaded = False
            if is_hdf5 and _stage_raw_hdf5(slot, s, B):
                return B
            if is_hdf5:
                res_t = _try_load_hdf5_raw_volume_gpu(
                    imageset,
                    self.device,
                    slice_start=s,
                    slice_len=B,
                    pinned_out=pinned_bufs[slot][:B],
                )
                if res_t is not None:
                    loaded = True
            elif cbf_fn is not None:
                try:
                    paths_b = [
                        imageset.get_path(i).encode("utf-8") for i in range(s, s + B)
                    ]
                    c_paths = (ctypes.c_char_p * B)(*paths_b)
                    rc = cbf_fn(
                        c_paths,
                        B,
                        n_elems,
                        pinned_bufs[slot].numpy().ctypes.data,
                        n_threads,
                    )
                    if rc == 0:
                        loaded = True
                except Exception:
                    loaded = False

            if not loaded:
                frames_np = [
                    flumpy.to_numpy(imageset.get_raw_data(i)[0])
                    for i in range(s, s + B)
                ]
                chunk_np = np.stack(frames_np, axis=0)
                if chunk_np.dtype == np.uint32:
                    chunk_np = chunk_np.view(np.int32)
                elif chunk_np.dtype != np.int32:
                    chunk_np = chunk_np.astype(np.int32)
                pinned_bufs[slot][:B].copy_(torch.from_numpy(chunk_np))

            with torch.cuda.stream(copy_stream):
                gpu_bufs[slot][:B].copy_(pinned_bufs[slot][:B], non_blocking=True)
                h2d_events[slot].record(copy_stream)
            return B

        slabs = [(s, min(32, len(indices) - s)) for s in range(0, len(indices), 32)]
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_load_and_stage, 0, slabs[0][0], slabs[0][1])
            for idx, (s, B) in enumerate(slabs):
                slot = idx % 2
                fut.result()
                if idx + 1 < len(slabs):
                    ns, nB = slabs[idx + 1]
                    fut = pool.submit(_load_and_stage, (idx + 1) % 2, ns, nB)
                torch.cuda.current_stream(self.device).wait_event(h2d_events[slot])
                self._run_batch_volume_gpu(
                    imageset=imageset,
                    function=function,
                    vol_t=gpu_bufs[slot][:B],
                    pixel_labeller=pixel_labeller,
                    slab_start=s,
                )
                compute_done_events[slot].record(
                    torch.cuda.current_stream(self.device)
                )

    def _run_batch_volume_gpu(
        self,
        imageset,
        function: ExtractPixelsFromImageGPU,
        vol_t: torch.Tensor,
        pixel_labeller: list[PixelListLabellerGPU],
        slab_start: int = 0,
    ):
        B, H, W = vol_t.shape
        first_frame = imageset.get_array_range()[0] + slab_start
        tr0, tr1 = function._trusted_ranges[0]

        use_mask_cache = len(imageset.indices()) <= 64 and slab_start == 0
        dev_key = (self.device.type, self.device.index if self.device.index is not None else 0)
        z0 = int(imageset.indices()[slab_start])
        m_key = (function._path_key, z0, B, float(tr0), float(tr1), dev_key)
        base_mask_vol = (
            _HDF5_GPU_MASK_CACHE.get(m_key)
            if (use_mask_cache and function._path_key)
            else None
        )
        if base_mask_vol is not None:
            _HDF5_GPU_MASK_CACHE.move_to_end(m_key)
        else:
            base_sm = function._base_static_mask_tensors[0]
            base_mask_vol = base_sm.unsqueeze(0) & (vol_t >= tr0) & (vol_t <= tr1)
            if use_mask_cache and function._path_key and self.device.type == "cuda":
                if len(_HDF5_GPU_MASK_CACHE) >= 8:
                    _HDF5_GPU_MASK_CACHE.popitem(last=False)
                _HDF5_GPU_MASK_CACHE[m_key] = base_mask_vol

        if function._static_mask_tensors is not None:
            mask_vol = base_mask_vol & function._static_mask_tensors[0].unsqueeze(0)
        else:
            mask_vol = base_mask_vol

        rejected_flags = [
            imageset.is_marked_for_rejection(slab_start + i) for i in range(B)
        ]
        if function._masker is not None:
            cloned_mask = False
            for i in range(B):
                if not rejected_flags[i]:
                    angle = function._scan.get_angle_from_image_index(
                        slab_start + i + function._image_range_start, deg=True
                    )
                    ext = function._masker.project_extrema(function._detector, angle)
                    if any(len(e) > 0 for e in ext):
                        dyn_mask_tuple = function._masker.get_mask(
                            function._detector, angle
                        )
                        if len(dyn_mask_tuple[0]) > 0:
                            if not cloned_mask and mask_vol is base_mask_vol:
                                mask_vol = base_mask_vol.clone()
                                cloned_mask = True
                            dm_t = flex_to_tensor(
                                dyn_mask_tuple[0],
                                device=self.device,
                                dtype=torch.bool,
                            )
                            mask_vol[i] &= dm_t

        if function._pedestal_tensors is None and function._gain_tensors is None:
            work_vol = vol_t
        else:
            work_vol = vol_t.to(torch.float32)
            if function._pedestal_tensors is not None:
                p_t = function._pedestal_tensors[0]
                if p_t.ndim == 0:
                    work_vol = work_vol - float(p_t.item())
                else:
                    work_vol = work_vol - p_t.to(torch.float32).unsqueeze(0)
            if function._gain_tensors is not None:
                g_t = function._gain_tensors[0]
                if g_t.ndim == 0:
                    work_vol = work_vol * float(1.0 / g_t.item())
                else:
                    work_vol = work_vol / g_t.to(torch.float32).unsqueeze(0)

        if self.region_of_interest is not None:
            x0, x1, y0, y1 = self.region_of_interest
            assert x0 < x1 and y0 < y1, "Invalid region of interest"
            assert x0 >= 0 and y0 >= 0 and x1 <= W and y1 <= H, "ROI out of bounds"
            thresh_roi = self.threshold_function.compute_threshold_batch_tensor(
                work_vol[:, y0:y1, x0:x1].contiguous(),
                mask_vol[:, y0:y1, x0:x1].contiguous(),
                imageset=imageset,
                i_panel=0,
                region_of_interest=self.region_of_interest,
            )
            out_vol = torch.zeros_like(mask_vol)
            out_vol[:, y0:y1, x0:x1] = thresh_roi
        else:
            out_vol = self.threshold_function.compute_threshold_batch_tensor(
                work_vol,
                mask_vol,
                imageset=imageset,
                i_panel=0,
                region_of_interest=None,
            )

        if any(rejected_flags):
            for i, rej in enumerate(rejected_flags):
                if rej:
                    out_vol[i].zero_()

        # Count strong pixels per frame before torch.nonzero, so max_strong is
        # checked before a huge nonzero. Sum the bool volume as uint8 into int32
        # (a plain .sum() first casts the whole volume to int64), and copy into
        # pinned memory waiting only on the compute stream: a blocking .cpu() to
        # pageable memory stalls behind the next slab's H2D copy on copy_stream,
        # which serialises the load/compute pipeline (~1.3 s on 720 Pilatus 6M frames).
        counts_dev = out_vol.view(torch.uint8).view(B, -1).sum(1, dtype=torch.int32)
        counts_host = torch.empty(B, dtype=torch.int32, pin_memory=True)
        counts_host.copy_(counts_dev, non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()
        frame_counts = counts_host.tolist()

        max_strong = (
            int(math.ceil(self.max_strong_pixel_fraction * (H * W)))
            if self.max_strong_pixel_fraction < 1
            else None
        )

        for i, n_strong in enumerate(frame_counts):
            if max_strong is not None and n_strong > max_strong:
                raise RuntimeError(
                    f"\n  The number of strong pixels found ({n_strong}) is greater than the\n"
                    f"  maximum allowed ({max_strong}). Try changing spot finding parameters\n"
                )
            frame_log = first_frame + i + 1
            logger.info(f"Found {n_strong} strong pixels on image {frame_log}")
            if self.compute_mean_background and not rejected_flags[i]:
                bg_sel = mask_vol[i] & (~out_vol[i])
                bg_vals = work_vol[i][bg_sel]
                avg_bg = (
                    float(bg_vals.to(torch.float64).mean().item())
                    if bg_vals.numel() > 0
                    else 0.0
                )
                logger.info(f"Average background on image {frame_log} is {avg_bg}")

        if sum(frame_counts) > 0:
            zyx = torch.nonzero(out_vol, as_tuple=False)
            if function._pedestal_tensors is None and function._gain_tensors is None:
                vals = work_vol[out_vol].to(torch.float64)
            else:
                vals = vol_t[out_vol].to(torch.float64)
                if function._pedestal_tensors is not None:
                    p_t = function._pedestal_tensors[0]
                    vals = vals - (p_t if p_t.ndim == 0 else p_t[zyx[:, 1], zyx[:, 2]])
                if function._gain_tensors is not None:
                    g_t = function._gain_tensors[0]
                    vals = vals / (g_t if g_t.ndim == 0 else g_t[zyx[:, 1], zyx[:, 2]])
            if first_frame != 0:
                zyx[:, 0] += first_frame
            coords = zyx
        else:
            coords = torch.empty((0, 3), dtype=torch.int64, device=self.device)
            vals = torch.empty((0,), dtype=torch.float64, device=self.device)

        pixel_labeller[0].add_volume(
            first_frame=first_frame,
            last_frame=first_frame + B,
            size=(H, W),
            coords=coords,
            values=vals,
        )

    def _run_parallel_raw_prefetch(
        self,
        imageset,
        function: ExtractPixelsFromImageGPU,
        indices: list[int],
        pixel_labeller: list[PixelListLabellerGPU],
        nproc: int,
    ):
        # Process first frame on main process (handles global_threshold=Auto and inspects raw dtype/shapes)
        first_idx = indices[0]
        first_raw = imageset.get_raw_data(first_idx)
        panel_specs = []
        frame_nbytes = 0
        for p in range(len(first_raw)):
            arr = flumpy.to_numpy(first_raw[p])
            panel_specs.append((arr.shape, arr.dtype, frame_nbytes, arr.nbytes))
            frame_nbytes += arr.nbytes

        if imageset.is_marked_for_rejection(first_idx):
            first_tensors = None
        else:
            first_tensors = [
                flex_to_tensor(first_raw[p], device=self.device)
                for p in range(len(first_raw))
            ]
        first_res, first_logs = function.process_raw_tensors(
            first_idx, first_tensors, emit_log=False
        )
        del first_raw, first_tensors

        rem_indices = indices[1:]
        num_slots = min(len(rem_indices), max(8, nproc * 2))
        shm = shared_memory.SharedMemory(create=True, size=num_slots * frame_nbytes)
        ctx = multiprocessing.get_context("fork")
        task_q = ctx.Queue()
        free_q = ctx.Queue()
        done_q = ctx.Queue()

        for s in range(num_slots):
            free_q.put(s)
        for idx in rem_indices:
            task_q.put(idx)
        for _ in range(nproc):
            task_q.put(None)

        shm_name = shm.name

        def _worker():
            local_shm = shared_memory.SharedMemory(name=shm_name)
            buf = local_shm.buf
            try:
                while True:
                    idx = task_q.get()
                    if idx is None:
                        break
                    try:
                        if imageset.is_marked_for_rejection(idx):
                            done_q.put((idx, -1, None))
                            continue
                        raw = imageset.get_raw_data(idx)
                        np_panels = [flumpy.to_numpy(raw[p]) for p in range(len(raw))]
                        slot = free_q.get()
                        base = slot * frame_nbytes
                        for p, (shape, dtype, offset, nbytes) in enumerate(panel_specs):
                            dst = np.ndarray(
                                shape, dtype=dtype, buffer=buf, offset=base + offset
                            )
                            np.copyto(dst, np_panels[p])
                        done_q.put((idx, slot, None))
                    except Exception as e:
                        done_q.put((idx, -2, str(e)))
            finally:
                local_shm.close()

        workers = [ctx.Process(target=_worker) for _ in range(nproc)]
        results_by_idx: dict[int, tuple[list[GPUFramePixelList], list[str]]] = {
            first_idx: (first_res, first_logs)
        }
        try:
            for w in workers:
                w.start()

            parent_buf = shm.buf
            error_msg = None
            for _ in range(len(rem_indices)):
                idx, slot, err = done_q.get()
                if slot == -2 or err is not None:
                    error_msg = err or f"Worker failed on image index {idx}"
                    continue
                if slot == -1:
                    raw_tensors = None
                else:
                    base = slot * frame_nbytes
                    raw_tensors = []
                    for shape, dtype, offset, _ in panel_specs:
                        arr = np.ndarray(
                            shape, dtype=dtype, buffer=parent_buf, offset=base + offset
                        )
                        raw_tensors.append(torch.from_numpy(arr).to(self.device))
                    free_q.put(slot)

                if error_msg is None:
                    try:
                        res, logs = function.process_raw_tensors(
                            idx, raw_tensors, emit_log=False
                        )
                        results_by_idx[idx] = (res, logs)
                    except Exception as e:
                        error_msg = str(e)

            if error_msg is not None:
                raise RuntimeError(error_msg)
        finally:
            for w in workers:
                w.join(timeout=2.0)
                if w.is_alive():
                    w.terminate()
            shm.close()
            shm.unlink()

        for idx in indices:
            res, logs = results_by_idx[idx]
            for msg in logs:
                logger.info(msg)
            assert len(pixel_labeller) == len(res), "Inconsistent size"
            for plabeller, plist in zip(pixel_labeller, res):
                plabeller.add(plist)
            res.clear()

    def _find_spots_2d_no_shoeboxes(self, imageset):
        function = ExtractPixelsFromImage2DNoShoeboxesGPU(
            imageset=imageset,
            threshold_function=self.threshold_function,
            mask=self.mask,
            max_strong_pixel_fraction=self.max_strong_pixel_fraction,
            compute_mean_background=self.compute_mean_background,
            region_of_interest=self.region_of_interest,
            min_spot_size=self.min_spot_size,
            max_spot_size=self.max_spot_size,
            filter_spots=self.filter_spots,
            device=self.device,
        )

        indices = list(range(len(imageset)))
        reflections = flex.reflection_table()

        logger.info("Extracting strong spots from images on GPU")
        for task in indices:
            reflections.extend(function(task)[0])

        return reflections, None


class _SpotFinderGPUMixin:
    """
    Overrides _find_spots_in_imageset on SpotFinder/TOFSpotFinder/LaueSpotFinder
    to use ExtractSpotsGPU on the GPU.
    """

    def find_spots(self, experiments: ExperimentList) -> flex.reflection_table:
        imagesets = []
        for experiment in experiments:
            if experiment.imageset not in imagesets:
                imagesets.append(experiment.imageset)

        reflections = flex.reflection_table()
        for j, imageset in enumerate(imagesets):
            logger.info(
                "-" * 80 + f"\nFinding strong spots in imageset {j}\n" + "-" * 80
            )
            table, hot_mask = self._find_spots_in_imageset(imageset)

            table["id"] = flex.int(table.nrows(), -1)
            for i, experiment in enumerate(experiments):
                if experiment.imageset is not imageset:
                    continue
                if not self.is_stills and experiment.scan:
                    z0, z1 = experiment.scan.get_array_range()
                    z = table["xyzobs.px.value"].parts()[2]
                    table["id"].set_selected((z > z0) & (z < z1), i)
                    if experiment.identifier:
                        table.experiment_identifiers()[i] = experiment.identifier
                else:
                    table["id"] = flex.int(table.nrows(), j)
                    if experiment.identifier:
                        table.experiment_identifiers()[j] = experiment.identifier
            missed = table["id"] == -1
            assert missed.count(True) == 0, (
                f"Failed to remap {missed.count(True)} experiment IDs"
            )

            reflections.extend(table)
            if self.write_hot_mask:
                if not imageset.external_lookup.mask.data.empty():
                    for m1, m2 in zip(hot_mask, imageset.external_lookup.mask.data):
                        m1 &= m2.data()
                    imageset.external_lookup.mask.data = ImageBool(hot_mask)
                else:
                    imageset.external_lookup.mask.data = ImageBool(hot_mask)
                imageset.external_lookup.mask.filename = "%s_%d.pickle" % (
                    self.hot_mask_prefix,
                    i,
                )
                with open(imageset.external_lookup.mask.filename, "wb") as outfile:
                    pickle.dump(hot_mask, outfile, protocol=pickle.HIGHEST_PROTOCOL)

        reflections.set_flags(
            flex.size_t_range(len(reflections)), reflections.flags.strong
        )

        if "shoebox" in reflections:
            reflections.is_overloaded(experiments)

        reflections = self._post_process(reflections)
        return reflections

    def _find_spots_in_imageset(self, imageset):
        skip_cpu_mask_gen = False
        if (
            isinstance(self.mask_generator, functools.partial)
            and self.mask_generator.func is dials.util.masking.generate_mask
        ):
            fp = self.mask_generator.keywords.get("params")
            if (
                fp is not None
                and getattr(fp, "border", 0) == 0
                and not getattr(fp, "untrusted", None)
                and getattr(fp, "d_min", None) is None
                and getattr(fp, "d_max", None) is None
                and not getattr(fp, "resolution_range", None)
                and not getattr(getattr(fp, "ice_rings", None), "filter", False)
            ):
                skip_cpu_mask_gen = True

        if skip_cpu_mask_gen:
            mask = self.mask
        else:
            mask = self.mask_generator(imageset)
            if self.mask is not None:
                mask = tuple(m1 & m2 for m1, m2 in zip(mask, self.mask))

        extract_spots = ExtractSpotsGPU(
            threshold_function=self.threshold_function,
            mask=mask,
            region_of_interest=self.region_of_interest,
            max_strong_pixel_fraction=self.max_strong_pixel_fraction,
            compute_mean_background=self.compute_mean_background,
            mp_method=self.mp_method,
            mp_nproc=self.mp_nproc,
            mp_njobs=self.mp_njobs,
            mp_chunksize=self.mp_chunksize,
            min_spot_size=self.min_spot_size,
            max_spot_size=self.max_spot_size,
            filter_spots=self.filter_spots,
            no_shoeboxes_2d=self.no_shoeboxes_2d,
            min_chunksize=self.min_chunksize,
            write_hot_pixel_mask=self.write_hot_mask,
            force_2d=getattr(self, "force_2d", False),
            output_shoeboxes=getattr(self, "output_shoeboxes", True),
            device=getattr(self, "device", None),
        )

        if isinstance(imageset, ImageSequence):
            max_scan_range = imageset.get_array_range()
        else:
            max_scan_range = (0, len(imageset))

        if not self.scan_range or self.scan_range[0] is None:
            scan_range = [(max_scan_range[0] + 1, max_scan_range[1])]
        else:
            scan_range = self.scan_range

        hot_pixels = tuple(flex.size_t() for _ in range(len(imageset.get_detector())))
        reflections = flex.reflection_table()
        for j0, j1 in scan_range:
            if j1 < j0:
                raise Sorry("Scan range must be in ascending order")
            elif j0 < max_scan_range[0] or j1 > max_scan_range[1]:
                raise Sorry(
                    f"Scan range must be within image range {max_scan_range[0] + 1}..{max_scan_range[1]}"
                )

            logger.info(f"\nFinding spots in image {j0} to {j1}...")
            j0 -= 1
            if isinstance(imageset, ImageSequence):
                j0 -= imageset.get_array_range()[0]
                j1 -= imageset.get_array_range()[0]
            if len(imageset) == 1 or (j0 == 0 and j1 == len(imageset)):
                r, h = extract_spots(imageset)
            else:
                r, h = extract_spots(imageset[j0:j1])
            reflections.extend(r)
            if h is not None:
                for h1, h2 in zip(hot_pixels, h):
                    h1.extend(h2)

        hot_mask = self._create_hot_mask(imageset, hot_pixels)
        return reflections, hot_mask


class SpotFinderGPU(_SpotFinderGPUMixin, SpotFinder):
    def __init__(
        self,
        *args,
        force_2d: bool = False,
        output_shoeboxes: bool = True,
        device: torch.device | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.force_2d = force_2d
        self.output_shoeboxes = output_shoeboxes
        self.device = device or get_default_device()


class TOFSpotFinderGPU(_SpotFinderGPUMixin, TOFSpotFinder):
    def __init__(self, *args, force_2d: bool = False, device: torch.device | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_2d = force_2d
        self.device = device or get_default_device()


class LaueSpotFinderGPU(_SpotFinderGPUMixin, LaueSpotFinder):
    def __init__(self, *args, force_2d: bool = False, device: torch.device | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_2d = force_2d
        self.device = device or get_default_device()


class SpotFinderFactoryGPU(SpotFinderFactory):
    """
    Factory class to create GPU-accelerated spot finders.
    """

    @staticmethod
    def from_parameters(params=None, experiments=None, is_stills=False, device: torch.device | None = None):
        if params is None:
            params = phil_scope.fetch(source=parse("")).extract()

        output_shoeboxes = getattr(getattr(params, "output", None), "shoeboxes", True)
        if params.spotfinder.force_2d and output_shoeboxes is False:
            no_shoeboxes_2d = True
        elif experiments is not None and output_shoeboxes is False:
            no_shoeboxes_2d = False
            all_stills = True
            for experiment in experiments:
                if isinstance(experiment.imageset, ImageSequence):
                    all_stills = False
                    break
            if all_stills:
                no_shoeboxes_2d = True
        else:
            no_shoeboxes_2d = False

        mask = SpotFinderFactory.load_image(params.spotfinder.lookup.mask)
        params.spotfinder.lookup.mask = mask

        threshold_function = SpotFinderFactoryGPU.configure_threshold(params, device=device)

        mask_generator = functools.partial(
            dials.util.masking.generate_mask, params=params.spotfinder.filter
        )

        if params.spotfinder.mp.method == "none":
            params.spotfinder.mp.method = None

        contains_tof_experiments = False
        if experiments is not None:
            for experiment in experiments:
                if experiment.scan is None:
                    continue
                if experiment.scan.has_property("time_of_flight"):
                    contains_tof_experiments = True
                elif contains_tof_experiments:
                    raise RuntimeError("All experiment scans must contain time_of_flight")

        if experiments is not None and experiments.all_tof():
            if params.spotfinder.filter.max_separation < 6:
                logger.info("Increasing max allowed peak-centroid distance to 6px")
                params.spotfinder.filter.max_separation = 6
            filter_spots = SpotFinderFactoryGPU.configure_filter(params)

            return TOFSpotFinderGPU(
                experiments=experiments,
                threshold_function=threshold_function,
                mask=params.spotfinder.lookup.mask,
                filter_spots=filter_spots,
                scan_range=params.spotfinder.scan_range,
                write_hot_mask=params.spotfinder.write_hot_mask,
                hot_mask_prefix=params.spotfinder.hot_mask_prefix,
                mp_method=params.spotfinder.mp.method,
                mp_nproc=params.spotfinder.mp.nproc,
                mp_njobs=params.spotfinder.mp.njobs,
                mp_chunksize=params.spotfinder.mp.chunksize,
                max_strong_pixel_fraction=params.spotfinder.filter.max_strong_pixel_fraction,
                compute_mean_background=params.spotfinder.compute_mean_background,
                region_of_interest=params.spotfinder.region_of_interest,
                mask_generator=mask_generator,
                min_spot_size=params.spotfinder.filter.min_spot_size,
                max_spot_size=params.spotfinder.filter.max_spot_size,
                min_chunksize=params.spotfinder.mp.min_chunksize,
                rs_proximity_threshold_multiplier=params.spotfinder.tof.rs_proximity_threshold_multiplier,
                force_2d=params.spotfinder.force_2d,
                device=device,
            )

        if experiments is not None and experiments.all_laue():
            filter_spots = SpotFinderFactoryGPU.configure_filter(params)

            return LaueSpotFinderGPU(
                experiments=experiments,
                threshold_function=threshold_function,
                mask=params.spotfinder.lookup.mask,
                filter_spots=filter_spots,
                scan_range=params.spotfinder.scan_range,
                write_hot_mask=params.spotfinder.write_hot_mask,
                hot_mask_prefix=params.spotfinder.hot_mask_prefix,
                mp_method=params.spotfinder.mp.method,
                mp_nproc=params.spotfinder.mp.nproc,
                mp_njobs=params.spotfinder.mp.njobs,
                mp_chunksize=params.spotfinder.mp.chunksize,
                max_strong_pixel_fraction=params.spotfinder.filter.max_strong_pixel_fraction,
                compute_mean_background=params.spotfinder.compute_mean_background,
                region_of_interest=params.spotfinder.region_of_interest,
                mask_generator=mask_generator,
                min_spot_size=params.spotfinder.filter.min_spot_size,
                max_spot_size=params.spotfinder.filter.max_spot_size,
                min_chunksize=params.spotfinder.mp.min_chunksize,
                initial_wavelength=params.spotfinder.laue.initial_wavelength,
                force_2d=params.spotfinder.force_2d,
                device=device,
            )

        filter_spots = SpotFinderFactoryGPU.configure_filter(params)

        return SpotFinderGPU(
            threshold_function=threshold_function,
            mask=params.spotfinder.lookup.mask,
            filter_spots=filter_spots,
            scan_range=params.spotfinder.scan_range,
            write_hot_mask=params.spotfinder.write_hot_mask,
            hot_mask_prefix=params.spotfinder.hot_mask_prefix,
            mp_method=params.spotfinder.mp.method,
            mp_nproc=params.spotfinder.mp.nproc,
            mp_njobs=params.spotfinder.mp.njobs,
            mp_chunksize=params.spotfinder.mp.chunksize,
            max_strong_pixel_fraction=params.spotfinder.filter.max_strong_pixel_fraction,
            compute_mean_background=params.spotfinder.compute_mean_background,
            region_of_interest=params.spotfinder.region_of_interest,
            mask_generator=mask_generator,
            min_spot_size=params.spotfinder.filter.min_spot_size,
            max_spot_size=params.spotfinder.filter.max_spot_size,
            no_shoeboxes_2d=no_shoeboxes_2d,
            min_chunksize=params.spotfinder.mp.min_chunksize,
            is_stills=is_stills,
            force_2d=params.spotfinder.force_2d,
            output_shoeboxes=bool(output_shoeboxes),
            device=device,
        )

    @staticmethod
    def configure_threshold(params, device: torch.device | None = None):
        alg = params.spotfinder.threshold.algorithm
        if alg == "dispersion_extended":
            return DispersionThresholdExtGPU(params, extended=True, device=device)
        elif alg == "dispersion":
            return DispersionThresholdExtGPU(params, extended=False, device=device)
        elif alg == "radial_profile":
            return RadialProfileSpotFinderThresholdGPU(params, device=device)
        else:
            Algorithm = dials.extensions.SpotFinderThreshold.load(alg)
            return Algorithm(params)

    @staticmethod
    def configure_filter(params):
        filters = []
        if params.spotfinder.filter.max_separation is not None:
            filters.append(
                PeakCentroidDistanceFilterGPU(params.spotfinder.filter.max_separation)
            )
        if params.spotfinder.filter.background_gradient.filter:
            bg_filter_params = params.spotfinder.filter.background_gradient
            filters.append(
                BackgroundGradientFilter(
                    background_size=bg_filter_params.background_size,
                    gradient_cutoff=bg_filter_params.gradient_cutoff,
                )
            )
        if params.spotfinder.filter.spot_density.filter:
            filters.append(SpotDensityFilterGPU())

        return FilterRunnerGPU(filters)
