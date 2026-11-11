"""Why does dials_gpu.integrate read this HDF5 sweep slowly?

Usage: libtbx.python check_hdf5_read.py refined.expt
Performs the same reads as integration's slab path
(_read_chunk_into in dials_gpu/integration/integrator.py) and reports, step by step,
whether the fast multithreaded HDF5 loader accepts the data or falls back to dxtbx.
"""

import sys
import time

import numpy as np
import torch
from dxtbx.model.experiment_list import ExperimentListFactory

from dials_gpu.spot_finding import finder as F

expts = ExperimentListFactory.from_json_file(sys.argv[1], check_format=True)
iset = expts[0].imageset
path = iset.get_path(0)
print(f"imageset type: {type(iset).__name__}, {len(iset)} frames, path {path}")
print(f"h5py available: {F.h5py is not None}; hdf5plugin available: {F.hdf5plugin is not None}; "
      f"bitshuffle decompressor: {F._get_bshuf_decompress() is not None}")

sub = iset[0:64]
print(f"\n1. _hdf5_slice(imageset[0:64]): {F._hdf5_slice(sub, 0, None)}   (type {type(sub).__name__})")

layout = F._hdf5_frame_layout(path)
if layout is None:
    print("2. _hdf5_frame_layout: None  <-- the fast loader does not understand this file's layout")
else:
    H, W, dtype, top, segs = layout
    print(f"2. _hdf5_frame_layout: {W}x{H} {dtype}, invalid_top={top}, {len(segs)} segment(s), first {segs[0]}")
    try:
        chunks, pieces = F._hdf5_get_chunks(path, layout, 0, 64)
        if chunks is not None:
            print(f"3. _hdf5_get_chunks: {len(chunks)} bitshuffle-LZ4 chunks (multithreaded decode)")
        else:
            d = pieces[0][1]
            plist = d.id.get_create_plist()
            filters = [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]
            print(f"3. _hdf5_get_chunks: NOT one bitshuffle-LZ4 chunk per frame -> slow h5py slice reads "
                  f"(chunks {d.chunks}, filter ids {filters}; fast path needs chunks (1,{H},{W}) and filter [32008])")
    except Exception as e:
        print(f"3. _hdf5_get_chunks raised {type(e).__name__}: {e}  <-- loader returns None")

t = time.perf_counter()
vol = F._try_load_hdf5_raw_volume_gpu(iset[0:64], torch.device("cpu"))
t_fast = (time.perf_counter() - t) / 64
if vol is None:
    print(f"\n4. _try_load_hdf5_raw_volume_gpu: None  <-- integration falls back to dxtbx, frame by frame")
else:
    print(f"\n4. _try_load_hdf5_raw_volume_gpu: {tuple(vol.shape)} {vol.dtype}, {t_fast * 1e3:.1f} ms/frame (cold)")
    t = time.perf_counter()
    F._HDF5_COMPRESSED_CACHE.clear()
    F._PINNED_HOST_VOL_TAG.clear()
    vol = F._try_load_hdf5_raw_volume_gpu(iset[64:128], torch.device("cpu"))
    print(f"   next 64 frames: {(time.perf_counter() - t) / 64 * 1e3:.1f} ms/frame")

t = time.perf_counter()
for k in range(64, 72):
    arr = iset.get_raw_data(k)[0]
t_dx = (time.perf_counter() - t) / 8
print(f"5. dxtbx get_raw_data: {t_dx * 1e3:.1f} ms/frame, type {type(arr).__name__}")
if vol is not None:
    from dials.array_family import flex  # noqa: F401
    a = arr.as_numpy_array() if hasattr(arr, "as_numpy_array") else np.asarray(arr)
    same = np.array_equal(a, vol[7].numpy()) if vol.shape[0] > 7 else "n/a"
    print(f"   fast loader matches dxtbx on frame 71: {same}")
print(f"\nAt dxtbx speed, 7200 frames = {t_dx * 7200:.0f} s; integration's Read time was 251 s.")
