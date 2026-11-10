"""Copy an NXmx file, rewriting /entry/data/data as bitshuffle-LZ4, one frame per chunk.

Everything else (groups, datasets, attributes, links) is copied unchanged. Used to turn
the uncompressed output of dxtbx.any2nexus into the compressed layout that mxi reads
(mxi reads bitshuffle-LZ4, LZ4 or uncompressed images; any2nexus writes gzip or none).

Usage (in the DIALS environment, which has h5py and hdf5plugin):
    python3 benchmarks/repack_nxmx_bslz4.py raw.nxs out.nxs
"""

import sys
import time

import h5py
import hdf5plugin

src_path, dst_path = sys.argv[1], sys.argv[2]
DATA = "/entry/data/data"

with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
    for k, v in src.attrs.items():
        dst.attrs[k] = v

    def copy_group(sg, dg):
        for k, v in sg.attrs.items():
            dg.attrs[k] = v
        for name in sg:
            link = sg.get(name, getlink=True)
            if isinstance(link, h5py.SoftLink):
                dg[name] = h5py.SoftLink(link.path)
                continue
            if isinstance(link, h5py.ExternalLink):
                dg[name] = h5py.ExternalLink(link.filename, link.path)
                continue
            obj = sg[name]
            if isinstance(obj, h5py.Group):
                copy_group(obj, dg.create_group(name))
            elif obj.name == DATA:
                n, h, w = obj.shape
                out = dg.create_dataset(name, shape=obj.shape, dtype=obj.dtype, chunks=(1, h, w),
                                        **hdf5plugin.Bitshuffle(cname="lz4"))
                for k, v in obj.attrs.items():
                    out.attrs[k] = v
                t0 = time.time()
                for i in range(n):
                    out[i] = obj[i]
                    if (i + 1) % 100 == 0:
                        print(f"  {i + 1}/{n} frames, {time.time() - t0:.0f} s", flush=True)
            else:
                sg.copy(obj, dg, name=name)

    copy_group(src, dst)
print("done")
