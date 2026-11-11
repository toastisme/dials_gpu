"""Check whether dials_gpu.integrate can use its fast slabbed path for a dataset.

Usage: dials.python check_slab_path.py refined.expt refined.refl
(the refined.* files from compare_stages: <workdir>/<dataset>/dials_gpu/)
Mirrors the conditions in dials_gpu/integration/integrator.py
(_try_run_slabbed_volume_integration and IntegratorGPU.fit_profiles).
"""

import sys

import numpy as np
from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentListFactory

expts = ExperimentListFactory.from_json_file(sys.argv[1], check_format=True)
print(f"experiments: {len(expts)}" + ("  <-- >1 disables the single-pass profile path" if len(expts) > 1 else ""))
for i, e in enumerate(expts):
    iset = e.imageset
    det = iset.get_detector()
    path0 = iset.get_path(0)
    H, W = det[0].get_image_size()[::-1]
    checks = {
        "single panel": len(det) == 1,
        "no dynamic mask (goniometer shadowing)": not iset.has_dynamic_mask(),
        "no pedestal map": iset.external_lookup.pedestal.data.empty(),
        "no gain map": iset.external_lookup.gain.data.empty(),
        "format .nxs/.h5/.hdf5/.cbf/.cbf.gz": bool(path0) and path0.endswith((".nxs", ".h5", ".hdf5", ".cbf", ".cbf.gz")),
        "more than 64 frames": len(iset) > 64,
    }
    print(f"\nexperiment {i}: {len(det)} panel(s), {W}x{H}, {len(iset)} frames, {path0}")
    for name, ok in checks.items():
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    print(f"  full-sweep int32 volume would be {len(iset) * H * W * 4 / 1e9:.1f} GB (cached only if <= 24 GB)")

if len(sys.argv) > 2:
    refl = flex.reflection_table.from_file(sys.argv[2])
    print(f"\n{sys.argv[2]}: {len(refl)} reflections (bboxes are recomputed by integration; this is indicative)")
    if "bbox" in refl:
        _, _, _, _, z0, z1 = refl["bbox"].parts()
        dz = z1.as_numpy_array() - z0.as_numpy_array()
        print(f"  bbox frames: median {np.median(dz):.0f}, 99th pct {np.percentile(dz, 99):.0f}, max {dz.max()}")
        n_wide = int((dz > 64).sum())
        print(f"  [{'ok' if n_wide == 0 else 'WARN'}] reflections spanning > 64 frames: {n_wide}"
              + ("  (disables the single-pass circular-slab profile fitting)" if n_wide else ""))
