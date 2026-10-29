# dials_gpu

GPU-accelerated implementations of [DIALS](https://dials.github.io/) crystallographic data-processing algorithms using PyTorch (`torch`).

## Command-Line Tools

- `dials_gpu.find_spots`: Drop-in GPU replacement for `dials.find_spots` supporting all standard DIALS PHIL parameters and spot-finding algorithms (`dispersion_extended`, `dispersion`, `radial_profile`, 2D/3D connected component labelling, centroiding, intensity summation, shoebox extraction, and spot filtering).

## Usage

Activate the DIALS environment and run `dials_gpu.find_spots` with the exact same arguments as `dials.find_spots`:

```bash
source /home/davidm/diffraction/dials/dials
dials_gpu.find_spots imported.expt output.reflections=strong.refl
# or directly on image files:
dials_gpu.find_spots test_data/C2sum_1_0001.cbf.gz
```

---

## Performance Optimization Tips

### 1. In-Process Python API for RL / ML Training Loops (Fastest: ~50 ms per 25-image `.nxs` file)
Spawning a new CLI subprocess (`dials_gpu.find_spots ...`) on every RL step incurs ~1.5–2.0 s of Python, `dxtbx`, and CUDA context initialization overhead. For Reinforcement Learning (RL) or hyperparameter optimization, call [`from_observations_gpu`](src/dials_gpu/command_line/find_spots.py) directly inside your long-lived Python training process:

```python
from dxtbx.model.experiment_list import ExperimentList, ExperimentListFactory
from iotbx.phil import parse
from dials.command_line.find_spots import phil_scope
from dials_gpu.command_line.find_spots import from_observations_gpu

# 1. Load ExperimentList once per dataset (from imported.expt or .nxs) and keep in memory
expts = ExperimentList.from_file("imported.expt")
# Or directly from a NeXus file:
# expts = ExperimentListFactory.from_filenames(["301_NiD_00001_w00017.nxs"])

# 2. Pre-fetch the base PHIL scope once; clone/update parameters on each RL step
base_phil = phil_scope.fetch(source=parse("spotfinder.mp.nproc=1"))

def rl_step(sigma_b: float, sigma_s: float, kernel_xy: int, min_spot_size: int):
    params = base_phil.extract()
    params.spotfinder.threshold.dispersion.sigma_background = sigma_b
    params.spotfinder.threshold.dispersion.sigma_strong = sigma_s
    params.spotfinder.threshold.dispersion.kernel_size = (kernel_xy, kernel_xy)
    params.spotfinder.filter.min_spot_size = min_spot_size

    # Runs in ~50 ms for 25 frames (2162x2068) when cached in GPU VRAM,
    # or ~145 ms from the host RAM compressed chunk cache.
    reflections = from_observations_gpu(expts, params)
    return reflections
```

### 2. Understanding & Ordering Dataset Sweeps for the Two-Tier HDF5 Cache
For single-panel HDF5/NeXus (`.nxs` / `.h5`) datasets (up to 64 frames per scan, such as 25-image wedges), `dials_gpu` uses a two-tier cache in [`finder.py`](src/dials_gpu/spot_finding/finder.py):
- **Tier-1 GPU VRAM Cache (`_HDF5_GPU_VOLUME_CACHE` & `_HDF5_GPU_MASK_CACHE`, default: 8 volumes, ~3.5 GB VRAM)**:
  Caches the 8 most recently used uncompressed `(B, H, W)` `int32` image volumes and base static/trusted-range masks directly on the GPU.
  - **Tip**: Group multiple RL steps/episodes on the same `.nxs` file (or cycle over a working batch of $\le 8$ `.nxs` files at a time) before rotating to the next batch of files. This achieves a 100% GPU VRAM cache hit rate (**~50 ms per 25-image wedge**, or **~490 images/s**).
- **Tier-2 Host RAM Compressed Cache (`_HDF5_COMPRESSED_CACHE`, default: 100 files, ~1.8 GB RAM)**:
  Caches the raw bitshuffle-LZ4 compressed HDF5 chunks in system RAM and decompresses all 25 frames in parallel across CPU threads directly into a pinned host memory buffer (`_PINNED_HOST_VOL_CACHE`) for DMA transfer to GPU (**~145 ms per 25-image wedge**, vs ~606 ms in `dxtbx`).
  - All 90 `.nxs` files in a typical `i19` dataset fit simultaneously inside the 100-file RAM cache (`~1.65 GB`).

### 3. Modifying `phil_scope` Parameters Without Re-Parsing Strings
- Calling `phil_scope.fetch(source=parse(...)).extract()` repeatedly inside a tight loop spends ~5–8 ms parsing PHIL strings on CPU. Instead, call `base_phil = phil_scope.fetch()` once at startup, and mutate attributes on `params = base_phil.extract()` directly (as shown above).
- **Important**: Always call `phil_scope.fetch().extract()` rather than `phil_scope.extract()` directly; `fetch()` resolves default empty scope templates such as `spotfinder.filter.untrusted`.

### 4. `spotfinder.mp.nproc` Settings: `.nxs`/`.h5` vs `.cbf`/`.cbf.gz`
- **For `.nxs` / `.h5` (HDF5 / NeXus) files**: Keep `spotfinder.mp.nproc=1` (the default). `dials_gpu` automatically uses its internal persistent thread pool (`_BSHUF_POOL`) to decompress bitshuffle-LZ4 chunks in parallel without `fork` overhead (and avoids `libhdf5` fork deadlocks).
- **For multi-file `.cbf` / `.cbf.gz` scans (e.g., 720-image Pilatus datasets)**: Set `spotfinder.mp.nproc=4` (or `8`) on the CLI or in `params`:
  ```bash
  dials_gpu.find_spots imported.expt spotfinder.mp.nproc=4
  ```
  This overlaps multi-core CPU CBF decompression in shared memory with GPU thresholding.

### 5. Skip Disk Serialization (`as_file`) During RL Training
- When using `do_spotfinding` or the CLI, writing `strong.refl` to disk serializes all C++ `Shoebox` pixel arrays. In RL training where you only need spot counts, centroids (`reflections["xyzobs.px.value"]`), intensities (`reflections["intensity.sum.value"]`), or to pass the `flex.reflection_table` in-memory to `dials.index`, use `from_observations_gpu(expts, params)` directly to avoid disk I/O altogether.

### 6. Customizing Cache Sizes for High-VRAM GPUs
If your GPU has ample free VRAM (e.g. 24–80 GB) and you want to keep more than 8 `.nxs` volumes resident in GPU memory simultaneously, you can inspect or clear the caches from [`dials_gpu.spot_finding.finder`](src/dials_gpu/spot_finding/finder.py):
```python
from dials_gpu.spot_finding.finder import (
    _HDF5_GPU_VOLUME_CACHE,
    _HDF5_GPU_MASK_CACHE,
    _HDF5_COMPRESSED_CACHE,
)
```

---

## Benchmarks (25-image Eiger CdTe `.nxs` wedge, `2162 × 2068`)

| Mode | Time per 25-image `.nxs` | Throughput | Speedup vs CPU (`nproc=1`) | Speedup vs CPU (`nproc=4`) |
| :--- | ---: | ---: | ---: | ---: |
| `dials.find_spots` (CPU, `nproc=1`) | `3,712.6 ms` | `6.7 img/s` | `1.0×` | `0.42×` |
| `dials.find_spots` (CPU, `nproc=4`) | `1,562.2 ms` | `16.0 img/s` | `2.4×` | `1.0×` |
| `dials_gpu` (Cold Read from `.nxs`) | `173.0 ms` | `144.5 img/s` | `21.5×` | `9.0×` |
| `dials_gpu` (Tier-2 Host RAM Cache Hit) | `146.5 ms` | `170.6 img/s` | `25.3×` | `10.7×` |
| **`dials_gpu` (Tier-1 GPU VRAM Cache Hit)** | **`50.8 ms`** | **`492.5 img/s`** | **`73.1×`** | **`30.8×`** |
