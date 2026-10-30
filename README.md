# dials_gpu

GPU-accelerated implementations of [DIALS](https://dials.github.io/) crystallographic data-processing algorithms using PyTorch (`torch`).

## Command-Line Tools

- `dials_gpu.find_spots`: Drop-in GPU replacement for `dials.find_spots` supporting all standard DIALS PHIL parameters and spot-finding algorithms (`dispersion_extended`, `dispersion`, `radial_profile`, 2D/3D connected component labelling, centroiding, intensity summation, shoebox extraction, and spot filtering).
- `dials_gpu.index`: Drop-in GPU replacement for `dials.index` supporting all standard DIALS PHIL parameters and indexing algorithms (`fft3d`, `fft1d`, `real_space_grid_search`, `low_res_spot_match`, `find_max_cell`, `optimise_basis_vectors`, `candidate_orientation_matrices`, `AssignIndicesGlobal` / `AssignIndicesLocal`, `non_primitive_basis`, rotation sequences, stills, and multi-lattice indexing), with all internal candidate orientation evaluation (`choose_best_orientation_matrix` / `ModelEvaluation`) and macro-cycle refinement (`Indexer.refine`) mapped to the GPU.
- `dials_gpu.refine`: Drop-in GPU replacement for `dials.refine` supporting all standard DIALS PHIL parameters and refinement algorithms (scan-static and scan-varying refinement, stills and spherical-relp stills refinement, single/multi-panel and hierarchical detectors, `mcd` / `tukey` / `sauter_poon` GPU centroid outlier rejection, `LevMar` / `GaussNewton` / `SimpleLBFGS` / `LBFGScurvs` / `SparseLevMar` engines, restraints, constraints, and `AutoReduce`).

## Usage

Activate the DIALS environment and run `dials_gpu.find_spots`, `dials_gpu.index`, and `dials_gpu.refine` with the exact same arguments as `dials.find_spots`, `dials.index`, and `dials.refine`:

```bash
source /home/davidm/diffraction/dials/dials

# 1. Spot-finding
dials_gpu.find_spots imported.expt output.reflections=strong.refl

# 2. Indexing (default fft3d, fft1d, or real_space_grid_search)
dials_gpu.index imported.expt strong.refl
dials_gpu.index imported.expt strong.refl indexing.method=fft1d
dials_gpu.index imported.expt strong.refl indexing.method=real_space_grid_search \
    known_symmetry.unit_cell=11.39,13.30,15.82,90,99.02,90 known_symmetry.space_group=P21/n

# 3. Refinement (scan-static or scan-varying)
dials_gpu.refine indexed.expt indexed.refl scan_varying=False
dials_gpu.refine indexed.expt indexed.refl scan_varying=True
```

---

## Performance Optimization Tips

### 1. In-Process Python API for RL / ML Training Loops (Spot-Finding + Indexing + Refinement in ~330 ms per `.nxs` file)
Spawning new CLI subprocesses (`dials_gpu.find_spots ...`, `dials_gpu.index ...`, `dials_gpu.refine ...`) on every RL step incurs ~1.5–2.0 s of Python, `dxtbx`, and CUDA context initialization overhead. For Reinforcement Learning (RL) or hyperparameter optimization, call [`from_observations_gpu`](src/dials_gpu/command_line/find_spots.py), [`index`](src/dials_gpu/command_line/index.py), and [`RefinerFactoryGPU`](src/dials_gpu/refinement/refiner.py) directly inside your long-lived Python training process:

```python
import copy
from dxtbx.model.experiment_list import ExperimentList
from iotbx.phil import parse
from dials.command_line.find_spots import phil_scope as spot_phil_scope
from dials.command_line.index import phil_scope as index_phil_scope
from dials.command_line.refine import working_phil as refine_phil_scope
from dials_gpu.command_line.find_spots import from_observations_gpu
from dials_gpu.command_line.index import index as index_gpu
from dials_gpu.refinement import RefinerFactoryGPU

# 1. Load ExperimentList once per dataset (from imported.expt or .nxs) and keep in memory
expts = ExperimentList.from_file("imported.expt")

# 2. Pre-fetch the base PHIL scopes once; clone/update parameters on each RL step
base_spot_phil = spot_phil_scope.fetch(source=parse("spotfinder.mp.nproc=1"))
base_index_phil = index_phil_scope.fetch(
    source=parse(
        "indexing.method=fft3d\n"
        "indexing.basis_vector_combinations.max_refine=5\n"
        "indexing.refinement_protocol.n_macro_cycles=2"
    )
)
base_refine_phil = refine_phil_scope.fetch(
    source=parse("refinement.parameterisation.scan_varying=False")
)

def rl_episode(sigma_b: float, sigma_s: float, kernel_xy: int, min_spot_size: int):
    spot_params = base_spot_phil.extract()
    spot_params.spotfinder.threshold.dispersion.sigma_background = sigma_b
    spot_params.spotfinder.threshold.dispersion.sigma_strong = sigma_s
    spot_params.spotfinder.threshold.dispersion.kernel_size = (kernel_xy, kernel_xy)
    spot_params.spotfinder.filter.min_spot_size = min_spot_size

    # Spot-finding on GPU (~50 ms warm)
    strong_refl = from_observations_gpu(expts, spot_params)

    # Indexing on GPU (~220–275 ms with RL fast refinement settings)
    idx_params = base_index_phil.extract()
    indexed_expts, indexed_refl = index_gpu(
        copy.deepcopy(expts), [strong_refl], idx_params
    )

    # Refinement on GPU (~64 ms scan-static, ~118 ms scan-varying)
    refiner = RefinerFactoryGPU.from_parameters_data_experiments(
        base_refine_phil.extract(), indexed_refl, indexed_expts
    )
    refiner.run()
    return refiner.get_experiments(), refiner.rmsds()
```

### 2. Optimizing `dials_gpu.index` and `dials_gpu.refine` for Many Small `.expt` Files (RL / ML Workloads)
In `dials_gpu.index`, all basis-vector search, indexing, and refinement steps run on the GPU:
- **Basis-vector search & index assignment** (`find_max_cell`, `fft3d` / `fft1d` / `real_space_grid_search`, `optimise_basis_vectors`, `candidate_orientation_matrices`, `assign_indices`, `non_primitive_basis`) execute on the GPU in **~40–60 ms total** (down from ~1,630 ms on CPU for `fft3d`, a **38.3× speedup**).
- **Candidate orientation evaluation (`choose_best_orientation_matrix`) & macro-cycle refinement (`Indexer.refine`)** use [`RefinerFactoryGPU`](src/dials_gpu/refinement/refiner.py) and [`ExperimentsPredictorFactoryGPU`](src/dials_gpu/refinement/predictor.py), computing reflections predictions, analytical state/prediction derivatives, dense Jacobians, and centroid outlier rejection (`MCD` / `Tukey` / `SauterPoon`) on the GPU without subprocess fork overhead.

To achieve maximum indexing and refinement throughput on many small `.expt` files:
- **Reduce `max_refine` and `n_macro_cycles` in `dials_gpu.index`**: Because the winning orientation matrix is almost always among the top 3–5 candidates, setting `max_refine=5` and `n_macro_cycles=2` reduces `dials_gpu.index` runtime to **`278–299 ms/file`** (**6.1–6.6× faster** than CPU `dials.index`, or **`217–229 ms/file` / ~8× faster** with `max_refine=3, n_macro_cycles=1`) while producing identical unit cells (`0.00000` max unit cell difference).
- **Avoid `multiprocessing.fork()` in `dials.refine` on small wedges**: Standard CPU `dials.refine` defaults to `refinement.reflections.outlier.nproc=4`, spending ~560 ms per `.expt` file forking worker processes for MCD outlier rejection. `RefinerFactoryGPU` runs MCD, Tukey, and Sauter-Poon outlier rejection in-process on the GPU in < 2 ms, completing full scan-static refinement in **`64.0 ms/file`** (**11.1× faster** than default CPU `dials.refine`) and scan-varying refinement in **`118.0 ms/file`** (**7.1× faster** than default CPU `dials.refine`).

### 3. Understanding & Ordering Dataset Sweeps for the Two-Tier HDF5 Cache
For single-panel HDF5/NeXus (`.nxs` / `.h5`) datasets (up to 64 frames per scan, such as 25-image wedges), `dials_gpu` uses a two-tier cache in [`finder.py`](src/dials_gpu/spot_finding/finder.py):
- **Tier-1 GPU VRAM Cache (`_HDF5_GPU_VOLUME_CACHE` & `_HDF5_GPU_MASK_CACHE`, default: 2 volumes, ~1.2 GB VRAM)**:
  Caches the most recently used uncompressed `(B, H, W)` `int32` image volumes and base static/trusted-range masks directly on the GPU while leaving ample VRAM for `FFT3DGPU` (`256³` complex grids).
  - **Tip**: Run multiple RL steps/episodes on the same `.nxs` file before advancing to the next file. This achieves a 100% GPU VRAM cache hit rate (**~50 ms per 25-image wedge**, or **~490 images/s**).
- **Tier-2 Host RAM Compressed Cache (`_HDF5_COMPRESSED_CACHE`, default: 100 files, ~1.8 GB RAM)**:
  Caches the raw bitshuffle-LZ4 compressed HDF5 chunks in system RAM and decompresses all 25 frames in parallel across CPU threads directly into a pinned host memory buffer (`_PINNED_HOST_VOL_CACHE`) for DMA transfer to GPU (**~145 ms per 25-image wedge**, vs ~606 ms in `dxtbx`).
  - All 90 `.nxs` files in a typical `i19` dataset fit simultaneously inside the 100-file RAM cache (`~1.65 GB`).

### 4. Modifying `phil_scope` Parameters Without Re-Parsing Strings
- Calling `phil_scope.fetch(source=parse(...)).extract()` repeatedly inside a tight loop spends ~5–8 ms parsing PHIL strings on CPU. Instead, call `base_phil = phil_scope.fetch()` once at startup, and mutate attributes on `params = base_phil.extract()` directly (as shown above).
- **Important**: Always call `phil_scope.fetch().extract()` rather than `phil_scope.extract()` directly; `fetch()` resolves default empty scope templates such as `spotfinder.filter.untrusted`.

### 5. `spotfinder.mp.nproc` Settings: `.nxs`/`.h5` vs `.cbf`/`.cbf.gz`
- **For `.nxs` / `.h5` (HDF5 / NeXus) files**: Keep `spotfinder.mp.nproc=1` (the default). `dials_gpu` automatically uses its internal persistent thread pool (`_BSHUF_POOL`) to decompress bitshuffle-LZ4 chunks in parallel without `fork` overhead (and avoids `libhdf5` fork deadlocks).
- **For multi-file `.cbf` / `.cbf.gz` scans (e.g., 720-image Pilatus datasets)**: Set `spotfinder.mp.nproc=4` (or `8`) on the CLI or in `params`:
  ```bash
  dials_gpu.find_spots imported.expt spotfinder.mp.nproc=4
  ```
  This overlaps multi-core CPU CBF decompression in shared memory with GPU thresholding.

---

## Benchmarks

### 1. Spot-Finding (`dials_gpu.find_spots` on 25-image `i19` Eiger CdTe `.nxs` wedge, `2162 × 2068`)

| Mode | Time per 25-image `.nxs` | Throughput | Speedup vs CPU (`nproc=1`) | Speedup vs CPU (`nproc=4`) |
| :--- | ---: | ---: | ---: | ---: |
| `dials.find_spots` (CPU, `nproc=1`) | `3,712.6 ms` | `6.7 img/s` | `1.0×` | `0.42×` |
| `dials.find_spots` (CPU, `nproc=4`) | `1,562.2 ms` | `16.0 img/s` | `2.4×` | `1.0×` |
| `dials_gpu.find_spots` (Cold Read from `.nxs`) | `173.0 ms` | `144.5 img/s` | `21.5×` | `9.0×` |
| `dials_gpu.find_spots` (Tier-2 Host RAM Cache Hit) | `146.5 ms` | `170.6 img/s` | `25.3×` | `10.7×` |
| **`dials_gpu.find_spots` (Tier-1 GPU VRAM Cache Hit)** | **`50.8 ms`** | **`492.5 img/s`** | **`73.1×`** | **`30.8×`** |

### 2. Indexing Per-Algorithm Breakdown (`dials_gpu.index` on `i19` & `i04_weak_data`)

| Indexing Algorithm / Stage | CPU (`dials.index`) | GPU (`dials_gpu.index`) | Speedup | Accuracy / Equivalence |
| :--- | ---: | ---: | ---: | :--- |
| `FFT3D` (3D FFT + periodic 6-connected flood fill, `256³`) | `1,593.0 ms` | **`11.5 ms`** | **`138.5×`** | 100% identical void counts, volumes & centres (`3.3e-16`) |
| `FFT3D.find_basis_vectors` (full pipeline incl. grouping) | `1,611.1 ms` | **`42.1 ms`** | **`38.3×`** | 100% identical candidate basis vectors |
| `RealSpaceGridSearch.score_vectors` (cosine grid scoring) | `493.0 ms` | **`0.45 ms`** | **`1,095×`** | `1e-14` max score difference |
| `RealSpaceGridSearch.find_basis_vectors` (full pipeline) | `543.9 ms` | **`14.6 ms`** | **`37.3×`** | 100% identical unique basis vectors |
| `FFT1D.find_basis_vectors` (batched hemisphere 1D RFFT) | `99.7 ms` | **`39.0 ms`** | **`2.5×`** | `1.3e-15` power spectrum match; identical vectors |
| `AssignIndicesGlobal` (`116,082` spots, `i04_weak_data`) | `74.1 ms` | **`12.6 ms`** | **`5.9×`** | 100% identical Miller indices & crystal IDs |

### 3. End-to-End Indexing across 10 `i19` `.expt` Files (`301_Nidppe_f3_phi0`)

| Configuration | CPU (`dials.index`) | GPU (`dials_gpu.index`) | Speedup vs CPU | Total Indexed Spots (10 files) | Max Unit Cell Diff vs CPU |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `fft3d` (default: `max_refine=50, n_macro_cycles=5`) | `2,874.8 ms/file` | **`1,299.4 ms/file`** | **`2.21×`** | `2,133` (CPU) / `2,133` (GPU) | `0.00000` |
| `real_space_grid_search` (default: `max_refine=50, n_macro=5`) | `1,751.2 ms/file` | **`899.6 ms/file`** | **`1.95×`** | `2,782` (CPU) / `2,782` (GPU) | `0.00000` |
| `fft1d` (default: `max_refine=50, n_macro_cycles=5`) | `1,198.2 ms/file` | **`1,000.9 ms/file`** | **`1.20×`** | `1,583` (CPU) / `1,583` (GPU) | `0.00000` |
| **`fft3d` (RL Fast: `max_refine=5, n_macro_cycles=2`)** | `1,846.8 ms/file` | **`278.5 ms/file`** | **`6.63×`** (`10.3×` vs default CPU) | `1,395` (CPU) / `1,395` (GPU) | `0.00000` |
| **`fft1d` (RL Fast: `max_refine=5, n_macro_cycles=2`)** | `440.4 ms/file` | **`229.5 ms/file`** | **`1.92×`** (`5.2×` vs default CPU) | `1,600` (CPU) / `1,600` (GPU) | `0.00000` |
| **`fft3d` (RL Ultra: `max_refine=3, n_macro_cycles=1`)** | `1,769.8 ms/file` | **`217.3 ms/file`** | **`8.15×`** (`13.2×` vs default CPU) | `1,250` (CPU) / `1,250` (GPU) | `0.00000` |

### 4. Refinement (`dials_gpu.refine` on `i19` 25-image `.expt` Files & `from-xds-all` 27k Reflections)

| Refinement Configuration | CPU (`dials.refine`) | GPU (`dials_gpu.refine`) | Speedup vs CPU | Max RMSD Diff vs CPU |
| :--- | ---: | ---: | ---: | ---: |
| `i19` Small Wedge (`scan_varying=False`, default PHIL) | `710.6 ms/file` | **`64.0 ms/file`** | **`11.10×`** | `2.69e-15` |
| `i19` Small Wedge (`scan_varying=True`, default PHIL) | `837.3 ms/file` | **`118.0 ms/file`** | **`7.10×`** | `3.59e-15` |
| `i19` Small Wedge (`scan_varying=False`, `outlier.nproc=1`) | `150.1 ms/file` | **`64.0 ms/file`** | **`2.34×`** | `2.69e-15` |
| `i19` Small Wedge (`scan_varying=True`, `outlier.nproc=1`) | `256.6 ms/file` | **`118.0 ms/file`** | **`2.17×`** | `3.59e-15` |
| `from-xds-all` (`26,993` refl, `scan_varying=False, nproc=1`) | `2,477.4 ms` | **`660.1 ms`** | **`3.75×`** | `2.07e-15` |
| `from-xds-all` (`26,993` refl, `scan_varying=True, nproc=1`) | `6,553.6 ms` | **`2,890.6 ms`** | **`2.27×`** | `4.27e-15` |
| Target Analytical Gradients + Dense Jacobian (`26,993` refl, `sv=False`) | `20.11 ms/step` | **`5.69 ms/step`** | **`3.53×`** | `< 1e-15` |
| Target Analytical Gradients + Dense Jacobian (`26,993` refl, `sv=True`) | `20.24 ms/step` | **`5.61 ms/step`** | **`3.61×`** | `< 1e-15` |
