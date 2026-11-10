# dials_gpu

GPU-accelerated implementations of [DIALS](https://dials.github.io/) crystallographic data-processing algorithms using PyTorch (`torch`).

## Command-Line Tools

- `dials_gpu.find_spots`: Drop-in GPU replacement for `dials.find_spots` supporting all standard DIALS PHIL parameters and spot-finding algorithms (`dispersion_extended`, `dispersion`, `radial_profile`, 2D/3D connected component labelling, centroiding, intensity summation, shoebox extraction, and spot filtering).
- `dials_gpu.index`: Drop-in GPU replacement for `dials.index` supporting all standard DIALS PHIL parameters and indexing algorithms (`fft3d`, `fft1d`, `real_space_grid_search`, `low_res_spot_match`, `find_max_cell`, `optimise_basis_vectors`, `candidate_orientation_matrices`, `AssignIndicesGlobal` / `AssignIndicesLocal`, `non_primitive_basis`, rotation sequences, stills, and multi-lattice indexing), with all internal candidate orientation evaluation (`choose_best_orientation_matrix` / `ModelEvaluation`) and macro-cycle refinement (`Indexer.refine`) mapped to the GPU.
- `dials_gpu.refine`: Drop-in GPU replacement for `dials.refine` supporting all standard DIALS PHIL parameters and refinement algorithms (scan-static and scan-varying refinement, stills and spherical-relp stills refinement, single/multi-panel and hierarchical detectors, `mcd` / `tukey` / `sauter_poon` GPU centroid outlier rejection, `LevMar` / `GaussNewton` / `SimpleLBFGS` / `LBFGScurvs` / `SparseLevMar` engines, restraints, constraints, and `AutoReduce`).
- `dials_gpu.integrate` (and alias `dials_gpu.interate`): Drop-in GPU replacement for `dials.integrate` supporting all standard DIALS PHIL parameters and integration algorithms (`3d`, `flat3d`, `2d`, `single2d`, `stills`, `3d_threaded` integrators; `glm` Cantoni & Ronchetti robust Poisson IRLS background, `simple` `null`/`nsigma`/`tukey`/`truncated` + `constant2d`/`constant3d`/`linear2d`/`linear3d` background, `auto` / `null` background; `simple` 3D shoebox centroiding with parallax correction; 3D/2D summation integration; `gaussian_rs` scan-static and scan-varying profile modelling and fitting; Lorentz-polarization and QE corrections; `SignificanceFilter`, `kapton_2019` absorption correction, and `frame_slice_shoeboxes`).
- `dials_gpu.scale`: Drop-in GPU replacement for `dials.scale` supporting all standard DIALS PHIL parameters and scaling algorithms (`physical`, `KB`, `array`, and `dose_decay` scaling models; 1D/2D/3D Gaussian smoothers, spherical harmonic `SHScaleComponent` absorption surface, `SingleScaleFactor`, `SingleBScaleFactor`, `QuadraticBScaleComponent`; `RefinerCalculatorGPU` multi-component product rule; `IhTableGPU` vectorized symmetry group blocking and `scatter_add_` `calc_Ih`; `ScalingTargetGPU` and `ScalingTargetFixedIHGPU` analytical L-BFGS gradients and dense Gauss-Newton / Levenberg-Marquardt Jacobians; `NormDevOutlierRejectionGPU`, `SimpleNormDevOutlierRejectionGPU`, and `TargetedOutlierRejectionGPU`; `SingleDatasetIntensityCombinerGPU` and `MultiDatasetIntensityCombinerGPU`; `ErrorModelRefineryGPU` and `ErrorModelTargetRegressionGPU`; `SingleScalerGPU`, `MultiScalerGPU`, `TargetScalerGPU`, `ScaleAndFilterAlgorithmGPU`, and cross-validation).

## Building from Source

Requirements: Linux, an NVIDIA GPU with CUDA, a DIALS source/bootstrap installation, and a CUDA build of PyTorch (which provides Triton) installed into the DIALS Python.

```bash
source /path/to/dials/dials      # activate the DIALS environment
./build.sh                       # compile the C++ extensions, check they load, pip install -e .
./build.sh --ext-only            # only recompile the C++ extensions
```

[`build.sh`](build.sh) finds the DIALS headers and libraries from the active environment, compiles [`_cbf_read.cpp`](src/dials_gpu/spot_finding/_cbf_read.cpp) and [`_shoebox_pack.cpp`](src/dials_gpu/integration/_shoebox_pack.cpp) (`-O3 -march=native` by default; override with `CXX` / `CXXFLAGS`), and installs the `dials_gpu.*` commands. **Re-run it after rebuilding or updating DIALS**: `_shoebox_pack_ext.so` is compiled against DIALS C++ headers and must match the installed DIALS.

## Usage

Activate the DIALS environment and run `dials_gpu.find_spots`, `dials_gpu.index`, `dials_gpu.refine`, `dials_gpu.integrate`, and `dials_gpu.scale` with the exact same arguments as `dials.find_spots`, `dials.index`, `dials.refine`, `dials.integrate`, and `dials.scale`:

```bash
source /path/to/dials/dials

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

# 4. Integration (full 3D + GLM + profile fitting, or fast summation-only)
dials_gpu.integrate refined.expt refined.refl
dials_gpu.integrate refined.expt refined.refl integration.profile.fitting=False

# 5. Scaling (single sweep, multi-sweep, targeted scaling, or scale-and-filter)
dials_gpu.scale integrated.refl integrated.expt
dials_gpu.scale sweep1.refl sweep1.expt sweep2.refl sweep2.expt model=physical
dials_gpu.scale integrated.refl integrated.expt model=dose_decay full_matrix=False
```

---

## Performance Optimization Tips

### 1. In-Process Python API for RL / ML Training Loops (Full Spot-Finding → Indexing → Refinement → Integration → Scaling Pipeline)
Spawning new CLI subprocesses (`dials_gpu.find_spots ...`, `dials_gpu.index ...`, `dials_gpu.refine ...`, `dials_gpu.integrate ...`, `dials_gpu.scale ...`) on every RL step incurs ~1.5–2.0 s of Python, `dxtbx`, and CUDA context initialization overhead. For Reinforcement Learning (RL) or hyperparameter optimization, call [`from_observations_gpu`](src/dials_gpu/command_line/find_spots.py), [`index`](src/dials_gpu/command_line/index.py), [`RefinerFactoryGPU`](src/dials_gpu/refinement/refiner.py), [`run_integration`](src/dials_gpu/command_line/integrate.py), and [`run_scaling`](src/dials_gpu/command_line/scale.py) directly inside your long-lived Python training process:

```python
import copy
from dxtbx.model.experiment_list import ExperimentList
from iotbx.phil import parse
from dials.command_line.find_spots import phil_scope as spot_phil_scope
from dials.command_line.index import phil_scope as index_phil_scope
from dials.command_line.refine import working_phil as refine_phil_scope
from dials.command_line.integrate import phil_scope as integrate_phil_scope
from dials.command_line.scale import phil_scope as scale_phil_scope
from dials_gpu.command_line.find_spots import from_observations_gpu
from dials_gpu.command_line.index import index as index_gpu
from dials_gpu.refinement import RefinerFactoryGPU
from dials_gpu.command_line.integrate import run_integration as integrate_gpu
from dials_gpu.command_line.scale import run_scaling as scale_gpu

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
base_integrate_phil = integrate_phil_scope.fetch(
    source=parse("integration.mp.nproc=1\nintegration.profile.fitting=False")
)
base_scale_phil = scale_phil_scope.fetch(
    source=parse("scaling_options.nproc=1\nscaling_options.full_matrix=False\noutput.html=None")
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
    refined_expts = refiner.get_experiments()
    refined_refl = refiner.get_matches()

    # Integration on GPU (~154 ms summation-only, ~286 ms with profile fitting)
    integrated_expts, integrated_refl, _ = integrate_gpu(
        base_integrate_phil.extract(), refined_expts, refined_refl
    )

    # Scaling on GPU (~65–85 ms/file single sweep, or ~185–700 ms joint multi-sweep)
    scaled_expts, scaled_refl = scale_gpu(
        base_scale_phil.extract(), integrated_expts, [integrated_refl]
    )
    return scaled_expts, scaled_refl
```

### 2. Optimizing `dials_gpu.index`, `dials_gpu.refine`, and `dials_gpu.integrate` for Many Small `.expt` Files (RL / ML Workloads)
In `dials_gpu.index`, `dials_gpu.refine`, and `dials_gpu.integrate`, the compute-intensive steps run on the GPU and OpenMP C++ without subprocess fork overhead:
- **Basis-vector search & index assignment** (`find_max_cell`, `fft3d` / `fft1d` / `real_space_grid_search`, `optimise_basis_vectors`, `candidate_orientation_matrices`, `assign_indices`, `non_primitive_basis`) execute on the GPU in **~40–60 ms total** (down from ~1,630 ms on CPU for `fft3d`, a **38.3× speedup**).
- **Candidate orientation evaluation (`choose_best_orientation_matrix`) & macro-cycle refinement (`Indexer.refine`)** use [`RefinerFactoryGPU`](src/dials_gpu/refinement/refiner.py) and [`ExperimentsPredictorFactoryGPU`](src/dials_gpu/refinement/predictor.py), computing reflections predictions, analytical state/prediction derivatives, dense Jacobians, and centroid outlier rejection (`MCD` / `Tukey` / `SauterPoon`) on the GPU without subprocess fork overhead.
- **Direct 3D volume shoebox extraction, histogram-compressed GPU background & lock-free C++17 `std::thread` profile fitting (`dials_gpu.integrate`)**:
  - For single-panel HDF5/NeXus sweeps (`.nxs`) and `.cbf` / `.cbf.gz` rotation scans, [`IntegratorGPU`](src/dials_gpu/integration/integrator.py) reuses the decoded 3D `(B, H, W)` volume directly from the spot-finding cache (`_HDF5_GPU_VOLUME_CACHE` / `_HOST_VOL_CACHE`) for small sweeps, or streams 48-frame slabs directly via C++ multi-threaded decompression (`_get_cbf_batch_reader` / `_try_load_hdf5_raw_volume_gpu`) for large sweeps, extracting all shoeboxes (`extract_shoeboxes_from_volume` / `extract_wide_shoeboxes_slice_from_volume` in [`_shoebox_pack.cpp`](src/dials_gpu/integration/_shoebox_pack.cpp)) in **~1.5 ms** per small wedge (`0.93 s` across 720 frames).
  - [`_shoebox_pack.cpp`](src/dials_gpu/integration/_shoebox_pack.cpp) executes `parallel_compute_bbox_3d_fast`, `parallel_compute_mask_3d_fast`, `parallel_mask_neighbouring` (**46.6× faster** than CPU), `match_with_reference_indices` (**20.2× faster** than CPU), and `ParallelGaussianRS` (`fit_parallel_cols` / `model_parallel_cols`, restricting `get_pixel_lab_coord` + `quad_to_grid` polygon clipping to the 2D column bounding box of valid foreground pixels) using a C++17 `std::thread` + atomic work-stealing pool, avoiding OpenMP `libgomp` vs PyTorch `libiomp5` spin-wait contention.
  - [`GLMBackgroundGPU`](src/dials_gpu/integration/background.py) compresses ~10M background pixels into a `(n_refl, 128)` integer count histogram + sparse tail (`extract_glm_constant3d_hist`) and runs Cantoni & Ronchetti Huber IRLS on GPU in **18.2 ms** for 5,336 shoeboxes (**12.0× faster** than CPU, `max diff = 0.00e+00`).
  - [`SimpleBackgroundGPU`](src/dials_gpu/integration/background.py) (`tukey` / `nsigma` / `truncated` / `null` + `constant3d`) runs in **13.4 ms** for 5,336 shoeboxes (**17.3× faster** than CPU).
  - **The slabbed large-sweep path (single-panel sweeps of more than 64 images) does not support dynamic masks.** If the imageset has a goniometer shadow mask (`imageset.has_dynamic_mask()`), integration falls back to a much slower per-task path. A plain `dials.import` of DLS mini-kappa Pilatus data such as `C2sum_1` sets `dynamic_shadowing=Auto`, which enables the shadow mask: on 720-frame `C2sum_1` this makes `dials_gpu.integrate` take **82–97 s instead of 6.4 s** (slower than `dials.integrate` on 20 cores, 42 s). Import with `dynamic_shadowing=False` when shadowing is not needed. [`benchmarks/compare_stages.py`](benchmarks/compare_stages.py) warns when an input has a dynamic mask.
  - Keep `integration.mp.nproc=1` (the default in `dials_gpu.integrate`) on small wedges to avoid multiprocessing fork overhead and reuse the in-process volume cache; set `integration.profile.fitting=False` if only summation intensities (`intensity.sum.value` / `intensity.sum.variance`) are needed for RL reward signals (**`154.1 ms/file`** on `i19`, **`7.285 s`** on 720-frame `C2sum_1`).

To achieve maximum indexing and refinement throughput on many small `.expt` files:
- **Reduce `max_refine` and `n_macro_cycles` in `dials_gpu.index`**: Because the winning orientation matrix is almost always among the top 3–5 candidates, setting `max_refine=5` and `n_macro_cycles=2` reduces `dials_gpu.index` runtime to **`278–299 ms/file`** (**6.1–6.6× faster** than CPU `dials.index`, or **`217–229 ms/file` / ~8× faster** with `max_refine=3, n_macro_cycles=1`) while producing identical unit cells (`0.00000` max unit cell difference).
- **Avoid `multiprocessing.fork()` in `dials.refine` on small wedges**: Standard CPU `dials.refine` defaults to `refinement.reflections.outlier.nproc=4`, spending ~560 ms per `.expt` file forking worker processes for MCD outlier rejection. `RefinerFactoryGPU` runs MCD, Tukey, and Sauter-Poon outlier rejection in-process on the GPU in < 2 ms, completing full scan-static refinement in **`64.0 ms/file`** (**11.1× faster** than default CPU `dials.refine`) and scan-varying refinement in **`118.0 ms/file`** (**7.1× faster** than default CPU `dials.refine`).

### 3. Understanding & Ordering Dataset Sweeps for the Two-Tier HDF5 Cache
For single-panel HDF5/NeXus (`.nxs` / `.h5`) datasets (up to 64 frames per scan, such as 25-image wedges), `dials_gpu` uses a two-tier cache in [`finder.py`](src/dials_gpu/spot_finding/finder.py):
- **Tier-1 GPU VRAM Cache (`_HDF5_GPU_VOLUME_CACHE` & `_HDF5_GPU_MASK_CACHE`, default: 8 volumes)**:
  Caches the most recently used uncompressed `(B, H, W)` `int32` image volumes and base static/trusted-range masks directly on the GPU for small sweeps ($\le 64$ frames), and automatically clears when switching to large $>64$-frame sweeps.
  - **Tip**: Run multiple RL steps/episodes on the same `.nxs` file (or across $\le 8$ small sweeps) before advancing. This achieves a 100% GPU VRAM cache hit rate (**~50.8 ms per 25-image wedge**, or **~492 images/s**).
- **Tier-2 Host RAM Compressed Cache (`_HDF5_COMPRESSED_CACHE`, default: 100 files, ~1.8 GB RAM)**:
  Caches the raw bitshuffle-LZ4 compressed HDF5 chunks in system RAM and decompresses all 25 frames in parallel across CPU threads directly into a pinned host memory buffer (`_PINNED_HOST_VOL_CACHE`) for DMA transfer to GPU (**~145 ms per 25-image wedge**, vs ~606 ms in `dxtbx`).
  - All 90 `.nxs` files in a typical `i19` dataset fit simultaneously inside the 100-file RAM cache (`~1.65 GB`).

### 4. Modifying `phil_scope` Parameters Without Re-Parsing Strings
- Calling `phil_scope.fetch(source=parse(...)).extract()` repeatedly inside a tight loop spends ~5–8 ms parsing PHIL strings on CPU. Instead, call `base_phil = phil_scope.fetch()` once at startup, and mutate attributes on `params = base_phil.extract()` directly (as shown above).
- **Important**: Always call `phil_scope.fetch().extract()` rather than `phil_scope.extract()` directly; `fetch()` resolves default empty scope templates such as `spotfinder.filter.untrusted`.

### 5. Large Sweeps (`>64` Frames) & Multi-File `.cbf` / `.cbf.gz` Scans
- **3-Stage Overlapped Pipeline (`_run_pipelined_batch_volume_gpu`)**: For single-panel sweeps with $>64$ frames (both HDF5 `.nxs` and multi-file `.cbf.gz` / `.cbf` datasets like 720-frame Pilatus `C2sum_1`), `dials_gpu.find_spots` automatically uses a 3-stage double-buffered pipeline:
  1. **CPU Decompression Stage**: Multi-threaded C++ `libdeflate` + 8× unrolled `x-CBF_BYTE_OFFSET` decoding ([`_cbf_read.cpp`](src/dials_gpu/spot_finding/_cbf_read.cpp), `1.045 s` for 720 `.cbf.gz` frames across 20 threads) or `bshuf_decompress_lz4` (`1.175 s` for 720 `.nxs` frames) directly into double-buffered pinned host memory (`_PIPELINE_PINNED_CACHE`).
  2. **PCIe DMA Stage**: Asynchronous host-to-device copy on a dedicated CUDA `copy_stream` into double-buffered GPU slabs `(32, H, W)`.
  3. **GPU Compute Stage**: Batched Triton `DispersionExtended` thresholding (pass-2 horizontal sums are only computed for 256-pixel row blocks within the kernel height of a candidate pixel), per-frame strong-pixel counts (a `uint8` sum copied into pinned memory, so the `max_strong_pixel_fraction` check runs before `torch.nonzero` without stalling behind the next slab's host-to-device copy), then `torch.nonzero` on the default compute stream, followed by single-pass GPU 3D connected-components (`35 ms`) and C++ `Shoebox` construction (`36 ms`), completing all 720 frames of `C2sum_1` (`112,023` spots) in **`2.98 s` warm (`4.14 s` first call)** in-process, or `7.1 s` as a `dials_gpu.find_spots` command (including Python and CUDA start-up; `dials.find_spots`: `23.4 s`). Re-measured 2026-11-09; see [§9](#9-reproducing-the-dials-vs-dials_gpu-vs-mxi-comparison).

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
| **`C2sum_1` (`~111k` indexed refl, `scan_varying=False` static)**¹ | `4,153 ms` | **`1,215 ms`** (`1,225 ms` first call) | **`3.42×`** | not re-measured |
| **`C2sum_1` (`~111k` indexed refl, `scan_varying=Auto` 2-macrocycle)**¹ | `23,362 ms` | **`3,540 ms`** (`3,559 ms` second call) | **`6.60×`** | not re-measured |
| Target Analytical Gradients + GPU Normal Eqns (`112,023` refl, `sv=False`) | `20.11 ms/step` | **`9.80 ms/step`** | **`2.05×`** | `< 1e-15` |
| Target Analytical Gradients + GPU Normal Eqns (`112,023` refl, `sv=True`) | `1,150.0 ms/step` | **`25.60 ms/step`** | **`44.9×`** | `< 1e-15` |

¹ Re-measured 2026-11-09 with the in-process method of [`benchmarks/compare_stages.py`](benchmarks/compare_stages.py) (, each implementation refining its own indexing result); see [§9](#9-reproducing-the-dials-vs-dials_gpu-vs-mxi-comparison). The earlier CPU static figure (`8,420 ms`) is not reproduced on current DIALS.

### 5. Integration (`dials_gpu.integrate` on `i19` 25-image `.nxs` Sweeps)

#### 5a. Per-Algorithm / Kernel Micro-Benchmarks (`N = 5,336` 3D Shoeboxes)

| Integration Kernel / Algorithm | CPU (`dials.integrate`) | GPU (`dials_gpu.integrate`) | Speedup | Max Numerical Diff vs CPU |
| :--- | ---: | ---: | ---: | ---: |
| `SimpleBackground (tukey + constant3d)` | `231.70 ms` | **`13.37 ms`** | **`17.33×`** | `4.75e-08` |
| `GLMBackground (constant3d)` (Cantoni & Ronchetti Huber Poisson IRLS) | `217.53 ms` | **`18.19 ms`** | **`11.96×`** | `0.00e+00` |
| `Summation Integration` (3D shoebox foreground/background summation) | `9.73 ms` | **`2.32 ms`** | **`4.19×`** | `1.21e-03` |
| `ProfileModelFactory.create` (`sigma_b`, `sigma_m`, `zeta`, `partiality`) | `55.38 ms` | **`28.02 ms`** | **`1.98×`** | `< 1e-7` |
| `SimpleCentroid` (3D shoebox centre-of-mass + `px_to_mm`) | `31.74 ms` | **`21.21 ms`** | **`1.50×`** | `2.06e-05` |

#### 5b. End-to-End Integration across 5 `i19` 25-image `.nxs` Datasets (`301_Nidppe_f3_phi0`) and 720-frame `C2sum_1`

| Configuration | CPU (`dials.integrate`) | GPU (`dials_gpu.integrate`) | Speedup vs CPU | Total Integrated Reflections |
| :--- | ---: | ---: | ---: | :--- |
| **`i19` Default (`3D + GLM + Profile Fitting`, `nproc=1` vs GPU)** | `2,687.1 ms/file` | **`110.3 ms/file`** (`47.0 ms` warm) | **`24.36×`** | `13,587` (CPU) / `13,587` (GPU) |
| **`i19` Default (`3D + GLM + Profile Fitting`, `nproc=4` vs GPU)** | `2,820.1 ms/file` | **`110.3 ms/file`** (`47.0 ms` warm) | **`25.57×`** | `13,680` (CPU) / `13,587` (GPU) |
| **`i19` Summation Only (`3D + GLM, profile.fitting=False`)** | `1,485.8 ms/file` | **`85.0 ms/file`** | **`17.48×`** | `13,367` (CPU) / `13,367` (GPU) |
| **`C2sum_1` Default (`3D + GLM + Profile Fitting`, 720 frames)**² | `210.2 s` (`nproc=1`) / `42.2 s` (`nproc=20`) | **`6.35 s`** (`6.43 s` second call) | **`33.1×`** vs `nproc=1` / **`6.65×`** vs `nproc=20` | `325,319` (CPU) / `325,316` (GPU) profile-fitted |
| **`C2sum_1` Default, imported with `dynamic_shadowing=Auto`**³ | `42.2 s` (`nproc=20`) | `82–97 s` | **`0.4–0.5×` (slower)** | `325,316` (GPU) profile-fitted |
| **`C2sum_1` Summation Only (`3D + GLM, profile.fitting=False`)** | `118.420 s` | **`4.510 s`** | **`26.26×`** | `313,792` (CPU) / `313,792` (GPU) — not re-measured |

² Re-measured 2026-11-09 ([§9](#9-reproducing-the-dials-vs-dials_gpu-vs-mxi-comparison)) on data imported with `dynamic_shadowing=False`, which the original figure also (implicitly) used. The earlier "`41.66 s` cached" CPU figure corresponds to `dials.integrate` with all 20 cores.
³ A plain `dials.import` of this DLS mini-kappa Pilatus data enables goniometer shadow masking; the slabbed fast path does not support dynamic masks (see [Performance Tip 2](#2-optimizing-dials_gpuindex-dials_gpurefine-and-dials_gpuintegrate-for-many-small-expt-files-rl--ml-workloads)).

### 6. Scaling (`dials_gpu.scale` on `i19` 25-image `.nxs` Sweeps & 50k Reflection Blocks)

#### 6a. Per-Algorithm / Kernel Micro-Benchmarks (`N = 50,000` Reflections)

| Scaling Kernel / Algorithm | CPU (`dials.scale`) | GPU (`dials_gpu.scale`) | Speedup | Max Numerical Diff vs CPU |
| :--- | ---: | ---: | ---: | ---: |
| `ScalingTarget.compute_functional_gradients` (L-BFGS analytical gradients) | `15.48 ms` | **`0.35 ms`** | **`43.81×`** | `6.29e-09` |
| `ScalingTarget.compute_residuals_and_gradients` (LevMar dense Jacobian) | `46.56 ms` | **`1.22 ms`** | **`38.03×`** | `5.68e-13` |
| `RefinerCalculator` (multi-component product rule scales + derivatives) | `15.93 ms` | **`1.26 ms`** | **`12.63×`** | `8.88e-16` |
| `SmoothBScaleComponent1D` (1D Gaussian smoother B-factor + derivatives) | `5.48 ms` | **`0.48 ms`** | **`11.45×`** | `4.44e-16` |
| `SmoothScaleComponent1D` (1D Gaussian smoother scale + derivatives) | `3.70 ms` | **`0.38 ms`** | **`9.64×`** | `6.66e-16` |
| `NormDevOutlierRejection` (iterative round-robin z-score outlier rejection) | `50.94 ms` | **`14.16 ms`** | **`3.60×`** | `0` outlier diffs |
| `IhTableBlock.calc_Ih` (`scatter_add_` symmetry group intensity averaging) | `0.63 ms` | **`0.26 ms`** | **`2.44×`** | `4.55e-13` |

#### 6b. End-to-End Scaling across `i19` 25-image `.nxs` Sweeps (`301_Nidppe_f3_phi0`) and 720-frame `C2sum_1`

| Scaling Configuration | CPU (`dials.scale`) | GPU (`dials_gpu.scale`) | Speedup vs CPU | Mean Relative Scale Diff |
| :--- | ---: | ---: | ---: | ---: |
| **Single-file (`w00017`, `physical`, full pipeline)** | `202.16 ms` | **`47.16 ms`** | **`4.29×`** | `4.70e-08` |
| **Single-file loop (`dose_decay, SimpleLBFGS, full_mat=False`)** | `179.35 ms/file` | **`48.92 ms/file`** | **`3.67×`** | `6.89e-10` |
| **Single-file loop (`dose_decay, SimpleLBFGS, full_mat=True`)** | `186.37 ms/file` | **`54.47 ms/file`** | **`3.42×`** | `2.53e-10` |
| **Single-file loop (`physical, SimpleLBFGS, full_mat=False`)** | `202.16 ms/file` | **`63.75 ms/file`** | **`3.17×`** | `5.12e-08` |
| **Single-file loop (`physical, SimpleLBFGS, full_mat=True`)** | `206.74 ms/file` | **`68.52 ms/file`** | **`3.02×`** | `4.70e-08` |
| **Joint 5-sweep (`KB, full_mat=False`)** | `463.76 ms` | **`125.93 ms`** | **`3.68×`** | `7.52e-15` |
| **Joint 5-sweep (`dose_decay, full_mat=False`)** | `637.21 ms` | **`193.88 ms`** | **`3.29×`** | `1.77e-09` |
| **`C2sum_1` (720 frames, `~325k` refls, `physical, full_mat=True`)**⁴ | `15,315 ms` | **`1,299 ms`** warm (`1,347 ms` first call) | **`11.79×`** | merging statistics identical (CC½ `0.9991` both) |

⁴ Re-measured 2026-11-09 ([§9](#9-reproducing-the-dials-vs-dials_gpu-vs-mxi-comparison)); each implementation scales its own integration result.

---

## 7. Comparison Against `mxi` (CPU & GPU) and Standard `dials`

[`benchmarks/benchmark_mxi_comparison.py`](benchmarks/benchmark_mxi_comparison.py) benchmarks [`dials`](https://github.com/dials/dials) (CPU), `dials_gpu` (PyTorch/Triton/CUDA), and [`mxi`](https://github.com/graeme-winter/mxi) v1.1.2 (C++20/OpenMP + optional `--gpu` CUDA kernels for `find` and `integrate`) across two complementary regimes:
1. **`i19` (`301_Nidppe_f3_phi0`)**: 5 small 25-frame Eiger 2X 4M NeXus sweeps (`2068 × 2162`, space group `P 1 21/n 1`, unit cell `11.39, 13.30, 15.82, 90, 99.02, 90`).
2. **`C2sum_1`**: 720-frame Pilatus 6M rotation sweep (`2463 × 2527`, space group `C 1 2 1`, unit cell `53.17, 61.25, 69.29, 90, 93.04, 90`, `112,023` strong spots, `~327,000` integrated reflections).

### 7a. Per-Stage Execution Time Comparison

#### Dataset 1: `i19` (25-Image Sweeps, Mean per Sweep)

| Pipeline Stage | `dials` (CPU) | `dials_gpu` (GPU) | `mxi` (CPU 20T) | `mxi --gpu` (Active / Total w/ Cold Init) |
| :--- | ---: | ---: | ---: | ---: |
| **Spotfinding (`find_spots` / `find`)** | `3,905.9 ms` (`3.71 s` 1T) | **`50.8 ms`** warm (`342.4 ms` cold avg) | `472.0–593.0 ms` | `469.0–550.4 ms` (`2,327 ms` w/ CUDA init) |
| **Indexing (`index`, `fft3d`)** | `2,022.7 ms` | **`565.0 ms`** (`217.3 ms` RL fast) | `1,844.0–2,097.4 ms` | *(CPU only)* |
| **Refinement (`refine`)** | `2,671.1 ms` (`710.6 ms` static) | **`164.6–188.1 ms`** (`64.0 ms` static) | **`15.2–23.0 ms`** (static 17p) | *(CPU only)* |
| **Integration (`integrate`)** | `2,687.1 ms` | **`91.1–110.3 ms`** (`47.0 ms` warm) | `164.0–183.0 ms` | `165.0–313.0 ms` (`102 ms` GPU prf vs `18 ms` CPU) |
| **Scaling (`scale`, 5 sweeps / 1 sweep)** | `825.2 ms` / `202.2 ms` | `240.6 ms` / **`47.2 ms`** | **`85.8–135.8 ms`** / **`13.0 ms`** | *(CPU only)* |

#### Dataset 2: `C2sum_1` (720-Frame Pilatus 6M Rotation Sweep)

| Pipeline Stage | `dials` (CPU) | `dials_gpu` (GPU) | `mxi` (CPU 20T) | `mxi --gpu` (Active / Total w/ Cold Init) |
| :--- | ---: | ---: | ---: | ---: |
| **Spotfinding (`112,023` spots)** | `23.0 s` (20T) / `32.4 s` (8T) | **`2.98 s` warm** (`4.14 s` first call) | `12.171 s` (`12.089 s` active) | `6.780 s` active (`8.695 s` total) |
| **Indexing (`~112k` spots)** | `47.3 s` (`23.6 s` `max_refine=5`) | **`5.98 s`** (`max_refine=5`) / `22.2 s` (default) | **`7.629 s`** | *(CPU only)* |
| **Refinement (`~111k` reflections)** | `23.4 s` (`4.15 s` static) | **`1.22 s`** (static) / **`3.54 s`** (SV) | **`2.051 s`** (`0.223 s` static + `1.652 s` SV) | *(CPU only)* |
| **Integration (`~325k` reflections)** | `210.2 s` (`nproc=1`) / `42.2 s` (`nproc=20`) | **`6.35 s`** (single-pass circular slab + C++; needs `dynamic_shadowing=False`, see §5b) | `7.008 s` (`3.008 s` CPU prf) | `5.206 s` active (`7.602 s` total; `1.488 s` GPU prf) |
| **Scaling (`~325k` obs, `~49k` unique)** | `15.3 s` | **`1.30 s` warm** (`1.35 s` first call; **`11.8×`** vs CPU) | `1.555 s` (`0.625 s` fit + `0.564 s` sym) | *(CPU only)* |

The `dials` and `dials_gpu` columns were re-measured on 2026-11-09 (DIALS `3.dev.1556`, in-process, same machine: i9-13900H, 20 CPUs, RTX 2000 Ada Laptop 8 GB) with the in-process method of [`benchmarks/compare_stages.py`](benchmarks/compare_stages.py); see [§9](#9-reproducing-the-dials-vs-dials_gpu-vs-mxi-comparison). The `mxi` columns are from the original run and were not re-measured.

---

### 7b. Numerical Closeness to Standard CPU `dials`

| Stage & Metric | `dials_gpu` vs `dials` (CPU) | `mxi` (CPU) vs `dials` (CPU) | `mxi --gpu` vs `mxi` (CPU) |
| :--- | :--- | :--- | :--- |
| **`i19` Spotfinding (`578` spots on `w00017`)** | **100.00% match**, RMSD `(0.0000, 0.0000, 0.0000)` | **100.00% match**, RMSD `(0.0000, 0.0000, 0.0000)` | **100.00% match**, identical (`578` spots) |
| **`C2sum_1` Spotfinding (`112,023` spots)** | All `112,023` spots match, RMSD `(0.0000, 0.0000, 0.0000)`; DIALS ≥ `3.dev.1556` finds `112,041` (`99.98%` matched): its union-find spot labelling (dials commit `09ffd9c34`) splits 18 more spots | **100.00% match**, RMSD `(0.0000, 0.0000, 0.0000)` | **100.00% match**, identical (`112,023` spots) |
| **`C2sum_1` Indexed Cell ($a, b, c, \beta$)** | `53.1716, 61.2469, 69.2901, 93.0460°` ($\Delta < 10^{-4}\text{ \AA}$) | `53.1708, 61.2130, 69.2985, 93.0518°` ($\Delta b = 0.034\text{ \AA}$) | *(CPU only)* |
| **`i19` Integrated Intensities ($r(I_{\text{sum}}), r(I_{\text{prf}})$)** | $r_{\text{sum}} = \mathbf{1.000000}$, $r_{\text{prf}} = \mathbf{1.000000}$ | $r_{\text{sum}} = 0.851116$, $r_{\text{prf}} = 0.869369$ | $r_{\text{sum}} = \mathbf{1.000000}$, $r_{\text{prf}} = \mathbf{0.999998}$ |
| **`C2sum_1` Integrated Intensities ($r(I_{\text{sum}}), r(I_{\text{prf}})$)** | $r_{\text{sum}} = \mathbf{1.000000}$, $r_{\text{prf}} = \mathbf{1.000000}$ (`0.999947` full pipe; re-measured 2026-11-09 full pipe against `dials.integrate` with `nproc=1`: `1.000000` / `1.000000` on `313,788` / `325,314` matched reflections; DIALS with `nproc>1` differs slightly from its own `nproc=1` result, see §9) | $r_{\text{sum}} = \mathbf{0.999624}$, $r_{\text{prf}} = \mathbf{0.999615}$ | $r_{\text{sum}} = \mathbf{1.000000}$, $r_{\text{prf}} = \mathbf{1.000000}$ |

---

### 7c. Overall Post-Scaling Merging Statistics (`iotbx.merging_statistics`)

| Metric | `C2sum_1` `dials` (CPU) | `C2sum_1` `dials_gpu` | `C2sum_1` `mxi` | `i19` (5-Sweep) `dials` | `i19` (5-Sweep) `dials_gpu` | `i19` (1-Sweep `w00017`) `mxi` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Total Observations ($N_{\text{obs}}$)** | `325,124` | `325,140` | `327,859` | `3,900` | `3,900` | `977` |
| **Unique Reflections ($N_{\text{uniq}}$)** | `49,115` | `49,118` | `48,585` | `3,418` | `3,418` | `970` |
| **Multiplicity** | `6.62` | `6.62` | `6.75` | `1.14` | `1.14` | `1.01` |
| **Completeness (%)** | `73.0%` | `73.0%` | `73.0%` | `18.0%` | `18.0%` | `5.3%` |
| **$\langle I/\sigma(I)\rangle$** | `11.03` | `11.02` | `11.15` | `2.70` | `2.70` | `2.82` |
| **$R_{\text{merge}}$** | **`0.0616`** | **`0.0617`** | `0.0697` | **`0.2266`** | **`0.2266`** | `0.0239` |
| **$R_{\text{meas}}$** | **`0.0670`** | **`0.0670`** | `0.0756` | **`0.3193`** | **`0.3192`** | `0.0337` |
| **$R_{\text{pim}}$** | **`0.0259`** | **`0.0259`** | `0.0290` | **`0.2247`** | **`0.2247`** | `0.0239` |
| **$CC_{1/2}$** | **`0.9974`** | **`0.9975`** | `0.9965` | **`0.8840`** | **`0.8841`** | `0.9986` |

These statistics come from `compute_merging_stats` in `benchmark_mxi_comparison.py` (profile intensities with its own filtering). [`compare_stages.py`](benchmarks/compare_stages.py) uses DIALS's `scaled_data_as_miller_array` filtering instead, so its absolute values differ; on the 2026-11-09 `C2sum_1` run it gives identical statistics for both implementations (CC½ `0.9991`, R_merge `0.0658`, completeness `73.0%`, multiplicity `6.62`, ⟨I/σ⟩ `16.81` vs `16.82`).

---

## 8. When Using a GPU Is (and Is Not) Recommended

Profiling both `dials_gpu` and `mxi --gpu` against multi-threaded CPU baselines highlights six concrete regimes where **offloading to the GPU is not recommended** (or requires size-based CPU fallbacks):

1. **Small-Sweep Profile Fitting with Host-Side Shoebox Packing (`mxi integrate --gpu` on 25-image `i19` sweeps, $N \approx 1,000$ reflections)**:
   - On a 25-frame `i19` sweep (`1,045` predicted reflections), `mxi integrate` on CPU (20 threads) completes profile fitting in **`18 ms`** (`183 ms` total), whereas `mxi integrate --gpu` takes **`102 ms`** for profile fitting (`313 ms` total) — **5.7× slower on GPU**.
   - *Cause*: `mxi integrate --gpu` extracts shoeboxes and fits backgrounds on the CPU, then packs `1,045` variable-sized shoeboxes into padded `float32` slabs on the host (`94 ms`) and transfers `60 MB` over PCIe (`6 ms`) to run a **`4 ms`** CUDA kernel. Even on 720-frame `C2sum_1` (`360,892` shoeboxes), host-side reference preparation (`630 ms`) + CPU packing of `4.96 GB` (`683 ms`) + PCIe transfer (`627 ms`) consumes `1.94 s` of overhead to feed a `283 ms` GPU kernel.
2. **Per-Thread CUDA Allocation Across Many Worker Threads on Small Sweeps (`mxi find --gpu` on 25-image `i19` sweeps)**:
   - On a 25-frame `i19` sweep with 20 threads (`-j 20`), `mxi find --gpu` allocates a separate `thread_local` CUDA `Workspace` (`~27 MB` across 4 `cudaMalloc` calls + `cudaStreamCreate` per thread = `540 MB` total) for just 1–2 frames per thread, spending `2.727 thread-s` waiting on CUDA driver locks (`469 ms` wall, or `2,327 ms` with cold CUDA context init) vs `472 ms` on CPU.
   - *Contrast*: `dials_gpu.find_spots` processes all 25 frames in a single batched 3D tensor `(25, H, W)` on one stream in **`50.8 ms`** (**9.2× faster** than `mxi find --gpu`).
3. **Cold CLI Subprocess Invocations on Small Datasets**:
   - Initializing a new CUDA context (`cudaFree(0)` / `torch.cuda.init()`) costs **`1.5–2.4 s`** per OS process. For any stage that completes in `< 1 s` on CPU (such as single-sweep `refine`, `scale`, or 25-image `find`/`integrate`), spawning a fresh GPU subprocess per file is dominated by CUDA driver initialization unless a persistent in-process worker pool is used.
4. **Small-$N$ Iterative Optimizers ($N < 4,000$ Reflections in L-BFGS / LevMar)**:
   - Each L-BFGS or Levenberg-Marquardt iteration launches ~20–30 small elementwise and `scatter_add_` kernels (`~5–10 µs` launch latency each). Below ~4,000 reflections, CPU SIMD operating in L1/L2 cache is faster per iteration than GPU kernel launch overhead (`dials_gpu` therefore enforces size thresholds such as `gpu_threshold=4096` in `AssignIndicesGlobalGPU` and `numel >= 8192` in `sum_in_groups_gpu`).
5. **Small Normal-Equation Linear Solves ($P < 100$ Parameters) & Variable Projection**:
   - Once the normal matrix $J^T W J$ ($P \times P$, where $P = 11\dots 70$) is accumulated, solving the tiny symmetric system on CPU via Cholesky/SVD takes `< 0.2 ms`, whereas transferring to/from `cuSOLVER` adds `0.5–1.0 ms` latency. Furthermore, `mxi scale` eliminates $I_h$ analytically via variable projection on the CPU, solving a compact $P \times P$ Levenberg-Marquardt system in `6 ms` (1 sweep) to `625 ms` (328k reflections) without needing GPU acceleration.

6. **Integration of Large Sweeps with a Dynamic (Goniometer Shadow) Mask**:
   - `dials_gpu.integrate`'s slabbed path for single-panel sweeps of more than 64 images does not support dynamic masks. On 720-frame `C2sum_1` imported with the default `dynamic_shadowing=Auto`, integration takes `82–97 s` on the GPU vs `42 s` for `dials.integrate` on 20 cores. Import with `dynamic_shadowing=False` when shadowing is not needed (`6.4 s`).

---

## 9. Reproducing the `dials` vs `dials_gpu` vs `mxi` Comparison

[`benchmarks/compare_stages.py`](benchmarks/compare_stages.py) repeats the stage-by-stage comparison above on any machine, for one or more imported `.expt` files. For each file it runs `find_spots → index → refine → integrate → scale` with each of:

- **`dials`** and **`dials_gpu`**: in-process (Python API, like the tables above), each in its own worker process;
- **[`mxi`](https://github.com/APS-24ID/mxi)** (optional): one `mxi <step>` command per stage, reading and writing the same DIALS `.expt` / `.refl` files, on the CPU and, for spot finding and integration, with `--gpu` when the build has CUDA.

It also times static refinement and, optionally, single-process `dials` integration. **mxi reads NXmx (HDF5) images only**: for CBF data such as `C2sum_1` it stops at spot finding ("reads NXmx only"), which the report shows as `FAILED`. To include mxi, convert the CBF images to NXmx (see [Converting CBF to NXmx](#converting-cbf-to-nxmx-for-mxi)).

### Building mxi

mxi is standalone C++ (no DIALS needed); it needs CMake ≥ 3.20, a C++20 compiler, HDF5's C library, and for `--gpu` the CUDA toolkit (≥ 11.6). The DIALS conda environment provides HDF5:

```bash
git clone https://github.com/APS-24ID/mxi.git && cd mxi
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DMXI_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native \
      -DHDF5_ROOT=/path/to/dials/conda_base          # or a system libhdf5-dev
cmake --build build -j
ctest --test-dir build                               # optional, ~2 min
build/mxi find --version                             # "mxi find 1.2.0 (CUDA)"
```

Add `-DMXI_FFTW=ON` (with FFTW installed) for mxi's faster indexing transform, as its release builds use; without it mxi uses a built-in radix-2 FFT. The reference results below used mxi 1.2.0 with CUDA and without FFTW.

### Converting CBF to NXmx (for mxi)

[cbf2nxmx](https://github.com/graeme-winter/cbf2nxmx) converts CBFs with full imgCIF headers, but not mini-CBF files such as `C2sum_1`'s (Pilatus header only: "no _axis table"). For those, write NXmx with DIALS' own `dxtbx.any2nexus` from a DIALS import, then repack the images as bitshuffle-LZ4, one frame per chunk, which mxi reads ([`benchmarks/repack_nxmx_bslz4.py`](benchmarks/repack_nxmx_bslz4.py)):

```bash
dials.import /data/C2sum_1/C2sum_1_*.cbf.gz dynamic_shadowing=False output.experiments=c2sum/imported.expt
dxtbx.any2nexus c2sum/imported.expt output_file=raw.nxs compression=None dtype=int32 \
    nexus_details.instrument_name="DLS PILATUS 6M S/N 60-0100" nexus_details.source_name="Diamond Light Source" \
    nexus_details.start_time=2016-05-12T15:55:57.904 nexus_details.end_time_estimated=2016-05-12T16:01:57.351 \
    nexus_details.sample_name=C2sum_1                                  # ~18 GB, uncompressed
python3 benchmarks/repack_nxmx_bslz4.py raw.nxs C2sum_1.nxs             # ~1.2 GB; raw.nxs can then go
dials.import C2sum_1.nxs output.experiments=c2sum_nxs/imported.expt     # the input for all three
```

(The `nexus_details` values are descriptive metadata `any2nexus` requires, taken from the CBF headers.) For `C2sum_1` this gives the same beam, detector, goniometer, scan, trusted range and mask in DIALS as the CBF import, and identical pixel values.

### Running the comparison

```bash
source /path/to/dials/dials
./build.sh                                   # build and install dials_gpu against this DIALS

# Import with DIALS (an NXmx file for all three; CBF for dials and dials_gpu only)
dials.import /data/i19/301_NiD_00001_w00019.nxs output.experiments=i19_w19/imported.expt
dials.import /data/C2sum_1/C2sum_1_*.cbf.gz dynamic_shadowing=False output.experiments=c2sum/imported.expt

# Compare (several .expt files can be given; each is reported separately)
libtbx.python benchmarks/compare_stages.py i19_w19/imported.expt --mxi /path/to/mxi/build/mxi \
    --space-group P21/n --unit-cell 11.39,13.30,15.82,90,99.02,90 --cpu-integrate-nproc1 --workdir compare_i19
libtbx.python benchmarks/compare_stages.py c2sum/imported.expt --implementations dials dials_gpu \
    --space-group C2 --unit-cell 53.17,61.25,69.29,90,93.04,90 --index-phil "max_refine=5" \
    --cpu-integrate-nproc1 --workdir compare_c2sum
```

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `--implementations ...` | all three | any of `dials dials_gpu mxi`; by default mxi is included when it is found |
| `--mxi PATH` | `$MXI`, or `mxi` on the `PATH` | the mxi program |
| `--space-group SG`, `--unit-cell a,b,c,α,β,γ` | none | known symmetry for **all** implementations: `dials` `space_group=`/`unit_cell=`; mxi `index --known-cell --known-space-group` and `scale --space-group`. Without them everything runs in P1, where mxi's scaling has nothing to fit (reported `INVALID`) |
| `--nproc N` | all CPUs | processes for `dials` spot finding and integration |
| `--cpu-integrate-nproc1` | off | also run `dials` integration with `nproc=1` (~3.5 min on `C2sum_1`): times it, and uses it as the like-for-like reference for the integration closeness checks (**recommended**) |
| `--gpu-repeats N` / `--cpu-repeats N` / `--mxi-repeats N` | `2` / `1` / `2` | runs per stage; `dials_gpu` reports its first call and its best warm call, mxi its best run |
| `--mxi-threads N` | all CPUs | mxi `-j` |
| `--no-mxi-gpu` | off | skip mxi's `--gpu` spot finding and integration |
| `--find-phil`, `--index-phil`, `--refine-phil`, `--integrate-phil`, `--scale-phil` | none | extra PHIL for `dials` and `dials_gpu`, e.g. `"max_refine=5"` |
| `--mxi-find-args`, `--mxi-index-args`, `--mxi-refine-args`, `--mxi-integrate-args`, `--mxi-scale-args` | none | extra arguments for that mxi command |
| `--timeout S` | `14400` | time limit for each implementation on each dataset |

Output: `<workdir>/report.md` (timing table, closeness checks, merging statistics, and any failures with their tracebacks or logs), `<workdir>/results.json`, and each stage's outputs and logs under `<workdir>/<dataset>/{dials,dials_gpu,mxi}/`.

How to read it:
- **Stage times exclude Python/CUDA process start-up.** Command-line runs are slower by roughly 1.5–4 s per `dials_gpu` command; [`benchmarks/benchmark_pipeline.py`](benchmarks/benchmark_pipeline.py) measures the same pipeline through the command-line tools.
- **"Warm" `dials_gpu` calls repeat the same data in the same process** and can reuse in-process caches; for processing one new dataset, the first-call time is the conservative figure.
- **mxi times are the `total` each mxi command reports with `--timing`** (its own work, including reading and writing the `.expt` / `.refl` files), best of its runs; the command's wall-clock time, which adds process start-up and, for `--gpu`, CUDA initialisation, is shown in brackets. mxi's speed-up uses the faster of its CPU and GPU runs.
- **Failures are never reported as times.** A stage that raises or exits non-zero is `FAILED` (with the error or log), a worker or command killed by a signal is `CRASHED`, one that exceeds `--timeout` is `TIMEOUT`, and later stages are `SKIPPED`. A stage that exits cleanly but leaves no usable result is `INVALID`: no strong spots, no indexed reflections or crystal, no reflections used in refinement, nothing integrated, or scale factors that are non-finite or all exactly 1. (On a P1 run of the i19 sweep, `dials.scale` returned normally with every scale factor `NaN`, and mxi's scaling, with no symmetry equivalents, left them all at 1: both are reported `INVALID`, not as fast times.) The script exits with code 1 if any stage did not succeed.
- **Closeness checks** (spot matching, unit cells, RMSDs, intensity correlations, merging statistics) flag a `dials_gpu` result outside the tolerances in `benchmark_pipeline.py` as `DIVERGENT`. mxi is an independent implementation, so its comparison with `dials` is shown for information, without verdicts.
- **Integration closeness is best judged against `dials` with `nproc=1`.** With `nproc>1`, DIALS splits the sweep into jobs and treats reflections near job boundaries differently, so its own `nproc=20` and `nproc=1` results differ (`r = 0.9987` for summation intensities on `C2sum_1`); `dials_gpu` reproduces `nproc=1`. With `--cpu-integrate-nproc1` the report compares against `nproc=1` and shows the `nproc=N` comparison and DIALS's own `nproc=N` vs `nproc=1` difference for reference; without it, a low correlation against `nproc=N` alone is not evidence of a `dials_gpu` problem.
- **Dynamic masks are flagged** in the report, since they disable the fast integration path.

Three-way reference result for `C2sum_1` (720 images; 2026-11-10; i9-13900H, 20 CPUs; RTX 2000 Ada Laptop GPU 8 GB; DIALS `3.dev.1573`; mxi 1.2.0 with CUDA, without FFTW; space group C2 with the known cell; `max_refine=5` for `dials` and `dials_gpu`; `--cpu-integrate-nproc1`). `dials` and mxi read the NXmx conversion above; **`dials_gpu` figures are from the original CBF images** (the run below), because `dials_gpu.integrate` currently produces no profile-fitted reflections on this NXmx file (summation intensities are correct) and its scaling then fails. The images and geometry are identical, but `dials_gpu`'s times therefore include CBF decompression instead of HDF5 reading:

| Stage | `dials` | `dials_gpu` warm (first call) | mxi CPU | mxi GPU | `dials_gpu` speed-up | mxi speed-up (best) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Spot finding | `26.28 s` | **`2.88 s`** (`4.06 s`) | `13.24 s` (`13.28 s`) | `6.74 s` (`6.86 s`) | `9.1×` | `3.9×` |
| Indexing | `22.23 s` | `6.00 s` (`5.88 s`) | **`5.28 s`** (`5.33 s`) | – | `3.7×` | `4.2×` |
| Refinement, static | `4.60 s` | `1.25 s` (`1.32 s`) | **`1.20 s`** (`1.27 s`) | – | `3.7×` | `3.8×` |
| Refinement, scan-varying | `25.08 s` | `3.60 s` (`3.54 s`) | **`2.12 s`** (`2.17 s`) | – | `7.0×` | `11.8×` |
| Integration (`dials` `nproc=1`) | `195.48 s` | `6.55 s` (`6.31 s`) | `7.32 s` (`7.51 s`) | **`5.92 s`** (`6.25 s`) | `29.8×` | `33.0×` |
| Integration (`dials` `nproc=20`) | `44.64 s` | `6.55 s` (`6.31 s`) | `7.32 s` (`7.51 s`) | **`5.92 s`** (`6.25 s`) | `6.8×` | `7.5×` |
| Scaling | `15.80 s` | `1.33 s` (`1.46 s`) | **`1.04 s`** (`1.12 s`) | – | `11.9×` | `15.2×` |

mxi times are its `--timing` totals, with the command's wall-clock time in brackets. Closeness to `dials`: `dials_gpu` (on CBF) is identical at every stage (integration `r = 1.00000` against `dials` `nproc=1`, identical merging statistics). mxi finds the same 112,023 spots (`dials` finds 18 more, see §7b), agrees on the cell to 0.05% after refinement (it indexes in the primitive setting), with RMSDs 0.264 / 0.239 px, 0.105° vs 0.236 / 0.231 px, 0.102°, and integrated intensities `r = 0.9997` (summation and profile fitting) against `dials` `nproc=1`; its scaled data differ more (CC½ `0.957` vs `0.999`, ⟨I/σ⟩ `3.0` vs `16.8`, R_merge `0.296` vs `0.066`, from 155k vs 325k observations), which has not been investigated.

Reference result for `C2sum_1`, `dials` vs `dials_gpu` on the CBF images: the output of the second command above, run before mxi support was added (2026-11-10; i9-13900H, 20 CPUs; RTX 2000 Ada Laptop GPU 8 GB; DIALS `3.dev.1573`; torch `2.12.0+cu130`; `C2sum_1` 720 images imported with `dynamic_shadowing=False`):

| Stage | `dials` | `dials_gpu` first call | `dials_gpu` warm | Speed-up (warm) |
| :--- | ---: | ---: | ---: | ---: |
| Spot finding (`dials` `nproc=20`) | `22.17 s` | `4.06 s` | `2.88 s` | `7.69×` |
| Indexing (`space_group=C2`, `max_refine=5`) | `22.71 s` | `5.88 s` | `6.00 s` | `3.79×` |
| Refinement, static | `4.17 s` | `1.32 s` | `1.25 s` | `3.33×` |
| Refinement, `scan_varying=Auto` | `22.93 s` | `3.54 s` | `3.60 s` | `6.36×` |
| Integration (`dials` `nproc=20`) | `44.45 s` | `6.31 s` | `6.55 s` | `6.79×` |
| Integration (`dials` `nproc=1`) | `208.72 s` | `6.31 s` | `6.55 s` | `31.88×` |
| Scaling | `14.62 s` | `1.46 s` | `1.33 s` | `10.96×` |

Closeness on this run:
- **Spot finding:** all 112,023 `dials_gpu` spots match `dials` spots with zero centroid difference; `dials` finds 18 more (see §7b).
- **Indexing / refinement:** 111,251 vs 111,263 reflections indexed; unit cells within 0.001%; identical refined RMSDs (x, y, φ = 0.236 px, 0.231 px, 0.102°).
- **Integration vs `dials` `nproc=1`:** `r = 1.00000` for summation (313,792 vs 313,796) and profile fitting (325,316 vs 325,319). Against `dials` `nproc=20`, `r = 0.973` / `0.984`, driven by a few hundred reflections that DIALS's job splitting integrates differently (DIALS's own `nproc=20` vs `nproc=1`: `r = 0.9987` / `0.9997`).
- **Scaling:** scaled intensities `r = 1.00000` on 325,181 matched reflections; merging statistics CC½ `0.9990` vs `0.9991`, R_merge `0.0658`, completeness `73.0%`, multiplicity `6.62`, ⟨I/σ⟩ `16.82` for both.

Reference result for the i19 sweep `301_NiD_00001_w00019` (25 images, Eiger 2X 4M NXmx), from the first command above (2026-11-10; same machine; DIALS `3.dev.1573`; mxi 1.2.0 with CUDA, without FFTW). On a dataset this small the GPU has little work, the in-process `dials_gpu` warm times reuse caches, and every mxi time includes reading and writing its files, so treat these as a check that all three run and agree rather than as throughput figures:

| Stage | `dials` | `dials_gpu` warm (first) | mxi CPU | mxi GPU |
| :--- | ---: | ---: | ---: | ---: |
| Spot finding | `1.51 s` | `0.07 s` (`0.78 s`) | `0.55 s` | `0.52 s` |
| Indexing | `3.38 s` | `2.02 s` (`2.17 s`) | `1.85 s` | – |
| Refinement, static | `0.35 s` | `0.07 s` (`0.09 s`) | `0.01 s` | – |
| Refinement, default | `0.73 s` | `0.17 s` (`0.17 s`) | `0.01 s` | – |
| Integration (`dials` `nproc=1`) | `2.13 s` | `0.06 s` (`0.09 s`) | `0.14 s` | `0.27 s` |
| Scaling (`P 1 21/n 1`) | `0.23 s` | `0.09 s` (`0.09 s`) | `0.07 s` | – |

Closeness to `dials`: `dials_gpu` is identical at every stage (446 spots, 192 indexed, the same refined RMSDs, integrated intensities `r = 1.00000` against `dials` `nproc=1`, scaled `r = 0.99987`, CC½ `0.9640` both). mxi finds the same 446 spots and agrees on the cell to 0.17%, with similar refined RMSDs (0.265 / 0.241 px, 0.049° vs 0.273 / 0.277 px, 0.044°); it integrates fewer reflections (973 vs 1,363 by summation; `r = 0.9997` on those matched) and its scaled data have CC½ `0.920` vs `0.964` and ⟨I/σ⟩ `1.19` vs `2.11`.
