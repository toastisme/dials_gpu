# dials_gpu

GPU-accelerated implementations of [DIALS](https://dials.github.io/) crystallographic data-processing algorithms using PyTorch (`torch`).

## Command-Line Tools

- `dials_gpu.find_spots`: Drop-in GPU replacement for `dials.find_spots` supporting all standard DIALS PHIL parameters and spot-finding algorithms (`dispersion_extended`, `dispersion`, `radial_profile`, 2D/3D connected component labelling, centroiding, intensity summation, shoebox extraction, and spot filtering).
- `dials_gpu.index`: Drop-in GPU replacement for `dials.index` supporting all standard DIALS PHIL parameters and indexing algorithms (`fft3d`, `fft1d`, `real_space_grid_search`, `low_res_spot_match`, `find_max_cell`, `optimise_basis_vectors`, `candidate_orientation_matrices`, `AssignIndicesGlobal` / `AssignIndicesLocal`, `non_primitive_basis`, rotation sequences, stills, and multi-lattice indexing), with all internal candidate orientation evaluation (`choose_best_orientation_matrix` / `ModelEvaluation`) and macro-cycle refinement (`Indexer.refine`) mapped to the GPU.
- `dials_gpu.refine`: Drop-in GPU replacement for `dials.refine` supporting all standard DIALS PHIL parameters and refinement algorithms (scan-static and scan-varying refinement, stills and spherical-relp stills refinement, single/multi-panel and hierarchical detectors, `mcd` / `tukey` / `sauter_poon` GPU centroid outlier rejection, `LevMar` / `GaussNewton` / `SimpleLBFGS` / `LBFGScurvs` / `SparseLevMar` engines, restraints, constraints, and `AutoReduce`).
- `dials_gpu.integrate` (and alias `dials_gpu.interate`): Drop-in GPU replacement for `dials.integrate` supporting all standard DIALS PHIL parameters and integration algorithms (`3d`, `flat3d`, `2d`, `single2d`, `stills`, `3d_threaded` integrators; `glm` Cantoni & Ronchetti robust Poisson IRLS background, `simple` `null`/`nsigma`/`tukey`/`truncated` + `constant2d`/`constant3d`/`linear2d`/`linear3d` background, `auto` / `null` background; `simple` 3D shoebox centroiding with parallax correction; 3D/2D summation integration; `gaussian_rs` scan-static and scan-varying profile modelling and fitting; Lorentz-polarization and QE corrections; `SignificanceFilter`, `kapton_2019` absorption correction, and `frame_slice_shoeboxes`).
- `dials_gpu.scale`: Drop-in GPU replacement for `dials.scale` supporting all standard DIALS PHIL parameters and scaling algorithms (`physical`, `KB`, `array`, and `dose_decay` scaling models; 1D/2D/3D Gaussian smoothers, spherical harmonic `SHScaleComponent` absorption surface, `SingleScaleFactor`, `SingleBScaleFactor`, `QuadraticBScaleComponent`; `RefinerCalculatorGPU` multi-component product rule; `IhTableGPU` vectorized symmetry group blocking and `scatter_add_` `calc_Ih`; `ScalingTargetGPU` and `ScalingTargetFixedIHGPU` analytical L-BFGS gradients and dense Gauss-Newton / Levenberg-Marquardt Jacobians; `NormDevOutlierRejectionGPU`, `SimpleNormDevOutlierRejectionGPU`, and `TargetedOutlierRejectionGPU`; `SingleDatasetIntensityCombinerGPU` and `MultiDatasetIntensityCombinerGPU`; `ErrorModelRefineryGPU` and `ErrorModelTargetRegressionGPU`; `SingleScalerGPU`, `MultiScalerGPU`, `TargetScalerGPU`, `ScaleAndFilterAlgorithmGPU`, and cross-validation).

## Usage

Activate the DIALS environment and run `dials_gpu.find_spots`, `dials_gpu.index`, `dials_gpu.refine`, `dials_gpu.integrate`, and `dials_gpu.scale` with the exact same arguments as `dials.find_spots`, `dials.index`, `dials.refine`, `dials.integrate`, and `dials.scale`:

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
  3. **GPU Compute Stage**: Batched Triton `DispersionExtended` thresholding + `torch.nonzero` + `torch.bincount` on the default compute stream, followed by single-pass GPU 3D connected-components (`35 ms`) and C++ `Shoebox` construction (`36 ms`), completing all 720 frames of `C2sum_1` (`112,023` spots) in **`2.832 s` warm (`4.061 s` cold)**.

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
| **`C2sum_1` (`112,023` refl, `scan_varying=False` static)** | `8,420.0 ms` | **`1,435.0 ms`** | **`5.87×`** | `< 1e-14` |
| **`C2sum_1` (`112,023` refl, `scan_varying=Auto` 2-macrocycle)** | `25,554.0 ms` | **`3,878.9 ms`** | **`6.59×`** | `< 1e-14` |
| Target Analytical Gradients + GPU Normal Eqns (`112,023` refl, `sv=False`) | `20.11 ms/step` | **`9.80 ms/step`** | **`2.05×`** | `< 1e-15` |
| Target Analytical Gradients + GPU Normal Eqns (`112,023` refl, `sv=True`) | `1,150.0 ms/step` | **`25.60 ms/step`** | **`44.9×`** | `< 1e-15` |

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
| **`i19` Default (`3D + GLM + Profile Fitting`, `nproc=1` vs GPU)** | `2,838.3 ms/file` | **`286.2 ms/file`** | **`9.44×`** | `13,587` (CPU) / `13,587` (GPU) |
| **`i19` Default (`3D + GLM + Profile Fitting`, `nproc=4` vs GPU)** | `2,820.1 ms/file` | **`286.2 ms/file`** | **`9.85×`** | `13,680` (CPU) / `13,587` (GPU) |
| **`i19` Summation Only (`3D + GLM, profile.fitting=False`)** | `1,485.8 ms/file` | **`154.1 ms/file`** | **`9.64×`** | `13,367` (CPU) / `13,367` (GPU) |
| **`C2sum_1` Default (`3D + GLM + Profile Fitting`, 720 frames)** | `233.802 s` | **`12.439 s`** | **`18.80×`** | `326,994` (CPU) / `326,994` (GPU) |
| **`C2sum_1` Summation Only (`3D + GLM, profile.fitting=False`)** | `118.420 s` | **`7.285 s`** | **`16.26×`** | `313,792` (CPU) / `313,792` (GPU) |

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

#### 6b. End-to-End Scaling across 10 `i19` 25-image `.nxs` Sweeps (`301_Nidppe_f3_phi0`, 13,242 Reflections)

| Scaling Configuration | CPU (`dials.scale`) | GPU (`dials_gpu.scale`) | Speedup vs CPU | Mean Relative Scale Diff |
| :--- | ---: | ---: | ---: | ---: |
| **Single-file loop (`dose_decay, SimpleLBFGS, full_mat=True`)** | `203.68 ms/file` | **`66.12 ms/file`** | **`3.08×`** | `6.11e-08` |
| **Single-file loop (`physical, SimpleLBFGS, full_mat=False`)** | `213.90 ms/file` | **`76.30 ms/file`** | **`2.80×`** | `8.11e-08` |
| **Single-file loop (`physical, SimpleLBFGS, full_mat=True`)** | `224.23 ms/file` | **`84.47 ms/file`** | **`2.65×`** | `7.34e-08` |
| **Joint 10-sweep (`dose_decay, full_mat=False`)** | `1,799.31 ms` | **`676.81 ms`** | **`2.66×`** | `4.29e-02` |
| **Joint 5-sweep (`dose_decay, full_mat=False`)** | `617.47 ms` | **`240.59 ms`** | **`2.57×`** | `2.40e-09` |
| **Joint 5-sweep (`KB, full_mat=False`)** | `468.92 ms` | **`189.06 ms`** | **`2.48×`** | `9.09e-15` |
| **Joint 10-sweep (`KB, full_mat=False`)** | `1,016.42 ms` | **`485.53 ms`** | **`2.09×`** | `5.65e-07` |
| **Joint 10-sweep (`physical, full_mat=False`)** | `1,384.96 ms` | **`735.45 ms`** | **`1.88×`** | `4.06e-02` |
| **Joint 10-sweep (`physical, full_mat=True`)** | `1,298.18 ms` | **`698.69 ms`** | **`1.86×`** | `4.26e-02` |

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
| **Integration (`integrate`)** | `2,838.3 ms` | **`286.2–314.8 ms`** (`154.1 ms` sum) | **`164.0–183.0 ms`** | `165.0–313.0 ms` (`102 ms` GPU prf vs `18 ms` CPU) |
| **Scaling (`scale`, 5 sweeps / 1 sweep)** | `825.2 ms` / `175.8 ms` | `321.3 ms` / **`66.1 ms`** | **`135.8 ms`** / **`13.0 ms`** | *(CPU only)* |

#### Dataset 2: `C2sum_1` (720-Frame Pilatus 6M Rotation Sweep)

| Pipeline Stage | `dials` (CPU) | `dials_gpu` (GPU) | `mxi` (CPU 20T) | `mxi --gpu` (Active / Total w/ Cold Init) |
| :--- | ---: | ---: | ---: | ---: |
| **Spotfinding (`112,023` spots)** | `18.146 s` (20T) / `36.408 s` (8T) | **`2.832 s` warm** (`3.479 s` cold) | `12.171 s` (`12.089 s` active) | `6.780 s` active (`8.695 s` total) |
| **Indexing (`112,023` spots)** | `49.336 s` (`19.003 s` `max_refine=5`) | **`10.761 s`** (`max_refine=5`) | **`7.629 s`** | *(CPU only)* |
| **Refinement (`~100k` reflections)** | `25.554 s` (`8.420 s` static) | **`1.435 s`** (static) / **`3.879 s`** (SV) | **`2.051 s`** (`0.223 s` static + `1.652 s` SV) | *(CPU only)* |
| **Integration (`~327k` reflections)** | `233.802 s` | **`12.439 s`** (`8.05 s` Pass 2; **`7.285 s`** sum-only) | **`7.008 s`** (`3.008 s` CPU prf) | **`5.206 s`** active (`7.602 s` total; `1.488 s` GPU prf) |
| **Scaling (`~327k` obs, `~49k` unique)** | `15.404 s` | **`3.609 s`** (`4.27×` vs `dials.scale`) | **`1.555 s`** (`0.625 s` fit + `0.564 s` sym) | *(CPU only)* |

---

### 7b. Numerical Closeness to Standard CPU `dials`

| Stage & Metric | `dials_gpu` vs `dials` (CPU) | `mxi` (CPU) vs `dials` (CPU) | `mxi --gpu` vs `mxi` (CPU) |
| :--- | :--- | :--- | :--- |
| **`i19` Spotfinding (`578` spots on `w00017`)** | **100.00% match**, RMSD `(0.0000, 0.0000, 0.0000)` | **100.00% match**, RMSD `(0.0000, 0.0000, 0.0000)` | **100.00% match**, identical (`578` spots) |
| **`C2sum_1` Spotfinding (`112,023` spots)** | **100.00% match**, RMSD `(0.0000, 0.0000, 0.0000)` | **100.00% match**, RMSD `(0.0000, 0.0000, 0.0000)` | **100.00% match**, identical (`112,023` spots) |
| **`C2sum_1` Indexed Cell ($a, b, c, \beta$)** | `53.1716, 61.2469, 69.2901, 93.0460°` ($\Delta < 10^{-4}\text{ \AA}$) | `53.1708, 61.2130, 69.2985, 93.0518°` ($\Delta b = 0.034\text{ \AA}$) | *(CPU only)* |
| **`i19` Integrated Intensities ($r(I_{\text{sum}}), r(I_{\text{prf}})$)** | $r_{\text{sum}} = \mathbf{1.000000}$, $r_{\text{prf}} = \mathbf{1.000000}$ | $r_{\text{sum}} = 0.851116$, $r_{\text{prf}} = 0.869369$ | $r_{\text{sum}} = \mathbf{1.000000}$, $r_{\text{prf}} = \mathbf{0.999998}$ |
| **`C2sum_1` Integrated Intensities ($r(I_{\text{sum}}), r(I_{\text{prf}})$)** | $r_{\text{sum}} = \mathbf{1.000000}$, $r_{\text{prf}} = \mathbf{1.000000}$ (`0.999947` full pipe) | $r_{\text{sum}} = \mathbf{0.999624}$, $r_{\text{prf}} = \mathbf{0.999615}$ | $r_{\text{sum}} = \mathbf{1.000000}$, $r_{\text{prf}} = \mathbf{1.000000}$ |

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

---

## 8. When Using a GPU Is (and Is Not) Recommended

Profiling both `dials_gpu` and `mxi --gpu` against multi-threaded CPU baselines highlights five concrete regimes where **offloading to the GPU is not recommended** (or requires size-based CPU fallbacks):

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

