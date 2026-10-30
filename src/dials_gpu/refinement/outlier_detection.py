from __future__ import annotations

import logging
import math
import random

from dials.array_family import flex
from dxtbx import flumpy
import libtbx
import numpy as np
import torch
from dxtbx.model.experiment_list import ExperimentList
from scitbx.array_family import flex
from scitbx.math import five_number_summary

from dials.algorithms.refinement import DialsRefineConfigError
from dials.algorithms.refinement.outlier_detection.mcd import MCD
from dials.algorithms.refinement.outlier_detection.outlier_base import (
    CentroidOutlier,
    CentroidOutlierFactory,
    phil_scope,
    phil_str,
)
from dials.algorithms.refinement.outlier_detection.sauter_poon import SauterPoon
from dials.algorithms.refinement.outlier_detection.tukey import Tukey
from dials.algorithms.refinement.reflection_manager import (
    BlockCalculator,
    LaueReflectionManager,
    ReflectionManager,
    ReflectionManagerFactory,
    StillsReflectionManager,
    TOFReflectionManager,
)
from dials.algorithms.statistics.fast_mcd import FastMCD, maha_dist_sq
from dials_refinement_helpers_ext import mcd_consistency

logger = logging.getLogger(__name__)


def _batched_means_and_cov_gpu(H: torch.Tensor):
    """Compute sample means, covariance matrices, and determinants for a batch of subsets.

    Args:
        H: (B, h, p) float64 tensor on GPU/CPU.
    Returns:
        T: (B, p) float64 tensor of sample means.
        S: (B, p, p) float64 tensor of sample covariance matrices (divisor h - 1).
        detS: (B,) float64 tensor of determinants.
    """
    h = H.shape[1]
    T = H.mean(dim=1)  # (B, p)
    diff = H - T.unsqueeze(1)  # (B, h, p)
    S = torch.bmm(diff.transpose(1, 2), diff) / (h - 1)  # (B, p, p)
    detS = torch.linalg.det(S)  # (B,)
    return T, S, detS


def _batched_concentration_step_gpu(h: int, data_t: torch.Tensor, T: torch.Tensor, S: torch.Tensor):
    """Perform a batched concentration step for B trials simultaneously.

    Args:
        h: subset size (int)
        data_t: (N, p) float64 tensor of observations
        T: (B, p) float64 tensor of centers
        S: (B, p, p) float64 tensor of covariance matrices
    Returns:
        Hnew: (B, h, p) float64 tensor of the h observations with smallest Mahalanobis distance
    """
    # diff: (B, N, p)
    diff = data_t.unsqueeze(0) - T.unsqueeze(1)
    # Solve S @ X = diff^T -> X = S^{-1} @ diff^T
    # diff.transpose(1, 2): (B, p, N)
    sol = torch.linalg.solve(S, diff.transpose(1, 2))  # (B, p, N)
    d2s = (diff * sol.transpose(1, 2)).sum(dim=-1)  # (B, N)
    # Select h smallest distances per trial
    # Use stable sort to match flex.sort_permutation tie-breaking
    _, idx = torch.topk(d2s, k=h, dim=1, largest=False, sorted=True)  # (B, h)
    # Gather observations: data_t[idx] -> (B, h, p)
    Hnew = data_t[idx]
    return Hnew


class FastMCDGPU(FastMCD):
    """GPU-accelerated FAST-MCD algorithm of Rousseeuw and van Driessen.

    Draws initial random permutations using `scitbx.array_family.flex.random_permutation`
    in the exact same order as CPU `FastMCD` so random seeds produce identical subsets,
    then executes all `n_trials` (500) concentration steps, covariance matrices,
    determinants, and Mahalanobis distances in batched PyTorch `float64` tensors on GPU.
    """

    def __init__(
        self,
        data,
        alpha=0.5,
        max_n_groups=5,
        min_group_size=300,
        n_trials=500,
        k1=2,
        k2=2,
        k3=100,
        device=None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        super().__init__(
            data,
            alpha=alpha,
            max_n_groups=max_n_groups,
            min_group_size=min_group_size,
            n_trials=n_trials,
            k1=k1,
            k2=k2,
            k3=k3,
        )

    def _cols_to_tensor(self, cols) -> torch.Tensor:
        """Convert list of flex.double columns into an (N, p) float64 tensor on device."""
        np_cols = [flumpy.to_numpy(c) for c in cols]
        arr = np.column_stack(np_cols)
        return torch.as_tensor(arr, dtype=torch.float64, device=self._device)

    def _draw_initial_T0_S0_batch(self, n_trials: int, data, data_t: torch.Tensor):
        """Draw `n_trials` initial (p+1)-subsets using `flex.random_permutation` in the exact
        same order as CPU `FastMCD.form_initial_subset`, returning batched `T0` and `S0` tensors.
        """
        n = len(data[0])
        p = self._p
        subset_size = p + 1

        # Draw all n_trials permutations from flex RNG
        perms_np = np.empty((n_trials, subset_size), dtype=np.int64)
        extra_trials = []
        for t in range(n_trials):
            perm = flex.random_permutation(n)
            perm_np = flumpy.to_numpy(perm)
            perms_np[t] = perm_np[:subset_size]
            # Keep full permutation only if degenerate (rare)
            extra_trials.append(perm_np)

        perms_t = torch.as_tensor(perms_np, dtype=torch.long, device=self._device)
        J_batch = data_t[perms_t]  # (n_trials, p+1, p)
        T0, S0, detS0 = _batched_means_and_cov_gpu(J_batch)

        # Handle rare degenerate initial subsets where detS0 <= 0.0
        bad_mask = ~(detS0 > 0.0)
        if bad_mask.any():
            bad_indices = torch.nonzero(bad_mask, as_tuple=False).squeeze(-1).cpu().tolist()
            for t_idx in bad_indices:
                perm_np = extra_trials[t_idx]
                i = 1
                det_val = 0.0
                while not det_val > 0.0:
                    sub_sz = p + 1 + i
                    idx_t = torch.as_tensor(perm_np[:sub_sz], dtype=torch.long, device=self._device)
                    J_single = data_t[idx_t].unsqueeze(0)
                    T_s, S_s, det_s = _batched_means_and_cov_gpu(J_single)
                    det_val = float(det_s[0].item())
                    i += 1
                T0[t_idx] = T_s[0]
                S0[t_idx] = S_s[0]

        return T0, S0

    def small_dataset_estimate(self):
        """Batched GPU implementation of `FastMCD.small_dataset_estimate`."""
        data_t = self._cols_to_tensor(self._data)  # (N, p)
        h = self._h

        # Step 1: Draw n_trials initial (p+1) subsets and take 1 concentration step -> H1
        T0, S0 = self._draw_initial_T0_S0_batch(self._n_trials, self._data, data_t)
        Hcurr = _batched_concentration_step_gpu(h, data_t, T0, S0)
        Tcurr, Scurr, detScurr = _batched_means_and_cov_gpu(Hcurr)

        # Step 2: Perform k1 concentration steps across all n_trials in parallel
        for _ in range(self._k1):
            Hcurr = _batched_concentration_step_gpu(h, data_t, Tcurr, Scurr)
            Tcurr, Scurr, detScurr = _batched_means_and_cov_gpu(Hcurr)

        # Step 3: Select 10 trials with lowest determinant
        n_best = min(10, self._n_trials)
        _, best_idx = torch.topk(detScurr, k=n_best, largest=False, sorted=True)
        T_best = Tcurr[best_idx]  # (10, p)
        S_best = Scurr[best_idx]  # (10, p, p)
        det_best = detScurr[best_idx]  # (10,)

        # Step 4: Take up to k3 concentration steps until convergence
        for _ in range(self._k3):
            H_best = _batched_concentration_step_gpu(h, data_t, T_best, S_best)
            T_new, S_new, det_new = _batched_means_and_cov_gpu(H_best)
            if torch.all(det_new == det_best):
                break
            T_best, S_best, det_best = T_new, S_new, det_new

        # Step 5: Pick minimum determinant among the 10 converged trials
        win_idx = int(torch.argmin(det_best).item())
        Tbest_np = T_best[win_idx].detach().cpu().numpy()
        Sbest_np = S_best[win_idx].detach().cpu().numpy()

        Tbest = flex.double(Tbest_np.tolist())
        Sbest = flex.double(np.ascontiguousarray(Sbest_np))
        return Tbest, Sbest

    def large_dataset_estimate(self):
        """Batched GPU implementation of `FastMCD.large_dataset_estimate`."""
        ngroups = int(self._n / self._min_group_size)
        if ngroups < self._max_n_groups:
            sample_size = self._n
        else:
            ngroups = self._max_n_groups
            sample_size = self._min_group_size * self._max_n_groups

        # Sample the data and split into groups using exact flex RNG calls
        sampled = self.sample_data(self._data, sample_size=sample_size)
        groups = self.split_into_groups(sample=sampled, ngroups=ngroups)

        n_trials = self._n_trials // ngroups
        h_frac = self._h / self._n

        T_group_best = []
        S_group_best = []

        for group in groups:
            h_sub = int(len(group[0]) * h_frac)
            group_t = self._cols_to_tensor(group)
            T0, S0 = self._draw_initial_T0_S0_batch(n_trials, group, group_t)
            Hcurr = _batched_concentration_step_gpu(h_sub, group_t, T0, S0)
            Tcurr, Scurr, detScurr = _batched_means_and_cov_gpu(Hcurr)

            for _ in range(self._k1):
                Hcurr = _batched_concentration_step_gpu(h_sub, group_t, Tcurr, Scurr)
                Tcurr, Scurr, detScurr = _batched_means_and_cov_gpu(Hcurr)

            n_keep = min(10, n_trials)
            _, best_idx = torch.topk(detScurr, k=n_keep, largest=False, sorted=True)
            T_group_best.append(Tcurr[best_idx])
            S_group_best.append(Scurr[best_idx])

        # Merge the 10 best trials from each group -> (10 * ngroups, p)
        Tcurr = torch.cat(T_group_best, dim=0)
        Scurr = torch.cat(S_group_best, dim=0)
        sampled_t = self._cols_to_tensor(sampled)
        h_mrgd = int(sample_size * h_frac)

        for _ in range(self._k2):
            Hcurr = _batched_concentration_step_gpu(h_mrgd, sampled_t, Tcurr, Scurr)
            Tcurr, Scurr, detScurr = _batched_means_and_cov_gpu(Hcurr)

        # Sort merged trials by lowest determinant and work with whole dataset
        size = self._n * self._p
        if size <= 100000:
            k4 = self._k3
        elif size <= 200000:
            k4 = 10
        elif size <= 300000:
            k4 = 9
        elif size <= 400000:
            k4 = 8
        elif size <= 500000:
            k4 = 7
        elif size <= 600000:
            k4 = 6
        elif size <= 700000:
            k4 = 5
        elif size <= 800000:
            k4 = 4
        elif size <= 900000:
            k4 = 3
        elif size <= 1000000:
            k4 = 2
        else:
            k4 = 1

        n_reps = 1 if self._n > 5000 else min(10, Tcurr.shape[0])
        _, best_idx = torch.topk(detScurr, k=n_reps, largest=False, sorted=True)
        T_best = Tcurr[best_idx]
        S_best = Scurr[best_idx]
        det_best = detScurr[best_idx]

        data_t = self._cols_to_tensor(self._data)
        for _ in range(k4):
            H_best = _batched_concentration_step_gpu(self._h, data_t, T_best, S_best)
            T_new, S_new, det_new = _batched_means_and_cov_gpu(H_best)
            if torch.all(det_new == det_best):
                break
            T_best, S_best, det_best = T_new, S_new, det_new

        win_idx = int(torch.argmin(det_best).item())
        Tbest_np = T_best[win_idx].detach().cpu().numpy()
        Sbest_np = S_best[win_idx].detach().cpu().numpy()

        Tbest = flex.double(Tbest_np.tolist())
        Sbest = flex.double(np.ascontiguousarray(Sbest_np))
        return Tbest, Sbest


class MCDGPU(MCD):
    """GPU-accelerated Minimum Covariance Determinant (MCD) outlier detector."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Always run in-process on GPU (avoiding ProcessPoolExecutor fork overhead)
        self.nproc = 1

    def _detect_outliers(self, cols):
        fast_mcd = FastMCDGPU(
            cols,
            alpha=self._alpha,
            max_n_groups=self._max_n_groups,
            min_group_size=self._min_group_size,
            n_trials=self._n_trials,
            k1=self._k1,
            k2=self._k2,
            k3=self._k3,
        )

        # get location and MCD scatter estimate
        T, S = fast_mcd.get_corrected_T_and_S()

        # get squared Mahalanobis distances
        d2s = maha_dist_sq(cols, T, S)

        # compare to the threshold
        outliers = d2s > self._mahasq_cutoff

        return outliers


class TukeyGPU(Tukey):
    """Tukey outlier detector running in-process (nproc=1)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nproc = 1


class SauterPoonGPU(SauterPoon):
    """SauterPoon outlier detector running in-process (nproc=1)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nproc = 1


class CentroidOutlierFactoryGPU(CentroidOutlierFactory):
    @staticmethod
    def from_parameters_and_colnames(params, colnames):
        method = params.outlier.algorithm
        if method == "null":
            return None
        elif method == "tukey":
            outlier_detector = TukeyGPU
            algo_params = params.outlier.tukey
        elif method == "mcd":
            outlier_detector = MCDGPU
            algo_params = params.outlier.mcd
        elif method == "sauter_poon":
            outlier_detector = SauterPoonGPU
            algo_params = params.outlier.sauter_poon
        else:
            raise RuntimeError("outlier.algorithm not recognised")

        kwargs = {
            k: v for k, v in algo_params.__dict__.items() if not k.startswith("_")
        }

        if params.outlier.separate_images:
            if params.outlier.separate_blocks:
                logger.info("Resetting separate_blocks=False")
                params.outlier.separate_blocks = False
            if params.outlier.separate_experiments:
                logger.info("Resetting separate_experiments=False")
                params.outlier.separate_experiments = False

        if not params.outlier.separate_blocks:
            params.outlier.block_width = None

        # On GPU, always keep nproc=1 to avoid ProcessPoolExecutor fork overhead and CUDA re-init issues
        if params.outlier.nproc is libtbx.Auto:
            params.outlier.nproc = 1

        od = outlier_detector(
            cols=colnames,
            min_num_obs=params.outlier.minimum_number_of_reflections,
            separate_experiments=params.outlier.separate_experiments,
            separate_panels=params.outlier.separate_panels,
            separate_images=params.outlier.separate_images,
            block_width=params.outlier.block_width,
            nproc=1,
            **kwargs,
        )
        return od


class ReflectionManagerFactoryGPU(ReflectionManagerFactory):
    @staticmethod
    def from_parameters_reflections_experiments(
        params, reflections, experiments, do_stills=False
    ):
        if params.random_seed is not None:
            random.seed(params.random_seed)
            flex.set_random_seed(params.random_seed)
            logger.debug("Random seed set to %d", params.random_seed)

        if experiments.all_laue() or experiments.all_tof():
            return ReflectionManagerFactoryGPU.laue_manager(
                experiments, reflections, params
            )
        elif do_stills:
            return ReflectionManagerFactoryGPU.stills_manager(
                experiments, reflections, params
            )
        else:
            return ReflectionManagerFactoryGPU.rotation_scan_manager(
                experiments, reflections, params
            )

    @staticmethod
    def stills_manager(
        experiments: ExperimentList,
        reflections: flex.reflection_table,
        params: libtbx.phil.scope_extract,
    ) -> StillsReflectionManager:
        refman = StillsReflectionManager

        if params.outlier.algorithm in ("auto", libtbx.Auto):
            params.outlier.algorithm = "sauter_poon"
        if params.outlier.sauter_poon.px_sz is libtbx.Auto:
            params.outlier.sauter_poon.px_sz = experiments.detectors()[0][
                0
            ].get_pixel_size()

        if params.outlier.algorithm in ("null", None):
            outlier_detector = None
        else:
            colnames = ReflectionManagerFactoryGPU.get_col_names(params, do_stills=True)
            params.outlier.block_width = None
            outlier_detector = CentroidOutlierFactoryGPU.from_parameters_and_colnames(
                params, colnames
            )

        if params.weighting_strategy.override == "statistical":
            raise DialsRefineConfigError(
                'The "statistical" weighting strategy is not compatible '
                "with stills refinement"
            )
        if params.weighting_strategy.override == "constant":
            params.weighting_strategy.override = "constant_stills"

        weighting_strategy = (
            ReflectionManagerFactoryGPU.get_weighting_strategy_override(params)
        )

        return refman(
            reflections=reflections,
            experiments=experiments,
            nref_per_degree=params.reflections_per_degree,
            max_sample_size=params.maximum_sample_size,
            min_sample_size=params.minimum_sample_size,
            close_to_spindle_cutoff=params.close_to_spindle_cutoff,
            scan_margin=params.scan_margin,
            outlier_detector=outlier_detector,
            weighting_strategy_override=weighting_strategy,
        )

    @staticmethod
    def rotation_scan_manager(
        experiments: ExperimentList,
        reflections: flex.reflection_table,
        params: libtbx.phil.scope_extract,
    ) -> ReflectionManager:
        refman = ReflectionManager

        if params.outlier.algorithm in ("auto", libtbx.Auto):
            params.outlier.algorithm = "mcd"
        if params.outlier.algorithm == "sauter_poon":
            if params.outlier.sauter_poon.px_sz is libtbx.Auto:
                params.outlier.sauter_poon.px_sz = experiments.detectors()[0][
                    0
                ].get_pixel_size()

        if params.weighting_strategy.override in ["stills", "external_deltapsi"]:
            msg = (
                f'The "{params.weighting_strategy.override}" weighting strategy is not compatible with '
                "scan refinement"
            )
            raise DialsRefineConfigError(msg)

        if params.outlier.algorithm in ("null", None):
            outlier_detector = None
        else:
            colnames = ReflectionManagerFactoryGPU.get_col_names(params)
            outlier_detector = CentroidOutlierFactoryGPU.from_parameters_and_colnames(
                params, colnames
            )

        weighting_strategy = (
            ReflectionManagerFactoryGPU.get_weighting_strategy_override(params)
        )

        return refman(
            reflections=reflections,
            experiments=experiments,
            nref_per_degree=params.reflections_per_degree,
            max_sample_size=params.maximum_sample_size,
            min_sample_size=params.minimum_sample_size,
            close_to_spindle_cutoff=params.close_to_spindle_cutoff,
            scan_margin=params.scan_margin,
            outlier_detector=outlier_detector,
            weighting_strategy_override=weighting_strategy,
        )

    @staticmethod
    def laue_manager(
        experiments: ExperimentList,
        reflections: flex.reflection_table,
        params: libtbx.phil.scope_extract,
    ) -> LaueReflectionManager:
        all_tof_experiments = False
        for expt in experiments:
            if expt.scan is not None and expt.scan.has_property("time_of_flight"):
                all_tof_experiments = True
            elif all_tof_experiments:
                raise ValueError(
                    "Cannot refine ToF and non-ToF experiments at the same time"
                )

        if all_tof_experiments:
            refman = TOFReflectionManager
        else:
            refman = LaueReflectionManager

        if params.outlier.algorithm in ("auto", libtbx.Auto):
            params.outlier.algorithm = "mcd"
        if params.outlier.sauter_poon.px_sz is libtbx.Auto:
            params.outlier.sauter_poon.px_sz = experiments.detectors()[0][
                0
            ].get_pixel_size()

        if params.outlier.algorithm in ("null", None):
            outlier_detector = None
        else:
            colnames = ReflectionManagerFactoryGPU.get_col_names(params, do_stills=True)
            params.outlier.block_width = None
            outlier_detector = CentroidOutlierFactoryGPU.from_parameters_and_colnames(
                params, colnames
            )

        if params.weighting_strategy.override == "statistical":
            params.weighting_strategy.override = "statistical_laue"
        elif params.weighting_strategy.override == "constant":
            params.weighting_strategy.override = "constant_laue"

        if params.weighting_strategy.override is not None:
            if params.weighting_strategy.override not in [
                "constant_laue",
                "statistical_laue",
            ]:
                raise ValueError(
                    f"{params.weighting_strategy.override} not compatible with Laue data"
                )

        weighting_strategy = (
            ReflectionManagerFactoryGPU.get_weighting_strategy_override(params)
        )

        return refman(
            reflections=reflections,
            experiments=experiments,
            nref_per_degree=params.reflections_per_degree,
            max_sample_size=params.maximum_sample_size,
            min_sample_size=params.minimum_sample_size,
            close_to_spindle_cutoff=params.close_to_spindle_cutoff,
            scan_margin=params.scan_margin,
            outlier_detector=outlier_detector,
            weighting_strategy_override=weighting_strategy,
            wavelength_weight=params.weighting_strategy.wavelength_weight,
        )


__all__ = [
    "BlockCalculator",
    "CentroidOutlier",
    "CentroidOutlierFactory",
    "CentroidOutlierFactoryGPU",
    "FastMCD",
    "FastMCDGPU",
    "LaueReflectionManager",
    "MCD",
    "MCDGPU",
    "ReflectionManager",
    "ReflectionManagerFactory",
    "ReflectionManagerFactoryGPU",
    "SauterPoon",
    "SauterPoonGPU",
    "StillsReflectionManager",
    "TOFReflectionManager",
    "Tukey",
    "TukeyGPU",
    "phil_scope",
    "phil_str",
]
