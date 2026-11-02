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
from dials.algorithms.refinement.analysis.centroid_analysis import CentroidAnalyser
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
    DEG2RAD,
    RAD2DEG,
    BlockCalculator,
    LaueReflectionManager,
    ReflectionManager,
    ReflectionManagerFactory,
    StillsReflectionManager,
    TOFReflectionManager,
)
from dials.algorithms.statistics.fast_mcd import FastMCD, maha_dist_sq
from dials.util import tabulate
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
    p = H.shape[2]
    T = H.mean(dim=1)  # (B, p)
    diff = H - T.unsqueeze(1)  # (B, h, p)
    S = torch.bmm(diff.transpose(1, 2), diff) / (h - 1)  # (B, p, p)
    if p == 3:
        detS = (torch.cross(S[:, 0], S[:, 1], dim=-1) * S[:, 2]).sum(dim=-1)
    elif p == 2:
        detS = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] * S[:, 1, 0]
    else:
        detS = torch.linalg.det(S)  # (B,)
    return T, S, detS


def _batched_concentration_step_gpu(
    h: int,
    data_t: torch.Tensor,
    T: torch.Tensor,
    S: torch.Tensor,
    g_idx: torch.Tensor | None = None,
):
    """Perform a batched concentration step for B trials simultaneously.

    Args:
        h: subset size (int)
        data_t: (N, p) or (G, N_g, p) float64 tensor of observations
        T: (B, p) float64 tensor of centers
        S: (B, p, p) float64 tensor of covariance matrices
        g_idx: optional (B, 1) group index tensor when data_t is (G, N_g, p)
    Returns:
        Hnew: (B, h, p) float64 tensor of the h observations with smallest Mahalanobis distance
    """
    p = S.shape[-1]
    if data_t.ndim == 3:
        G, N_g, _ = data_t.shape
        B = T.shape[0]
        n_trials = B // G
        diff = (data_t.unsqueeze(1) - T.view(G, n_trials, 1, p)).reshape(B, N_g, p)
    else:
        # diff: (B, N, p)
        diff = data_t.unsqueeze(0) - T.unsqueeze(1)
    if p == 3:
        cof = torch.cross(
            torch.roll(S, -1, dims=1), torch.roll(S, -2, dims=1), dim=-1
        )
        det = (S[:, 0] * cof[:, 0]).sum(dim=-1)
        Sinv = cof.transpose(1, 2) / det.unsqueeze(-1).unsqueeze(-1)
        d2s = (diff * torch.bmm(diff, Sinv)).sum(dim=-1)
    elif p == 2:
        s00, s01 = S[:, 0, 0], S[:, 0, 1]
        s10, s11 = S[:, 1, 0], S[:, 1, 1]
        det = s00 * s11 - s01 * s10
        inv_det = (1.0 / det).unsqueeze(-1).unsqueeze(-1)
        Sinv = (
            torch.stack(
                [
                    torch.stack([s11, -s01], dim=-1),
                    torch.stack([-s10, s00], dim=-1),
                ],
                dim=-2,
            )
            * inv_det
        )
        d2s = (diff * torch.bmm(diff, Sinv)).sum(dim=-1)
    else:
        sol = torch.linalg.solve(S, diff.transpose(1, 2))  # (B, p, N)
        d2s = (diff * sol.transpose(1, 2)).sum(dim=-1)  # (B, N)
    # Select h smallest distances per trial
    _, idx = torch.topk(d2s, k=h, dim=1, largest=False, sorted=True)  # (B, h)
    # Gather observations -> (B, h, p)
    if data_t.ndim == 3:
        Hnew = data_t[g_idx, idx]
    else:
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

    def _draw_initial_T0_S0_batch(
        self, n_trials: int, n_obs: int, data_t: torch.Tensor
    ):
        """Draw `n_trials` initial (p+1)-subsets using `flex.random_permutation` in the exact
        same order as CPU `FastMCD.form_initial_subset`, returning batched `T0` and `S0` tensors.
        """
        p = self._p
        subset_size = p + 1

        # Draw all n_trials permutations from flex RNG
        perms_np = np.empty((n_trials, subset_size), dtype=np.int64)
        extra_trials = []
        for t in range(n_trials):
            perm = flex.random_permutation(n_obs)
            perms_np[t] = flumpy.to_numpy(perm)[:subset_size]
            # Keep full permutation only if degenerate (rare)
            extra_trials.append(perm)

        perms_t = torch.as_tensor(perms_np, dtype=torch.long, device=self._device)
        J_batch = data_t[perms_t]  # (n_trials, p+1, p)
        T0, S0, detS0 = _batched_means_and_cov_gpu(J_batch)

        # Handle rare degenerate initial subsets where detS0 <= 0.0
        bad_mask = ~(detS0 > 0.0)
        if bad_mask.any():
            bad_indices = (
                torch.nonzero(bad_mask, as_tuple=False).squeeze(-1).cpu().tolist()
            )
            for t_idx in bad_indices:
                perm_np = flumpy.to_numpy(extra_trials[t_idx])
                i = 1
                det_val = 0.0
                while not det_val > 0.0:
                    sub_sz = p + 1 + i
                    idx_t = torch.as_tensor(
                        perm_np[:sub_sz], dtype=torch.long, device=self._device
                    )
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
        T0, S0 = self._draw_initial_T0_S0_batch(
            self._n_trials, len(self._data[0]), data_t
        )
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
        for step_i in range(self._k3):
            H_best = _batched_concentration_step_gpu(h, data_t, T_best, S_best)
            T_new, S_new, det_new = _batched_means_and_cov_gpu(H_best)
            if ((step_i & 3) == 3 or step_i == self._k3 - 1) and torch.all(
                det_new == det_best
            ):
                T_best, S_best, det_best = T_new, S_new, det_new
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

        data_t = self._cols_to_tensor(self._data)
        p = self._p

        # Sample the data and split into groups on GPU using exact flex RNG calls
        rows = flex.random_selection(self._n, sample_size)
        rows_t = torch.as_tensor(
            flumpy.to_numpy(rows).astype(np.int64), dtype=torch.long, device=self._device
        )
        sampled_t = data_t[rows_t]

        perm_split = flex.random_permutation(sample_size)
        perm_split_t = torch.as_tensor(
            flumpy.to_numpy(perm_split).astype(np.int64),
            dtype=torch.long,
            device=self._device,
        )
        permuted_t = sampled_t[perm_split_t]

        n_trials = self._n_trials // ngroups
        h_frac = self._h / self._n
        n_keep = min(10, n_trials)
        rem = sample_size % ngroups

        if rem == 0:
            # All ngroups have identical size g_len -> batch all ngroups * n_trials together
            g_len = sample_size // ngroups
            h_sub = int(g_len * h_frac)
            groups_t = permuted_t.view(ngroups, g_len, p)
            total_trials = ngroups * n_trials
            subset_size = p + 1

            perms_np = np.empty((total_trials, subset_size), dtype=np.int64)
            extra_trials = []
            for t in range(total_trials):
                perm = flex.random_permutation(g_len)
                perms_np[t] = flumpy.to_numpy(perm)[:subset_size]
                extra_trials.append(perm)

            perms_t = torch.as_tensor(perms_np, dtype=torch.long, device=self._device)
            g_idx = (
                torch.arange(ngroups, device=self._device)
                .repeat_interleave(n_trials)
                .unsqueeze(1)
            )
            J_batch = groups_t[g_idx, perms_t]
            T0, S0, detS0 = _batched_means_and_cov_gpu(J_batch)

            bad_mask = ~(detS0 > 0.0)
            if bad_mask.any():
                bad_indices = (
                    torch.nonzero(bad_mask, as_tuple=False).squeeze(-1).cpu().tolist()
                )
                for t_idx in bad_indices:
                    g_i = t_idx // n_trials
                    perm_np = flumpy.to_numpy(extra_trials[t_idx])
                    i = 1
                    det_val = 0.0
                    while not det_val > 0.0:
                        sub_sz = p + 1 + i
                        idx_t = torch.as_tensor(
                            perm_np[:sub_sz], dtype=torch.long, device=self._device
                        )
                        J_single = groups_t[g_i, idx_t].unsqueeze(0)
                        T_s, S_s, det_s = _batched_means_and_cov_gpu(J_single)
                        det_val = float(det_s[0].item())
                        i += 1
                    T0[t_idx] = T_s[0]
                    S0[t_idx] = S_s[0]

            Hcurr = _batched_concentration_step_gpu(
                h_sub, groups_t, T0, S0, g_idx=g_idx
            )
            Tcurr, Scurr, detScurr = _batched_means_and_cov_gpu(Hcurr)

            for _ in range(self._k1):
                Hcurr = _batched_concentration_step_gpu(
                    h_sub, groups_t, Tcurr, Scurr, g_idx=g_idx
                )
                Tcurr, Scurr, detScurr = _batched_means_and_cov_gpu(Hcurr)

            _, best_idx = torch.topk(
                detScurr.view(ngroups, n_trials),
                k=n_keep,
                dim=1,
                largest=False,
                sorted=True,
            )
            Tcurr = (
                Tcurr.view(ngroups, n_trials, p)
                .gather(1, best_idx.unsqueeze(-1).expand(ngroups, n_keep, p))
                .reshape(ngroups * n_keep, p)
            )
            Scurr = (
                Scurr.view(ngroups, n_trials, p, p)
                .gather(
                    1,
                    best_idx.unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(ngroups, n_keep, p, p),
                )
                .reshape(ngroups * n_keep, p, p)
            )
        else:
            blocksize = int(sample_size / ngroups)
            blocksizes = [blocksize] * (ngroups - rem) + [blocksize + 1] * rem
            T_group_best = []
            S_group_best = []
            start = 0
            for b_len in blocksizes:
                end = start + b_len
                group_t = permuted_t[start:end]
                start = end
                h_sub = int(b_len * h_frac)
                T0, S0 = self._draw_initial_T0_S0_batch(n_trials, b_len, group_t)
                Hcurr = _batched_concentration_step_gpu(h_sub, group_t, T0, S0)
                Tcurr, Scurr, detScurr = _batched_means_and_cov_gpu(Hcurr)

                for _ in range(self._k1):
                    Hcurr = _batched_concentration_step_gpu(
                        h_sub, group_t, Tcurr, Scurr
                    )
                    Tcurr, Scurr, detScurr = _batched_means_and_cov_gpu(Hcurr)

                _, best_idx = torch.topk(detScurr, k=n_keep, largest=False, sorted=True)
                T_group_best.append(Tcurr[best_idx])
                S_group_best.append(Scurr[best_idx])

            Tcurr = torch.cat(T_group_best, dim=0)
            Scurr = torch.cat(S_group_best, dim=0)

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

        for step_i in range(k4):
            H_best = _batched_concentration_step_gpu(self._h, data_t, T_best, S_best)
            T_new, S_new, det_new = _batched_means_and_cov_gpu(H_best)
            if ((step_i & 3) == 3 or step_i == k4 - 1) and torch.all(
                det_new == det_best
            ):
                T_best, S_best, det_best = T_new, S_new, det_new
                break
            T_best, S_best, det_best = T_new, S_new, det_new

        win_idx = int(torch.argmin(det_best).item())
        Tbest_np = T_best[win_idx].detach().cpu().numpy()
        Sbest_np = S_best[win_idx].detach().cpu().numpy()

        Tbest = flex.double(Tbest_np.tolist())
        Sbest = flex.double(np.ascontiguousarray(Sbest_np))
        return Tbest, Sbest


class _GPUCentroidOutlierMixin:
    """Mixin for CentroidOutlier subclasses that avoids 4 successive 25-column
    reflection_table.select() deep copies during job splitting by operating on
    1D index arrays and slicing only the 3 residual columns per job.
    """

    def __call__(self, reflections):
        if self.nproc > 1:
            return super().__call__(reflections)

        logger.info(
            "Detecting centroid outliers using the %s algorithm", type(self).__name__
        )
        for col in self._cols:
            assert col in reflections

        sel = reflections.get_flags(reflections.flags.predicted)
        all_indices = sel.iselection()
        if len(all_indices) == 0:
            return super().__call__(reflections)

        all_idx_np = flumpy.to_numpy(all_indices).astype(np.int64)
        id_all_np = flumpy.to_numpy(reflections["id"])[all_idx_np]
        panel_all_np = flumpy.to_numpy(reflections["panel"])[all_idx_np]
        nexp = int(id_all_np.max()) + 1

        jobs = []
        if self._separate_experiments:
            for iexp in range(nexp):
                mask = id_all_np == iexp
                jobs.append(
                    {
                        "id": iexp,
                        "panel": "all",
                        "idx_np": all_idx_np[mask],
                        "panel_np": panel_all_np[mask],
                    }
                )
        else:
            jobs.append(
                {
                    "id": "all",
                    "panel": "all",
                    "idx_np": all_idx_np,
                    "panel_np": panel_all_np,
                }
            )

        jobs2 = []
        if self._separate_panels:
            for job in jobs:
                idx_np = job["idx_np"]
                p_np = job["panel_np"]
                iexp = job["id"]
                max_p = int(p_np.max()) if len(p_np) > 0 else -1
                for ipanel in range(max_p + 1):
                    mask = p_np == ipanel
                    jobs2.append(
                        {
                            "id": iexp,
                            "panel": ipanel,
                            "idx_np": idx_np[mask],
                        }
                    )
        else:
            jobs2 = jobs

        jobs3 = []
        if self.get_block_width() is not None:
            phi_full_np = flumpy.to_numpy(reflections["xyzobs.mm.value"])[:, 2]
            for job in jobs2:
                idx_np = job["idx_np"]
                iexp = job["id"]
                ipanel = job["panel"]
                if len(idx_np) == 0:
                    jobs3.append(job)
                    continue
                phi = phi_full_np[idx_np]
                phi_low = float(phi.min())
                phi_range = float(phi.max()) - phi_low
                if phi_range == 0.0:
                    jobs3.append(job)
                    continue
                bw = self.get_block_width(iexp)
                if bw is None:
                    jobs3.append(job)
                    continue
                nblocks = max(1, int(round(math.degrees(phi_range / bw))))
                real_width = phi_range / nblocks
                block_end = 0.0
                for iblock in range(nblocks - 1):
                    block_start = iblock * real_width
                    block_end = (iblock + 1) * real_width
                    mask = (phi >= (phi_low + block_start)) & (
                        phi < (phi_low + block_end)
                    )
                    jobs3.append(
                        {
                            "id": iexp,
                            "panel": ipanel,
                            "idx_np": idx_np[mask],
                            "phi_start": math.degrees(phi_low + block_start),
                            "phi_end": math.degrees(phi_low + block_end),
                        }
                    )
                mask = phi >= (phi_low + block_end)
                jobs3.append(
                    {
                        "id": iexp,
                        "panel": ipanel,
                        "idx_np": idx_np[mask],
                        "phi_start": math.degrees(phi_low + block_end),
                        "phi_end": math.degrees(phi_low + phi_range),
                    }
                )
        elif self._separate_images:
            images_full = (
                reflections["xyzobs.px.value"].parts()[2].iround()
            )
            images_full_np = flumpy.to_numpy(images_full)
            phi_full_np = flumpy.to_numpy(reflections["xyzobs.mm.value"])[:, 2]
            for job in jobs2:
                idx_np = job["idx_np"]
                iexp = job["id"]
                ipanel = job["panel"]
                if len(idx_np) == 0:
                    jobs3.append(job)
                    continue
                images = images_full_np[idx_np]
                images_low = int(images.min())
                images_high = int(images.max())
                for iblock in range(images_low, images_high + 1):
                    mask = images == iblock
                    sub_idx = idx_np[mask]
                    phi = phi_full_np[sub_idx]
                    jobs3.append(
                        {
                            "id": iexp,
                            "panel": ipanel,
                            "idx_np": sub_idx,
                            "phi_start": float(phi.min()) if len(phi) > 0 else 0.0,
                            "phi_end": float(phi.max()) if len(phi) > 0 else 0.0,
                        }
                    )
        else:
            jobs3 = jobs2

        header = ["Job"]
        if self._separate_experiments:
            header.append("Exp\nid")
        if self._separate_panels:
            header.append("Panel\nid")
        if self.get_block_width() is not None:
            header.append("Block range\n(deg)")
        header.extend(["Nref", "Nout", "%out"])
        rows = []

        col_arrays = [reflections[col] for col in self._cols]
        for i, job in enumerate(jobs3):
            idx_np = job["idx_np"]
            nref = len(idx_np)
            indices = flumpy.from_numpy(np.ascontiguousarray(idx_np, dtype=np.uint64))
            flex.set_random_seed(42)
            if nref >= self._min_num_obs:
                cols = [c.select(indices) for c in col_arrays]
                outliers = self._detect_outliers(cols)
                ioutliers = indices.select(outliers)
            else:
                ioutliers = flex.size_t()

            iexp = job["id"]
            ipanel = job["panel"]
            nout = len(ioutliers)
            if nout > 0:
                reflections.set_flags(ioutliers, reflections.flags.centroid_outlier)
                self.nreject += nout

            row = [str(i + 1)]
            if self._separate_experiments:
                row.append(str(iexp))
            if self._separate_panels:
                row.append(str(ipanel))
            if self.get_block_width() is not None:
                try:
                    row.append("{phi_start:.2f} - {phi_end:.2f}".format(**job))
                except KeyError:
                    row.append(f"{0.0:.2f} - {0.0:.2f}")
            p100 = (nout / nref * 100.0) if nref > 0 else 0
            row.extend([str(nref), str(nout), f"{p100:3.1f}"])
            rows.append(row)

        if self.nreject == 0:
            return False
        logger.info(f"{self.nreject} reflections have been flagged as outliers")
        logger.debug("Outlier rejections per job:")
        logger.debug(tabulate(rows, header))
        return True


class MCDGPU(_GPUCentroidOutlierMixin, MCD):
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


class TukeyGPU(_GPUCentroidOutlierMixin, Tukey):
    """Tukey outlier detector running in-process (nproc=1)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nproc = 1


class SauterPoonGPU(_GPUCentroidOutlierMixin, SauterPoon):
    """SauterPoon outlier detector running in-process (nproc=1)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nproc = 1


class BlockCalculatorGPU(BlockCalculator):
    """Vectorized BlockCalculator avoiding O(n_blocks * n_refl) Python loops."""

    def per_width(self, width, deg=True):
        if deg:
            width *= DEG2RAD
        self._create_block_columns()

        phi_obs_np = flumpy.to_numpy(self._reflections["xyzobs.mm.value"])[:, 2]
        id_np = flumpy.to_numpy(self._reflections["id"])
        block_np = flumpy.to_numpy(self._reflections["block"])
        bc_np = flumpy.to_numpy(self._reflections["block_centre"])

        for iexp, exp in enumerate(self._experiments):
            if len(self._experiments) == 1 and iexp == 0:
                isel_np = None
                exp_phi_np = phi_obs_np
            else:
                isel_np = np.nonzero(id_np == iexp)[0]
                if len(isel_np) == 0:
                    continue
                exp_phi_np = phi_obs_np[isel_np]

            start, stop = exp.scan.get_oscillation_range(deg=False)
            nblocks = int(abs(stop - start) / width) + 1
            if stop > start:
                _width = width + 1e-11
            elif stop == start:
                _width = 1e-11
            else:
                _width = -width + 1e-11

            half_width = width * (0.5 - 1e-11)
            block_starts = np.array(
                [start + n * _width for n in range(nblocks)], dtype=np.float64
            )
            block_centres = np.array(
                [
                    exp.scan.get_array_index_from_angle(e + half_width, deg=False)
                    for e in block_starts
                ],
                dtype=np.float64,
            )

            if _width > 0 and nblocks > 1:
                b_idx = np.searchsorted(block_starts, exp_phi_np, side="right") - 1
                clipped = np.clip(b_idx, 0, nblocks - 1)
                valid = (b_idx >= 0) & (exp_phi_np <= block_starts[clipped] + _width)
                if isel_np is None:
                    if valid.all():
                        block_np[:] = b_idx
                        bc_np[:] = block_centres[b_idx]
                    else:
                        block_np[valid] = b_idx[valid]
                        bc_np[valid] = block_centres[b_idx[valid]]
                else:
                    v_isel = isel_np[valid]
                    block_np[v_isel] = b_idx[valid]
                    bc_np[v_isel] = block_centres[b_idx[valid]]
            else:
                for b_num, (b_start, b_cent) in enumerate(
                    zip(block_starts, block_centres)
                ):
                    if _width >= 0:
                        m = (b_start <= exp_phi_np) & (exp_phi_np <= (b_start + _width))
                    else:
                        m = (b_start <= exp_phi_np) & (exp_phi_np <= (b_start + _width))
                    if isel_np is None:
                        block_np[m] = b_num
                        bc_np[m] = b_cent
                    else:
                        sub_i = isel_np[m]
                        block_np[sub_i] = b_num
                        bc_np[sub_i] = b_cent

        return self._reflections

    def per_image(self):
        self._create_block_columns()
        phi_obs = self._reflections["xyzobs.mm.value"].parts()[2]
        id_np = flumpy.to_numpy(self._reflections["id"])
        block_np = flumpy.to_numpy(self._reflections["block"])
        bc_np = flumpy.to_numpy(self._reflections["block_centre"])

        for iexp, exp in enumerate(self._experiments):
            sel = self._reflections["id"] == iexp
            isel = sel.iselection()
            if len(isel) == 0:
                continue
            exp_phi = phi_obs.select(isel)
            frames = exp.scan.get_array_index_from_angle(exp_phi, deg=False)
            frames_np = flumpy.to_numpy(flex.floor(frames).iround()).astype(np.int64)
            start = int(frames_np.min())
            if len(self._experiments) == 1 and len(isel) == len(self._reflections):
                block_np[:] = frames_np - start
                bc_np[:] = frames_np + 0.5
            else:
                isel_np = flumpy.to_numpy(isel).astype(np.int64)
                block_np[isel_np] = frames_np - start
                bc_np[isel_np] = frames_np + 0.5

        return self._reflections


class CentroidAnalyserGPU(CentroidAnalyser):
    """Vectorized CentroidAnalyser that avoids 3 full reflection_table deep copies
    and O(n_blocks * n_refl) histogram loops when determining block sizes.
    """

    def __init__(self, reflections, av_callback=flex.mean, debug=False):
        if debug:
            super().__init__(reflections, av_callback=av_callback, debug=debug)
            return

        self._average_residuals = False
        self._spectral_analysis = False
        self._av_callback = av_callback
        self._raw_reflections = reflections
        self._reflections = None

        h_np = flumpy.to_numpy(reflections["miller_index"])
        xyzcal_np = flumpy.to_numpy(reflections["xyzcal.mm"])
        valid_mask = ((h_np[:, 0] != 0) | (h_np[:, 1] != 0) | (h_np[:, 2] != 0)) & ~(
            (xyzcal_np[:, 0] == 0.0) & (xyzcal_np[:, 1] == 0.0)
        )
        id_np = flumpy.to_numpy(reflections["id"])[valid_mask]
        phi_deg_all = (
            flumpy.to_numpy(reflections["xyzobs.mm.value"])[valid_mask, 2] * RAD2DEG
        )
        self._nexp = int(id_np.max()) + 1 if len(id_np) > 0 else 0
        self._results = []

        for iexp in range(self._nexp):
            phi_np = phi_deg_all if self._nexp == 1 else phi_deg_all[id_np == iexp]
            if len(phi_np) == 0:
                self._results.append({})
                continue
            phi_min = float(phi_np.min())
            phi_max = float(phi_np.max())
            phi_range = (phi_min, phi_max)
            phi_width = phi_max - phi_min
            ideal_block_size = 1.0
            old_nblocks = 0
            while True:
                nblocks = int(phi_width // ideal_block_size)
                if nblocks == old_nblocks:
                    nblocks -= 1
                nblocks = max(nblocks, 1)
                block_size = phi_width / nblocks
                if nblocks == 1:
                    nr_np = np.array([len(phi_np)], dtype=np.int32)
                    nr = flumpy.from_numpy(nr_np)
                    break
                edges = phi_min + np.arange(1, nblocks, dtype=np.float64) * block_size
                bin_idx = np.searchsorted(edges, phi_np, side="right")
                nr_np = np.bincount(bin_idx, minlength=nblocks).astype(np.int32)
                min_nr = int(nr_np.min())
                if min_nr >= 50:
                    nr = flumpy.from_numpy(np.ascontiguousarray(nr_np))
                    break
                fac = 2 if min_nr < 5 else (50 / min_nr)
                ideal_block_size *= fac
                old_nblocks = nblocks

            self._results.append(
                {
                    "block_size": block_size,
                    "nref_per_block": nr,
                    "nblocks": nblocks,
                    "phi_range": phi_range,
                }
            )

    def __call__(
        self, calc_average_residuals=True, calc_periodograms=True, spans=(4, 4)
    ):
        if not calc_average_residuals and not calc_periodograms:
            return self._results
        if self._reflections is None:
            super().__init__(
                self._raw_reflections, av_callback=self._av_callback, debug=False
            )
        return super().__call__(
            calc_average_residuals=calc_average_residuals,
            calc_periodograms=calc_periodograms,
            spans=spans,
        )


class _GPUReflectionManagerMixin:
    """Mixin for ReflectionManager avoiding redundant 25-column deep copies and sorts."""

    def get_centroid_analyser(self, debug=False):
        return CentroidAnalyserGPU(self._reflections, debug=debug)

    def print_stats_on_matches(self):
        if not logger.isEnabledFor(logging.INFO):
            return
        used = self._reflections.get_flags(self._reflections.flags.used_in_refinement)
        if used.count(False) == 0:
            l = self._reflections
        else:
            l = self.get_matches()
        nref = len(l)
        if nref == 0:
            logger.warning(
                "Unable to calculate summary statistics for zero observations"
            )
            return
        try:
            x_resid = l["x_resid"]
            y_resid = l["y_resid"]
            phi_resid = l["phi_resid"]
            w_x, w_y, w_phi = l["xyzobs.mm.weights"].parts()
        except KeyError:
            return super().print_stats_on_matches()

        msg = (
            f"\nSummary statistics for {nref} observations" + " matched to predictions:"
        )
        header = ["", "Min", "Q1", "Med", "Q3", "Max"]
        rows = []
        row_data = five_number_summary(x_resid)
        rows.append(["Xc - Xo (mm)"] + [f"{e:.4g}" for e in row_data])
        row_data = five_number_summary(y_resid)
        rows.append(["Yc - Yo (mm)"] + [f"{e:.4g}" for e in row_data])
        row_data = five_number_summary(phi_resid)
        rows.append(["Phic - Phio (deg)"] + [f"{e * RAD2DEG:.4g}" for e in row_data])
        row_data = five_number_summary(w_x)
        rows.append(["X weights"] + [f"{e:.4g}" for e in row_data])
        row_data = five_number_summary(w_y)
        rows.append(["Y weights"] + [f"{e:.4g}" for e in row_data])
        row_data = five_number_summary(w_phi)
        rows.append(["Phi weights"] + [f"{e * DEG2RAD**2:.4g}" for e in row_data])

        logger.info(msg)
        logger.info(tabulate(rows, header, numalign="right") + "\n")

    def finalise(self, analysis=None):
        logger.debug("Finalising the Reflection Manager")

        mask = self._reflections.get_flags(self._reflections.flags.predicted)
        self._reflections.set_flags(mask, self._reflections.flags.used_in_refinement)

        self.print_stats_on_matches()

        mask = self._reflections.get_flags(self._reflections.flags.centroid_outlier)
        self._reflections.unset_flags(mask, self._reflections.flags.centroid_outlier)
        mask = self._indexed.get_flags(self._indexed.flags.centroid_outlier)
        self._indexed.unset_flags(mask, self._indexed.flags.centroid_outlier)

        if self._outlier_detector is None:
            rejection_occurred = False
        else:
            if self._outlier_detector.get_block_width() is libtbx.Auto:
                if analysis is None:
                    self._outlier_detector.set_block_width(18.0)
                else:
                    widths = [e.get("block_size") for e in analysis]
                    widths = [max(e, 18.0) if e is not None else None for e in widths]
                    self._outlier_detector.set_block_width(widths)
            rejection_occurred = self._outlier_detector(self._reflections)

        ioutliers = self._reflections.get_flags(
            self._reflections.flags.centroid_outlier
        )
        ioutliers = self._reflections["iobs"].select(ioutliers)
        iobs_np = flumpy.to_numpy(self._indexed["iobs"])
        if len(iobs_np) > 1 and np.any(iobs_np[:-1] > iobs_np[1:]):
            self._indexed.sort("iobs")
        self._indexed.set_flags(ioutliers, self._indexed.flags.centroid_outlier)

        has_pred = self._reflections.get_flags(self._reflections.flags.predicted)
        inlier = ~self._reflections.get_flags(self._reflections.flags.centroid_outlier)
        keep_sel = has_pred & inlier
        if keep_sel.count(False) > 0:
            self._reflections = self._reflections.select(keep_sel)
        self._reflections.set_flags(
            flex.bool(len(self._reflections), True),
            self._reflections.flags.used_in_refinement,
        )

        logger.info("%d reflections remain in the manager", len(self._reflections))
        if len(self._reflections) == 0:
            raise DialsRefineConfigError("No reflections available for refinement")

        if rejection_occurred:
            self.print_stats_on_matches()

        self._create_working_set()
        logger.debug("Working set size = %d observations", self.get_sample_size())

    def _create_working_set(self):
        working_isel = flex.size_t()
        all_kept = True
        for iexp, exp in enumerate(self._experiments):
            sel = self._reflections["id"] == iexp
            isel = sel.iselection()
            nrefs = sample_size = len(isel)

            if (
                exp.scan
                and exp.scan.has_property("oscillation")
                and self._nref_per_degree
            ):
                sequence_range_rad = exp.scan.get_oscillation_range(deg=False)
                width = abs(sequence_range_rad[1] - sequence_range_rad[0]) * RAD2DEG
                if self._nref_per_degree is libtbx.Auto:
                    turns = width / 360.0
                    if turns > 1:
                        approx_nref_1_turn = int(math.ceil(nrefs / turns))
                        sample_size = int(max(approx_nref_1_turn, 100.0 * width))
                else:
                    sample_size = int(self._nref_per_degree * max(round(width), 1.0))

            sample_size = max(sample_size, self._min_sample_size)
            if self._max_sample_size:
                sample_size = min(sample_size, self._max_sample_size)

            if sample_size < nrefs:
                all_kept = False
                isel = isel.select(flex.random_selection(nrefs, sample_size))
            working_isel.extend(isel)

        if all_kept and len(working_isel) == len(self._reflections):
            w_np = flumpy.to_numpy(working_isel)
            if len(w_np) <= 1 or np.all(w_np[:-1] < w_np[1:]):
                self._free_reflections = self._reflections[0:0]
                return

        free_sel = flex.bool(len(self._reflections), True)
        free_sel.set_selected(working_isel, False)
        self._free_reflections = self._reflections.select(free_sel)
        self._reflections = self._reflections.select(working_isel)


class ReflectionManagerGPU(_GPUReflectionManagerMixin, ReflectionManager):
    pass


class StillsReflectionManagerGPU(_GPUReflectionManagerMixin, StillsReflectionManager):
    pass


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
        refman = StillsReflectionManagerGPU

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
        refman = ReflectionManagerGPU

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
