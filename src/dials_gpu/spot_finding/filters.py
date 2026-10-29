from __future__ import annotations

import logging
import numpy as np
import torch
from dxtbx import flumpy

from dials.algorithms.spot_finding.factory import (
    BackgroundGradientFilter,
    PeakCentroidDistanceFilter,
    SpotDensityFilter,
)
from dials.array_family import flex

logger = logging.getLogger("dials.algorithms.spot_finding.factory")


class PeakCentroidDistanceFilterGPU:
    """
    GPU-accelerated PeakCentroidDistanceFilter.
    Can filter directly on GPU tensors (peak_coords, xyzobs_px_value) or
    fall back to CPU flex.observation/flex.shoebox.
    """

    def __init__(self, maxd: float):
        self.maxd = float(maxd)

    def run_gpu(self, flags_t: torch.Tensor, peak_coords: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        dist = torch.linalg.norm(peak_coords - centroids, dim=1)
        return flags_t & (dist <= self.maxd)

    def __call__(self, flags, **kwargs):
        if isinstance(flags, torch.Tensor):
            num_before = int(flags.sum().item())
            flags = self.run_gpu(flags, kwargs["peak_coords"], kwargs["centroids"])
            num_after = int(flags.sum().item())
            logger.info(
                f"Filtered {num_after} of {num_before} spots by peak-centroid distance"
            )
            return flags

        return PeakCentroidDistanceFilter(self.maxd)(flags, **kwargs)


class SpotDensityFilterGPU:
    """
    SpotDensityFilter operating on GPU centroid tensors or flex.observation.
    """

    def __init__(self, nbins: int = 50, gradient_cutoff: float = 0.002):
        self.nbins = nbins
        self.gradient_cutoff = gradient_cutoff

    def run_gpu(self, flags_t: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        if centroids.shape[0] == 0:
            return flags_t
        obs_x_np = np.ascontiguousarray(centroids[:, 0].detach().cpu().numpy(), dtype=np.float64)
        obs_y_np = np.ascontiguousarray(centroids[:, 1].detach().cpu().numpy(), dtype=np.float64)
        obs_x = flumpy.from_numpy(obs_x_np)
        obs_y = flumpy.from_numpy(obs_y_np)

        H, xedges, yedges = np.histogram2d(obs_x_np, obs_y_np, bins=self.nbins)
        H_flex = flex.double(H.flatten().astype(np.float64))
        n_slots = min(int(flex.max(H_flex)), 30)
        if n_slots <= 0:
            return flags_t
        hist = flex.histogram(H_flex, n_slots=n_slots)

        slots = hist.slots()
        cumulative_hist = flex.long(len(slots))
        for i, slot in enumerate(slots):
            cumulative_hist[i] = slot
            if i > 0:
                cumulative_hist[i] += cumulative_hist[i - 1]

        cumulative_hist = cumulative_hist.as_double() / flex.max(cumulative_hist.as_double())

        cutoff = None
        gradients = flex.double()
        for i in range(len(slots) - 1):
            x1 = cumulative_hist[i]
            x2 = cumulative_hist[i + 1]
            g = (x2 - x1) / hist.slot_width()
            gradients.append(g)
            if (
                cutoff is None
                and i > 0
                and g < self.gradient_cutoff
                and gradients[i - 1] < self.gradient_cutoff
            ):
                cutoff = hist.slot_centers()[i - 1] - 0.5 * hist.slot_width()

        if cutoff is not None:
            sel = np.column_stack(np.where(H > cutoff))
            cx = centroids[:, 0]
            cy = centroids[:, 1]
            for ix, iy in sel:
                in_bin = (cx > float(xedges[ix])) & (cx < float(xedges[ix + 1])) & (cy > float(yedges[iy])) & (cy < float(yedges[iy + 1]))
                flags_t = flags_t & (~in_bin)
        return flags_t

    def __call__(self, flags, **kwargs):
        if isinstance(flags, torch.Tensor):
            num_before = int(flags.sum().item())
            flags = self.run_gpu(flags, kwargs["centroids"])
            num_after = int(flags.sum().item())
            logger.info(f"Filtered {num_after} of {num_before} spots by spot density")
            return flags

        return SpotDensityFilter(self.nbins, self.gradient_cutoff)(flags, **kwargs)


class FilterRunnerGPU:
    """
    GPU-aware FilterRunner replacement.
    """

    def __init__(self, filters=None):
        self.filters = filters or []

    @property
    def needs_shoeboxes_for_filtering(self) -> bool:
        return any(isinstance(f, BackgroundGradientFilter) for f in self.filters)

    def run_gpu(
        self,
        num_spots: int,
        peak_coords: torch.Tensor,
        centroids: torch.Tensor,
        device: torch.device,
        sequence=None,
        shoeboxes=None,
    ) -> torch.Tensor:
        flags_t = torch.ones((num_spots,), dtype=torch.bool, device=device)
        if num_spots == 0 or not self.filters:
            return flags_t

        for f in self.filters:
            if isinstance(f, (PeakCentroidDistanceFilterGPU, SpotDensityFilterGPU)):
                flags_t = f(flags_t, peak_coords=peak_coords, centroids=centroids, sequence=sequence)
            elif isinstance(f, PeakCentroidDistanceFilter):
                flags_t = PeakCentroidDistanceFilterGPU(f.maxd)(
                    flags_t, peak_coords=peak_coords, centroids=centroids
                )
            elif isinstance(f, SpotDensityFilter):
                flags_t = SpotDensityFilterGPU(f.nbins, f.gradient_cutoff)(
                    flags_t, centroids=centroids
                )
            elif isinstance(f, BackgroundGradientFilter):
                # Convert flags to flex.bool, run BackgroundGradientFilter, convert back
                flags_np = np.ascontiguousarray(flags_t.detach().cpu().numpy(), dtype=bool)
                flags_flex = flumpy.from_numpy(flags_np)
                flags_flex = f(flags_flex, sequence=sequence, shoeboxes=shoeboxes)
                flags_t = torch.from_numpy(flumpy.to_numpy(flags_flex)).to(device=device, dtype=torch.bool)
            else:
                flags_np = np.ascontiguousarray(flags_t.detach().cpu().numpy(), dtype=bool)
                flags_flex = flumpy.from_numpy(flags_np)
                flags_flex = f(flags_flex, sequence=sequence, shoeboxes=shoeboxes)
                flags_t = torch.from_numpy(flumpy.to_numpy(flags_flex)).to(device=device, dtype=torch.bool)

        return flags_t

    def __call__(self, flags=None, **kwargs):
        if isinstance(flags, torch.Tensor):
            return self.run_gpu(
                num_spots=flags.shape[0],
                peak_coords=kwargs["peak_coords"],
                centroids=kwargs["centroids"],
                device=flags.device,
                sequence=kwargs.get("sequence"),
                shoeboxes=kwargs.get("shoeboxes"),
            )
        # Fallback for standard DIALS interface
        if flags is None:
            shoeboxes = kwargs.get("shoeboxes")
            observations = kwargs.get("observations")
            length = len(shoeboxes) if shoeboxes is not None else len(observations)
            flags = flex.bool(length, True)
        for f in self.filters:
            flags = f(flags, **kwargs)
        return flags
