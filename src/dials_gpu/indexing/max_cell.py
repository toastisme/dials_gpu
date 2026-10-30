from __future__ import annotations

import logging
import math

import numpy as np
import torch
from scitbx.array_family import flex

from dials.algorithms.indexing import DialsIndexError
from dials.algorithms.indexing.nearest_neighbor import NeighborAnalysis
from dials.algorithms.spot_finding.per_image_analysis import ice_rings_selection

logger = logging.getLogger(__name__)


class NeighborAnalysisGPU(NeighborAnalysis):
    """
    GPU-accelerated 3D nearest-neighbor analysis using PyTorch torch.cdist.
    Matches dials.algorithms.indexing.nearest_neighbor.NeighborAnalysis.
    """

    def __init__(
        self,
        reflections,
        step_size=45,
        tolerance=1.5,
        max_height_fraction=0.25,
        percentile=None,
        histogram_binning="linear",
        nn_per_bin=5,
        convert_reflections_z_to_deg=True,
        device=None,
    ):
        self.tolerance = tolerance
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        near = 10
        self.NNBIN = nn_per_bin
        self.histogram_binning = histogram_binning

        direct = flex.double()

        if "entering" in reflections:
            entering_flags = reflections["entering"].as_numpy_array()
        else:
            entering_flags = np.ones(len(reflections), dtype=bool)
        rs_vectors = reflections["rlp"]
        rs_np = rs_vectors.as_numpy_array()

        z_flex = reflections["xyzobs.mm.value"].parts()[2]
        if convert_reflections_z_to_deg:
            z_flex = z_flex * (180.0 / math.pi)
        z_np = z_flex.as_numpy_array()
        imgset_np = reflections["imageset_id"].as_numpy_array()

        all_d_spacings = 1.0 / rs_vectors.norms()
        d_spacings = flex.double()

        rs_t = torch.from_numpy(rs_np).to(device=self.device, dtype=torch.float64)
        max_imgset = int(imgset_np.max()) if len(imgset_np) > 0 else -1

        for imageset_id in range(max_imgset + 1):
            sel_imageset = imgset_np == imageset_id
            if not np.any(sel_imageset):
                continue
            z_sub = z_np[sel_imageset]
            z_min = float(z_sub.min())
            z_max = float(z_sub.max())
            d_z = z_max - z_min
            n_steps = max(int(math.ceil(d_z / step_size)), 1)

            for n in range(n_steps):
                sel_step = (
                    sel_imageset
                    & (z_np >= (z_min + n * step_size))
                    & (z_np < (z_min + (n + 1) * step_size))
                )

                for entering in (True, False):
                    sel_entering = sel_step & (entering_flags == entering)
                    idx_entering = np.flatnonzero(sel_entering)
                    if idx_entering.size == 0:
                        continue

                    if idx_entering.size == 1:
                        # Single point has no distinct neighbor -> distance inf -> direct 0.0
                        direct.append(0.0)
                    else:
                        idx_t = torch.from_numpy(idx_entering).to(device=self.device)
                        pts = rs_t[idx_t]
                        dist_mat = torch.cdist(pts, pts)
                        dist_mat.fill_diagonal_(float("inf"))
                        min_dists = dist_mat.min(dim=1).values
                        inv_dists = torch.where(
                            min_dists > 0,
                            1.0 / min_dists,
                            torch.zeros_like(min_dists),
                        )
                        direct.extend(
                            flex.double(
                                np.ascontiguousarray(
                                    inv_dists.cpu().numpy(), dtype=np.float64
                                )
                            )
                        )
                    d_spacings.extend(all_d_spacings)

        assert len(direct) > near, (
            f"Too few spots ({len(direct)}) for nearest neighbour analysis."
        )

        perm = flex.sort_permutation(direct)
        direct = direct.select(perm)
        d_spacings = d_spacings.select(perm)

        sel = direct > 1
        direct = direct.select(sel)
        d_spacings = d_spacings.select(sel)

        if percentile is None:
            n = int(math.floor(0.99 * len(direct)))
            direct = direct[:n]
            d_spacings = d_spacings[:n]

        if self.histogram_binning == "log":
            hst = flex.histogram(
                flex.log10(direct), n_slots=int(len(direct) / self.NNBIN)
            )
            self.slot_start = flex.double(
                [10 ** (s - 0.5 * hst.slot_width()) for s in hst.slot_centers()]
            )
            self.slot_end = flex.double(
                [10 ** (s + 0.5 * hst.slot_width()) for s in hst.slot_centers()]
            )
            self.slot_width = self.slot_end - self.slot_start
        else:
            hst = flex.histogram(direct, n_slots=int(len(direct) / self.NNBIN))
            self.slot_start = hst.slot_centers() - 0.5 * hst.slot_width()
            self.slot_end = hst.slot_centers() + 0.5 * hst.slot_width()
            self.slot_width = hst.slot_width()

        self.relative_frequency = hst.slots().as_double() / self.slot_width
        highest_bin_height = flex.max(self.relative_frequency)

        if percentile is not None:
            perm = flex.sort_permutation(direct, reverse=True)
            self.max_cell = (
                self.tolerance * direct[perm[int((1 - percentile) * len(direct))]]
            )
        else:
            isel = (
                self.relative_frequency.as_double()
                > (max_height_fraction * highest_bin_height)
            ).iselection()
            self.max_cell = (
                self.tolerance * self.slot_end[int(flex.max(isel.as_double()))]
            )

        self.reciprocal_lattice_vectors = rs_vectors
        self.d_spacings = d_spacings
        self.direct = direct
        self.histogram = hst


def find_max_cell_gpu(
    reflections,
    max_cell_multiplier=1.3,
    step_size=45,
    nearest_neighbor_percentile=None,
    histogram_binning="linear",
    nn_per_bin=5,
    max_height_fraction=0.25,
    filter_ice=True,
    filter_overlaps=True,
    overlaps_border=0,
    convert_reflections_z_to_deg=True,
    device=None,
):
    logger.debug("Finding suitable max_cell based on %i reflections", len(reflections))
    if filter_ice:
        ice_sel = ice_rings_selection(reflections)
        reflections = reflections.select(~ice_sel)
        logger.debug(
            "Rejecting %i reflections at ice ring resolution", ice_sel.count(True)
        )
        if not reflections.size():
            raise DialsIndexError(
                "No spots left for max cell analysis after ice-ring filtering"
            )

    if filter_overlaps and "bbox" in reflections:
        overlap_sel = flex.bool(len(reflections), False)
        overlaps = reflections.find_overlaps(border=overlaps_border)
        for item in overlaps.edges():
            i0 = overlaps.source(item)
            i1 = overlaps.target(item)
            overlap_sel[i0] = True
            overlap_sel[i1] = True
        logger.debug("Rejecting %i overlapping bounding boxes", overlap_sel.count(True))
        reflections = reflections.select(~overlap_sel)
    logger.debug("%i reflections remain for max_cell identification", len(reflections))

    if not len(reflections):
        raise DialsIndexError(
            "Too few spots remaining for nearest neighbour analysis (%d)"
            % len(reflections)
        )

    try:
        nn = NeighborAnalysisGPU(
            reflections,
            step_size=step_size,
            max_height_fraction=max_height_fraction,
            tolerance=max_cell_multiplier,
            percentile=nearest_neighbor_percentile,
            histogram_binning=histogram_binning,
            nn_per_bin=nn_per_bin,
            convert_reflections_z_to_deg=convert_reflections_z_to_deg,
            device=device,
        )
    except AssertionError as e:
        raise DialsIndexError("Failure in nearest neighbour analysis:\n" + str(e))

    return nn
