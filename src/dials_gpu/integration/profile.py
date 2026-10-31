from __future__ import annotations

import logging
import math

import numpy as np
import torch
from dxtbx import flumpy

from dials.algorithms.profile_model.factory import ProfileModelFactory
from dials.algorithms.profile_model.gaussian_rs import (
    BBoxCalculator2D,
    BBoxCalculator3D,
    CoordinateSystem,
    CoordinateSystem2d,
    MaskCalculator2D,
    MaskCalculator3D,
    Model as GaussianRSProfileModel,
    PartialityCalculator2D,
    PartialityCalculator3D,
     zeta_factor,
)
from dials.algorithms.profile_model.gaussian_rs.calculator import (
    ComputeEsdBeamDivergence,
    ComputeEsdReflectingRange,
    FractionOfObservedIntensity,
    ProfileModelCalculator,
    ScanVaryingProfileModelCalculator,
    _select_reflections_for_sigma_calc,
)
from dials.array_family import flex
from scitbx import simplex

logger = logging.getLogger(__name__)


def _get_device(device=None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _bbox_to_numpy(bbox_flex) -> np.ndarray:
    if len(bbox_flex) == 0:
        return np.empty((0, 6), dtype=np.int64)
    return np.column_stack([p.as_numpy_array() for p in bbox_flex.parts()])


def compute_zeta_gpu(experiment, s1_flex, device=None) -> flex.double:
    """
    Compute the zeta factor for an array of s1 vectors on the GPU using PyTorch.
    Matches `dials.algorithms.profile_model.gaussian_rs.zeta_factor` (Kabsch 2010):
      e1 = normalize(s1 x s0)
      zeta = dot(normalize(m2), e1)
    """
    gonio = experiment.goniometer
    beam = experiment.beam
    if gonio is None or beam is None or len(s1_flex) == 0:
        return flex.double(len(s1_flex), 0.0)

    dev = _get_device(device)
    s0_np = np.array(beam.get_s0(), dtype=np.float64)
    m2_np = np.array(gonio.get_rotation_axis(), dtype=np.float64)
    m2_norm = np.linalg.norm(m2_np)
    if m2_norm > 0:
        m2_np = m2_np / m2_norm

    s1_np = s1_flex.as_numpy_array().reshape(-1, 3)
    s1_t = torch.as_tensor(s1_np, dtype=torch.float64, device=dev)
    s0_t = torch.as_tensor(s0_np, dtype=torch.float64, device=dev)
    m2_t = torch.as_tensor(m2_np, dtype=torch.float64, device=dev)

    cross = torch.linalg.cross(s1_t, s0_t.unsqueeze(0))
    cross_norm = torch.linalg.norm(cross, dim=1, keepdim=True).clamp(min=1e-30)
    e1 = cross / cross_norm
    zeta = torch.sum(e1 * m2_t.unsqueeze(0), dim=1)
    return flex.double(np.ascontiguousarray(zeta.cpu().numpy(), dtype=np.float64))


def compute_d_gpu(experiments, reflections, device=None) -> flex.double:
    """
    Compute resolution d = 1 / |s1 - s0| for reflections on the GPU using PyTorch.
    """
    n = len(reflections)
    if n == 0:
        return flex.double()

    dev = _get_device(device)
    s1_np = reflections["s1"].as_numpy_array().reshape(-1, 3)
    s1_t = torch.as_tensor(s1_np, dtype=torch.float64, device=dev)

    if len(experiments) == 1:
        s0_np = np.array(experiments[0].beam.get_s0(), dtype=np.float64)
        s0_t = torch.as_tensor(s0_np, dtype=torch.float64, device=dev).unsqueeze(0)
    else:
        exp_ids = reflections["id"].as_numpy_array()
        s0_list = [
            np.array(exp.beam.get_s0(), dtype=np.float64)
            if exp.beam is not None
            else np.zeros(3, dtype=np.float64)
            for exp in experiments
        ]
        s0_table = torch.as_tensor(
            np.stack(s0_list, axis=0), dtype=torch.float64, device=dev
        )
        ids_t = torch.as_tensor(
            np.clip(exp_ids, 0, len(experiments) - 1), dtype=torch.long, device=dev
        )
        s0_t = s0_table[ids_t]

    diff = s1_t - s0_t
    d_star = torch.linalg.norm(diff, dim=1).clamp(min=1e-30)
    d = 1.0 / d_star
    return flex.double(np.ascontiguousarray(d.cpu().numpy(), dtype=np.float64))


def compute_partiality_gpu(
    experiments, reflections, sigma_m: float | list[float], device=None
) -> flex.double:
    """
    Compute Gaussian RS partiality on the GPU using PyTorch:
      c = |zeta| / (sqrt(2) * sigma_m)
      p = 0.5 * (erf(c * (phib - phi)) - erf(c * (phia - phi)))
    """
    n = len(reflections)
    if n == 0:
        return flex.double()

    dev = _get_device(device)
    # Check if any experiment is 2D/stills (no scan or oscillation == 0)
    for exp in experiments:
        if (
            exp.scan is None
            or exp.goniometer is None
            or exp.scan.get_oscillation()[1] == 0
        ):
            # Fallback to DIALS PartialityCalculator2D/3D via model
            result = flex.double(n, 1.0)
            return result

    s1_np = reflections["s1"].as_numpy_array().reshape(-1, 3)
    bbox_np = _bbox_to_numpy(reflections["bbox"])
    xyz_mm_np = reflections["xyzcal.mm"].as_numpy_array().reshape(-1, 3)
    exp_ids = (
        reflections["id"].as_numpy_array()
        if "id" in reflections
        else np.zeros(n, dtype=np.int64)
    )

    s1_t = torch.as_tensor(s1_np, dtype=torch.float64, device=dev)
    z0_t = torch.as_tensor(bbox_np[:, 4], dtype=torch.float64, device=dev)
    z1_t = torch.as_tensor(bbox_np[:, 5], dtype=torch.float64, device=dev)
    phi_t = torch.as_tensor(xyz_mm_np[:, 2], dtype=torch.float64, device=dev)
    ids_t = torch.as_tensor(
        np.clip(exp_ids, 0, len(experiments) - 1), dtype=torch.long, device=dev
    )

    s0_list = []
    m2_list = []
    osc0_list = []
    dosc_list = []
    zmin_list = []
    zmax_list = []
    sigm_list = []
    for i, exp in enumerate(experiments):
        s0_list.append(np.array(exp.beam.get_s0(), dtype=np.float64))
        m2 = np.array(exp.goniometer.get_rotation_axis(), dtype=np.float64)
        m2_norm = np.linalg.norm(m2)
        if m2_norm > 0:
            m2 = m2 / m2_norm
        m2_list.append(m2)
        osc = exp.scan.get_oscillation(deg=False)
        arr_range = exp.scan.get_array_range()
        osc0_list.append(osc[0] - arr_range[0] * osc[1])
        dosc_list.append(osc[1])
        zmin_list.append(arr_range[0])
        zmax_list.append(arr_range[1])
        sm = sigma_m[i] if isinstance(sigma_m, (list, tuple)) else float(sigma_m)
        sigm_list.append(sm)

    s0_t = torch.as_tensor(np.stack(s0_list), dtype=torch.float64, device=dev)[ids_t]
    m2_t = torch.as_tensor(np.stack(m2_list), dtype=torch.float64, device=dev)[ids_t]
    osc0_t = torch.as_tensor(osc0_list, dtype=torch.float64, device=dev)[ids_t]
    dosc_t = torch.as_tensor(dosc_list, dtype=torch.float64, device=dev)[ids_t]
    zmin_t = torch.as_tensor(zmin_list, dtype=torch.float64, device=dev)[ids_t]
    zmax_t = torch.as_tensor(zmax_list, dtype=torch.float64, device=dev)[ids_t]
    sigm_t = torch.as_tensor(sigm_list, dtype=torch.float64, device=dev)[ids_t]

    cross = torch.linalg.cross(s1_t, s0_t)
    e1 = cross / torch.linalg.norm(cross, dim=1, keepdim=True).clamp(min=1e-30)
    zeta = torch.abs(torch.sum(e1 * m2_t, dim=1))

    za = torch.maximum(zmin_t, z0_t)
    zb = torch.minimum(zmax_t, z1_t)
    phia = osc0_t + za * dosc_t
    phib = osc0_t + zb * dosc_t

    c = zeta / (math.sqrt(2.0) * sigm_t.clamp(min=1e-30))
    p = 0.5 * (
        torch.special.erf(c * (phib - phi_t)) - torch.special.erf(c * (phia - phi_t))
    )
    invalid = (zb <= za) | (sigm_t <= 0)
    p = torch.where(invalid, torch.zeros_like(p), p)
    return flex.double(np.ascontiguousarray(p.cpu().numpy(), dtype=np.float64))


class ComputeEsdBeamDivergenceGPU:
    """
    GPU-accelerated calculation of the E.S.D of the beam divergence (`sigma_b`).
    Port of `dials.algorithms.profile_model.gaussian_rs.calculator.ComputeEsdBeamDivergence`.
    Uses PyTorch on GPU to compute the intensity-weighted angular variance of all
    shoebox pixels around the reflection centroid.
    """

    def __init__(self, detector, reflections, centroid_definition="s1", device=None):
        assert centroid_definition in ["com", "s1"]
        self._device = _get_device(device)
        shoeboxes = reflections["shoebox"]
        s1_all = reflections["s1"]

        # Extract valid shoebox pixel coordinates and intensities, then evaluate
        # the angular variance in a single batched GPU tensor operation.
        all_px_x = []
        all_px_y = []
        all_vals = []
        all_panels = []
        all_refl_idx = []
        refl_s1_list = []

        n_refls = len(reflections)
        valid_refl_count = 0
        for r in range(n_refls):
            sb = shoeboxes[r]
            mask_np = sb.mask.as_numpy_array()
            nz, ny, nx = mask_np.shape
            valid_mask = mask_np != 0
            if not np.any(valid_mask):
                continue
            data_np = sb.data.as_numpy_array()
            vals = data_np[valid_mask]
            total_val = float(np.sum(vals))
            if total_val <= 1.0:
                continue

            ks, js, is_ = np.nonzero(valid_mask)
            x0, _, y0, _, _, _ = sb.bbox
            px_x = x0 + is_ + 0.5
            px_y = y0 + js + 0.5

            all_px_x.append(px_x)
            all_px_y.append(px_y)
            all_vals.append(vals)
            all_panels.append(np.full(len(vals), sb.panel, dtype=np.int64))
            all_refl_idx.append(np.full(len(vals), valid_refl_count, dtype=np.int64))
            refl_s1_list.append(s1_all[r])
            valid_refl_count += 1

        if valid_refl_count == 0:
            raise RuntimeError("No reflections remaining for profile modelling")

        px_x_np = np.concatenate(all_px_x).astype(np.float64)
        px_y_np = np.concatenate(all_px_y).astype(np.float64)
        vals_np = np.concatenate(all_vals).astype(np.float64)
        panels_np = np.concatenate(all_panels)
        refl_idx_np = np.concatenate(all_refl_idx)

        dev = self._device
        px_x_t = torch.as_tensor(px_x_np, dtype=torch.float64, device=dev)
        px_y_t = torch.as_tensor(px_y_np, dtype=torch.float64, device=dev)
        vals_t = torch.as_tensor(vals_np, dtype=torch.float64, device=dev)
        panels_t = torch.as_tensor(panels_np, dtype=torch.long, device=dev)
        refl_idx_t = torch.as_tensor(refl_idx_np, dtype=torch.long, device=dev)

        # Panel geometry matrices on GPU: lab = origin + mm_x * fast + mm_y * slow
        # (including ParallaxCorrectedPxMmStrategy offset when active)
        n_panels = len(detector)
        orig_list = []
        fast_list = []
        slow_list = []
        norm_list = []
        px_sz_x = []
        px_sz_y = []
        mu_list = []
        t0_list = []
        has_parallax_list = []
        for p in detector:
            orig_list.append(np.array(p.get_origin(), dtype=np.float64))
            fast_list.append(np.array(p.get_fast_axis(), dtype=np.float64))
            slow_list.append(np.array(p.get_slow_axis(), dtype=np.float64))
            norm_list.append(np.array(p.get_normal(), dtype=np.float64))
            ps = p.get_pixel_size()
            px_sz_x.append(float(ps[0]))
            px_sz_y.append(float(ps[1]))
            strat_name = type(p.get_px_mm_strategy()).__name__
            is_parallax = "Parallax" in strat_name and p.get_mu() > 0 and p.get_thickness() > 0
            has_parallax_list.append(is_parallax)
            mu_list.append(float(p.get_mu()) if is_parallax else 0.0)
            t0_list.append(float(p.get_thickness()) if is_parallax else 0.0)

        orig_tab = torch.as_tensor(np.stack(orig_list), dtype=torch.float64, device=dev)
        fast_tab = torch.as_tensor(np.stack(fast_list), dtype=torch.float64, device=dev)
        slow_tab = torch.as_tensor(np.stack(slow_list), dtype=torch.float64, device=dev)
        norm_tab = torch.as_tensor(np.stack(norm_list), dtype=torch.float64, device=dev)
        sx_tab = torch.as_tensor(px_sz_x, dtype=torch.float64, device=dev)
        sy_tab = torch.as_tensor(px_sz_y, dtype=torch.float64, device=dev)
        mu_tab = torch.as_tensor(mu_list, dtype=torch.float64, device=dev)
        t0_tab = torch.as_tensor(t0_list, dtype=torch.float64, device=dev)
        par_tab = torch.as_tensor(has_parallax_list, dtype=torch.bool, device=dev)

        def _px_to_lab_gpu(px_x_in, px_y_in, pids_in):
            orig = orig_tab[pids_in]
            fast = fast_tab[pids_in]
            slow = slow_tab[pids_in]
            norm = norm_tab[pids_in]
            mm_x = px_x_in * sx_tab[pids_in]
            mm_y = px_y_in * sy_tab[pids_in]
            s1_0 = orig + mm_x.unsqueeze(1) * fast + mm_y.unsqueeze(1) * slow
            if any(has_parallax_list):
                is_par = par_tab[pids_in]
                mu = mu_tab[pids_in].clamp(min=1e-30)
                t0 = t0_tab[pids_in]
                s1_u = s1_0 / torch.linalg.norm(s1_0, dim=1, keepdim=True).clamp(min=1e-30)
                cos_t = torch.abs(torch.sum(s1_u * norm, dim=1)).clamp(min=1e-30)
                inv_mu = 1.0 / mu
                la = inv_mu - (t0 / cos_t + inv_mu) * torch.exp(-mu * t0 / cos_t)
                la = torch.where(is_par, la, torch.zeros_like(la))
                mm_x = mm_x - la * torch.sum(s1_u * fast, dim=1)
                mm_y = mm_y - la * torch.sum(s1_u * slow, dim=1)
                s1_0 = orig + mm_x.unsqueeze(1) * fast + mm_y.unsqueeze(1) * slow
            return s1_0

        s1_px = _px_to_lab_gpu(px_x_t, px_y_t, panels_t)

        if centroid_definition == "com":
            xyz_all = reflections["xyzobs.px.value"]
            cent_list = []
            for r in range(n_refls):
                sb = shoeboxes[r]
                m_np = sb.mask.as_numpy_array()
                valid_m = m_np != 0
                if not np.any(valid_m):
                    continue
                if float(np.sum(sb.data.as_numpy_array()[valid_m])) <= 1.0:
                    continue
                cent_list.append(detector[sb.panel].get_pixel_lab_coord(xyz_all[r][0:2]))
            refl_s1_t = torch.as_tensor(
                np.array(cent_list, dtype=np.float64),
                dtype=torch.float64,
                device=dev,
            )
        else:
            refl_s1_t = torch.as_tensor(
                np.array(refl_s1_list, dtype=np.float64),
                dtype=torch.float64,
                device=dev,
            )

        s1_cent = refl_s1_t[refl_idx_t]
        # Angle between s1_cent and s1_px: atan2(|a x b|, a . b) for high precision at small angles
        cross_norm = torch.linalg.norm(torch.linalg.cross(s1_cent, s1_px), dim=1)
        dot_prod = torch.sum(s1_cent * s1_px, dim=1)
        theta = torch.atan2(cross_norm, dot_prod)

        sum_v_theta2 = torch.zeros(valid_refl_count, dtype=torch.float64, device=dev)
        sum_v = torch.zeros(valid_refl_count, dtype=torch.float64, device=dev)
        sum_v_theta2.index_add_(0, refl_idx_t, vals_t * (theta * theta))
        sum_v.index_add_(0, refl_idx_t, vals_t)

        valid_r = sum_v > 1.0
        variances = (sum_v_theta2[valid_r] / (sum_v[valid_r] - 1.0)).cpu().numpy()
        if len(variances) == 0:
            raise RuntimeError("No reflections remaining for profile modelling")

        self._variances = flex.double(np.ascontiguousarray(variances, dtype=np.float64))
        self._sigma = math.sqrt(float(np.mean(variances)))

    def sigma(self) -> float:
        return self._sigma

    def variances(self) -> flex.double:
        return self._variances


class FractionOfObservedIntensityGPU:
    """
    GPU-accelerated calculation of fraction of observed intensity as a function of sigma_m.
    Matches `dials.algorithms.profile_model.gaussian_rs.calculator.FractionOfObservedIntensity`.
    """

    def __init__(self, crystal, beam, detector, goniometer, scan, reflections, device=None):
        self._device = _get_device(device)
        s0 = beam.get_s0()
        m2 = goniometer.get_rotation_axis()
        s1 = reflections["s1"]
        bbox = reflections["bbox"]
        phi = reflections["xyzcal.mm"].parts()[2]

        zeta = compute_zeta_gpu(
            type("Exp", (), {"beam": beam, "goniometer": goniometer})(),
            s1,
            device=self._device,
        )
        z0 = bbox.parts()[4]
        z1 = bbox.parts()[5]
        phi0 = scan.get_angle_from_array_index(z0.as_double(), deg=False)
        phi1 = scan.get_angle_from_array_index(z1.as_double(), deg=False)

        dphi0 = flex.abs(phi0 - phi)
        dphi1 = flex.abs(phi1 - phi)
        dphi = flex.min(dphi0, dphi1)
        self._Z_t = torch.as_tensor(
            (0.5 * dphi * flex.abs(zeta)).as_numpy_array(),
            dtype=torch.float64,
            device=self._device,
        )

    def __call__(self, sigma_m: float) -> float:
        if sigma_m <= 0:
            return 0.0
        val = torch.mean(torch.special.erf(self._Z_t / sigma_m)).item()
        return float(val)


class _ExtendedEstimatorGPU:
    """
    GPU-accelerated maximum-likelihood estimator for `sigma_m` (`algorithm="extended"`).
    Matches `ComputeEsdReflectingRange.ExtendedEstimator` in
    `dials/algorithms/profile_model/gaussian_rs/calculator.py`, evaluating the
    log-likelihood across all reflection frames on GPU using `torch.special.erf`
    and `index_add_`.
    """

    def __init__(self, crystal, beam, detector, goniometer, scan, reflections, device=None):
        self._device = _get_device(device)
        s1 = reflections["s1"]
        phi = reflections["xyzcal.mm"].parts()[2].as_numpy_array()
        bbox = _bbox_to_numpy(reflections["bbox"])
        shoeboxes = reflections["shoebox"]

        zeta_np = compute_zeta_gpu(
            type("Exp", (), {"beam": beam, "goniometer": goniometer})(),
            s1,
            device=self._device,
        ).as_numpy_array()

        osc = scan.get_oscillation(deg=False)
        arr_range = scan.get_array_range()
        osc0 = osc[0] - arr_range[0] * osc[1]
        dosc = osc[1]

        # Mask code Valid | Foreground = 1 | 4 = 5
        mask_code = 5

        e1_list = []
        e2_list = []
        n_list = []
        refl_idx_list = []
        k_list = []

        valid_refl_id = 0
        sqrt2 = math.sqrt(2.0)

        for r in range(len(reflections)):
            sb = shoeboxes[r]
            mask_np = flumpy.to_numpy(sb.mask)
            data_np = flumpy.to_numpy(sb.data)
            fg_valid = mask_np == mask_code
            # Sum foreground data per z-slice
            frame_sums = np.sum(np.where(fg_valid, data_np, 0.0), axis=(1, 2))
            pos_frames = np.where(frame_sums > 0)[0]
            if len(pos_frames) == 0:
                continue

            z0 = bbox[r, 4]
            phi_r = phi[r]
            abs_zeta = abs(zeta_np[r])
            factor = abs_zeta / sqrt2

            f_indices = z0 + pos_frames
            phi0_f = osc0 + f_indices * dosc
            phi1_f = phi0_f + dosc
            tau_f = (phi1_f + phi0_f) * 0.5 - phi_r
            dphi_half = (phi1_f - phi0_f) * 0.5

            e1_f = (tau_f + dphi_half) * factor
            e2_f = (tau_f - dphi_half) * factor
            n_f = frame_sums[pos_frames]

            e1_list.append(e1_f)
            e2_list.append(e2_f)
            n_list.append(n_f)
            refl_idx_list.append(np.full(len(pos_frames), valid_refl_id, dtype=np.int64))
            k_list.append(float(np.sum(n_f)))
            valid_refl_id += 1

        if valid_refl_id == 0:
            raise RuntimeError("0 reflections passes suitability criteria for profile modelling")

        dev = self._device
        self._e1_t = torch.as_tensor(
            np.concatenate(e1_list), dtype=torch.float64, device=dev
        )
        self._e2_t = torch.as_tensor(
            np.concatenate(e2_list), dtype=torch.float64, device=dev
        )
        self._n_t = torch.as_tensor(
            np.concatenate(n_list), dtype=torch.float64, device=dev
        )
        self._refl_idx_t = torch.as_tensor(
            np.concatenate(refl_idx_list), dtype=torch.long, device=dev
        )
        self._K_t = torch.as_tensor(k_list, dtype=torch.float64, device=dev)
        self._num_valid_refls = valid_refl_id

        start = math.log(0.1 * math.pi / 180.0)
        stop = math.log(1.0 * math.pi / 180.0)
        starting_simplex = [flex.double([start]), flex.double([stop])]
        optimizer = simplex.simplex_opt(
            1, matrix=starting_simplex, evaluator=self, tolerance=1e-3
        )
        self.sigma = math.exp(optimizer.get_solution()[0])

    def target(self, log_sigma: flex.double) -> float:
        sigma_m = math.exp(float(log_sigma[0]))
        z = (
            torch.special.erf(self._e1_t / sigma_m)
            - torch.special.erf(self._e2_t / sigma_m)
        ) * 0.5
        z = torch.clamp(z, min=1e-10)
        Z = torch.zeros(self._num_valid_refls, dtype=torch.float64, device=self._device)
        Z.index_add_(0, self._refl_idx_t, z)
        log_Z = torch.log(Z)
        L = (
            torch.sum(self._n_t * torch.log(z))
            - torch.sum(self._K_t * log_Z)
            + torch.sum(log_Z)
        )
        return -float(L.item())


class _BasicEstimatorGPU:
    """
    GPU-accelerated maximum-likelihood estimator for `sigma_m` (`algorithm="basic"`).
    Matches `ComputeEsdReflectingRange.Estimator` in
    `dials/algorithms/profile_model/gaussian_rs/calculator.py`.
    """

    def __init__(self, crystal, beam, detector, goniometer, scan, reflections, device=None):
        self._device = _get_device(device)
        s1 = reflections["s1"]
        phi = reflections["xyzcal.mm"].parts()[2].as_numpy_array()
        bbox = _bbox_to_numpy(reflections["bbox"])
        shoeboxes = reflections["shoebox"]

        zeta_np = compute_zeta_gpu(
            type("Exp", (), {"beam": beam, "goniometer": goniometer})(),
            s1,
            device=self._device,
        ).as_numpy_array()

        osc = scan.get_oscillation(deg=False)
        arr_range = scan.get_array_range()
        osc0 = osc[0] - arr_range[0] * osc[1]
        dosc = osc[1]
        mask_code = 5  # Valid | Foreground
        sqrt2 = math.sqrt(2.0)

        e1_list = []
        e2_list = []
        for r in range(len(reflections)):
            sb = shoeboxes[r]
            mask_np = flumpy.to_numpy(sb.mask)
            fg_counts = np.sum(mask_np == mask_code, axis=(1, 2))
            pos_frames = np.where(fg_counts > 0)[0]
            if len(pos_frames) == 0:
                continue

            z0 = bbox[r, 4]
            phi_r = phi[r]
            factor = abs(zeta_np[r]) / sqrt2
            f_indices = z0 + pos_frames
            phi0_f = osc0 + f_indices * dosc
            phi1_f = phi0_f + dosc
            tau_f = (phi1_f + phi0_f) * 0.5 - phi_r
            dphi_half = (phi1_f - phi0_f) * 0.5

            e1_list.append((tau_f + dphi_half) * factor)
            e2_list.append((tau_f - dphi_half) * factor)

        if not e1_list:
            raise RuntimeError("No foreground pixels in reflections")

        dev = self._device
        self._e1_t = torch.as_tensor(
            np.concatenate(e1_list), dtype=torch.float64, device=dev
        )
        self._e2_t = torch.as_tensor(
            np.concatenate(e2_list), dtype=torch.float64, device=dev
        )

        start = math.log(0.1 * math.pi / 180.0)
        stop = math.log(1.0 * math.pi / 180.0)
        starting_simplex = [flex.double([start]), flex.double([stop])]
        optimizer = simplex.simplex_opt(
            1, matrix=starting_simplex, evaluator=self, tolerance=1e-3
        )
        self.sigma = math.exp(optimizer.get_solution()[0])

    def target(self, log_sigma: flex.double) -> float:
        sigma_m = math.exp(float(log_sigma[0]))
        p = (
            torch.special.erf(self._e1_t / sigma_m)
            - torch.special.erf(self._e2_t / sigma_m)
        ) * 0.5
        p = torch.clamp(p, min=1e-10)
        return -float(torch.sum(torch.log(p)).item())


class ComputeEsdReflectingRangeGPU:
    """
    GPU-accelerated calculation of the E.S.D of the reflecting range (`sigma_m`).
    Port of `dials.algorithms.profile_model.gaussian_rs.calculator.ComputeEsdReflectingRange`.
    """

    def __init__(
        self,
        crystal,
        beam,
        detector,
        goniometer,
        scan,
        reflections,
        algorithm="extended",
        device=None,
    ):
        if algorithm == "basic":
            estimator = _BasicEstimatorGPU(
                crystal, beam, detector, goniometer, scan, reflections, device=device
            )
        elif algorithm == "crude":
            estimator = ComputeEsdReflectingRange.CrudeEstimator(
                crystal, beam, detector, goniometer, scan, reflections
            )
        elif algorithm == "extended":
            estimator = _ExtendedEstimatorGPU(
                crystal, beam, detector, goniometer, scan, reflections, device=device
            )
        else:
            raise RuntimeError(f'Unknown algorithm: "{algorithm}"')

        self._sigma = estimator.sigma

    def sigma(self) -> float:
        return self._sigma


class ProfileModelCalculatorGPU(ProfileModelCalculator):
    """
    GPU-accelerated calculator for Gaussian RS profile model parameters (`sigma_b`, `sigma_m`).
    Drop-in replacement for `dials.algorithms.profile_model.gaussian_rs.calculator.ProfileModelCalculator`.
    """

    def __init__(
        self,
        reflections,
        crystal,
        beam,
        detector,
        goniometer,
        scan,
        min_zeta=0.05,
        algorithm="extended",
        centroid_definition="s1",
        device=None,
    ):
        if len(reflections) == 0:
            raise RuntimeError("No reflections remaining for profile modelling")

        dev = _get_device(device)
        if goniometer is None or scan is None or scan.is_still():
            calc_b = ComputeEsdBeamDivergenceGPU(
                detector, reflections, centroid_definition, device=dev
            )
            self._sigma_b = calc_b.sigma()
            self._sigma_m = 0.0
        else:
            sub = _select_reflections_for_sigma_calc(reflections)
            zeta = compute_zeta_gpu(
                type("Exp", (), {"beam": beam, "goniometer": goniometer})(),
                sub["s1"],
                device=dev,
            )
            sub = sub.select(flex.abs(zeta) >= min_zeta)
            calc_b = ComputeEsdBeamDivergenceGPU(
                detector, sub, centroid_definition, device=dev
            )
            self._sigma_b = calc_b.sigma()
            calc_m = ComputeEsdReflectingRangeGPU(
                crystal,
                beam,
                detector,
                goniometer,
                scan,
                sub,
                algorithm=algorithm,
                device=dev,
            )
            self._sigma_m = calc_m.sigma()


class ScanVaryingProfileModelCalculatorGPU(ScanVaryingProfileModelCalculator):
    """
    GPU-accelerated calculator for scan-varying Gaussian RS profile model parameters.
    Drop-in replacement for `ScanVaryingProfileModelCalculator`.
    """

    def __init__(
        self,
        reflections,
        crystal,
        beam,
        detector,
        goniometer,
        scan,
        min_zeta=0.05,
        algorithm="extended",
        centroid_definition="s1",
        device=None,
    ):
        self._device = _get_device(device)
        super().__init__(
            reflections,
            crystal,
            beam,
            detector,
            goniometer,
            scan,
            min_zeta=min_zeta,
            algorithm=algorithm,
            centroid_definition=centroid_definition,
        )

    def _compute_for_subset(
        self,
        reflections,
        crystal,
        beam,
        detector,
        goniometer,
        scan,
        min_zeta=0.05,
        algorithm="extended",
        centroid_definition="s1",
    ):
        return ProfileModelCalculatorGPU(
            reflections,
            crystal,
            beam,
            detector,
            goniometer,
            scan,
            min_zeta=min_zeta,
            algorithm=algorithm,
            centroid_definition=centroid_definition,
            device=getattr(self, "_device", None),
        )


class GaussianRSProfileModelGPU(GaussianRSProfileModel):
    """
    GPU-accelerated Gaussian RS profile model subclassing DIALS `gaussian_rs.Model`
    so `isinstance(profile, ProfileModelExt)` and C++ extensions remain 100% compatible.
    """

    def compute_mask(
        self,
        reflections,
        crystal,
        beam,
        detector,
        goniometer=None,
        scan=None,
        image_volume=None,
        **kwargs,
    ):
        if image_volume is None:
            from dials.algorithms.profile_model.gaussian_rs import MaskCalculator
            from dials_gpu.integration import _shoebox_pack_ext

            delta_b = self._n_sigma * self._sigma_b
            delta_m = self._n_sigma * self._sigma_m
            mask_foreground = MaskCalculator(
                crystal, beam, detector, goniometer, scan, delta_b, delta_m
            )
            _shoebox_pack_ext.parallel_compute_mask(
                mask_foreground,
                reflections["shoebox"],
                reflections["s1"],
                reflections["xyzcal.px"].parts()[2],
                reflections["panel"],
            )
            return None
        return super().compute_mask(
            reflections,
            crystal,
            beam,
            detector,
            goniometer=goniometer,
            scan=scan,
            image_volume=image_volume,
            **kwargs,
        )

    @classmethod
    def create_from_reflections(
        cls,
        params,
        reflections,
        crystal,
        beam,
        detector,
        goniometer=None,
        scan=None,
        profile=None,
    ):
        if not len(reflections) >= params.gaussian_rs.min_spots.overall:
            if scan is not None:
                num_degrees = scan.get_num_images() * scan.get_oscillation()[1]
                spots_per_degree = len(reflections) / num_degrees
                if not spots_per_degree >= params.gaussian_rs.min_spots.per_degree:
                    raise RuntimeError(
                        f"Too few reflections for profile modelling: got {len(reflections)}"
                    )
            else:
                raise RuntimeError(
                    f"Too few reflections for profile modelling: expected >= {params.gaussian_rs.min_spots.overall}, got {len(reflections)}"
                )
        if not len(reflections) > 0:
            raise RuntimeError("No reflections remaining for profile modelling.")

        if [
            params.gaussian_rs.parameters.sigma_b,
            params.gaussian_rs.parameters.sigma_m,
        ].count(None) == 1:
            raise RuntimeError("sigma_b and sigma_m parameters must both be set")
        if (
            params.gaussian_rs.parameters.sigma_b is not None
            and params.gaussian_rs.parameters.sigma_m is not None
        ):
            return cls(
                params=params,
                n_sigma=params.gaussian_rs.parameters.n_sigma,
                sigma_b=params.gaussian_rs.parameters.sigma_b,
                sigma_m=params.gaussian_rs.parameters.sigma_m,
                deg=True,
            )

        if not params.gaussian_rs.scan_varying:
            Calculator = ProfileModelCalculatorGPU
        else:
            Calculator = ScanVaryingProfileModelCalculatorGPU
        calculator = Calculator(
            reflections,
            crystal,
            beam,
            detector,
            goniometer,
            scan,
            params.gaussian_rs.filter.min_zeta,
            algorithm=params.gaussian_rs.sigma_m_algorithm,
            centroid_definition=params.gaussian_rs.centroid_definition,
        )
        return cls(
            params=params,
            n_sigma=params.gaussian_rs.parameters.n_sigma,
            sigma_b=calculator.sigma_b(),
            sigma_m=calculator.sigma_m(),
        )


class ProfileModelFactoryGPU(ProfileModelFactory):
    """
    GPU-accelerated factory for creating profile models on experiments.
    """

    @staticmethod
    def create(params, experiments, reflections=None):
        if params.profile.algorithm is None:
            if all(e.scan is None for e in experiments):
                params.profile.algorithm = "nave"
            else:
                params.profile.algorithm = "gaussian_rs"

        if params.profile.algorithm == "gaussian_rs":
            if reflections is not None:
                assert len(reflections) > 0
                reflections_list = reflections.split_by_experiment_id()
                assert len(reflections_list) == len(experiments)
                for expr, refl in zip(experiments, reflections_list):
                    expr.profile = GaussianRSProfileModelGPU.create(
                        params.profile,
                        refl,
                        expr.crystal,
                        expr.beam,
                        expr.detector,
                        expr.goniometer,
                        expr.scan,
                        expr.profile,
                    )
            else:
                for expr in experiments:
                    expr.profile = GaussianRSProfileModelGPU.create(
                        params.profile,
                        None,
                        expr.crystal,
                        expr.beam,
                        expr.detector,
                        expr.goniometer,
                        expr.scan,
                        expr.profile,
                    )
            return experiments

        return ProfileModelFactory.create(params, experiments, reflections)
