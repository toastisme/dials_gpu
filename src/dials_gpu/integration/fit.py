from __future__ import annotations

import math

import numpy as np
import torch

from dials.algorithms.integration import Corrections, CorrectionsMulti
from dials.algorithms.integration.fit import ProfileFitter
from dials.array_family import flex
from dials_gpu.integration.background import _get_device


class ProfileFitterGPU:
    """
    GPU-accelerated Iterative Reweighted Least Squares (IRLS) profile fitter
    using PyTorch (`torch.float64`).
    Matches `dials.algorithms.integration.fit.ProfileFitter` (`dials/algorithms/integration/fit/fitting.h`),
    supporting both single-reflection fitting (`3D` profile `p`) and multi-reflection
    deconvolution (`4D` profile `p` of shape `(N, Z, Y, X)`).
    """

    def __init__(self, d, b, m, p, eps: float = 1e-3, maxiter: int = 10, device=None):
        self._device = _get_device(device)
        assert maxiter >= 1 and eps >= 0.0

        d_np = d.as_numpy_array() if hasattr(d, "as_numpy_array") else np.asarray(d)
        b_np = b.as_numpy_array() if hasattr(b, "as_numpy_array") else np.asarray(b)
        m_np = m.as_numpy_array() if hasattr(m, "as_numpy_array") else np.asarray(m)
        p_np = p.as_numpy_array() if hasattr(p, "as_numpy_array") else np.asarray(p)

        if p_np.ndim == 3:
            p_np = p_np[np.newaxis, ...]
        assert p_np.ndim == 4
        n_refl = p_np.shape[0]
        assert d_np.shape == p_np.shape[1:]
        assert b_np.shape == d_np.shape
        assert m_np.shape == d_np.shape

        dev = self._device
        d_t = torch.as_tensor(d_np.ravel(), dtype=torch.float64, device=dev)
        b_t = torch.as_tensor(b_np.ravel(), dtype=torch.float64, device=dev)
        m_t = torch.as_tensor(m_np.ravel().astype(bool), dtype=torch.bool, device=dev)
        p_t = torch.as_tensor(
            p_np.reshape(n_refl, -1), dtype=torch.float64, device=dev
        )

        self._maxiter = int(maxiter)
        self._eps = float(eps)

        if n_refl == 1:
            self._fit_1_gpu(d_t, b_t, m_t, p_t[0])
        else:
            self._fit_n_gpu(d_t, b_t, m_t, p_t)

    def _fit_1_gpu(
        self,
        d_t: torch.Tensor,
        b_t: torch.Tensor,
        m_t: torch.Tensor,
        p_t: torch.Tensor,
    ):
        d_m = d_t[m_t]
        b_m = b_t[m_t]
        p_m = p_t[m_t]

        sumd = float(torch.sum(d_m).item())
        sumb = float(torch.sum(b_m).item())
        sump = float(torch.sum(p_m).item())
        if sump <= 0.0:
            raise RuntimeError("ProfileFitterGPU: sum(p) <= 0")

        I0 = sumd - sumb
        V0 = 0.0

        active = p_m > 0.0
        d_act = d_m[active]
        b_act = b_m[active]
        p_act = p_m[active]
        diff_db = d_act - b_act
        p2_act = p_act * p_act
        abs_b_act = torch.abs(b_act)

        niter = self._maxiter
        error = 0.0
        for it in range(self._maxiter):
            v = 1e-10 + abs_b_act + torch.abs(I0 * p_act)
            sum2 = float(torch.sum(p2_act / v).item())
            if sum2 <= 0.0:
                raise RuntimeError("ProfileFitterGPU: sum2 <= 0")
            sum1 = float(torch.sum(diff_db * p_act / v).item())
            I_val = sum1 / sum2
            V_val = abs(I_val) + abs(sumb)
            error = abs(I_val - I0)
            I0 = I_val
            V0 = V_val
            if error < self._eps:
                niter = it
                break

        if niter >= self._maxiter:
            I0 = sumd - sumb
            V0 = abs(I0) + abs(sumb)

        # Pearson correlation between x = I0 * p_m + b_m and y = d_m over m_t
        n_pts = d_m.numel()
        if n_pts > 0:
            x = I0 * p_m + b_m
            y = d_m
            xb = torch.mean(x)
            yb = torch.mean(y)
            dx = x - xb
            dy = y - yb
            sx = torch.sum(dx * dx)
            sy = torch.sum(dy * dy)
            sxy = torch.sum(dx * dy)
            denom = torch.sqrt(sx * sy).item()
            corr = float(sxy.item() / denom) if denom > 0.0 else 0.0
        else:
            corr = 0.0

        self._intensity = flex.double([I0])
        self._variance = flex.double([V0])
        self._correlation = corr
        self._niter = niter
        self._error = error

    def _fit_n_gpu(
        self,
        d_t: torch.Tensor,
        b_t: torch.Tensor,
        m_t: torch.Tensor,
        p_t: torch.Tensor,
    ):
        n_refl = p_t.shape[0]
        N_total = float(d_t.numel())
        d_m = d_t[m_t]
        b_m = b_t[m_t]
        p_m = p_t[:, m_t]  # (M, K)

        sumb = float(torch.sum(b_m).item())
        sump = torch.sum(p_m, dim=1)
        if torch.any(sump <= 0.0):
            raise RuntimeError("inversion_in_place: singular matrix")

        I0 = torch.ones(n_refl, dtype=torch.float64, device=d_t.device)
        diff_db = d_m - b_m

        niter = self._maxiter
        error = 0.0
        for it in range(self._maxiter):
            weights_I = torch.clamp(torch.abs(I0), min=1.0 / N_total)
            v = b_m + torch.sum(weights_I.unsqueeze(1) * p_m, dim=0)
            if torch.any(v <= 0.0):
                raise RuntimeError("ProfileFitterGPU: v <= 0")
            w_inv = 1.0 / v  # (K,)
            A = (p_m * w_inv.unsqueeze(0)) @ p_m.T  # (M, M)
            rhs = torch.sum(p_m * (diff_db * w_inv).unsqueeze(0), dim=1)  # (M,)
            I_new = torch.linalg.solve(A, rhs)
            diff = I_new - I0
            error = float(torch.sum(diff * diff).item())
            I0 = I_new
            if error < self._eps * self._eps:
                niter = it
                break

        V0 = torch.abs(I0) + sumb
        n_pts = d_m.numel()
        if n_pts > 0:
            x = b_m + torch.sum(I0.unsqueeze(1) * p_m, dim=0)
            y = d_m
            xb = torch.mean(x)
            yb = torch.mean(y)
            dx = x - xb
            dy = y - yb
            sx = torch.sum(dx * dx)
            sy = torch.sum(dy * dy)
            sxy = torch.sum(dx * dy)
            denom = torch.sqrt(sx * sy).item()
            corr = float(sxy.item() / denom) if denom > 0.0 else 0.0
        else:
            corr = 0.0

        self._intensity = flex.double(
            np.ascontiguousarray(I0.cpu().numpy(), dtype=np.float64)
        )
        self._variance = flex.double(
            np.ascontiguousarray(V0.cpu().numpy(), dtype=np.float64)
        )
        self._correlation = corr
        self._niter = niter
        self._error = error

    def intensity(self) -> flex.double:
        return self._intensity

    def variance(self) -> flex.double:
        return self._variance

    def correlation(self) -> float:
        return self._correlation

    def niter(self) -> int:
        return self._niter

    def maxiter(self) -> int:
        return self._maxiter

    def error(self) -> float:
        return self._error


def lp_correction_gpu(s0, p_n, p_f: float, m2, s1_flex, device=None) -> flex.double:
    """
    Compute the Lorentz-Polarization (`LP`) correction on the GPU using PyTorch (`torch.float64`).
    Matches `dials.algorithms.integration.lp_correction(s0, pn, pf, m2, s1)` (Kabsch 2010):
      L = |m2 . (s1 x s0)| / (|s1| * |s0|)
      P1 = (1 - 2*pf) * (1 - (pn . s1 / |s1|)^2)
      P2 = pf * (1 + (s1 . s0 / (|s1| * |s0|))^2)
      LP = L / (P1 + P2)
    """
    dev = _get_device(device)
    is_single = not isinstance(s1_flex, flex.vec3_double)
    if is_single:
        s1_np = np.array([ tuple(s1_flex) ], dtype=np.float64)
    else:
        if len(s1_flex) == 0:
            return flex.double()
        s1_np = s1_flex.as_numpy_array().reshape(-1, 3)

    s0_t = torch.as_tensor(np.array(s0, dtype=np.float64), dtype=torch.float64, device=dev)
    pn_t = torch.as_tensor(np.array(p_n, dtype=np.float64), dtype=torch.float64, device=dev)
    m2_t = torch.as_tensor(np.array(m2, dtype=np.float64), dtype=torch.float64, device=dev)
    s1_t = torch.as_tensor(s1_np, dtype=torch.float64, device=dev)

    s0_len = torch.linalg.norm(s0_t).clamp(min=1e-30)
    s1_len = torch.linalg.norm(s1_t, dim=1).clamp(min=1e-30)
    s1_s0_len = s1_len * s0_len

    cross = torch.linalg.cross(s1_t, s0_t.unsqueeze(0))
    L_val = torch.abs(torch.sum(m2_t.unsqueeze(0) * cross, dim=1)) / s1_s0_len

    pn_dot_s1 = torch.sum(pn_t.unsqueeze(0) * s1_t, dim=1) / s1_len
    P1 = (1.0 - 2.0 * float(p_f)) * (1.0 - pn_dot_s1 * pn_dot_s1)

    s1_dot_s0 = torch.sum(s1_t * s0_t.unsqueeze(0), dim=1) / s1_s0_len
    P2 = float(p_f) * (1.0 + s1_dot_s0 * s1_dot_s0)

    P_val = (P1 + P2).clamp(min=1e-30)
    lp = L_val / P_val

    if is_single:
        return float(lp[0].item())
    return flex.double(np.ascontiguousarray(lp.cpu().numpy(), dtype=np.float64))


def qe_correction_gpu(mu: float, t0: float, s1_flex, d_n, device=None) -> flex.double:
    """
    Compute the detector quantum efficiency (`QE`) correction on the GPU using PyTorch.
    Matches `dials.algorithms.integration.qe_correction(mu, t0, s1, dn)`:
      cos_angle = (s1 / |s1|) . (dn / |dn|)
      qe = 1 - exp(-mu * t0 / cos_angle)  if cos_angle > 0 else 0.0
    """
    assert mu >= 0 and t0 >= 0
    dev = _get_device(device)
    is_single = not isinstance(s1_flex, flex.vec3_double)
    if is_single:
        s1_np = np.array([ tuple(s1_flex) ], dtype=np.float64)
    else:
        if len(s1_flex) == 0:
            return flex.double()
        s1_np = s1_flex.as_numpy_array().reshape(-1, 3)

    s1_t = torch.as_tensor(s1_np, dtype=torch.float64, device=dev)
    dn_t = torch.as_tensor(np.array(d_n, dtype=np.float64), dtype=torch.float64, device=dev)

    s1_u = s1_t / torch.linalg.norm(s1_t, dim=1, keepdim=True).clamp(min=1e-30)
    dn_u = dn_t / torch.linalg.norm(dn_t).clamp(min=1e-30)

    cos_ang = torch.sum(s1_u * dn_u.unsqueeze(0), dim=1)
    qe = torch.where(
        cos_ang > 0.0,
        1.0 - torch.exp(-float(mu) * float(t0) / cos_ang.clamp(min=1e-30)),
        torch.zeros_like(cos_ang),
    )

    if is_single:
        return float(qe[0].item())
    return flex.double(np.ascontiguousarray(qe.cpu().numpy(), dtype=np.float64))


class CorrectionsGPU:
    """
    GPU-accelerated LP and QE corrections class matching `dials.algorithms.integration.Corrections`.
    """

    def __init__(self, beam, goniometer, detector=None, device=None):
        self._device = _get_device(device)
        self._s0 = beam.get_s0()
        self._pn = beam.get_polarization_normal()
        self._pf = beam.get_polarization_fraction()
        self._m2 = goniometer.get_rotation_axis() if goniometer is not None else (0.0, 0.0, 0.0)
        self._detector = detector

    def lp(self, s1):
        return lp_correction_gpu(
            self._s0, self._pn, self._pf, self._m2, s1, device=self._device
        )

    def qe(self, s1, panel):
        assert self._detector is not None
        is_single = not isinstance(s1, flex.vec3_double)
        if is_single:
            p = self._detector[int(panel)]
            return qe_correction_gpu(
                p.get_mu(), p.get_thickness(), s1, p.get_normal(), device=self._device
            )

        n = len(s1)
        if n == 0:
            return flex.double()
        s1_np = s1.as_numpy_array().reshape(-1, 3)
        panel_np = panel.as_numpy_array()
        dev = self._device

        mu_list = [float(p.get_mu()) for p in self._detector]
        t0_list = [float(p.get_thickness()) for p in self._detector]
        dn_list = [np.array(p.get_normal(), dtype=np.float64) for p in self._detector]

        pids = torch.as_tensor(panel_np.astype(np.int64), dtype=torch.long, device=dev)
        mu_t = torch.as_tensor(mu_list, dtype=torch.float64, device=dev)[pids]
        t0_t = torch.as_tensor(t0_list, dtype=torch.float64, device=dev)[pids]
        dn_t = torch.as_tensor(np.stack(dn_list), dtype=torch.float64, device=dev)[pids]
        s1_t = torch.as_tensor(s1_np, dtype=torch.float64, device=dev)

        s1_u = s1_t / torch.linalg.norm(s1_t, dim=1, keepdim=True).clamp(min=1e-30)
        dn_u = dn_t / torch.linalg.norm(dn_t, dim=1, keepdim=True).clamp(min=1e-30)
        cos_ang = torch.sum(s1_u * dn_u, dim=1)
        qe = torch.where(
            cos_ang > 0.0,
            1.0 - torch.exp(-mu_t * t0_t / cos_ang.clamp(min=1e-30)),
            torch.zeros_like(cos_ang),
        )
        # Note: if mu == 0, C++ Panel::get_qe or CorrectionsMulti sets qe = 1.0 in reflection_table.compute_corrections
        return flex.double(np.ascontiguousarray(qe.cpu().numpy(), dtype=np.float64))


def compute_corrections_gpu(
    reflections: flex.reflection_table, experiments, device=None
) -> flex.double:
    """
    GPU-accelerated calculation of `lp` and `qe` columns on a reflection table.
    Matches `reflections.compute_corrections(experiments)` in `dials/array_family/flex_ext.py`.
    """
    n_refl = len(reflections)
    if n_refl == 0:
        reflections["lp"] = flex.double()
        reflections["qe"] = flex.double()
        return flex.double()

    # Check if any experiment lacks goniometer (stills)
    if any(exp.goniometer is None or exp.beam is None for exp in experiments):
        return reflections.compute_corrections(experiments)

    dev = _get_device(device)
    s1_np = reflections["s1"].as_numpy_array().reshape(-1, 3)
    exp_ids = (
        reflections["id"].as_numpy_array()
        if "id" in reflections
        else np.zeros(n_refl, dtype=np.int64)
    )
    panel_ids = (
        reflections["panel"].as_numpy_array()
        if "panel" in reflections
        else np.zeros(n_refl, dtype=np.int64)
    )

    s0_list = []
    pn_list = []
    pf_list = []
    m2_list = []
    for exp in experiments:
        s0_list.append(np.array(exp.beam.get_s0(), dtype=np.float64))
        pn_list.append(np.array(exp.beam.get_polarization_normal(), dtype=np.float64))
        pf_list.append(float(exp.beam.get_polarization_fraction()))
        m2_list.append(np.array(exp.goniometer.get_rotation_axis(), dtype=np.float64))

    ids_t = torch.as_tensor(
        np.clip(exp_ids, 0, len(experiments) - 1), dtype=torch.long, device=dev
    )
    s0_t = torch.as_tensor(np.stack(s0_list), dtype=torch.float64, device=dev)[ids_t]
    pn_t = torch.as_tensor(np.stack(pn_list), dtype=torch.float64, device=dev)[ids_t]
    pf_t = torch.as_tensor(pf_list, dtype=torch.float64, device=dev)[ids_t]
    m2_t = torch.as_tensor(np.stack(m2_list), dtype=torch.float64, device=dev)[ids_t]
    s1_t = torch.as_tensor(s1_np, dtype=torch.float64, device=dev)

    s0_len = torch.linalg.norm(s0_t, dim=1).clamp(min=1e-30)
    s1_len = torch.linalg.norm(s1_t, dim=1).clamp(min=1e-30)
    s1_s0_len = s1_len * s0_len

    cross = torch.linalg.cross(s1_t, s0_t)
    L_val = torch.abs(torch.sum(m2_t * cross, dim=1)) / s1_s0_len
    pn_dot_s1 = torch.sum(pn_t * s1_t, dim=1) / s1_len
    P1 = (1.0 - 2.0 * pf_t) * (1.0 - pn_dot_s1 * pn_dot_s1)
    s1_dot_s0 = torch.sum(s1_t * s0_t, dim=1) / s1_s0_len
    P2 = pf_t * (1.0 + s1_dot_s0 * s1_dot_s0)
    lp_t = L_val / (P1 + P2).clamp(min=1e-30)

    lp_flex = flex.double(np.ascontiguousarray(lp_t.cpu().numpy(), dtype=np.float64))
    reflections["lp"] = lp_flex

    if experiments[-1].detector[0].get_mu() > 0:
        corr_multi = CorrectionsMulti()
        for exp in experiments:
            if (
                exp.goniometer is not None
                and exp.scan is not None
                and exp.scan.get_oscillation()[1] != 0.0
            ):
                corr_multi.append(Corrections(exp.beam, exp.goniometer, exp.detector))
            else:
                corr_multi.append(Corrections(exp.beam, exp.detector))
        qe_flex = corr_multi.qe(
            reflections["id"], reflections["s1"], reflections["panel"]
        )
        reflections["qe"] = qe_flex
    return lp_flex
