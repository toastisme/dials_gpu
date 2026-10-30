from __future__ import annotations

import math
import numpy as np
import torch
from dxtbx import flumpy

from dials.algorithms.refinement.prediction.managed_predictors import (
    ExperimentsPredictorFactory,
    LaueExperimentsPredictor,
    ScansExperimentsPredictor,
    StillsExperimentsPredictor,
    TOFExperimentsPredictor,
)
from dials.array_family import flex


def _rodrigues_rotate_batch(
    v: torch.Tensor, axis: torch.Tensor, angle: torch.Tensor
) -> torch.Tensor:
    """Rotate 3-vectors ``v`` (..., 3) around unit ``axis`` (..., 3) by ``angle`` (...) radians."""
    cos_a = torch.cos(angle).unsqueeze(-1)
    sin_a = torch.sin(angle).unsqueeze(-1)
    dot = (v * axis).sum(dim=-1, keepdim=True)
    cross = torch.cross(axis, v, dim=-1)
    return v * cos_a + cross * sin_a + axis * dot * (1.0 - cos_a)


def predict_scans_tensors_gpu(
    h: torch.Tensor,
    entering: torch.Tensor,
    phi_obs: torch.Tensor,
    UB: torch.Tensor,
    s0: torch.Tensor,
    D: torch.Tensor,
    F: torch.Tensor,
    S: torch.Tensor,
    e0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict scan reflections on GPU in float64.

    Args:
        h: (N, 3) Miller indices (float64)
        entering: (N,) bool tensor
        phi_obs: (N,) observed rotation angle in radians (float64)
        UB: (N, 3, 3) or (1, 3, 3) setting matrix
        s0: (N, 3) or (1, 3) beam vector
        D: (N, 3, 3) or (1, 3, 3) inverse detector d-matrix
        F: (N, 3, 3) or (1, 3, 3) goniometer fixed rotation matrix
        S: (N, 3, 3) or (1, 3, 3) goniometer setting rotation matrix
        e0: (N, 3) or (1, 3) unit datum spindle axis

    Returns:
        (x_mm, y_mm, phi_calc, s1, predicted_mask)
    """
    # pstar0 = F * UB * h
    UB_h = torch.matmul(UB, h.unsqueeze(-1)).squeeze(-1)
    pstar0 = torch.matmul(F, UB_h.unsqueeze(-1)).squeeze(-1)

    # s0 in datum frame: s0_datum = S^T * s0
    s0_datum = torch.matmul(S.transpose(-1, -2), s0.unsqueeze(-1)).squeeze(-1)
    m2 = e0 / torch.linalg.norm(e0, dim=-1, keepdim=True)

    m1_raw = torch.cross(m2, s0_datum, dim=-1)
    m1 = m1_raw / torch.linalg.norm(m1_raw, dim=-1, keepdim=True)
    m3_raw = torch.cross(m1, m2, dim=-1)
    m3 = m3_raw / torch.linalg.norm(m3_raw, dim=-1, keepdim=True)

    s0_d_m2 = (s0_datum * m2).sum(dim=-1)
    s0_d_m3 = (s0_datum * m3).sum(dim=-1)

    pstar0_len_sq = (pstar0 * pstar0).sum(dim=-1)
    s0_len_sq = (s0_datum * s0_datum).sum(dim=-1)

    pstar0_d_m1 = (pstar0 * m1).sum(dim=-1)
    pstar0_d_m2 = (pstar0 * m2).sum(dim=-1)
    pstar0_d_m3 = (pstar0 * m3).sum(dim=-1)

    pstar_d_m3 = (-0.5 * pstar0_len_sq - pstar0_d_m2 * s0_d_m2) / s0_d_m3
    rho_sq = pstar0_len_sq - pstar0_d_m2 * pstar0_d_m2
    rad = rho_sq - pstar_d_m3 * pstar_d_m3

    valid_ewald = (pstar0_len_sq <= 4.0 * s0_len_sq) & (rad >= 0.0)
    pstar_d_m1_abs = torch.sqrt(torch.clamp(rad, min=0.0))
    sign = torch.where(entering, 1.0, -1.0)
    pstar_d_m1 = sign * pstar_d_m1_abs

    cosphi = pstar_d_m1 * pstar0_d_m1 + pstar_d_m3 * pstar0_d_m3
    sinphi = pstar_d_m1 * pstar0_d_m3 - pstar_d_m3 * pstar0_d_m1
    phi_raw = torch.atan2(sinphi, cosphi)

    # Match 2*pi turn to phi_obs: resid = fmod(phi_raw - phi_obs, 2*pi) wrapped to [-pi, pi]
    two_pi = 2.0 * math.pi
    diff = torch.remainder(phi_raw - phi_obs + math.pi, two_pi) - math.pi
    phi_calc = phi_obs + diff

    # Rotated pstar = S * R(m2, phi_raw) * pstar0
    m2_expanded = m2.expand_as(pstar0)
    pstar_rot = _rodrigues_rotate_batch(pstar0, m2_expanded, phi_raw)
    pstar = torch.matmul(S, pstar_rot.unsqueeze(-1)).squeeze(-1)
    s1 = s0 + pstar

    # Detector intersection: pv = D * s1 = (u, v, w)
    pv = torch.matmul(D, s1.unsqueeze(-1)).squeeze(-1)
    u = pv[:, 0]
    v = pv[:, 1]
    w = pv[:, 2]
    predicted_mask = valid_ewald & (w > 0.0)

    w_safe = torch.where(predicted_mask, w, torch.ones_like(w))
    x_mm = torch.where(predicted_mask, u / w_safe, torch.zeros_like(u))
    y_mm = torch.where(predicted_mask, v / w_safe, torch.zeros_like(v))

    return x_mm, y_mm, phi_calc, s1, predicted_mask


def predict_stills_tensors_gpu(
    h: torch.Tensor,
    UB: torch.Tensor,
    s0: torch.Tensor,
    D: torch.Tensor,
    spherical_relp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict stills reflections on GPU in float64.

    Returns:
        (x_mm, y_mm, delpsi, s1, predicted_mask)
    """
    q = torch.matmul(UB, h.unsqueeze(-1)).squeeze(-1)
    qq = (q * q).sum(dim=-1)
    q_norm = torch.sqrt(qq)
    s0_len = torch.linalg.norm(s0, dim=-1)
    unit_s0 = s0 / s0_len.unsqueeze(-1)

    e1_raw = torch.cross(q, unit_s0, dim=-1)
    e1 = e1_raw / torch.linalg.norm(e1_raw, dim=-1, keepdim=True)
    c0_raw = torch.cross(unit_s0, e1, dim=-1)
    c0 = c0_raw / torch.linalg.norm(c0_raw, dim=-1, keepdim=True)

    a = 0.5 * qq / s0_len
    tmp = qq - a * a
    valid_ewald = (q_norm > 0.0) & (tmp > 0.0)
    b = torch.sqrt(torch.clamp(tmp, min=0.0))
    r = -a.unsqueeze(-1) * unit_s0 + b.unsqueeze(-1) * c0

    q0 = q / torch.clamp(q_norm.unsqueeze(-1), min=1e-30)
    q1_raw = torch.cross(q0, e1, dim=-1)
    q1 = q1_raw / torch.linalg.norm(q1_raw, dim=-1, keepdim=True)
    delpsi = -torch.atan2((r * q1).sum(dim=-1), (r * q0).sum(dim=-1))

    if spherical_relp:
        s1_raw = s0 + q
        s1_raw_len_sq = (s1_raw * s1_raw).sum(dim=-1)
        valid_ewald = valid_ewald & (s1_raw_len_sq > 0.0)
        s1 = (
            s1_raw
            / torch.clamp(torch.sqrt(s1_raw_len_sq).unsqueeze(-1), min=1e-30)
            * s0_len.unsqueeze(-1)
        )
    else:
        s1_raw = s0 + r
        s1 = (
            s1_raw
            / torch.linalg.norm(s1_raw, dim=-1, keepdim=True)
            * s0_len.unsqueeze(-1)
        )

    pv = torch.matmul(D, s1.unsqueeze(-1)).squeeze(-1)
    u = pv[:, 0]
    v = pv[:, 1]
    w = pv[:, 2]
    predicted_mask = valid_ewald & (w > 0.0)

    w_safe = torch.where(predicted_mask, w, torch.ones_like(w))
    x_mm = torch.where(predicted_mask, u / w_safe, torch.zeros_like(u))
    y_mm = torch.where(predicted_mask, v / w_safe, torch.zeros_like(v))

    return x_mm, y_mm, delpsi, s1, predicted_mask


def _can_use_gpu_px_mm(experiments) -> bool:
    if not torch.cuda.is_available():
        return False
    for expt in experiments:
        if expt.detector is None:
            return False
        for panel in expt.detector:
            if panel.get_projection_2d() is not None:
                pass
            sname = type(panel.get_px_mm_strategy()).__name__
            if sname not in ("SimplePxMmStrategy", "ParallaxCorrectedPxMmStrategy"):
                return False
    return True


class ScansExperimentsPredictorGPU(ScansExperimentsPredictor):
    """GPU-accelerated ScansExperimentsPredictor with C++ fallback."""

    def __init__(self, experiments):
        super().__init__(experiments)


class StillsExperimentsPredictorGPU(StillsExperimentsPredictor):
    """GPU-accelerated StillsExperimentsPredictor with C++ fallback."""

    def __init__(self, experiments, spherical_relp=False):
        super().__init__(experiments)
        self.spherical_relp_model = spherical_relp


class ExperimentsPredictorFactoryGPU(ExperimentsPredictorFactory):
    @staticmethod
    def from_experiments(experiments, force_stills=False, spherical_relp=False):
        assert experiments.all_same_type(), (
            "Cannot create ExperimentsPredictor for a mixture of experiments with different types"
        )

        if experiments.all_tof():
            return TOFExperimentsPredictor(experiments)
        elif experiments.all_laue():
            return LaueExperimentsPredictor(experiments)

        if not force_stills:
            for exp in experiments:
                if exp.goniometer is None:
                    force_stills = True
                    break

        if force_stills:
            return StillsExperimentsPredictorGPU(
                experiments, spherical_relp=spherical_relp
            )
        return ScansExperimentsPredictorGPU(experiments)
