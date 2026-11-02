from __future__ import annotations

import math
import numpy as np
import torch
from dxtbx import flumpy
from rstbx.symmetry.constraints.parameter_reduction import symmetrize_reduce_enlarge
from scitbx import matrix

from dials.algorithms.refinement.parameterisation import (
    build_prediction_parameterisation as _dials_build_prediction_parameterisation,
)
from dials.algorithms.refinement.parameterisation.autoreduce import (
    count_associated_refs,
    id_associated_refs,
)
from dials.algorithms.refinement.parameterisation.crystal_parameters import (
    CrystalUnitCellMixin,
)
from dials.algorithms.refinement.parameterisation.prediction_parameters import (
    XYPhiPredictionParameterisation,
    XYPhiPredictionParameterisationSparse,
)
from dials.algorithms.refinement.parameterisation.prediction_parameters_stills import (
    SphericalRelpStillsPredictionParameterisation,
    SphericalRelpStillsPredictionParameterisationSparse,
    StillsPredictionParameterisation,
    StillsPredictionParameterisationSparse,
)
from dials.algorithms.refinement.parameterisation.scan_varying_crystal_parameters import (
    ScanVaryingCrystalUnitCellParameterisation,
)
from dials.algorithms.refinement.parameterisation.scan_varying_model_parameters import (
    ScanVaryingModelParameterisation,
)
from dials.algorithms.refinement.parameterisation.scan_varying_prediction_parameters import (
    ScanVaryingPredictionParameterisation,
    ScanVaryingPredictionParameterisationSparse,
)
from dials.array_family import flex
from dials_gpu.refinement.predictor import _rodrigues_rotate_batch


def _get_cached_constraint_matrix(uc_param) -> np.ndarray:
    """Return cached (6, n_indep) linear constraint matrix C for unit cell parameterisation."""
    sg = uc_param._model.get_space_group()
    hall = sg.type().hall_symbol()
    if getattr(uc_param, "_cached_sg_hall", None) != hall:
        S = symmetrize_reduce_enlarge(sg)
        n_indep = S.constraints.n_independent_params()
        C = np.zeros((6, n_indep), dtype=np.float64)
        for k in range(n_indep):
            e = [0.0] * n_indep
            e[k] = 1.0
            C[:, k] = S.enlarge(e)
        uc_param._cached_C = C
        uc_param._cached_sg_hall = hall
    return uc_param._cached_C


def _fast_crystal_uc_compose_core(uc_param, raw_vals):
    """Fast analytical replacement for CrystalUnitCellMixin._compose_core."""
    C = _get_cached_constraint_matrix(uc_param)
    vals = np.asarray(raw_vals, dtype=np.float64) * 1.0e-5
    g = C @ vals
    g0, g1, g2, g3, g4, g5 = (
        float(g[0]),
        float(g[1]),
        float(g[2]),
        float(g[3]),
        float(g[4]),
        float(g[5]),
    )
    if g2 <= 0.0:
        raise RuntimeError(
            "Unable to compose the crystal model. Please check that the "
            "experiments match the indexing of the reflections."
        )
    cstrz = math.sqrt(g2)
    bstrz = g5 / cstrz
    astrz = g4 / cstrz
    rad1 = g1 - bstrz * bstrz
    if rad1 <= 0.0:
        raise RuntimeError(
            "Unable to compose the crystal model. Please check that the "
            "experiments match the indexing of the reflections."
        )
    bstry = math.sqrt(rad1)
    astry = (g3 - astrz * bstrz) / bstry
    rad0 = g0 - astry * astry - astrz * astrz
    if rad0 <= 0.0:
        raise RuntimeError(
            "Unable to compose the crystal model. Please check that the "
            "experiments match the indexing of the reflections."
        )
    astrx = math.sqrt(rad0)

    newB = matrix.sqr((astrx, 0.0, 0.0, astry, bstry, 0.0, astrz, bstrz, cstrz))

    denom = g1 * g2 - g5 * g5
    if denom == 0.0:
        raise RuntimeError(
            "Unable to compose the crystal model. Please check that the "
            "experiments match the indexing of the reflections."
        )
    fac3 = -denom
    fac4 = g2 * g3 - g4 * g5
    fac5 = g3 - g4 * g5 / g2
    rad2 = -(g2 * g3 * g3 + g4 * (g1 * g4 - 2.0 * g3 * g5) + g0 * fac3) / denom
    if rad2 <= 0.0:
        raise RuntimeError(
            "Unable to compose the crystal model. Please check that the "
            "experiments match the indexing of the reflections."
        )
    sqrt_rad1 = bstry
    rad1_32 = rad1 * sqrt_rad1
    sqrt_rad2 = math.sqrt(rad2)
    factor_dg1 = (fac4 * fac4) / (fac3 * fac3 * sqrt_rad2)
    rat5_22 = g5 / (g2 * g2)
    fac1 = g5 * fac5
    fac2 = g1 * g4 - g3 * g5
    fac2sq = fac2 * fac2
    better_ratio = (fac2 / fac3) * (fac4 / fac3)

    dB_dg = np.zeros((9, 6), dtype=np.float64)
    dB_dg[0, 0] = 0.5 / sqrt_rad2

    dB_dg[0, 1] = 0.5 * factor_dg1
    dB_dg[3, 1] = -0.5 * fac5 / rad1_32
    dB_dg[4, 1] = 0.5 / sqrt_rad1

    dB_dg[0, 2] = 0.5 * fac2sq / (fac3 * fac3 * sqrt_rad2)
    dB_dg[3, 2] = -0.5 * rat5_22 * fac1 / rad1_32 + g4 * rat5_22 / sqrt_rad1
    dB_dg[4, 2] = 0.5 * rat5_22 * g5 / sqrt_rad1
    dB_dg[6, 2] = -0.5 * g4 / (g2 * cstrz)
    dB_dg[7, 2] = -0.5 * rat5_22 * cstrz
    dB_dg[8, 2] = 0.5 / cstrz

    dB_dg[0, 3] = fac4 / (fac3 * sqrt_rad2)
    dB_dg[3, 3] = 1.0 / sqrt_rad1

    dB_dg[0, 4] = fac2 / (fac3 * sqrt_rad2)
    dB_dg[3, 4] = -g5 / (g2 * sqrt_rad1)
    dB_dg[6, 4] = 1.0 / cstrz

    dB_dg[0, 5] = better_ratio / sqrt_rad2
    dB_dg[3, 5] = fac1 / (g2 * rad1_32) - g4 / (g2 * sqrt_rad1)
    dB_dg[4, 5] = -g5 / (g2 * sqrt_rad1)
    dB_dg[7, 5] = 1.0 / cstrz

    dB_indep = (dB_dg @ C).T * 1.0e-5
    dB_dval = [matrix.sqr(tuple(row)) for row in dB_indep]
    return newB, dB_dval


def _fast_sv_crystal_uc_compose(self, t):
    """Fast scan-varying crystal unit cell compose without redundant dead-code loop."""
    vals, weights, sumweights = zip(
        *(self._smoother.value_weight(t, pset) for pset in self._param)
    )
    inv_sumw = [1.0 / sw for sw in sumweights]
    dvals_dp = [e * isw for e, isw in zip(weights, inv_sumw)]
    self._B_at_t, dB_dval = _fast_crystal_uc_compose_core(self, vals)
    self._dstate_dp = [[None] * e.size for e in dvals_dp]
    for i, (dv, dB) in enumerate(zip(dvals_dp, dB_dval)):
        for j, e in dv:
            self._dstate_dp[i][j] = e * dB
    self._last_dB_dval = dB_dval
    self._last_dvals_dp = dvals_dp


def _patch_crystal_uc_parameterisations(pred_param):
    for p in pred_param.get_crystal_unit_cell_parameterisations():
        if isinstance(p, ScanVaryingCrystalUnitCellParameterisation):
            p.compose = _fast_sv_crystal_uc_compose.__get__(p, type(p))
        elif isinstance(p, CrystalUnitCellMixin):
            p._compose_core = _fast_crystal_uc_compose_core.__get__(p, type(p))


def _compute_smoother_weight_matrix(smoother, dummy_param, frames_np: np.ndarray) -> np.ndarray:
    """Compute normalized (B, num_samples) Gaussian smoother weight matrix for integer frames."""
    n_blocks = len(frames_np)
    n_samples = len(dummy_param)
    W_np = np.empty((n_blocks, n_samples), dtype=np.float64)
    for b_idx, f_val in enumerate(frames_np):
        _, weight, sumweight = smoother.value_weight(int(f_val), dummy_param)
        W_np[b_idx] = flumpy.to_numpy(weight.as_dense_vector()) * (1.0 / sumweight)
    return W_np


def _axis_angle_to_rot_and_drot_batch(
    axis, angles: np.ndarray, skip_derivatives: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute rotation matrices (B, 3, 3) and angle derivatives (B, 3, 3) for 1D angles (rad)."""
    ax = np.asarray(axis, dtype=np.float64)
    ax = ax / np.linalg.norm(ax)
    ux, uy, uz = float(ax[0]), float(ax[1]), float(ax[2])
    c = np.cos(angles)
    s = np.sin(angles)
    omc = 1.0 - c

    R = np.empty((len(angles), 3, 3), dtype=np.float64)
    R[:, 0, 0] = c + ux * ux * omc
    R[:, 0, 1] = ux * uy * omc - uz * s
    R[:, 0, 2] = ux * uz * omc + uy * s
    R[:, 1, 0] = uy * ux * omc + uz * s
    R[:, 1, 1] = c + uy * uy * omc
    R[:, 1, 2] = uy * uz * omc - ux * s
    R[:, 2, 0] = uz * ux * omc - uy * s
    R[:, 2, 1] = uz * uy * omc + ux * s
    R[:, 2, 2] = c + uz * uz * omc

    if skip_derivatives:
        return R, None

    dR = np.empty((len(angles), 3, 3), dtype=np.float64)
    dR[:, 0, 0] = -s + ux * ux * s
    dR[:, 0, 1] = ux * uy * s - uz * c
    dR[:, 0, 2] = ux * uz * s + uy * c
    dR[:, 1, 0] = uy * ux * s + uz * c
    dR[:, 1, 1] = -s + uy * uy * s
    dR[:, 1, 2] = uy * uz * s - ux * c
    dR[:, 2, 0] = uz * ux * s - uy * c
    dR[:, 2, 1] = uz * uy * s + ux * c
    dR[:, 2, 2] = -s + uz * uz * s
    return R, dR


def _compose_sv_ori_batch(
    xl_op, W_np: np.ndarray, skip_derivatives: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """Vectorized batch composition of ScanVaryingCrystalOrientationParameterisation."""
    U0 = np.asarray(xl_op._initial_state.elems, dtype=np.float64).reshape(3, 3)
    P = np.asarray([ps.value for ps in xl_op._param], dtype=np.float64)
    vals = (W_np @ P.T) * 1.0e-3

    R1, dR1 = _axis_angle_to_rot_and_drot_batch(
        xl_op._param[0].axis, vals[:, 0], skip_derivatives
    )
    R2, dR2 = _axis_angle_to_rot_and_drot_batch(
        xl_op._param[1].axis, vals[:, 1], skip_derivatives
    )
    R3, dR3 = _axis_angle_to_rot_and_drot_batch(
        xl_op._param[2].axis, vals[:, 2], skip_derivatives
    )

    R1_U0 = np.matmul(R1, U0)
    R21_U0 = np.matmul(R2, R1_U0)
    U_blocks = np.matmul(R3, R21_U0)
    xl_op._U_at_t = matrix.sqr(tuple(U_blocks[-1].reshape(-1)))

    if skip_derivatives:
        return U_blocks, None

    R32 = np.matmul(R3, R2)
    base_list = []
    if not xl_op._param[0].get_fixed():
        base_list.append(np.matmul(R32, np.matmul(dR1, U0)) * 1.0e-3)
    if not xl_op._param[1].get_fixed():
        base_list.append(np.matmul(R3, np.matmul(dR2, R1_U0)) * 1.0e-3)
    if not xl_op._param[2].get_fixed():
        base_list.append(np.matmul(dR3, R21_U0) * 1.0e-3)

    if base_list:
        dU_base = np.stack(base_list, axis=1)
    else:
        dU_base = np.empty((len(W_np), 0, 3, 3), dtype=np.float64)
    return U_blocks, dU_base


def _compose_sv_uc_batch(
    xl_ucp, W_np: np.ndarray, skip_derivatives: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """Vectorized batch composition of ScanVaryingCrystalUnitCellParameterisation."""
    C = _get_cached_constraint_matrix(xl_ucp)
    P = np.asarray([ps.value for ps in xl_ucp._param], dtype=np.float64)
    vals = (W_np @ P.T) * 1.0e-5
    g = vals @ C.T

    B_len = len(W_np)
    g0, g1, g2, g3, g4, g5 = g[:, 0], g[:, 1], g[:, 2], g[:, 3], g[:, 4], g[:, 5]
    if np.any(g2 <= 0.0):
        raise RuntimeError(
            "Unable to compose the crystal model. Please check that the "
            "experiments match the indexing of the reflections."
        )
    cstrz = np.sqrt(g2)
    bstrz = g5 / cstrz
    astrz = g4 / cstrz
    rad1 = g1 - bstrz * bstrz
    if np.any(rad1 <= 0.0):
        raise RuntimeError(
            "Unable to compose the crystal model. Please check that the "
            "experiments match the indexing of the reflections."
        )
    bstry = np.sqrt(rad1)
    astry = (g3 - astrz * bstrz) / bstry
    rad0 = g0 - astry * astry - astrz * astrz
    if np.any(rad0 <= 0.0):
        raise RuntimeError(
            "Unable to compose the crystal model. Please check that the "
            "experiments match the indexing of the reflections."
        )
    astrx = np.sqrt(rad0)

    B_blocks = np.zeros((B_len, 3, 3), dtype=np.float64)
    B_blocks[:, 0, 0] = astrx
    B_blocks[:, 1, 0] = astry
    B_blocks[:, 1, 1] = bstry
    B_blocks[:, 2, 0] = astrz
    B_blocks[:, 2, 1] = bstrz
    B_blocks[:, 2, 2] = cstrz
    xl_ucp._B_at_t = matrix.sqr(tuple(B_blocks[-1].reshape(-1)))

    if skip_derivatives:
        return B_blocks, None

    denom = g1 * g2 - g5 * g5
    if np.any(denom == 0.0):
        raise RuntimeError(
            "Unable to compose the crystal model. Please check that the "
            "experiments match the indexing of the reflections."
        )
    fac3 = -denom
    fac4 = g2 * g3 - g4 * g5
    fac5 = g3 - g4 * g5 / g2
    rad2 = -(g2 * g3 * g3 + g4 * (g1 * g4 - 2.0 * g3 * g5) + g0 * fac3) / denom
    if np.any(rad2 <= 0.0):
        raise RuntimeError(
            "Unable to compose the crystal model. Please check that the "
            "experiments match the indexing of the reflections."
        )
    sqrt_rad1 = bstry
    rad1_32 = rad1 * sqrt_rad1
    sqrt_rad2 = np.sqrt(rad2)
    factor_dg1 = (fac4 * fac4) / (fac3 * fac3 * sqrt_rad2)
    rat5_22 = g5 / (g2 * g2)
    fac1 = g5 * fac5
    fac2 = g1 * g4 - g3 * g5
    fac2sq = fac2 * fac2
    better_ratio = (fac2 / fac3) * (fac4 / fac3)

    dB_dg = np.zeros((B_len, 9, 6), dtype=np.float64)
    dB_dg[:, 0, 0] = 0.5 / sqrt_rad2
    dB_dg[:, 0, 1] = 0.5 * factor_dg1
    dB_dg[:, 3, 1] = -0.5 * fac5 / rad1_32
    dB_dg[:, 4, 1] = 0.5 / sqrt_rad1
    dB_dg[:, 0, 2] = 0.5 * fac2sq / (fac3 * fac3 * sqrt_rad2)
    dB_dg[:, 3, 2] = -0.5 * rat5_22 * fac1 / rad1_32 + g4 * rat5_22 / sqrt_rad1
    dB_dg[:, 4, 2] = 0.5 * rat5_22 * g5 / sqrt_rad1
    dB_dg[:, 6, 2] = -0.5 * g4 / (g2 * cstrz)
    dB_dg[:, 7, 2] = -0.5 * rat5_22 * cstrz
    dB_dg[:, 8, 2] = 0.5 / cstrz
    dB_dg[:, 0, 3] = fac4 / (fac3 * sqrt_rad2)
    dB_dg[:, 3, 3] = 1.0 / sqrt_rad1
    dB_dg[:, 0, 4] = fac2 / (fac3 * sqrt_rad2)
    dB_dg[:, 3, 4] = -g5 / (g2 * sqrt_rad1)
    dB_dg[:, 6, 4] = 1.0 / cstrz
    dB_dg[:, 0, 5] = better_ratio / sqrt_rad2
    dB_dg[:, 3, 5] = fac1 / (g2 * rad1_32) - g4 / (g2 * sqrt_rad1)
    dB_dg[:, 4, 5] = -g5 / (g2 * sqrt_rad1)
    dB_dg[:, 7, 5] = 1.0 / cstrz

    free_idx = [i for i, ps in enumerate(xl_ucp._param) if not ps.get_fixed()]
    if free_idx:
        C_free = C[:, free_idx] * 1.0e-5
        dB_base = np.einsum("bij,jk->bki", dB_dg, C_free).reshape(
            B_len, len(free_idx), 3, 3
        )
    else:
        dB_base = np.empty((B_len, 0, 3, 3), dtype=np.float64)
    return B_blocks, dB_base


def _compose_sv_beam_batch(
    bp, W_np: np.ndarray, skip_derivatives: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """Batch composition of ScanVaryingBeamParameterisation."""
    P = np.asarray([ps.value for ps in bp._param], dtype=np.float64)
    vals = W_np @ P.T
    B_len = len(W_np)
    free_idx = [i for i, ps in enumerate(bp._param) if not ps.get_fixed()]
    s0_blocks = np.empty((B_len, 3), dtype=np.float64)
    ds0_base = (
        np.empty((B_len, len(free_idx), 3), dtype=np.float64)
        if not skip_derivatives
        else None
    )
    ius0 = bp._initial_state["unit_s0"]
    ipn = bp._initial_state["polarization_normal"]
    mu1_set, mu2_set, _ = bp._param
    for b in range(B_len):
        (s0, _), ds0_dval = bp._compose_core(
            ius0,
            ipn,
            float(vals[b, 0]),
            float(vals[b, 1]),
            float(vals[b, 2]),
            mu1_axis=mu1_set.axis,
            mu2_axis=mu2_set.axis,
        )
        s0_blocks[b] = s0.elems
        if not skip_derivatives:
            for k, idx in enumerate(free_idx):
                ds0_base[b, k] = ds0_dval[idx].elems
    bp._s0_at_t = s0
    return s0_blocks, ds0_base


def _compose_sv_gonio_batch(
    gp, W_np: np.ndarray, skip_derivatives: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """Vectorized batch composition of ScanVaryingGoniometerParameterisation."""
    S0 = np.asarray(gp._initial_state.elems, dtype=np.float64).reshape(3, 3)
    P = np.asarray([ps.value for ps in gp._param], dtype=np.float64)
    vals = (W_np @ P.T) * 1.0e-3

    R1, dR1 = _axis_angle_to_rot_and_drot_batch(
        gp._param[0].axis, vals[:, 0], skip_derivatives
    )
    R2, dR2 = _axis_angle_to_rot_and_drot_batch(
        gp._param[1].axis, vals[:, 1], skip_derivatives
    )

    R1_S0 = np.matmul(R1, S0)
    S_blocks = np.matmul(R2, R1_S0)
    gp._S_at_t = matrix.sqr(tuple(S_blocks[-1].reshape(-1)))

    if skip_derivatives:
        return S_blocks, None

    base_list = []
    if not gp._param[0].get_fixed():
        base_list.append(np.matmul(R2, np.matmul(dR1, S0)) * 1.0e-3)
    if not gp._param[1].get_fixed():
        base_list.append(np.matmul(dR2, R1_S0) * 1.0e-3)
    if base_list:
        dS_base = np.stack(base_list, axis=1)
    else:
        dS_base = np.empty((len(W_np), 0, 3, 3), dtype=np.float64)
    return S_blocks, dS_base


def _compose_sv_det_batch(
    dp, exp, W_np: np.ndarray, skip_derivatives: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Batch composition of ScanVaryingDetectorParameterisationSinglePanel."""
    from dials.algorithms.refinement.parameterisation.scan_varying_detector_parameters import (
        Param,
    )

    P = np.asarray([ps.value for ps in dp._param], dtype=np.float64)
    vals = W_np @ P.T
    B_len = len(W_np)
    free_idx = [i for i, ps in enumerate(dp._param) if not ps.get_fixed()]
    d_blocks = np.empty((B_len, 3, 3), dtype=np.float64)
    D_blocks = np.empty((B_len, 3, 3), dtype=np.float64)
    dd_base = (
        np.empty((B_len, len(free_idx), 3, 3), dtype=np.float64)
        if not skip_derivatives
        else None
    )
    axes = [ps.axis for ps in dp._param]
    D_static = np.asarray(exp.detector[0].get_D_matrix(), dtype=np.float64).reshape(
        3, 3
    )
    for b in range(B_len):
        new_state, dd_dval = dp._compose_core(
            *(Param(float(vals[b, j]), axes[j]) for j in range(6))
        )
        f1, f2, f3 = new_state["d1"]
        s1, s2, s3 = new_state["d2"]
        o1, o2, o3 = new_state["origin"]
        d_mat = matrix.sqr((f1, s1, o1, f2, s2, o2, f3, s3, o3))
        d_blocks[b] = np.asarray(d_mat.elems, dtype=np.float64).reshape(3, 3)
        D_blocks[b] = D_static
        if not skip_derivatives:
            for k, idx in enumerate(free_idx):
                dd_base[b, k] = np.asarray(
                    dd_dval[idx].elems, dtype=np.float64
                ).reshape(3, 3)
    dp._d_at_t = d_mat
    return d_blocks, D_blocks, dd_base


class _FastModifyParametersMixin:
    """Mixin that avoids recomposing fixed model parameterisations on every step."""

    def _modify_parameters(self, vals, set_vals=False, set_esds=False, set_fix=False):
        assert [set_vals, set_esds, set_fix].count(True) == 1
        assert len(vals) == len(self)
        it = iter(vals)

        for model in (
            self._detector_parameterisations
            + self._beam_parameterisations
            + self._xl_orientation_parameterisations
            + self._xl_unit_cell_parameterisations
            + self._goniometer_parameterisations
        ):
            n_free = model.num_free()
            if n_free == 0:
                if set_esds:
                    model.set_param_esds([])
                continue
            tmp = [next(it) for _ in range(n_free)]
            if set_esds:
                model.set_param_esds(tmp)
            elif set_vals:
                model.set_param_vals(tmp)
            elif set_fix:
                current_fixes = model.get_fixed()
                free_indices = [i for i, e in enumerate(current_fixes) if not e]
                if len(free_indices) != n_free:
                    logger.warning(
                        "Unable to fix parameters for parameterisation with prefix %s",
                        model.model_identifier,
                    )
                    continue
                for i, fix in zip(free_indices, tmp):
                    if fix:
                        current_fixes[i] = True
                model.set_fixed(current_fixes)


class _GPUScanGradientsMixin(_FastModifyParametersMixin):
    """GPU analytical gradient computation for scan-static and scan-varying refinement."""

    def can_use_gpu_gradients(self) -> bool:
        return torch.cuda.is_available()

    def _ensure_cpu_sv_derivative_cache(self):
        if isinstance(self, ScanVaryingPredictionParameterisation) and getattr(
            self, "_sv_derivative_cache_dirty", False
        ):
            self._sv_derivative_cache_dirty = False
            ScanVaryingPredictionParameterisation.compose(
                self, self._sv_last_reflections, skip_derivatives=False
            )

    def get_gradients(self, reflections, callback=None):
        if (
            not self.can_use_gpu_gradients()
            or len(self) * len(reflections) > 25_000_000
        ):
            self._ensure_cpu_sv_derivative_cache()
            return super().get_gradients(reflections, callback=callback)
        if callback is count_associated_refs:
            dX, dY, dphi = self.get_gradients_gpu(reflections)
            if dX.numel() == 0:
                return [0] * len(self)
            mask = (dX.abs() > 1e-10) | (dY.abs() > 1e-10) | (dphi.abs() > 1e-10)
            return mask.sum(dim=1).cpu().tolist()
        if callback is id_associated_refs:
            dX, dY, dphi = self.get_gradients_gpu(reflections)
            mask_np = (
                ((dX.abs() > 1e-10) | (dY.abs() > 1e-10) | (dphi.abs() > 1e-10))
                .cpu()
                .numpy()
            )
            return [
                flumpy.from_numpy(np.ascontiguousarray(mask_np[i]))
                for i in range(len(self))
            ]
        if isinstance(
            self,
            (
                XYPhiPredictionParameterisationSparse,
                ScanVaryingPredictionParameterisationSparse,
            ),
        ):
            self._ensure_cpu_sv_derivative_cache()
            return super().get_gradients(reflections, callback=callback)
        dX, dY, dphi = self.get_gradients_gpu(reflections)
        stacked_np = torch.stack((dX, dY, dphi), dim=1).cpu().numpy()
        results = []
        for i in range(len(self)):
            r = {
                "dX_dp": flumpy.from_numpy(np.ascontiguousarray(stacked_np[i, 0])),
                "dY_dp": flumpy.from_numpy(np.ascontiguousarray(stacked_np[i, 1])),
                "dphi_dp": flumpy.from_numpy(np.ascontiguousarray(stacked_np[i, 2])),
            }
            results.append(callback(r) if callback is not None else r)
        return results

    def _get_cached_refl_tensors_scan(self, reflections, device: torch.device):
        """Return cached reflection-constant GPU tensors and integer index selections."""
        n_refl = len(reflections)
        n_expts = len(self._experiments)
        h_i32 = flumpy.to_numpy(reflections["miller_index"])
        exp_id_view = flumpy.to_numpy(reflections["id"])
        panel_id_view = flumpy.to_numpy(reflections["panel"])

        cache = getattr(self, "_gpu_refl_cache", None)
        if (
            cache is not None
            and cache["n_refl"] == n_refl
            and cache["device"] == device
            and np.array_equal(h_i32, cache["h_i32"])
            and np.array_equal(exp_id_view, cache["exp_id_np"])
            and np.array_equal(panel_id_view, cache["panel_id_np"])
        ):
            return cache

        h_np = h_i32.astype(np.float64)
        h_gpu = torch.as_tensor(h_np, dtype=torch.float64, device=device)
        exp_id_i64 = exp_id_view.astype(np.int64)
        panel_id_i64 = panel_id_view.astype(np.int64)
        exp_id_gpu = torch.as_tensor(exp_id_i64, device=device, dtype=torch.long)
        panel_id_gpu = torch.as_tensor(panel_id_i64, device=device, dtype=torch.long)

        # Fixed goniometer rotation F and datum axis e0 never change during refinement
        f_e0_flat = []
        for exp in self._experiments:
            f_e0_flat.extend(exp.goniometer.get_fixed_rotation())
            f_e0_flat.extend(exp.goniometer.get_rotation_axis_datum())
        f_e0_gpu = torch.tensor(
            f_e0_flat, dtype=torch.float64, device=device
        ).reshape(n_expts, 12)
        F_table = f_e0_gpu[:, :9].reshape(n_expts, 3, 3)
        e0_table = f_e0_gpu[:, 9:12].reshape(n_expts, 3)
        e0_unit_table = e0_table / torch.linalg.norm(e0_table, dim=-1, keepdim=True)

        if n_expts == 1:
            F_gpu = F_table
            e0_gpu = e0_table
            e0_unit = e0_unit_table
        else:
            F_gpu = F_table[exp_id_gpu]
            e0_gpu = e0_table[exp_id_gpu]
            e0_unit = e0_unit_table[exp_id_gpu]

        # Precompute integer index selections per unique tuple(exp_ids)
        isel_map = {}
        for p in (
            self._detector_parameterisations
            + self._beam_parameterisations
            + self._xl_orientation_parameterisations
            + self._xl_unit_cell_parameterisations
            + self._goniometer_parameterisations
        ):
            key = tuple(sorted(set(p.get_experiment_ids())))
            if key not in isel_map:
                if n_expts == 1 or len(key) == n_expts:
                    isel_map[key] = (n_refl, None)
                else:
                    idx_np = np.nonzero(np.isin(exp_id_i64, key))[0]
                    isel_gpu = (
                        torch.as_tensor(idx_np, dtype=torch.long, device=device)
                        if len(idx_np) > 0
                        else None
                    )
                    isel_map[key] = (len(idx_np), isel_gpu)

        cache = {
            "n_refl": n_refl,
            "device": device,
            "h_i32": h_i32.copy(),
            "exp_id_np": exp_id_view.copy(),
            "panel_id_np": panel_id_view.copy(),
            "h_gpu": h_gpu,
            "exp_id_gpu": exp_id_gpu,
            "panel_id_gpu": panel_id_gpu,
            "F_gpu": F_gpu,
            "e0_gpu": e0_gpu,
            "e0_unit": e0_unit,
            "isel_map": isel_map,
        }
        self._gpu_refl_cache = cache
        return cache

    def get_gradients_gpu(
        self, reflections, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute (dX_dp, dY_dp, dphi_dp) as (P, N) float64 tensors on GPU."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_refl = len(reflections)
        n_params = len(self)
        n_expts = len(self._experiments)
        if n_refl == 0 or n_params == 0:
            z = torch.zeros((n_params, n_refl), dtype=torch.float64, device=device)
            return z, z.clone(), z.clone()

        alloc_fn = torch.empty if n_expts == 1 else torch.zeros
        dX_all = alloc_fn((n_params, n_refl), dtype=torch.float64, device=device)
        dY_all = alloc_fn((n_params, n_refl), dtype=torch.float64, device=device)
        dphi_all = alloc_fn((n_params, n_refl), dtype=torch.float64, device=device)

        is_scan_varying = isinstance(self, ScanVaryingPredictionParameterisation)
        cache = self._get_cached_refl_tensors_scan(reflections, device)
        h_gpu = cache["h_gpu"]
        exp_id_gpu = cache["exp_id_gpu"]
        panel_id_gpu = cache["panel_id_gpu"]
        F_gpu = cache["F_gpu"]
        e0_gpu = cache["e0_gpu"]
        e0_unit = cache["e0_unit"]
        isel_map = cache["isel_map"]

        # Upload per-step reflection columns in a single H2D transfer
        s1_np = flumpy.to_numpy(reflections["s1"])
        xyzcal_np = flumpy.to_numpy(reflections["xyzcal.mm"])

        if is_scan_varying:
            imatch_np = (
                flumpy.to_numpy(reflections["imatch"]).astype(np.int64)
                if "imatch" in reflections
                else None
            )
            sv_comp = getattr(self, "_sv_composed", None)
            if sv_comp is not None and (
                (imatch_np is None and n_refl == sv_comp["nref"])
                or (imatch_np is not None)
            ):
                step_refl_np = np.empty((2, n_refl, 3), dtype=np.float64)
                step_refl_np[0] = s1_np
                step_refl_np[1] = xyzcal_np
                step_refl_gpu = torch.as_tensor(
                    step_refl_np, dtype=torch.float64, device=device
                )
                s1_gpu = step_refl_gpu[0]
                xyzcal_gpu = step_refl_gpu[1]
                phi_gpu = xyzcal_gpu[:, 2]

                layout = sv_comp.get("layout", sv_comp)
                if (
                    layout.get("global_block_idx_gpu") is None
                    or layout.get("gpu_device") != device
                ):
                    layout["global_block_idx_gpu"] = torch.as_tensor(
                        sv_comp["global_block_idx_np"],
                        dtype=torch.long,
                        device=device,
                    )
                    layout["W_gpu"] = {
                        pid: torch.as_tensor(w_arr, dtype=torch.float64, device=device)
                        for pid, w_arr in sv_comp["W_dict"].items()
                    }
                    layout["gpu_device"] = device
                sv_comp["global_block_idx_gpu"] = layout["global_block_idx_gpu"]
                sv_comp["W_gpu"] = layout["W_gpu"]

                b_idx_gpu = layout["global_block_idx_gpu"]
                if imatch_np is not None:
                    imatch_gpu = torch.as_tensor(
                        imatch_np, dtype=torch.long, device=device
                    )
                    b_idx_gpu = b_idx_gpu[imatch_gpu]

                U_blocks_gpu = torch.as_tensor(
                    sv_comp["U_blocks"], dtype=torch.float64, device=device
                )
                B_blocks_gpu = torch.as_tensor(
                    sv_comp["B_blocks"], dtype=torch.float64, device=device
                )
                UB_blocks_gpu = torch.as_tensor(
                    sv_comp["UB_blocks"], dtype=torch.float64, device=device
                )
                s0_blocks_gpu = torch.as_tensor(
                    sv_comp["s0_blocks"], dtype=torch.float64, device=device
                )
                S_blocks_gpu = torch.as_tensor(
                    sv_comp["S_blocks"], dtype=torch.float64, device=device
                )

                U_gpu = U_blocks_gpu[b_idx_gpu]
                B_gpu = B_blocks_gpu[b_idx_gpu]
                UB_gpu = UB_blocks_gpu[b_idx_gpu]
                if not self._varying_beams and n_expts == 1:
                    s0_gpu = s0_blocks_gpu[:1]
                else:
                    s0_gpu = s0_blocks_gpu[b_idx_gpu]
                if not self._varying_goniometers and n_expts == 1:
                    S_gpu = S_blocks_gpu[:1]
                else:
                    S_gpu = S_blocks_gpu[b_idx_gpu]

                if sv_comp["D_blocks"] is not None:
                    D_blocks_gpu = torch.as_tensor(
                        sv_comp["D_blocks"], dtype=torch.float64, device=device
                    )
                    D_gpu = D_blocks_gpu[b_idx_gpu]
                else:
                    D_table_gpu = torch.as_tensor(
                        sv_comp["D_table"], dtype=torch.float64, device=device
                    )
                    if n_expts == 1:
                        if D_table_gpu.shape[1] == 1:
                            D_gpu = D_table_gpu[:, 0]
                        else:
                            D_gpu = D_table_gpu[0, panel_id_gpu]
                    else:
                        D_gpu = D_table_gpu[exp_id_gpu, panel_id_gpu]
            else:
                b_idx_gpu = None
                step_refl_np = np.empty((18, n_refl, 3), dtype=np.float64)
                step_refl_np[0] = s1_np
                step_refl_np[1] = xyzcal_np
                step_refl_np[2:5] = (
                    flumpy.to_numpy(reflections["ub_matrix"])
                    .reshape(n_refl, 3, 3)
                    .transpose(1, 0, 2)
                )
                step_refl_np[5] = flumpy.to_numpy(reflections["s0_vector"])
                step_refl_np[6:9] = (
                    flumpy.to_numpy(reflections["D_matrix"])
                    .reshape(n_refl, 3, 3)
                    .transpose(1, 0, 2)
                )
                step_refl_np[9:12] = (
                    flumpy.to_numpy(reflections["S_matrix"])
                    .reshape(n_refl, 3, 3)
                    .transpose(1, 0, 2)
                )
                step_refl_np[12:15] = (
                    flumpy.to_numpy(reflections["u_matrix"])
                    .reshape(n_refl, 3, 3)
                    .transpose(1, 0, 2)
                )
                step_refl_np[15:18] = (
                    flumpy.to_numpy(reflections["b_matrix"])
                    .reshape(n_refl, 3, 3)
                    .transpose(1, 0, 2)
                )
                step_refl_gpu = torch.as_tensor(
                    step_refl_np, dtype=torch.float64, device=device
                )
                s1_gpu = step_refl_gpu[0]
                xyzcal_gpu = step_refl_gpu[1]
                phi_gpu = xyzcal_gpu[:, 2]
                UB_gpu = step_refl_gpu[2:5].permute(1, 0, 2)
                s0_gpu = step_refl_gpu[5]
                D_gpu = step_refl_gpu[6:9].permute(1, 0, 2)
                S_gpu = step_refl_gpu[9:12].permute(1, 0, 2)
                U_gpu = step_refl_gpu[12:15].permute(1, 0, 2)
                B_gpu = step_refl_gpu[15:18].permute(1, 0, 2)

            # Pack any static model parameterisations (e.g., static detector/beam in scan-varying refinement)
            flat_floats: list[float] = []
            ds_dp_slices = {}
            for p_group, elem_size in (
                (self._detector_parameterisations, 9),
                (self._beam_parameterisations, 3),
                (self._xl_orientation_parameterisations, 9),
                (self._xl_unit_cell_parameterisations, 9),
                (self._goniometer_parameterisations, 9),
            ):
                for p in p_group:
                    n_free = p.num_free()
                    if n_free == 0 or isinstance(p, ScanVaryingModelParameterisation):
                        continue
                    s_start = len(flat_floats)
                    if elem_size == 9 and p.is_multi_state():
                        exp_ids = p.get_experiment_ids()
                        n_panels = len(self._experiments[exp_ids[0]].detector)
                        for ip in range(n_panels):
                            for der in p.get_ds_dp(
                                multi_state_elt=ip, use_none_as_null=True
                            ):
                                if der is not None:
                                    flat_floats.extend(der.elems)
                                else:
                                    flat_floats.extend((0.0,) * 9)
                        ds_dp_slices[id(p)] = (
                            s_start,
                            len(flat_floats),
                            (n_panels, n_free, 3, 3),
                        )
                    else:
                        for der in p.get_ds_dp(use_none_as_null=True):
                            if der is not None:
                                flat_floats.extend(der.elems)
                        shape = (n_free, 3) if elem_size == 3 else (n_free, 3, 3)
                        ds_dp_slices[id(p)] = (s_start, len(flat_floats), shape)
            if flat_floats:
                packed_static_gpu = torch.tensor(
                    flat_floats, dtype=torch.float64, device=device
                )
                packed_ds_dp = {
                    pid: packed_static_gpu[s_start:s_end].reshape(shape)
                    for pid, (s_start, s_end, shape) in ds_dp_slices.items()
                }
            else:
                packed_ds_dp = {}
        else:
            step_refl_np = np.empty((2, n_refl, 3), dtype=np.float64)
            step_refl_np[0] = s1_np
            step_refl_np[1] = xyzcal_np
            step_refl_gpu = torch.as_tensor(
                step_refl_np, dtype=torch.float64, device=device
            )
            s1_gpu = step_refl_gpu[0]
            xyzcal_gpu = step_refl_gpu[1]
            phi_gpu = xyzcal_gpu[:, 2]

            # Pack all experiment state matrices and static ds_dp derivatives into ONE H2D transfer
            flat_floats: list[float] = []
            for exp in self._experiments:
                flat_floats.extend(exp.crystal.get_U())
                flat_floats.extend(exp.crystal.get_B())
                flat_floats.extend(exp.beam.get_s0())
                flat_floats.extend(exp.goniometer.get_setting_rotation())
            exp_block_len = 30 * n_expts

            max_panels = max(len(exp.detector) for exp in self._experiments)
            det_start = len(flat_floats)
            for exp in self._experiments:
                for panel in exp.detector:
                    flat_floats.extend(panel.get_D_matrix())
                for _ in range(max_panels - len(exp.detector)):
                    flat_floats.extend((0.0,) * 9)
            det_end = len(flat_floats)

            ds_dp_slices = {}
            for p_group, elem_size in (
                (self._detector_parameterisations, 9),
                (self._beam_parameterisations, 3),
                (self._xl_orientation_parameterisations, 9),
                (self._xl_unit_cell_parameterisations, 9),
                (self._goniometer_parameterisations, 9),
            ):
                for p in p_group:
                    n_free = p.num_free()
                    if n_free == 0:
                        continue
                    s_start = len(flat_floats)
                    if elem_size == 9 and p.is_multi_state():
                        exp_ids = p.get_experiment_ids()
                        n_panels = len(self._experiments[exp_ids[0]].detector)
                        for ip in range(n_panels):
                            for der in p.get_ds_dp(
                                multi_state_elt=ip, use_none_as_null=True
                            ):
                                if der is not None:
                                    flat_floats.extend(der.elems)
                                else:
                                    flat_floats.extend((0.0,) * 9)
                        ds_dp_slices[id(p)] = (
                            s_start,
                            len(flat_floats),
                            (n_panels, n_free, 3, 3),
                        )
                    else:
                        for der in p.get_ds_dp(use_none_as_null=True):
                            if der is not None:
                                flat_floats.extend(der.elems)
                        shape = (n_free, 3) if elem_size == 3 else (n_free, 3, 3)
                        ds_dp_slices[id(p)] = (s_start, len(flat_floats), shape)

            packed_gpu = torch.tensor(
                flat_floats, dtype=torch.float64, device=device
            )
            exp_block = packed_gpu[:exp_block_len].reshape(n_expts, 30)
            U_table = exp_block[:, 0:9].reshape(n_expts, 3, 3)
            B_table = exp_block[:, 9:18].reshape(n_expts, 3, 3)
            UB_table = torch.matmul(U_table, B_table)
            s0_table = exp_block[:, 18:21].reshape(n_expts, 3)
            S_table = exp_block[:, 21:30].reshape(n_expts, 3, 3)
            D_table = packed_gpu[det_start:det_end].reshape(n_expts, max_panels, 3, 3)

            if n_expts == 1:
                U_gpu = U_table
                B_gpu = B_table
                UB_gpu = UB_table
                s0_gpu = s0_table
                S_gpu = S_table
                if max_panels == 1:
                    D_gpu = D_table[:, 0]
                else:
                    D_gpu = D_table[0, panel_id_gpu]
            else:
                U_gpu = U_table[exp_id_gpu]
                B_gpu = B_table[exp_id_gpu]
                UB_gpu = UB_table[exp_id_gpu]
                s0_gpu = s0_table[exp_id_gpu]
                S_gpu = S_table[exp_id_gpu]
                D_gpu = D_table[exp_id_gpu, panel_id_gpu]

            packed_ds_dp = {
                pid: packed_gpu[s_start:s_end].reshape(shape)
                for pid, (s_start, s_end, shape) in ds_dp_slices.items()
            }

        # Common kinematic vectors on GPU
        if D_gpu.shape[0] == 1:
            pv_gpu = torch.matmul(s1_gpu, D_gpu[0].T)
        else:
            pv_gpu = torch.matmul(D_gpu, s1_gpu.unsqueeze(-1)).squeeze(-1)
        u_gpu = pv_gpu[:, 0]
        v_gpu = pv_gpu[:, 1]
        w_gpu = pv_gpu[:, 2]
        w_inv_gpu = 1.0 / w_gpu
        u_w_inv_gpu = u_gpu * w_inv_gpu
        v_w_inv_gpu = v_gpu * w_inv_gpu

        if UB_gpu.shape[0] == 1:
            UB_h = torch.matmul(h_gpu, UB_gpu[0].T)
        else:
            UB_h = torch.matmul(UB_gpu, h_gpu.unsqueeze(-1)).squeeze(-1)
        if F_gpu.shape[0] == 1:
            pstar0 = torch.matmul(UB_h, F_gpu[0].T)
        else:
            pstar0 = torch.matmul(F_gpu, UB_h.unsqueeze(-1)).squeeze(-1)
        relp_gpu = _rodrigues_rotate_batch(pstar0, e0_unit, phi_gpu)
        if S_gpu.shape[0] == 1:
            r_gpu = torch.matmul(relp_gpu, S_gpu[0].T)
            e_lab_gpu = torch.matmul(e0_gpu, S_gpu[0].T)
        else:
            r_gpu = torch.matmul(S_gpu, relp_gpu.unsqueeze(-1)).squeeze(-1)
            e_lab_gpu = torch.matmul(S_gpu, e0_gpu.unsqueeze(-1)).squeeze(-1)
        e_cross_r_gpu = torch.cross(e_lab_gpu.expand_as(r_gpu), r_gpu, dim=-1)
        inv_ewald_denom_gpu = -1.0 / (e_cross_r_gpu * s0_gpu).sum(dim=-1)

        param_idx = 0

        def _sel(t: torch.Tensor, isel: torch.Tensor | None) -> torch.Tensor:
            if isel is None or t.shape[0] == 1:
                return t
            return t[isel]

        # Helper to apply quotient rule and write into (dX_all, dY_all, dphi_all) without boolean sync
        def _store_grads(
            p_offset: int,
            isel: torch.Tensor | None,
            dpv: torch.Tensor,
            dphi: torch.Tensor | None,
            w_smoother_T: torch.Tensor | None = None,
        ):
            if dpv.ndim == 2:
                dpv = dpv.unsqueeze(1)
            k_count = dpv.shape[1]
            du = dpv[:, :, 0]
            dv = dpv[:, :, 1]
            dw = dpv[:, :, 2]
            if isel is None:
                w_inv_sub = w_inv_gpu.unsqueeze(1)
                u_w_sub = u_w_inv_gpu.unsqueeze(1)
                v_w_sub = v_w_inv_gpu.unsqueeze(1)
            else:
                w_inv_sub = w_inv_gpu[isel].unsqueeze(1)
                u_w_sub = u_w_inv_gpu[isel].unsqueeze(1)
                v_w_sub = v_w_inv_gpu[isel].unsqueeze(1)

            dX_base = (w_inv_sub * (du - dw * u_w_sub)).T
            dY_base = (w_inv_sub * (dv - dw * v_w_sub)).T

            if w_smoother_T is not None:
                n_samples = w_smoother_T.shape[0]
                n_total = k_count * n_samples
                if isel is None:
                    torch.mul(
                        dX_base.unsqueeze(1),
                        w_smoother_T.unsqueeze(0),
                        out=dX_all[p_offset : p_offset + n_total].view(
                            k_count, n_samples, -1
                        ),
                    )
                    torch.mul(
                        dY_base.unsqueeze(1),
                        w_smoother_T.unsqueeze(0),
                        out=dY_all[p_offset : p_offset + n_total].view(
                            k_count, n_samples, -1
                        ),
                    )
                    if dphi is not None:
                        if dphi.ndim == 1:
                            dphi = dphi.unsqueeze(1)
                        torch.mul(
                            dphi.T.unsqueeze(1),
                            w_smoother_T.unsqueeze(0),
                            out=dphi_all[p_offset : p_offset + n_total].view(
                                k_count, n_samples, -1
                            ),
                        )
                    else:
                        dphi_all[p_offset : p_offset + n_total].zero_()
                    return
                dX_out = (dX_base.unsqueeze(1) * w_smoother_T.unsqueeze(0)).reshape(
                    n_total, -1
                )
                dY_out = (dY_base.unsqueeze(1) * w_smoother_T.unsqueeze(0)).reshape(
                    n_total, -1
                )
                if dphi is not None:
                    if dphi.ndim == 1:
                        dphi = dphi.unsqueeze(1)
                    dphi_out = (
                        dphi.T.unsqueeze(1) * w_smoother_T.unsqueeze(0)
                    ).reshape(n_total, -1)
                else:
                    dphi_out = None
            else:
                n_total = k_count
                dX_out = dX_base
                dY_out = dY_base
                if dphi is not None:
                    if dphi.ndim == 1:
                        dphi = dphi.unsqueeze(1)
                    dphi_out = dphi.T
                else:
                    dphi_out = None

            if isel is None:
                dX_all[p_offset : p_offset + n_total] = dX_out
                dY_all[p_offset : p_offset + n_total] = dY_out
                if dphi_out is not None:
                    dphi_all[p_offset : p_offset + n_total] = dphi_out
                else:
                    dphi_all[p_offset : p_offset + n_total].zero_()
            else:
                dX_all[p_offset : p_offset + n_total, isel] = dX_out
                dY_all[p_offset : p_offset + n_total, isel] = dY_out
                if dphi_out is not None:
                    dphi_all[p_offset : p_offset + n_total, isel] = dphi_out

        # 1. Detector parameterisations
        for dp in self._detector_parameterisations:
            n_free = dp.num_free()
            if n_free == 0:
                continue
            key = tuple(sorted(set(dp.get_experiment_ids())))
            n_sub, isel = isel_map[key]
            if n_sub == 0:
                param_idx += n_free
                continue

            if not is_scan_varying or not isinstance(
                dp, ScanVaryingModelParameterisation
            ):
                pv_sub = _sel(pv_gpu, isel)
                D_sub = _sel(D_gpu, isel)
                if dp.is_multi_state():
                    M_panels = packed_ds_dp[id(dp)]
                    M_refl = M_panels[_sel(panel_id_gpu, isel)]
                    M_pv = torch.einsum("nkij,nj->nki", M_refl, pv_sub)
                else:
                    M = packed_ds_dp[id(dp)]
                    M_pv = torch.einsum("kij,nj->nki", M, pv_sub)
                if D_sub.shape[0] == 1:
                    dpv = -torch.matmul(M_pv, D_sub[0].T)
                else:
                    dpv = -torch.einsum("nij,nkj->nki", D_sub, M_pv)
                _store_grads(param_idx, isel, dpv, None)
            else:
                self._compute_sv_model_grads_gpu(
                    dp,
                    "detector",
                    param_idx,
                    n_sub,
                    isel,
                    imatch_np,
                    b_idx_gpu,
                    D_gpu,
                    pv_gpu,
                    s1_gpu,
                    U_gpu,
                    B_gpu,
                    F_gpu,
                    S_gpu,
                    e0_unit,
                    phi_gpu,
                    h_gpu,
                    relp_gpu,
                    r_gpu,
                    e_cross_r_gpu,
                    inv_ewald_denom_gpu,
                    _store_grads,
                    device,
                )
            param_idx += n_free

        # 2. Beam parameterisations
        for bp in self._beam_parameterisations:
            n_free = bp.num_free()
            if n_free == 0:
                continue
            key = tuple(sorted(set(bp.get_experiment_ids())))
            n_sub, isel = isel_map[key]
            if n_sub == 0:
                param_idx += n_free
                continue

            if not is_scan_varying or not isinstance(
                bp, ScanVaryingModelParameterisation
            ):
                ds0 = packed_ds_dp[id(bp)]
                r_sub = _sel(r_gpu, isel)
                inv_e_sub = _sel(inv_ewald_denom_gpu, isel)
                ecr_sub = _sel(e_cross_r_gpu, isel)
                D_sub = _sel(D_gpu, isel)
                dphi = torch.matmul(r_sub, ds0.T) * inv_e_sub.unsqueeze(1)
                rhs = ecr_sub.unsqueeze(1) * dphi.unsqueeze(-1) + ds0.unsqueeze(0)
                if D_sub.shape[0] == 1:
                    dpv = torch.matmul(rhs, D_sub[0].T)
                else:
                    dpv = torch.einsum("nij,nkj->nki", D_sub, rhs)
                _store_grads(param_idx, isel, dpv, dphi)
            else:
                self._compute_sv_model_grads_gpu(
                    bp,
                    "beam",
                    param_idx,
                    n_sub,
                    isel,
                    imatch_np,
                    b_idx_gpu,
                    D_gpu,
                    pv_gpu,
                    s1_gpu,
                    U_gpu,
                    B_gpu,
                    F_gpu,
                    S_gpu,
                    e0_unit,
                    phi_gpu,
                    h_gpu,
                    relp_gpu,
                    r_gpu,
                    e_cross_r_gpu,
                    inv_ewald_denom_gpu,
                    _store_grads,
                    device,
                )
            param_idx += n_free

        # 3. Crystal orientation parameterisations
        for xlop in self._xl_orientation_parameterisations:
            n_free = xlop.num_free()
            if n_free == 0:
                continue
            key = tuple(sorted(set(xlop.get_experiment_ids())))
            n_sub, isel = isel_map[key]
            if n_sub == 0:
                param_idx += n_free
                continue

            if not is_scan_varying or not isinstance(
                xlop, ScanVaryingModelParameterisation
            ):
                dU = packed_ds_dp[id(xlop)]
                B_sub = _sel(B_gpu, isel)
                F_sub = _sel(F_gpu, isel)
                S_sub = _sel(S_gpu, isel)
                D_sub = _sel(D_gpu, isel)
                h_sub = _sel(h_gpu, isel)
                e0_sub = _sel(e0_unit, isel)
                phi_sub = _sel(phi_gpu, isel)
                s1_sub = _sel(s1_gpu, isel)
                ecr_sub = _sel(e_cross_r_gpu, isel)
                inv_e_sub = _sel(inv_ewald_denom_gpu, isel)

                Bh = torch.matmul(B_sub, h_sub.unsqueeze(-1)).squeeze(-1)
                dU_Bh = torch.einsum("kij,nj->nki", dU, Bh)
                if F_sub.shape[0] == 1:
                    dpstar0 = torch.matmul(dU_Bh, F_sub[0].T)
                else:
                    dpstar0 = torch.einsum("nij,nkj->nki", F_sub, dU_Bh)
                drelp = _rodrigues_rotate_batch(
                    dpstar0, e0_sub.unsqueeze(1), phi_sub.unsqueeze(1)
                )
                if S_sub.shape[0] == 1:
                    dr = torch.matmul(drelp, S_sub[0].T)
                else:
                    dr = torch.einsum("nij,nkj->nki", S_sub, drelp)
                dphi = (dr * s1_sub.unsqueeze(1)).sum(dim=-1) * inv_e_sub.unsqueeze(1)
                rhs = dr + ecr_sub.unsqueeze(1) * dphi.unsqueeze(-1)
                if D_sub.shape[0] == 1:
                    dpv = torch.matmul(rhs, D_sub[0].T)
                else:
                    dpv = torch.einsum("nij,nkj->nki", D_sub, rhs)
                _store_grads(param_idx, isel, dpv, dphi)
            else:
                self._compute_sv_model_grads_gpu(
                    xlop,
                    "xl_orientation",
                    param_idx,
                    n_sub,
                    isel,
                    imatch_np,
                    b_idx_gpu,
                    D_gpu,
                    pv_gpu,
                    s1_gpu,
                    U_gpu,
                    B_gpu,
                    F_gpu,
                    S_gpu,
                    e0_unit,
                    phi_gpu,
                    h_gpu,
                    relp_gpu,
                    r_gpu,
                    e_cross_r_gpu,
                    inv_ewald_denom_gpu,
                    _store_grads,
                    device,
                )
            param_idx += n_free

        # 4. Crystal unit cell parameterisations
        for xlucp in self._xl_unit_cell_parameterisations:
            n_free = xlucp.num_free()
            if n_free == 0:
                continue
            key = tuple(sorted(set(xlucp.get_experiment_ids())))
            n_sub, isel = isel_map[key]
            if n_sub == 0:
                param_idx += n_free
                continue

            if not is_scan_varying or not isinstance(
                xlucp, ScanVaryingModelParameterisation
            ):
                dB = packed_ds_dp[id(xlucp)]
                U_sub = _sel(U_gpu, isel)
                F_sub = _sel(F_gpu, isel)
                S_sub = _sel(S_gpu, isel)
                D_sub = _sel(D_gpu, isel)
                h_sub = _sel(h_gpu, isel)
                e0_sub = _sel(e0_unit, isel)
                phi_sub = _sel(phi_gpu, isel)
                s1_sub = _sel(s1_gpu, isel)
                ecr_sub = _sel(e_cross_r_gpu, isel)
                inv_e_sub = _sel(inv_ewald_denom_gpu, isel)

                dB_h = torch.einsum("kij,nj->nki", dB, h_sub)
                FU_sub = torch.matmul(F_sub, U_sub)
                if FU_sub.shape[0] == 1:
                    dpstar0 = torch.matmul(dB_h, FU_sub[0].T)
                else:
                    dpstar0 = torch.einsum("nij,nkj->nki", FU_sub, dB_h)
                drelp = _rodrigues_rotate_batch(
                    dpstar0, e0_sub.unsqueeze(1), phi_sub.unsqueeze(1)
                )
                if S_sub.shape[0] == 1:
                    dr = torch.matmul(drelp, S_sub[0].T)
                else:
                    dr = torch.einsum("nij,nkj->nki", S_sub, drelp)
                dphi = (dr * s1_sub.unsqueeze(1)).sum(dim=-1) * inv_e_sub.unsqueeze(1)
                rhs = dr + ecr_sub.unsqueeze(1) * dphi.unsqueeze(-1)
                if D_sub.shape[0] == 1:
                    dpv = torch.matmul(rhs, D_sub[0].T)
                else:
                    dpv = torch.einsum("nij,nkj->nki", D_sub, rhs)
                _store_grads(param_idx, isel, dpv, dphi)
            else:
                self._compute_sv_model_grads_gpu(
                    xlucp,
                    "xl_unit_cell",
                    param_idx,
                    n_sub,
                    isel,
                    imatch_np,
                    b_idx_gpu,
                    D_gpu,
                    pv_gpu,
                    s1_gpu,
                    U_gpu,
                    B_gpu,
                    F_gpu,
                    S_gpu,
                    e0_unit,
                    phi_gpu,
                    h_gpu,
                    relp_gpu,
                    r_gpu,
                    e_cross_r_gpu,
                    inv_ewald_denom_gpu,
                    _store_grads,
                    device,
                )
            param_idx += n_free

        # 5. Goniometer parameterisations
        for gp in self._goniometer_parameterisations:
            n_free = gp.num_free()
            if n_free == 0:
                continue
            key = tuple(sorted(set(gp.get_experiment_ids())))
            n_sub, isel = isel_map[key]
            if n_sub == 0:
                param_idx += n_free
                continue

            if not is_scan_varying or not isinstance(
                gp, ScanVaryingModelParameterisation
            ):
                dS = packed_ds_dp[id(gp)]
                relp_sub = _sel(relp_gpu, isel)
                s1_sub = _sel(s1_gpu, isel)
                ecr_sub = _sel(e_cross_r_gpu, isel)
                inv_e_sub = _sel(inv_ewald_denom_gpu, isel)
                D_sub = _sel(D_gpu, isel)

                dr = torch.einsum("kij,nj->nki", dS, relp_sub)
                dphi = (dr * s1_sub.unsqueeze(1)).sum(dim=-1) * inv_e_sub.unsqueeze(1)
                rhs = dr + ecr_sub.unsqueeze(1) * dphi.unsqueeze(-1)
                if D_sub.shape[0] == 1:
                    dpv = torch.matmul(rhs, D_sub[0].T)
                else:
                    dpv = torch.einsum("nij,nkj->nki", D_sub, rhs)
                _store_grads(param_idx, isel, dpv, dphi)
            else:
                self._compute_sv_model_grads_gpu(
                    gp,
                    "goniometer",
                    param_idx,
                    n_sub,
                    isel,
                    imatch_np,
                    b_idx_gpu,
                    D_gpu,
                    pv_gpu,
                    s1_gpu,
                    U_gpu,
                    B_gpu,
                    F_gpu,
                    S_gpu,
                    e0_unit,
                    phi_gpu,
                    h_gpu,
                    relp_gpu,
                    r_gpu,
                    e_cross_r_gpu,
                    inv_ewald_denom_gpu,
                    _store_grads,
                    device,
                )
            param_idx += n_free

        return dX_all, dY_all, dphi_all

    def _compute_sv_model_grads_gpu(
        self,
        parameterisation,
        model_kind: str,
        param_idx: int,
        n_sub: int,
        isel: torch.Tensor | None,
        imatch_np: np.ndarray | None,
        b_idx_gpu: torch.Tensor | None,
        D_gpu,
        pv_gpu,
        s1_gpu,
        U_gpu,
        B_gpu,
        F_gpu,
        S_gpu,
        e0_unit,
        phi_gpu,
        h_gpu,
        relp_gpu,
        r_gpu,
        e_cross_r_gpu,
        inv_ewald_denom_gpu,
        _store_grads,
        device,
    ):
        """Compute scan-varying gradients for a single ScanVaryingModelParameterisation on GPU."""
        if n_sub == 0:
            return
        n_free = parameterisation.num_free()
        if n_free == 0:
            return

        def _sel(t: torch.Tensor) -> torch.Tensor:
            if isel is None or t.shape[0] == 1:
                return t
            return t[isel]

        sv_comp = getattr(self, "_sv_composed", None)
        pid = id(parameterisation)
        use_factored = (
            b_idx_gpu is not None
            and sv_comp is not None
            and pid in sv_comp["base_ders"]
        )

        if use_factored:
            b_sub = b_idx_gpu if isel is None else b_idx_gpu[isel]
            w_smoother_T = sv_comp["W_gpu"][pid].T[:, b_sub]
            base_der_gpu = torch.as_tensor(
                sv_comp["base_ders"][pid], dtype=torch.float64, device=device
            )
        else:
            self._ensure_cpu_sv_derivative_cache()
            w_smoother_T = None
            n_refl = phi_gpu.shape[0]
            n_cache = self._derivative_cache.nref or n_refl
            entry = self._derivative_cache._cache[parameterisation]

        if model_kind == "beam":
            if use_factored:
                ds0_refl = base_der_gpu[b_sub]
            else:
                V_all = np.zeros((n_cache, n_free, 3), dtype=np.float64)
                for k, p_data in enumerate(entry):
                    for pair in p_data:
                        idx = flumpy.to_numpy(pair.iselection).astype(np.int64)
                        V_all[idx, k] = pair.derivative.elems
                if imatch_np is not None:
                    V_all = V_all[imatch_np]
                ds0_refl = _sel(
                    torch.as_tensor(V_all, dtype=torch.float64, device=device)
                )
            r_sub = _sel(r_gpu)
            inv_e_sub = _sel(inv_ewald_denom_gpu)
            ecr_sub = _sel(e_cross_r_gpu)
            D_sub = _sel(D_gpu)
            dphi = (r_sub.unsqueeze(1) * ds0_refl).sum(dim=-1) * inv_e_sub.unsqueeze(1)
            rhs = ecr_sub.unsqueeze(1) * dphi.unsqueeze(-1) + ds0_refl
            if D_sub.shape[0] == 1:
                dpv = torch.matmul(rhs, D_sub[0].T)
            else:
                dpv = torch.einsum("nij,nkj->nki", D_sub, rhs)
            _store_grads(param_idx, isel, dpv, dphi, w_smoother_T)
        else:
            if use_factored:
                M_refl = base_der_gpu[b_sub]
            else:
                M_all = np.zeros((n_cache, n_free, 3, 3), dtype=np.float64)
                for k, p_data in enumerate(entry):
                    for pair in p_data:
                        idx = flumpy.to_numpy(pair.iselection).astype(np.int64)
                        M_all[idx, k] = np.array(
                            pair.derivative.elems, dtype=np.float64
                        ).reshape(3, 3)
                if imatch_np is not None:
                    M_all = M_all[imatch_np]
                M_refl = _sel(
                    torch.as_tensor(M_all, dtype=torch.float64, device=device)
                )
            D_sub = _sel(D_gpu)

            if model_kind == "detector":
                pv_sub = _sel(pv_gpu)
                M_pv = torch.einsum("nkij,nj->nki", M_refl, pv_sub)
                if D_sub.shape[0] == 1:
                    dpv = -torch.matmul(M_pv, D_sub[0].T)
                else:
                    dpv = -torch.einsum("nij,nkj->nki", D_sub, M_pv)
                _store_grads(param_idx, isel, dpv, None, w_smoother_T)
            elif model_kind == "xl_orientation":
                B_sub = _sel(B_gpu)
                F_sub = _sel(F_gpu)
                S_sub = _sel(S_gpu)
                h_sub = _sel(h_gpu)
                e0_sub = _sel(e0_unit)
                phi_sub = _sel(phi_gpu)
                s1_sub = _sel(s1_gpu)
                ecr_sub = _sel(e_cross_r_gpu)
                inv_e_sub = _sel(inv_ewald_denom_gpu)

                Bh = torch.matmul(B_sub, h_sub.unsqueeze(-1)).squeeze(-1)
                dU_Bh = torch.einsum("nkij,nj->nki", M_refl, Bh)
                if F_sub.shape[0] == 1:
                    dpstar0 = torch.matmul(dU_Bh, F_sub[0].T)
                else:
                    dpstar0 = torch.einsum("nij,nkj->nki", F_sub, dU_Bh)
                drelp = _rodrigues_rotate_batch(
                    dpstar0, e0_sub.unsqueeze(1), phi_sub.unsqueeze(1)
                )
                if S_sub.shape[0] == 1:
                    dr = torch.matmul(drelp, S_sub[0].T)
                else:
                    dr = torch.einsum("nij,nkj->nki", S_sub, drelp)
                dphi = (dr * s1_sub.unsqueeze(1)).sum(dim=-1) * inv_e_sub.unsqueeze(1)
                rhs = dr + ecr_sub.unsqueeze(1) * dphi.unsqueeze(-1)
                if D_sub.shape[0] == 1:
                    dpv = torch.matmul(rhs, D_sub[0].T)
                else:
                    dpv = torch.einsum("nij,nkj->nki", D_sub, rhs)
                _store_grads(param_idx, isel, dpv, dphi, w_smoother_T)
            elif model_kind == "xl_unit_cell":
                U_sub = _sel(U_gpu)
                F_sub = _sel(F_gpu)
                S_sub = _sel(S_gpu)
                h_sub = _sel(h_gpu)
                e0_sub = _sel(e0_unit)
                phi_sub = _sel(phi_gpu)
                s1_sub = _sel(s1_gpu)
                ecr_sub = _sel(e_cross_r_gpu)
                inv_e_sub = _sel(inv_ewald_denom_gpu)

                dB_h = torch.einsum("nkij,nj->nki", M_refl, h_sub)
                FU_sub = torch.matmul(F_sub, U_sub)
                dpstar0 = torch.einsum("nij,nkj->nki", FU_sub, dB_h)
                drelp = _rodrigues_rotate_batch(
                    dpstar0, e0_sub.unsqueeze(1), phi_sub.unsqueeze(1)
                )
                if S_sub.shape[0] == 1:
                    dr = torch.matmul(drelp, S_sub[0].T)
                else:
                    dr = torch.einsum("nij,nkj->nki", S_sub, drelp)
                dphi = (dr * s1_sub.unsqueeze(1)).sum(dim=-1) * inv_e_sub.unsqueeze(1)
                rhs = dr + ecr_sub.unsqueeze(1) * dphi.unsqueeze(-1)
                if D_sub.shape[0] == 1:
                    dpv = torch.matmul(rhs, D_sub[0].T)
                else:
                    dpv = torch.einsum("nij,nkj->nki", D_sub, rhs)
                _store_grads(param_idx, isel, dpv, dphi, w_smoother_T)
            elif model_kind == "goniometer":
                relp_sub = _sel(relp_gpu)
                s1_sub = _sel(s1_gpu)
                ecr_sub = _sel(e_cross_r_gpu)
                inv_e_sub = _sel(inv_ewald_denom_gpu)

                dr = torch.einsum("nkij,nj->nki", M_refl, relp_sub)
                dphi = (dr * s1_sub.unsqueeze(1)).sum(dim=-1) * inv_e_sub.unsqueeze(1)
                rhs = dr + ecr_sub.unsqueeze(1) * dphi.unsqueeze(-1)
                if D_sub.shape[0] == 1:
                    dpv = torch.matmul(rhs, D_sub[0].T)
                else:
                    dpv = torch.einsum("nij,nkj->nki", D_sub, rhs)
                _store_grads(param_idx, isel, dpv, dphi, w_smoother_T)


class XYPhiPredictionParameterisationGPU(
    _GPUScanGradientsMixin, XYPhiPredictionParameterisation
):
    pass


class XYPhiPredictionParameterisationSparseGPU(
    _GPUScanGradientsMixin, XYPhiPredictionParameterisationSparse
):
    pass


class _GPUScanVaryingComposeMixin(_GPUScanGradientsMixin):
    """Vectorized block-batched compose() and get_varying_UB() for scan-varying refinement."""

    def _get_cached_sv_block_layout(self, reflections):
        nref = len(reflections)
        exp_id_view = flumpy.to_numpy(reflections["id"])
        block_view = flumpy.to_numpy(reflections["block"])
        bc_view = flumpy.to_numpy(reflections["block_centre"])
        panel_view = flumpy.to_numpy(reflections["panel"])

        # Build signature of free sets so fixing a parameter invalidates base_ders shape
        free_sig = tuple(
            tuple(not ps.get_fixed() for ps in p._param)
            for p in (
                self._detector_parameterisations
                + self._beam_parameterisations
                + self._xl_orientation_parameterisations
                + self._xl_unit_cell_parameterisations
                + self._goniometer_parameterisations
            )
            if hasattr(p, "num_sets")
        )

        cache = getattr(self, "_sv_block_layout_cache", None)
        if (
            cache is not None
            and cache["nref"] == nref
            and cache["free_sig"] == free_sig
            and np.array_equal(exp_id_view, cache["exp_id_np"])
            and np.array_equal(block_view, cache["block_np"])
            and np.array_equal(bc_view, cache["bc_np"])
            and np.array_equal(panel_view, cache["panel_np"])
        ):
            return cache

        n_expts = len(self._experiments)
        exp_entries = []
        B_total = 0
        global_block_idx_np = np.empty(nref, dtype=np.int64)

        for iexp in range(n_expts):
            if n_expts == 1:
                isel_np = None
                blocks_sub = block_view
                bc_sub = bc_view
                panel_sub = panel_view
            else:
                isel_np = np.nonzero(exp_id_view == iexp)[0]
                if len(isel_np) == 0:
                    exp_entries.append(None)
                    continue
                blocks_sub = block_view[isel_np]
                bc_sub = bc_view[isel_np]
                panel_sub = panel_view[isel_np]

            b_min = int(blocks_sub.min())
            rel_b = (blocks_sub - b_min).astype(np.int64)
            counts = np.bincount(rel_b)
            occupied = np.nonzero(counts > 0)[0]
            B_iexp = len(occupied)

            first_pos = np.empty(len(counts), dtype=np.int64)
            first_pos[rel_b] = np.arange(len(blocks_sub), dtype=np.int64)
            bc_per_occ = bc_sub[first_pos[occupied]]

            block_remap = np.empty(len(counts), dtype=np.int64)
            block_remap[occupied] = np.arange(B_iexp, dtype=np.int64)
            local_b_idx = block_remap[rel_b]

            if not np.all(bc_sub == bc_per_occ[local_b_idx]):
                raise AssertionError(
                    "Failing: a block contains reflections that shouldn't be there"
                )

            frames_np = np.floor(bc_per_occ).astype(np.int64)
            start_b = B_total
            end_b = B_total + B_iexp
            B_total = end_b

            if n_expts == 1:
                global_block_idx_np = local_b_idx
            else:
                global_block_idx_np[isel_np] = start_b + local_b_idx

            exp_entries.append(
                (start_b, end_b, isel_np, panel_sub.astype(np.int64), frames_np)
            )

        W_dict = {}
        for p in (
            self._detector_parameterisations
            + self._beam_parameterisations
            + self._xl_orientation_parameterisations
            + self._xl_unit_cell_parameterisations
            + self._goniometer_parameterisations
        ):
            if hasattr(p, "num_sets") and p.num_free() > 0:
                W_p = np.zeros((B_total, p.num_samples()), dtype=np.float64)
                for iexp in p.get_experiment_ids():
                    entry = exp_entries[iexp]
                    if entry is not None:
                        start_b, end_b, _, _, frames_np = entry
                        W_p[start_b:end_b] = _compute_smoother_weight_matrix(
                            p._smoother, p._param[0], frames_np
                        )
                W_dict[id(p)] = W_p

        cache = {
            "nref": nref,
            "free_sig": free_sig,
            "exp_id_np": exp_id_view.copy(),
            "block_np": block_view.copy(),
            "bc_np": bc_view.copy(),
            "panel_np": panel_view.copy(),
            "B_total": B_total,
            "global_block_idx_np": global_block_idx_np,
            "exp_entries": exp_entries,
            "W_dict": W_dict,
        }
        self._sv_block_layout_cache = cache
        return cache

    def compose(self, reflections, skip_derivatives=False):
        """Vectorized batch composition of scan-varying models across all blocks."""
        self._prepare_for_compose(reflections, skip_derivatives)
        self._sv_last_reflections = reflections
        self._sv_derivative_cache_dirty = not skip_derivatives

        nref = len(reflections)
        if nref == 0:
            reflections["ub_matrix"] = flex.mat3_double(0)
            self._sv_composed = None
            return

        if "ub_matrix" not in reflections:
            reflections["ub_matrix"] = flex.mat3_double(nref)

        layout = self._get_cached_sv_block_layout(reflections)
        B_total = layout["B_total"]
        if B_total == 0:
            reflections["ub_matrix"] = (
                reflections["u_matrix"] * reflections["b_matrix"]
            )
            self._sv_composed = None
            return

        global_block_idx_np = layout["global_block_idx_np"]
        exp_entries = layout["exp_entries"]
        W_dict = layout["W_dict"]

        U_blocks = np.empty((B_total, 3, 3), dtype=np.float64)
        B_blocks = np.empty((B_total, 3, 3), dtype=np.float64)
        s0_blocks = np.empty((B_total, 3), dtype=np.float64)
        S_blocks = np.empty((B_total, 3, 3), dtype=np.float64)

        has_sv_det = any(
            hasattr(dp, "num_sets") for dp in self._detector_parameterisations
        )
        if has_sv_det:
            d_blocks = np.empty((B_total, 3, 3), dtype=np.float64)
            D_blocks = np.empty((B_total, 3, 3), dtype=np.float64)
            D_table = None
        else:
            d_blocks = None
            D_blocks = None
            max_panels = max(len(exp.detector) for exp in self._experiments)
            D_table = np.zeros(
                (len(self._experiments), max_panels, 3, 3), dtype=np.float64
            )

        base_ders = {}
        if not skip_derivatives:
            for p_group, shape_tail in (
                (self._detector_parameterisations, (3, 3)),
                (self._beam_parameterisations, (3,)),
                (self._xl_orientation_parameterisations, (3, 3)),
                (self._xl_unit_cell_parameterisations, (3, 3)),
                (self._goniometer_parameterisations, (3, 3)),
            ):
                for p in p_group:
                    if hasattr(p, "num_sets") and p.num_free() > 0:
                        n_free_sets = sum(
                            not ps.get_fixed() for ps in p._param
                        )
                        base_ders[id(p)] = np.zeros(
                            (B_total, n_free_sets, *shape_tail), dtype=np.float64
                        )

        self._current_frame = {}
        d_view = flumpy.to_numpy(reflections["d_matrix"]).reshape(nref, 3, 3)
        D_view = flumpy.to_numpy(reflections["D_matrix"]).reshape(nref, 3, 3)

        for iexp, exp in enumerate(self._experiments):
            entry = exp_entries[iexp]
            if entry is None:
                continue
            start_b, end_b, isel_np, panel_sub, frames_np = entry
            last_frame = int(frames_np[-1])

            xl_op = self._get_xl_orientation_parameterisation(iexp)
            xl_ucp = self._get_xl_unit_cell_parameterisation(iexp)
            bp = self._get_beam_parameterisation(iexp)
            dp = self._get_detector_parameterisation(iexp)
            gp = self._get_goniometer_parameterisation(iexp)

            # 1. Crystal orientation
            if xl_op is not None and hasattr(xl_op, "num_sets"):
                w_sub = (
                    W_dict[id(xl_op)][start_b:end_b]
                    if id(xl_op) in W_dict
                    else _compute_smoother_weight_matrix(
                        xl_op._smoother, xl_op._param[0], frames_np
                    )
                )
                u_b, du_b = _compose_sv_ori_batch(
                    xl_op, w_sub, skip_derivatives or xl_op.num_free() == 0
                )
                U_blocks[start_b:end_b] = u_b
                self._current_frame[xl_op] = last_frame
                if du_b is not None and id(xl_op) in base_ders:
                    base_ders[id(xl_op)][start_b:end_b] = du_b
            else:
                u_mat = (
                    xl_op.get_state().elems
                    if xl_op is not None
                    else exp.crystal.get_U()
                )
                U_blocks[start_b:end_b] = np.asarray(
                    u_mat, dtype=np.float64
                ).reshape(1, 3, 3)

            # 2. Crystal unit cell
            if xl_ucp is not None and hasattr(xl_ucp, "num_sets"):
                w_sub = (
                    W_dict[id(xl_ucp)][start_b:end_b]
                    if id(xl_ucp) in W_dict
                    else _compute_smoother_weight_matrix(
                        xl_ucp._smoother, xl_ucp._param[0], frames_np
                    )
                )
                b_b, db_b = _compose_sv_uc_batch(
                    xl_ucp, w_sub, skip_derivatives or xl_ucp.num_free() == 0
                )
                B_blocks[start_b:end_b] = b_b
                self._current_frame[xl_ucp] = last_frame
                if db_b is not None and id(xl_ucp) in base_ders:
                    base_ders[id(xl_ucp)][start_b:end_b] = db_b
            else:
                b_mat = (
                    xl_ucp.get_state().elems
                    if xl_ucp is not None
                    else exp.crystal.get_B()
                )
                B_blocks[start_b:end_b] = np.asarray(
                    b_mat, dtype=np.float64
                ).reshape(1, 3, 3)

            # 3. Beam
            if bp is not None and hasattr(bp, "num_sets"):
                w_sub = (
                    W_dict[id(bp)][start_b:end_b]
                    if id(bp) in W_dict
                    else _compute_smoother_weight_matrix(
                        bp._smoother, bp._param[0], frames_np
                    )
                )
                s0_b, ds0_b = _compose_sv_beam_batch(
                    bp, w_sub, skip_derivatives or bp.num_free() == 0
                )
                s0_blocks[start_b:end_b] = s0_b
                self._current_frame[bp] = last_frame
                if ds0_b is not None and id(bp) in base_ders:
                    base_ders[id(bp)][start_b:end_b] = ds0_b
            else:
                s0_vec = (
                    bp.get_state().elems if bp is not None else exp.beam.get_s0()
                )
                s0_blocks[start_b:end_b] = np.asarray(
                    s0_vec, dtype=np.float64
                ).reshape(1, 3)

            # 4. Goniometer
            if gp is not None and hasattr(gp, "num_sets"):
                w_sub = (
                    W_dict[id(gp)][start_b:end_b]
                    if id(gp) in W_dict
                    else _compute_smoother_weight_matrix(
                        gp._smoother, gp._param[0], frames_np
                    )
                )
                S_b, dS_b = _compose_sv_gonio_batch(
                    gp, w_sub, skip_derivatives or gp.num_free() == 0
                )
                S_blocks[start_b:end_b] = S_b
                self._current_frame[gp] = last_frame
                if dS_b is not None and id(gp) in base_ders:
                    base_ders[id(gp)][start_b:end_b] = dS_b
            else:
                S_mat = (
                    gp.get_state().elems
                    if gp is not None
                    else exp.goniometer.get_setting_rotation()
                )
                S_blocks[start_b:end_b] = np.asarray(
                    S_mat, dtype=np.float64
                ).reshape(1, 3, 3)

            # 5. Detector
            if dp is not None and hasattr(dp, "num_sets"):
                w_sub = (
                    W_dict[id(dp)][start_b:end_b]
                    if id(dp) in W_dict
                    else _compute_smoother_weight_matrix(
                        dp._smoother, dp._param[0], frames_np
                    )
                )
                d_b, D_b, dd_b = _compose_sv_det_batch(
                    dp, exp, w_sub, skip_derivatives or dp.num_free() == 0
                )
                d_blocks[start_b:end_b] = d_b
                D_blocks[start_b:end_b] = D_b
                self._current_frame[dp] = last_frame
                if dd_b is not None and id(dp) in base_ders:
                    base_ders[id(dp)][start_b:end_b] = dd_b
            else:
                n_panels = len(exp.detector)
                d_panels = np.empty((n_panels, 3, 3), dtype=np.float64)
                D_panels = np.empty((n_panels, 3, 3), dtype=np.float64)
                for pid in range(n_panels):
                    if dp is not None:
                        dmat = (
                            dp.get_state(multi_state_elt=pid)
                            if dp.is_multi_state()
                            else dp.get_state()
                        )
                        d_panels[pid] = np.asarray(
                            dmat.elems, dtype=np.float64
                        ).reshape(3, 3)
                    else:
                        d_panels[pid] = np.asarray(
                            exp.detector[pid].get_d_matrix(), dtype=np.float64
                        ).reshape(3, 3)
                    D_panels[pid] = np.asarray(
                        exp.detector[pid].get_D_matrix(), dtype=np.float64
                    ).reshape(3, 3)

                if D_table is not None:
                    D_table[iexp, :n_panels] = D_panels
                if has_sv_det:
                    # Mixed static and scan-varying detectors across experiments
                    d_blocks[start_b:end_b] = d_panels[0]
                    D_blocks[start_b:end_b] = D_panels[0]
                else:
                    if isel_np is None:
                        if n_panels == 1:
                            d_view[:] = d_panels
                            D_view[:] = D_panels
                        else:
                            d_view[:] = d_panels[panel_sub]
                            D_view[:] = D_panels[panel_sub]
                    else:
                        if n_panels == 1:
                            d_view[isel_np] = d_panels
                            D_view[isel_np] = D_panels
                        else:
                            d_view[isel_np] = d_panels[panel_sub]
                            D_view[isel_np] = D_panels[panel_sub]

        UB_blocks = np.matmul(U_blocks, B_blocks)
        same_refl = layout.get("populated_refl_id") == id(reflections)
        fixed_ori = not self._varying_xl_orientations and all(
            p.num_free() == 0 for p in self._xl_orientation_parameterisations
        )
        fixed_uc = not self._varying_xl_unit_cells and all(
            p.num_free() == 0 for p in self._xl_unit_cell_parameterisations
        )
        fixed_beam = not self._varying_beams and all(
            p.num_free() == 0 for p in self._beam_parameterisations
        )
        fixed_gonio = not self._varying_goniometers and all(
            p.num_free() == 0 for p in self._goniometer_parameterisations
        )

        if not (same_refl and fixed_ori):
            flumpy.to_numpy(reflections["u_matrix"]).reshape(nref, 3, 3)[:] = U_blocks[
                global_block_idx_np
            ]
        if not (same_refl and fixed_uc):
            flumpy.to_numpy(reflections["b_matrix"]).reshape(nref, 3, 3)[:] = B_blocks[
                global_block_idx_np
            ]
        flumpy.to_numpy(reflections["ub_matrix"]).reshape(nref, 3, 3)[:] = UB_blocks[
            global_block_idx_np
        ]
        if not (same_refl and fixed_beam):
            flumpy.to_numpy(reflections["s0_vector"]).reshape(nref, 3)[:] = s0_blocks[
                global_block_idx_np
            ]
        if not (same_refl and fixed_gonio):
            flumpy.to_numpy(reflections["S_matrix"]).reshape(nref, 3, 3)[:] = S_blocks[
                global_block_idx_np
            ]
        if has_sv_det:
            d_view[:] = d_blocks[global_block_idx_np]
            D_view[:] = D_blocks[global_block_idx_np]
        layout["populated_refl_id"] = id(reflections)

        self._sv_composed = {
            "nref": nref,
            "layout": layout,
            "U_blocks": U_blocks,
            "B_blocks": B_blocks,
            "UB_blocks": UB_blocks,
            "s0_blocks": s0_blocks,
            "S_blocks": S_blocks,
            "D_blocks": D_blocks,
            "D_table": D_table,
            "base_ders": base_ders,
            "W_dict": W_dict,
            "global_block_idx_np": global_block_idx_np,
            "global_block_idx_gpu": layout.get("global_block_idx_gpu"),
            "W_gpu": layout.get("W_gpu"),
            "gpu_device": layout.get("gpu_device"),
        }

    def get_varying_UB(self, obs_image_numbers, experiment_id):
        """Vectorized extraction of scan-point UB matrices after scan-varying refinement."""
        if not (self._varying_xl_unit_cells or self._varying_xl_orientations):
            return None

        xl_op = self._get_xl_orientation_parameterisation(experiment_id)
        xl_ucp = self._get_xl_unit_cell_parameterisation(experiment_id)
        frames_np = np.asarray(obs_image_numbers, dtype=np.int64)
        n_pts = len(frames_np)

        if xl_op is not None and hasattr(xl_op, "num_sets"):
            W_ori = _compute_smoother_weight_matrix(
                xl_op._smoother, xl_op._param[0], frames_np
            )
            U_arr, _ = _compose_sv_ori_batch(xl_op, W_ori, skip_derivatives=True)
            self._current_frame[xl_op] = int(frames_np[-1])
        else:
            u_mat = (
                xl_op.get_state().elems
                if xl_op is not None
                else self._experiments[experiment_id].crystal.get_U()
            )
            U_arr = np.broadcast_to(
                np.asarray(u_mat, dtype=np.float64).reshape(1, 3, 3), (n_pts, 3, 3)
            )

        if xl_ucp is not None and hasattr(xl_ucp, "num_sets"):
            W_uc = _compute_smoother_weight_matrix(
                xl_ucp._smoother, xl_ucp._param[0], frames_np
            )
            B_arr, _ = _compose_sv_uc_batch(xl_ucp, W_uc, skip_derivatives=True)
            self._current_frame[xl_ucp] = int(frames_np[-1])
        else:
            b_mat = (
                xl_ucp.get_state().elems
                if xl_ucp is not None
                else self._experiments[experiment_id].crystal.get_B()
            )
            B_arr = np.broadcast_to(
                np.asarray(b_mat, dtype=np.float64).reshape(1, 3, 3), (n_pts, 3, 3)
            )

        UB_arr = np.matmul(U_arr, B_arr).reshape(n_pts, 9)
        return [matrix.sqr(tuple(row)) for row in UB_arr]


class ScanVaryingPredictionParameterisationGPU(
    _GPUScanVaryingComposeMixin, ScanVaryingPredictionParameterisation
):
    pass


class ScanVaryingPredictionParameterisationSparseGPU(
    _GPUScanVaryingComposeMixin, ScanVaryingPredictionParameterisationSparse
):
    pass


class _GPUStillsGradientsMixin(_FastModifyParametersMixin):
    """GPU analytical gradient computation for stills refinement."""

    def can_use_gpu_gradients(self) -> bool:
        return torch.cuda.is_available()

    def get_gradients(self, reflections, callback=None):
        if (
            not self.can_use_gpu_gradients()
            or len(self) * len(reflections) > 25_000_000
        ):
            return super().get_gradients(reflections, callback=callback)
        if callback is count_associated_refs:
            dX, dY, dpsi = self.get_gradients_gpu(reflections)
            if dX.numel() == 0:
                return [0] * len(self)
            mask = (dX.abs() > 1e-10) | (dY.abs() > 1e-10) | (dpsi.abs() > 1e-10)
            return mask.sum(dim=1).cpu().tolist()
        if callback is id_associated_refs:
            dX, dY, dpsi = self.get_gradients_gpu(reflections)
            mask_np = (
                ((dX.abs() > 1e-10) | (dY.abs() > 1e-10) | (dpsi.abs() > 1e-10))
                .cpu()
                .numpy()
            )
            return [
                flumpy.from_numpy(np.ascontiguousarray(mask_np[i]))
                for i in range(len(self))
            ]
        if isinstance(
            self,
            (
                StillsPredictionParameterisationSparse,
                SphericalRelpStillsPredictionParameterisationSparse,
            ),
        ):
            return super().get_gradients(reflections, callback=callback)
        dX, dY, dpsi = self.get_gradients_gpu(reflections)
        stacked_np = torch.stack((dX, dY, dpsi), dim=1).cpu().numpy()
        results = []
        for i in range(len(self)):
            r = {
                "dX_dp": flumpy.from_numpy(np.ascontiguousarray(stacked_np[i, 0])),
                "dY_dp": flumpy.from_numpy(np.ascontiguousarray(stacked_np[i, 1])),
                "dDeltaPsi_dp": flumpy.from_numpy(
                    np.ascontiguousarray(stacked_np[i, 2])
                ),
            }
            results.append(callback(r) if callback is not None else r)
        return results

    def get_gradients_gpu(
        self, reflections, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute (dX_dp, dY_dp, dDeltaPsi_dp) as (P, N) float64 tensors on GPU."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_refl = len(reflections)
        n_params = len(self)
        dX_all = torch.zeros((n_params, n_refl), dtype=torch.float64, device=device)
        dY_all = torch.zeros((n_params, n_refl), dtype=torch.float64, device=device)
        dpsi_all = torch.zeros((n_params, n_refl), dtype=torch.float64, device=device)

        if n_refl == 0 or n_params == 0:
            return dX_all, dY_all, dpsi_all

        spherical_relp = isinstance(
            self, SphericalRelpStillsPredictionParameterisation
        )

        h_gpu = torch.as_tensor(
            reflections["miller_index"].as_vec3_double().as_numpy_array(),
            dtype=torch.float64,
            device=device,
        )
        s1_gpu = torch.as_tensor(
            flumpy.to_numpy(reflections["s1"]), dtype=torch.float64, device=device
        )
        delpsi_gpu = torch.as_tensor(
            flumpy.to_numpy(reflections["delpsical.rad"]),
            dtype=torch.float64,
            device=device,
        )
        exp_id_np = flumpy.to_numpy(reflections["id"]).astype(np.int64)
        panel_id_np = flumpy.to_numpy(reflections["panel"]).astype(np.int64)
        exp_id_gpu = torch.as_tensor(exp_id_np, device=device, dtype=torch.long)
        panel_id_gpu = torch.as_tensor(panel_id_np, device=device, dtype=torch.long)

        n_expts = len(self._experiments)
        U_table = torch.tensor(
            [list(exp.crystal.get_U()) for exp in self._experiments],
            dtype=torch.float64,
            device=device,
        ).reshape(n_expts, 3, 3)
        B_table = torch.tensor(
            [list(exp.crystal.get_B()) for exp in self._experiments],
            dtype=torch.float64,
            device=device,
        ).reshape(n_expts, 3, 3)
        UB_table = torch.matmul(U_table, B_table)
        s0_table = torch.tensor(
            [list(exp.beam.get_s0()) for exp in self._experiments],
            dtype=torch.float64,
            device=device,
        ).reshape(n_expts, 3)

        U_gpu = U_table[exp_id_gpu]
        B_gpu = B_table[exp_id_gpu]
        UB_gpu = UB_table[exp_id_gpu]
        s0_gpu = s0_table[exp_id_gpu]

        if n_expts == 1 and len(self._experiments[0].detector) == 1:
            D_single = torch.tensor(
                self._experiments[0].detector[0].get_D_matrix(),
                dtype=torch.float64,
                device=device,
            ).reshape(1, 3, 3)
            D_gpu = D_single.expand(n_refl, 3, 3)
        else:
            max_panels = max(len(exp.detector) for exp in self._experiments)
            D_table = torch.zeros(
                (n_expts, max_panels, 3, 3), dtype=torch.float64, device=device
            )
            for iexp, exp in enumerate(self._experiments):
                for ipanel, panel in enumerate(exp.detector):
                    D_table[iexp, ipanel] = torch.tensor(
                        panel.get_D_matrix(), dtype=torch.float64, device=device
                    ).reshape(3, 3)
            D_gpu = D_table[exp_id_gpu, panel_id_gpu]

        pv_gpu = torch.matmul(D_gpu, s1_gpu.unsqueeze(-1)).squeeze(-1)
        u_gpu = pv_gpu[:, 0]
        v_gpu = pv_gpu[:, 1]
        w_gpu = pv_gpu[:, 2]
        w_inv_gpu = 1.0 / w_gpu
        u_w_inv_gpu = u_gpu * w_inv_gpu
        v_w_inv_gpu = v_gpu * w_inv_gpu

        q_gpu = torch.matmul(UB_gpu, h_gpu.unsqueeze(-1)).squeeze(-1)
        qq_gpu = (q_gpu * q_gpu).sum(dim=-1)
        q_scalar_gpu = torch.sqrt(qq_gpu)
        r_gpu = s1_gpu - s0_gpu
        q0_gpu = q_gpu / q_scalar_gpu.unsqueeze(-1)
        nu_gpu = torch.linalg.norm(s0_gpu, dim=-1)
        wl_gpu = 1.0 / nu_gpu
        s0u_gpu = s0_gpu * wl_gpu.unsqueeze(-1)
        e1_raw = torch.cross(q0_gpu, s0u_gpu, dim=-1)
        e1_gpu = e1_raw / torch.linalg.norm(e1_raw, dim=-1, keepdim=True)
        c0_raw = torch.cross(s0u_gpu, e1_gpu, dim=-1)
        c0_gpu = c0_raw / torch.linalg.norm(c0_raw, dim=-1, keepdim=True)
        e1_cross_r_gpu = torch.cross(e1_gpu, r_gpu, dim=-1)
        inv_ewald_denom_gpu = -1.0 / (e1_cross_r_gpu * s0_gpu).sum(dim=-1)

        if spherical_relp:
            q_s0_gpu = q_gpu + s0_gpu
            ss_gpu = (q_s0_gpu * q_s0_gpu).sum(dim=-1)
            s_gpu = torch.sqrt(ss_gpu)
            inv_s_gpu = 1.0 / s_gpu
            inv_sss_gpu = 1.0 / (s_gpu * ss_gpu)

        exp_masks = {
            iexp: (
                torch.ones(n_refl, dtype=torch.bool, device=device)
                if n_expts == 1 and iexp == 0
                else (exp_id_gpu == iexp)
            )
            for iexp in range(n_expts)
        }

        def _store_grads(
            p_offset: int,
            mask: torch.Tensor | None,
            dpv: torch.Tensor,
            dpsi: torch.Tensor | None,
        ):
            if dpv.ndim == 2:
                dpv = dpv.unsqueeze(1)
            if mask is None or mask.all():
                w_inv_sub = w_inv_gpu.unsqueeze(1)
                u_w_sub = u_w_inv_gpu.unsqueeze(1)
                v_w_sub = v_w_inv_gpu.unsqueeze(1)
                du = dpv[:, :, 0]
                dv = dpv[:, :, 1]
                dw = dpv[:, :, 2]
                k_count = dpv.shape[1]
                dX_all[p_offset : p_offset + k_count] = (
                    w_inv_sub * (du - dw * u_w_sub)
                ).T
                dY_all[p_offset : p_offset + k_count] = (
                    w_inv_sub * (dv - dw * v_w_sub)
                ).T
                if dpsi is not None:
                    if dpsi.ndim == 1:
                        dpsi = dpsi.unsqueeze(1)
                    dpsi_all[p_offset : p_offset + k_count] = dpsi.T
            else:
                if not mask.any():
                    return
                w_inv_sub = w_inv_gpu[mask].unsqueeze(1)
                u_w_sub = u_w_inv_gpu[mask].unsqueeze(1)
                v_w_sub = v_w_inv_gpu[mask].unsqueeze(1)
                du = dpv[:, :, 0]
                dv = dpv[:, :, 1]
                dw = dpv[:, :, 2]
                k_count = dpv.shape[1]
                dX_all[p_offset : p_offset + k_count, mask] = (
                    w_inv_sub * (du - dw * u_w_sub)
                ).T
                dY_all[p_offset : p_offset + k_count, mask] = (
                    w_inv_sub * (dv - dw * v_w_sub)
                ).T
                if dpsi is not None:
                    if dpsi.ndim == 1:
                        dpsi = dpsi.unsqueeze(1)
                    dpsi_all[p_offset : p_offset + k_count, mask] = dpsi.T

        param_idx = 0

        # 1. Detector parameterisations
        for dp in self._detector_parameterisations:
            n_free = dp.num_free()
            if n_free == 0:
                continue
            exp_ids = dp.get_experiment_ids()
            mask = exp_masks[exp_ids[0]].clone()
            for eid in exp_ids[1:]:
                mask |= exp_masks[eid]
            if dp.is_multi_state():
                n_panels = len(self._experiments[exp_ids[0]].detector)
                M_panels = torch.zeros(
                    (n_panels, n_free, 3, 3), dtype=torch.float64, device=device
                )
                for ip in range(n_panels):
                    for k, der in enumerate(
                        dp.get_ds_dp(multi_state_elt=ip, use_none_as_null=True)
                    ):
                        if der is not None:
                            M_panels[ip, k] = torch.tensor(
                                der.elems, dtype=torch.float64, device=device
                            ).reshape(3, 3)
                M_refl = M_panels[panel_id_gpu[mask]]
                pv_sub = pv_gpu[mask]
                D_sub = D_gpu[mask]
                M_pv = torch.einsum("nkij,nj->nki", M_refl, pv_sub)
                dpv = -torch.einsum("nij,nkj->nki", D_sub, M_pv)
                _store_grads(param_idx, mask, dpv, None)
            else:
                ds_dp = [
                    der
                    for der in dp.get_ds_dp(use_none_as_null=True)
                    if der is not None
                ]
                M = torch.tensor(
                    [list(der.elems) for der in ds_dp],
                    dtype=torch.float64,
                    device=device,
                ).reshape(n_free, 3, 3)
                pv_sub = pv_gpu[mask]
                D_sub = D_gpu[mask]
                M_pv = torch.einsum("kij,nj->nki", M, pv_sub)
                dpv = -torch.einsum("nij,nkj->nki", D_sub, M_pv)
                _store_grads(param_idx, mask, dpv, None)
            param_idx += n_free

        # 2. Beam parameterisations
        for bp in self._beam_parameterisations:
            n_free = bp.num_free()
            if n_free == 0:
                continue
            exp_ids = bp.get_experiment_ids()
            mask = exp_masks[exp_ids[0]].clone()
            for eid in exp_ids[1:]:
                mask |= exp_masks[eid]
            ds_dp = [
                der for der in bp.get_ds_dp(use_none_as_null=True) if der is not None
            ]
            ds0 = torch.tensor(
                [list(der.elems) for der in ds_dp],
                dtype=torch.float64,
                device=device,
            ).reshape(n_free, 3)
            r_sub = r_gpu[mask]
            inv_e_sub = inv_ewald_denom_gpu[mask]
            ecr_sub = e1_cross_r_gpu[mask]
            D_sub = D_gpu[mask]
            dpsi = torch.matmul(r_sub, ds0.T) * inv_e_sub.unsqueeze(1)
            if spherical_relp:
                s0u_sub = s0u_gpu[mask]
                nu_sub = nu_gpu[mask]
                q_s0_sub = q_s0_gpu[mask]
                inv_s_sub = inv_s_gpu[mask]
                inv_sss_sub = inv_sss_gpu[mask]
                s0u_dot_ds0 = (s0u_sub.unsqueeze(1) * ds0.unsqueeze(0)).sum(
                    dim=-1, keepdim=True
                )
                term1 = (
                    s0u_dot_ds0 * q_s0_sub.unsqueeze(1)
                    + nu_sub.unsqueeze(1).unsqueeze(2) * ds0.unsqueeze(0)
                ) * inv_s_sub.unsqueeze(1).unsqueeze(2)
                qs0_dot_ds0 = (q_s0_sub.unsqueeze(1) * ds0.unsqueeze(0)).sum(
                    dim=-1, keepdim=True
                )
                term2 = (
                    nu_sub.unsqueeze(1).unsqueeze(2)
                    * q_s0_sub.unsqueeze(1)
                    * qs0_dot_ds0
                ) * inv_sss_sub.unsqueeze(1).unsqueeze(2)
                dpv = torch.einsum("nij,nkj->nki", D_sub, term1 - term2)
            else:
                wl_sub = wl_gpu[mask]
                c0_sub = c0_gpu[mask]
                e1_sub = e1_gpu[mask]
                q_sub = q_gpu[mask]
                delpsi_sub = delpsi_gpu[mask]
                scaled = wl_sub.unsqueeze(1).unsqueeze(2) * ds0.unsqueeze(0)
                ds0u = (scaled * c0_sub.unsqueeze(1)).sum(
                    dim=-1, keepdim=True
                ) * c0_sub.unsqueeze(1) + (scaled * e1_sub.unsqueeze(1)).sum(
                    dim=-1, keepdim=True
                ) * e1_sub.unsqueeze(1)
                de1_dp = torch.cross(c0_sub.unsqueeze(1), ds0u, dim=-1)
                cos_d = torch.cos(delpsi_sub).unsqueeze(1).unsqueeze(2)
                sin_d = torch.sin(delpsi_sub).unsqueeze(1).unsqueeze(2)
                e1_dot_q = (e1_sub * q_sub).sum(dim=-1, keepdim=True).unsqueeze(1)
                de1_dot_q = (de1_dp * q_sub.unsqueeze(1)).sum(dim=-1, keepdim=True)
                drde_dedp = (
                    torch.cross(de1_dp, q_sub.unsqueeze(1), dim=-1) * sin_d
                    + (de1_dot_q * e1_sub.unsqueeze(1) + e1_dot_q * de1_dp)
                    * (1.0 - cos_d)
                )
                rhs = (
                    ds0.unsqueeze(0)
                    + ecr_sub.unsqueeze(1) * dpsi.unsqueeze(-1)
                    + drde_dedp
                )
                dpv = torch.einsum("nij,nkj->nki", D_sub, rhs)
            _store_grads(param_idx, mask, dpv, dpsi)
            param_idx += n_free

        # 3 & 4. Crystal orientation and unit cell parameterisations
        for model_kind, param_list in (
            ("xl_orientation", self._xl_orientation_parameterisations),
            ("xl_unit_cell", self._xl_unit_cell_parameterisations),
        ):
            for xlp in param_list:
                n_free = xlp.num_free()
                if n_free == 0:
                    continue
                exp_ids = xlp.get_experiment_ids()
                mask = exp_masks[exp_ids[0]].clone()
                for eid in exp_ids[1:]:
                    mask |= exp_masks[eid]
                ds_dp = [
                    der
                    for der in xlp.get_ds_dp(use_none_as_null=True)
                    if der is not None
                ]
                M = torch.tensor(
                    [list(der.elems) for der in ds_dp],
                    dtype=torch.float64,
                    device=device,
                ).reshape(n_free, 3, 3)
                h_sub = h_gpu[mask]
                if model_kind == "xl_orientation":
                    B_sub = B_gpu[mask]
                    Bh = torch.matmul(B_sub, h_sub.unsqueeze(-1)).squeeze(-1)
                    dq = torch.einsum("kij,nj->nki", M, Bh)
                else:
                    U_sub = U_gpu[mask]
                    dB_h = torch.einsum("kij,nj->nki", M, h_sub)
                    dq = torch.einsum("nij,nkj->nki", U_sub, dB_h)

                e1_sub = e1_gpu[mask]
                delpsi_sub = delpsi_gpu[mask]
                s1_sub = s1_gpu[mask]
                inv_e_sub = inv_ewald_denom_gpu[mask]
                ecr_sub = e1_cross_r_gpu[mask]
                D_sub = D_gpu[mask]

                dr = _rodrigues_rotate_batch(
                    dq, e1_sub.unsqueeze(1), delpsi_sub.unsqueeze(1)
                )
                dpsi = (dr * s1_sub.unsqueeze(1)).sum(dim=-1) * inv_e_sub.unsqueeze(1)
                if spherical_relp:
                    nu_sub = nu_gpu[mask]
                    q_s0_sub = q_s0_gpu[mask]
                    inv_s_sub = inv_s_gpu[mask]
                    inv_sss_sub = inv_sss_gpu[mask]
                    term1 = (
                        nu_sub.unsqueeze(1).unsqueeze(2) * dq
                    ) * inv_s_sub.unsqueeze(1).unsqueeze(2)
                    qs0_dot_dq = (q_s0_sub.unsqueeze(1) * dq).sum(dim=-1, keepdim=True)
                    term2 = (
                        nu_sub.unsqueeze(1).unsqueeze(2)
                        * q_s0_sub.unsqueeze(1)
                        * qs0_dot_dq
                    ) * inv_sss_sub.unsqueeze(1).unsqueeze(2)
                    dpv = torch.einsum("nij,nkj->nki", D_sub, term1 - term2)
                else:
                    q_sub = q_gpu[mask]
                    q_scalar_sub = q_scalar_gpu[mask]
                    qq_sub = qq_gpu[mask]
                    q0_sub = q0_gpu[mask]
                    s0u_sub = s0u_gpu[mask]
                    q_dot_dq = (q_sub.unsqueeze(1) * dq).sum(dim=-1, keepdim=True)
                    dq0 = (
                        q_scalar_sub.unsqueeze(1).unsqueeze(2) * dq
                        - q_dot_dq * q0_sub.unsqueeze(1)
                    ) / qq_sub.unsqueeze(1).unsqueeze(2)
                    de1_dp = torch.cross(dq0, s0u_sub.unsqueeze(1), dim=-1)
                    dp_step = 1.0e-8
                    del_e1 = de1_dp * dp_step
                    e1f = e1_sub.unsqueeze(1) + del_e1 * 0.5
                    e1f_unit = e1f / torch.linalg.norm(e1f, dim=-1, keepdim=True)
                    rfwd = _rodrigues_rotate_batch(
                        q_sub.unsqueeze(1), e1f_unit, delpsi_sub.unsqueeze(1)
                    )
                    e1r = e1_sub.unsqueeze(1) - del_e1 * 0.5
                    e1r_unit = e1r / torch.linalg.norm(e1r, dim=-1, keepdim=True)
                    rrev = _rodrigues_rotate_batch(
                        q_sub.unsqueeze(1), e1r_unit, delpsi_sub.unsqueeze(1)
                    )
                    drde_dedp = (rfwd - rrev) * (1.0 / dp_step)
                    rhs = dr + ecr_sub.unsqueeze(1) * dpsi.unsqueeze(-1) + drde_dedp
                    dpv = torch.einsum("nij,nkj->nki", D_sub, rhs)
                _store_grads(param_idx, mask, dpv, dpsi)
                param_idx += n_free

        return dX_all, dY_all, dpsi_all


class StillsPredictionParameterisationGPU(
    _GPUStillsGradientsMixin, StillsPredictionParameterisation
):
    pass


class StillsPredictionParameterisationSparseGPU(
    _GPUStillsGradientsMixin, StillsPredictionParameterisationSparse
):
    pass


class SphericalRelpStillsPredictionParameterisationGPU(
    _GPUStillsGradientsMixin, SphericalRelpStillsPredictionParameterisation
):
    pass


class SphericalRelpStillsPredictionParameterisationSparseGPU(
    _GPUStillsGradientsMixin, SphericalRelpStillsPredictionParameterisationSparse
):
    pass


_CLASS_MAP = {
    XYPhiPredictionParameterisation: XYPhiPredictionParameterisationGPU,
    XYPhiPredictionParameterisationSparse: XYPhiPredictionParameterisationSparseGPU,
    ScanVaryingPredictionParameterisation: ScanVaryingPredictionParameterisationGPU,
    ScanVaryingPredictionParameterisationSparse: ScanVaryingPredictionParameterisationSparseGPU,
    StillsPredictionParameterisation: StillsPredictionParameterisationGPU,
    StillsPredictionParameterisationSparse: StillsPredictionParameterisationSparseGPU,
    SphericalRelpStillsPredictionParameterisation: SphericalRelpStillsPredictionParameterisationGPU,
    SphericalRelpStillsPredictionParameterisationSparse: SphericalRelpStillsPredictionParameterisationSparseGPU,
}


def build_prediction_parameterisation_gpu(
    params, experiments, reflection_manager=None, do_stills=False
):
    pred_param = _dials_build_prediction_parameterisation(
        params,
        experiments,
        reflection_manager=reflection_manager,
        do_stills=do_stills,
    )
    gpu_cls = _CLASS_MAP.get(type(pred_param))
    if gpu_cls is not None:
        pred_param.__class__ = gpu_cls
    _patch_crystal_uc_parameterisations(pred_param)
    return pred_param
