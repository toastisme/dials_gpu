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

    def get_gradients(self, reflections, callback=None):
        if not self.can_use_gpu_gradients() or len(self) * len(reflections) > 10_000_000:
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
        h_np = reflections["miller_index"].as_vec3_double().as_numpy_array()
        exp_id_np = np.array(flumpy.to_numpy(reflections["id"]), copy=True)
        panel_id_np = np.array(flumpy.to_numpy(reflections["panel"]), copy=True)

        cache = getattr(self, "_gpu_refl_cache", None)
        if (
            cache is not None
            and cache["n_refl"] == n_refl
            and cache["device"] == device
            and np.array_equal(h_np, cache["h_np"])
            and np.array_equal(exp_id_np, cache["exp_id_np"])
            and np.array_equal(panel_id_np, cache["panel_id_np"])
        ):
            return cache

        h_gpu = torch.as_tensor(
            h_np, dtype=torch.float64, device=device
        )
        exp_id_i64 = exp_id_np.astype(np.int64)
        panel_id_i64 = panel_id_np.astype(np.int64)
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
            "h_np": h_np.copy(),
            "exp_id_np": exp_id_np.copy(),
            "panel_id_np": panel_id_np.copy(),
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
        dX_all = torch.zeros((n_params, n_refl), dtype=torch.float64, device=device)
        dY_all = torch.zeros((n_params, n_refl), dtype=torch.float64, device=device)
        dphi_all = torch.zeros((n_params, n_refl), dtype=torch.float64, device=device)

        if n_refl == 0 or n_params == 0:
            return dX_all, dY_all, dphi_all

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
        n_expts = len(self._experiments)

        if is_scan_varying:
            imatch_np = (
                flumpy.to_numpy(reflections["imatch"]).astype(np.int64)
                if "imatch" in reflections
                else None
            )
            step_refl_np = np.empty((24, n_refl, 3), dtype=np.float64)
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
                step_refl_np[:18], dtype=torch.float64, device=device
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
        pv_gpu = torch.matmul(D_gpu, s1_gpu.unsqueeze(-1)).squeeze(-1)
        u_gpu = pv_gpu[:, 0]
        v_gpu = pv_gpu[:, 1]
        w_gpu = pv_gpu[:, 2]
        w_inv_gpu = 1.0 / w_gpu
        u_w_inv_gpu = u_gpu * w_inv_gpu
        v_w_inv_gpu = v_gpu * w_inv_gpu

        UB_h = torch.matmul(UB_gpu, h_gpu.unsqueeze(-1)).squeeze(-1)
        pstar0 = torch.matmul(F_gpu, UB_h.unsqueeze(-1)).squeeze(-1)
        relp_gpu = _rodrigues_rotate_batch(pstar0, e0_unit, phi_gpu)
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
                dX_all[p_offset : p_offset + k_count] = (
                    w_inv_sub * (du - dw * u_w_sub)
                ).T
                dY_all[p_offset : p_offset + k_count] = (
                    w_inv_sub * (dv - dw * v_w_sub)
                ).T
                if dphi is not None:
                    if dphi.ndim == 1:
                        dphi = dphi.unsqueeze(1)
                    dphi_all[p_offset : p_offset + k_count] = dphi.T
            else:
                w_inv_sub = w_inv_gpu[isel].unsqueeze(1)
                u_w_sub = u_w_inv_gpu[isel].unsqueeze(1)
                v_w_sub = v_w_inv_gpu[isel].unsqueeze(1)
                dX_all[p_offset : p_offset + k_count, isel] = (
                    w_inv_sub * (du - dw * u_w_sub)
                ).T
                dY_all[p_offset : p_offset + k_count, isel] = (
                    w_inv_sub * (dv - dw * v_w_sub)
                ).T
                if dphi is not None:
                    if dphi.ndim == 1:
                        dphi = dphi.unsqueeze(1)
                    dphi_all[p_offset : p_offset + k_count, isel] = dphi.T

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
        n_refl = phi_gpu.shape[0]
        n_cache = self._derivative_cache.nref or n_refl
        entry = self._derivative_cache._cache[parameterisation]

        def _sel(t: torch.Tensor) -> torch.Tensor:
            if isel is None or t.shape[0] == 1:
                return t
            return t[isel]

        if model_kind == "beam":
            V_all = np.zeros((n_cache, n_free, 3), dtype=np.float64)
            for k, p_data in enumerate(entry):
                for pair in p_data:
                    idx = flumpy.to_numpy(pair.iselection).astype(np.int64)
                    V_all[idx, k] = pair.derivative.elems
            if imatch_np is not None:
                V_all = V_all[imatch_np]
            ds0_refl = _sel(torch.as_tensor(V_all, dtype=torch.float64, device=device))
            r_sub = _sel(r_gpu)
            inv_e_sub = _sel(inv_ewald_denom_gpu)
            ecr_sub = _sel(e_cross_r_gpu)
            D_sub = _sel(D_gpu)
            dphi = (r_sub.unsqueeze(1) * ds0_refl).sum(dim=-1) * inv_e_sub.unsqueeze(1)
            rhs = ecr_sub.unsqueeze(1) * dphi.unsqueeze(-1) + ds0_refl
            dpv = torch.einsum("nij,nkj->nki", D_sub, rhs)
            _store_grads(param_idx, isel, dpv, dphi)
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
            M_refl = _sel(torch.as_tensor(M_all, dtype=torch.float64, device=device))
            D_sub = _sel(D_gpu)

            if model_kind == "detector":
                pv_sub = _sel(pv_gpu)
                M_pv = torch.einsum("nkij,nj->nki", M_refl, pv_sub)
                dpv = -torch.einsum("nij,nkj->nki", D_sub, M_pv)
                _store_grads(param_idx, isel, dpv, None)
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
                dpstar0 = torch.einsum("nij,nkj->nki", F_sub, dU_Bh)
                drelp = _rodrigues_rotate_batch(
                    dpstar0, e0_sub.unsqueeze(1), phi_sub.unsqueeze(1)
                )
                dr = torch.einsum("nij,nkj->nki", S_sub, drelp)
                dphi = (dr * s1_sub.unsqueeze(1)).sum(dim=-1) * inv_e_sub.unsqueeze(1)
                rhs = dr + ecr_sub.unsqueeze(1) * dphi.unsqueeze(-1)
                dpv = torch.einsum("nij,nkj->nki", D_sub, rhs)
                _store_grads(param_idx, isel, dpv, dphi)
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
                dr = torch.einsum("nij,nkj->nki", S_sub, drelp)
                dphi = (dr * s1_sub.unsqueeze(1)).sum(dim=-1) * inv_e_sub.unsqueeze(1)
                rhs = dr + ecr_sub.unsqueeze(1) * dphi.unsqueeze(-1)
                dpv = torch.einsum("nij,nkj->nki", D_sub, rhs)
                _store_grads(param_idx, isel, dpv, dphi)
            elif model_kind == "goniometer":
                relp_sub = _sel(relp_gpu)
                s1_sub = _sel(s1_gpu)
                ecr_sub = _sel(e_cross_r_gpu)
                inv_e_sub = _sel(inv_ewald_denom_gpu)

                dr = torch.einsum("nkij,nj->nki", M_refl, relp_sub)
                dphi = (dr * s1_sub.unsqueeze(1)).sum(dim=-1) * inv_e_sub.unsqueeze(1)
                rhs = dr + ecr_sub.unsqueeze(1) * dphi.unsqueeze(-1)
                dpv = torch.einsum("nij,nkj->nki", D_sub, rhs)
                _store_grads(param_idx, isel, dpv, dphi)


class XYPhiPredictionParameterisationGPU(
    _GPUScanGradientsMixin, XYPhiPredictionParameterisation
):
    pass


class XYPhiPredictionParameterisationSparseGPU(
    _GPUScanGradientsMixin, XYPhiPredictionParameterisationSparse
):
    pass


class ScanVaryingPredictionParameterisationGPU(
    _GPUScanGradientsMixin, ScanVaryingPredictionParameterisation
):
    pass


class ScanVaryingPredictionParameterisationSparseGPU(
    _GPUScanGradientsMixin, ScanVaryingPredictionParameterisationSparse
):
    pass


class _GPUStillsGradientsMixin(_FastModifyParametersMixin):
    """GPU analytical gradient computation for stills refinement."""

    def can_use_gpu_gradients(self) -> bool:
        return torch.cuda.is_available()

    def get_gradients(self, reflections, callback=None):
        if not self.can_use_gpu_gradients() or len(self) * len(reflections) > 10_000_000:
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
