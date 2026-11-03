"""
GPU-accelerated Scalers (SingleScalerGPU, MultiScalerGPU, TargetScalerGPU),
ScalerFactories, calc_sf_variances_gpu, and ScalingAlgorithmGPU.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
from cctbx import crystal, miller
from dxtbx import flumpy
from libtbx import Auto
from scitbx import sparse

from dials.algorithms.scaling.algorithm import (
    ScaleAndFilterAlgorithm,
    ScalingAlgorithm,
    prepare_input,
    scaling_algorithm,
    targeted_scaling_algorithm,
)
from dials.algorithms.scaling.error_model.error_model import extract_error_model_groups
from dials.algorithms.scaling.Ih_table import IhTable
from dials.algorithms.scaling.parameter_handler import ScalingParameterManagerGenerator
from dials.algorithms.scaling.scaler import (
    MultiScaler,
    MultiScalerBase,
    NullScaler,
    ScalerBase,
    SingleScaler,
    TargetScaler,
    calc_sf_variances,
)
from dials.algorithms.scaling.scaler_factory import (
    MultiScalerFactory,
    NullScalerFactory,
    ScalerFactory,
    SingleScalerFactory,
    TargetScalerFactory,
    create_scaler,
)
from dials.algorithms.scaling.scaling_library import (
    choose_initial_scaling_intensities,
    create_scaling_model,
    scaled_data_as_miller_array,
    set_image_ranges_in_scaling_models,
)
from dials.algorithms.scaling.scaling_utilities import (
    BadDatasetForScalingException,
    DialsMergingStatisticsError,
    Reasons,
    log_memory_usage,
    quasi_normalisation,
)
from dials.array_family import flex
from dials.util import tabulate
from dials.util.filter_reflections import sum_partial_reflections
from dials.util.observer import Subject

from .combine_intensities import (
    MultiDatasetIntensityCombinerGPU,
    SingleDatasetIntensityCombinerGPU,
)
from .components import (
    RefinerCalculatorGPU,
    SHScaleComponentGPU,
    _get_device,
    _get_minimization_device,
    align_axis_along_z_gpu,
    calc_crystal_frame_vectors_gpu,
    configure_absorption_component_gpu,
    evaluate_component_gpu,
)
from .error_model import run_error_model_refinement_gpu
from .ih_table import IhTableGPU, calc_Ih_gpu
from .outlier_rejection import (
    determine_Esq_outlier_index_arrays_gpu,
    determine_outlier_index_arrays_gpu,
)
from .refiner import scaling_refinery_gpu
from .target import ScalingTargetFixedIHGPU, ScalingTargetGPU

logger = logging.getLogger("dials")


def calc_sf_variances_gpu(
    components, var_cov, device: torch.device | str | None = None
):
    """GPU-accelerated drop-in replacement for calc_sf_variances."""
    comp_names = list(components.keys())
    if not comp_names:
        return flex.double([])

    n_refl_est = int(components[comp_names[0]].n_refl[0])
    dev = _get_minimization_device(n_refl_est, device)

    scales_list = []
    derivs_list = []
    n_param_tot = 0
    for name in comp_names:
        comp = components[name]
        n_param_tot += comp.n_params
        s_gpu, d_gpu = evaluate_component_gpu(
            comp, block_id=0, compute_derivatives=True, device=dev
        )
        scales_list.append(s_gpu)
        derivs_list.append(d_gpu)

    n_refl = scales_list[0].shape[0]
    if n_refl == 0 or n_param_tot == 0:
        return flex.double(n_refl, 0.0)

    scaled_blocks = []
    for i, d_i in enumerate(derivs_list):
        if d_i.shape[1] == 0:
            continue
        mult = torch.ones(n_refl, dtype=torch.float64, device=dev)
        for j, s_j in enumerate(scales_list):
            if i != j:
                mult = mult * s_j
        scaled_blocks.append(d_i * mult[:, None])

    if not scaled_blocks:
        return flex.double(n_refl, 0.0)

    J_gpu = torch.cat(scaled_blocks, dim=1)
    if isinstance(var_cov, sparse.matrix):
        v_np = np.asarray(
            flumpy.to_numpy(var_cov.as_dense_matrix()), dtype=np.float64
        ).reshape(var_cov.n_rows, var_cov.n_cols)
    else:
        v_np = np.asarray(flumpy.to_numpy(var_cov), dtype=np.float64).reshape(
            J_gpu.shape[1], J_gpu.shape[1]
        )
    V_gpu = torch.from_numpy(v_np).to(device=dev, dtype=torch.float64)
    sigmasq_gpu = torch.sum(torch.mm(J_gpu, V_gpu) * J_gpu, dim=1)
    sigmasq_np = np.ascontiguousarray(sigmasq_gpu.cpu().numpy(), dtype=np.float64)
    return flumpy.from_numpy(sigmasq_np)


class _GPUScalerMixin:
    """Mixin providing GPU-accelerated scaling, refinement, and sync methods."""

    @Subject.notify_event(event="performed_scaling")
    def perform_scaling(self, engine=None, max_iterations=None, tolerance=None):
        """Minimise the scaling model on the GPU."""
        self._perform_scaling(
            target_type=ScalingTargetGPU,
            engine=engine,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

    def _perform_scaling(
        self, target_type, engine=None, max_iterations=None, tolerance=None
    ):
        if target_type is not ScalingTargetFixedIHGPU and issubclass(
            target_type, ScalingTargetGPU
        ) is False:
            # Map CPU target classes to GPU target classes if passed by caller
            if target_type.__name__ == "ScalingTargetFixedIH":
                target_type = ScalingTargetFixedIHGPU
            else:
                target_type = ScalingTargetGPU

        target = target_type()
        shared = self.determine_shared_model_components()
        pmg = ScalingParameterManagerGenerator(
            self.active_scalers,
            target,
            self.params.scaling_refinery.refinement_order,
            shared=shared,
        )
        for apm in pmg.parameter_managers():
            cur_engine = engine or self.params.scaling_refinery.engine
            cur_max_iter = (
                max_iterations or self.params.scaling_refinery.max_iterations
            )
            st = time.time()
            refinery = scaling_refinery_gpu(
                engine=cur_engine,
                scaler=self,
                target=target,
                prediction_parameterisation=apm,
                max_iterations=cur_max_iter,
            )
            if tolerance:
                refinery.set_tolerance(tolerance)
            try:
                refinery.run()
            except RuntimeError as e:
                logger.error(e, exc_info=True)
            logger.info("Time taken for refinement %.2f", (time.time() - st))
            refinery.print_step_table()
            self._sync_gpu_blocks_to_cpu()
            self._update_after_minimisation(apm)
            logger.info("\n" + "=" * 80 + "\n")

    def _sync_gpu_blocks_to_cpu(self):
        """Sync GPU scales and Ih_values back to the pandas DataFrame in each block."""
        if not self.Ih_table:
            return
        for block in self.Ih_table.blocked_data_list:
            g_gpu = getattr(block, "_gpu_scales", None)
            if g_gpu is not None and g_gpu.shape[0] == block.size:
                block.Ih_table.loc[:, "inverse_scale_factor"] = g_gpu.cpu().numpy()
            Ih_gpu = getattr(block, "_gpu_Ih_values", None)
            if Ih_gpu is not None and Ih_gpu.shape[0] == block.size:
                block.Ih_table.loc[:, "Ih_values"] = Ih_gpu.cpu().numpy()


class SingleScalerGPU(_GPUScalerMixin, SingleScaler):
    """GPU-accelerated SingleScaler."""

    def update_for_minimisation_gpu(
        self, apm, block_id: int, calc_derivs: bool = True, sync_to_cpu: bool = False
    ):
        """Update scale factors, derivatives, and Ih on the GPU for block_id."""
        block = self.Ih_table.blocked_data_list[block_id]
        dev = _get_minimization_device(block.size)
        apm_i = apm.apm_list[0]
        scales_gpu, derivs_gpu = (
            RefinerCalculatorGPU.calculate_scales_and_derivatives_gpu(
                apm_i, block_id, device=dev, compute_derivatives=calc_derivs
            )
        )
        n_obs = scales_gpu.shape[0]
        if calc_derivs and derivs_gpu is not None:
            n_params = derivs_gpu.shape[1]
            block._gpu_derivs_blocks = [(0, n_obs, 0, n_params, derivs_gpu)]
            block._gpu_n_params = n_params
        else:
            n_params = 0
            block._gpu_derivs_blocks = None

        calc_Ih_gpu(block, scales_gpu=scales_gpu, sync_to_cpu=sync_to_cpu, device=dev)

        if sync_to_cpu and calc_derivs and derivs_gpu is not None:
            d_np = np.ascontiguousarray(derivs_gpu.cpu().numpy(), dtype=np.float64)
            d_sparse = sparse.matrix(n_obs, n_params)
            if n_obs > 0 and n_params > 0:
                b = flumpy.from_numpy(d_np)
                b.reshape(flex.grid(n_obs, n_params))
                d_sparse.assign_block(b, 0, 0)
            block.derivatives = d_sparse

    def update_for_minimisation(self, apm, block_id):
        self.update_for_minimisation_gpu(
            apm, block_id, calc_derivs=True, sync_to_cpu=True
        )

    def _get_valid_ih_table(
        self, sub_selection=None, anomalous=False, include_partiality=False
    ):
        """Build a lightweight reflection_table containing only the columns required by IhTableGPU,
        reusing cached ASU Miller indices and suitable reflection selections."""
        rt = self._reflection_table
        cache_key = (id(rt), id(self.suitable_refl_for_scaling_sel), rt.size())
        cache = getattr(self, "_gpu_valid_cache", None)
        if cache is None or cache["key"] != cache_key:
            isel = self.suitable_refl_for_scaling_sel.iselection()
            valid_hkl = rt["miller_index"].select(isel)
            valid_d = rt["d"].select(isel) if "d" in rt else None
            valid_part = rt["partiality"].select(isel) if "partiality" in rt else None
            cache = {
                "key": cache_key,
                "isel": isel,
                "hkl": valid_hkl,
                "d": valid_d,
                "partiality": valid_part,
                "asu_hkl": {},
            }
            self._gpu_valid_cache = cache

        anom_flag = bool(anomalous)
        asu_hkl = cache["asu_hkl"].get(anom_flag)
        if asu_hkl is None:
            cs = crystal.symmetry(space_group=self.experiment.crystal.get_space_group())
            ms = miller.set(
                crystal_symmetry=cs, indices=cache["hkl"], anomalous_flag=anom_flag
            )
            asu_hkl = ms.map_to_asu().indices()
            cache["asu_hkl"][anom_flag] = asu_hkl

        out = flex.reflection_table()
        if sub_selection is None:
            final_isel = cache["isel"]
            out["miller_index"] = cache["hkl"]
            out["asu_miller_index"] = asu_hkl
            if cache["d"] is not None:
                out["d"] = cache["d"]
            if include_partiality and cache["partiality"] is not None:
                out["partiality"] = cache["partiality"]
        else:
            final_isel = cache["isel"].select(sub_selection)
            out["miller_index"] = cache["hkl"].select(sub_selection)
            out["asu_miller_index"] = asu_hkl.select(sub_selection)
            if cache["d"] is not None:
                out["d"] = cache["d"].select(sub_selection)
            if include_partiality and cache["partiality"] is not None:
                out["partiality"] = cache["partiality"].select(sub_selection)

        out["intensity"] = rt["intensity"].select(final_isel)
        out["variance"] = rt["variance"].select(final_isel)
        out["inverse_scale_factor"] = rt["inverse_scale_factor"].select(final_isel)
        return out

    def _configure_model_and_datastructures(self, for_multi=False):
        sel_reflections = self.get_valid_reflections()
        model = self.experiment.scaling_model
        abs_comp = model._components.pop("absorption", None)
        try:
            model.configure_components(sel_reflections, self.experiment, self.params)
        finally:
            if abs_comp is not None:
                model._components["absorption"] = abs_comp
        if abs_comp is not None:
            configure_absorption_component_gpu(model, sel_reflections)

        free_set_percentage = 0
        if self.params.scaling_options.use_free_set and not for_multi:
            free_set_percentage = self.params.scaling_options.free_set_percentage
        self._global_Ih_table, self._free_Ih_table = self._create_global_Ih_table(
            free_set_percentage=free_set_percentage, anomalous=self.params.anomalous
        )
        rows = [[key, str(val.n_params)] for key, val in self.components.items()]
        logger.info("The following corrections will be applied to this dataset: \n")
        logger.info(tabulate(rows, ["correction", "n_parameters"]))

    def _create_Ih_table(self):
        """Create an IhTableGPU from the reflection table using the scaling selection."""
        self._Ih_table = IhTableGPU(
            [
                self._get_valid_ih_table(
                    sub_selection=self.scaling_selection,
                    anomalous=self.params.anomalous,
                )
            ],
            self.experiment.crystal.get_space_group(),
            indices_lists=[self.scaling_selection.iselection()],
            nblocks=self.params.scaling_options.nproc,
            anomalous=self.params.anomalous,
        )
        if self._experiment.scaling_model.error_model:
            self._Ih_table.update_weights(self._experiment.scaling_model.error_model)

    def _create_global_Ih_table(
        self, free_set_percentage=0, anomalous=False, remove_outliers=False
    ):
        sub_sel = ~self.outliers if remove_outliers else None
        sel_reflections = self._get_valid_ih_table(
            sub_selection=sub_sel,
            anomalous=anomalous,
            include_partiality=True,
        )
        free_Ih_table = None
        global_Ih_table = IhTableGPU(
            [sel_reflections],
            self.experiment.crystal.get_space_group(),
            nblocks=1,
            free_set_percentage=free_set_percentage,
            free_set_offset=self.params.scaling_options.free_set_offset,
            anomalous=anomalous,
        )
        if free_set_percentage:
            loc_indices = (
                global_Ih_table.blocked_data_list[-1].Ih_table["loc_indices"].to_numpy()
            )
            self.free_set_selection = flex.bool(self.n_suitable_refl, False)
            self.free_set_selection.set_selected(flumpy.from_numpy(loc_indices), True)
            global_Ih_table = IhTableGPU(
                [sel_reflections.select(~self.free_set_selection)],
                self.experiment.crystal.get_space_group(),
                [(~self.free_set_selection).iselection()],
                nblocks=1,
                anomalous=anomalous,
            )
            free_Ih_table = IhTableGPU(
                [sel_reflections.select(self.free_set_selection)],
                self.experiment.crystal.get_space_group(),
                [self.free_set_selection.iselection()],
                nblocks=1,
                anomalous=anomalous,
            )
        return global_Ih_table, free_Ih_table

    def _combine_intensities(self, use_Imid=None):
        try:
            if use_Imid is not None:
                logger.info(
                    "Using previously determined optimal intensity choice: %s\n",
                    {
                        use_Imid: str(round(use_Imid, 4)),
                        0: "profile intensities",
                        1: "summation intensities",
                    }[use_Imid],
                )
            else:
                logger.info("Performing profile/summation intensity optimisation.")
            combiner = SingleDatasetIntensityCombinerGPU(self, use_Imid)
        except DialsMergingStatisticsError as e:
            logger.info("Intensity combination failed with the error %s", e)
        else:
            intensity, variance = combiner.calculate_suitable_combined_intensities()
            isel = self.suitable_refl_for_scaling_sel.iselection()
            self._reflection_table["intensity"].set_selected(isel, intensity)
            self._reflection_table["variance"].set_selected(isel, variance)
            if self.global_Ih_table:
                self.global_Ih_table.update_data_in_blocks(
                    intensity, 0, column="intensity"
                )
                self.global_Ih_table.update_data_in_blocks(
                    variance, 0, column="variance"
                )
                self.global_Ih_table.update_weights(
                    self._experiment.scaling_model.error_model
                )
                for block in self.global_Ih_table.blocked_data_list:
                    calc_Ih_gpu(block, sync_to_cpu=True)
            if self._free_Ih_table:
                self._free_Ih_table.update_data_in_blocks(
                    intensity, 0, column="intensity"
                )
                self._free_Ih_table.update_data_in_blocks(
                    variance, 0, column="variance"
                )
                self._free_Ih_table.update_weights(
                    self._experiment.scaling_model.error_model
                )
                for block in self._free_Ih_table.blocked_data_list:
                    calc_Ih_gpu(block, sync_to_cpu=True)
            self.experiment.scaling_model.record_intensity_combination_Imid(
                combiner.max_key
            )

    @Subject.notify_event(event="performed_error_analysis")
    def perform_error_optimisation(self, update_Ih=True):
        Ih_table, _ = self._create_global_Ih_table(anomalous=True, remove_outliers=True)
        try:
            model = run_error_model_refinement_gpu(
                self._experiment.scaling_model.error_model,
                Ih_table,
                self.params.reflection_selection.min_partiality,
                use_stills_filtering=self.is_still,
            )
        except (ValueError, RuntimeError) as e:
            logger.info(e)
            logger.debug(e, exc_info=True)
        else:
            self._update_error_model(model, update_Ih=update_Ih)

    @Subject.notify_event(event="performed_outlier_rejection")
    def round_of_outlier_rejection(self):
        assert self.global_Ih_table is not None
        if self.params.scaling_options.outlier_rejection:
            outlier_indices = determine_outlier_index_arrays_gpu(
                self.global_Ih_table,
                self.params.scaling_options.outlier_rejection,
                self.params.scaling_options.outlier_zmax,
            )[0]
            outlier_indices = flumpy.from_numpy(outlier_indices)
            self.outliers = flex.bool(self.n_suitable_refl, False)
            self.outliers.set_selected(outlier_indices, True)
            if self.params.scaling_options.emax:
                Esq_outlier_indices = determine_Esq_outlier_index_arrays_gpu(
                    self.global_Ih_table,
                    self.experiment,
                    self.params.scaling_options.emax,
                )[0]
                self.outliers.set_selected(flumpy.from_numpy(Esq_outlier_indices), True)
            if self._free_Ih_table:
                free_outlier_indices = determine_outlier_index_arrays_gpu(
                    self._free_Ih_table,
                    self.params.scaling_options.outlier_rejection,
                    self.params.scaling_options.outlier_zmax,
                )[0]
                self.outliers.set_selected(
                    flumpy.from_numpy(free_outlier_indices), True
                )
                if self.params.scaling_options.emax:
                    free_Esq_outlier_indices = determine_Esq_outlier_index_arrays_gpu(
                        self._free_Ih_table,
                        self.experiment,
                        self.params.scaling_options.emax,
                    )[0]
                    self.outliers.set_selected(
                        flumpy.from_numpy(free_Esq_outlier_indices), True
                    )

    def expand_scales_to_all_reflections(self, caller=None, calc_cov=False):
        """Calculate scale factors (and optional variances) for all suitable reflections on GPU."""
        dev = _get_minimization_device(self.n_suitable_refl)
        self._reflection_table["inverse_scale_factor_variance"] = flex.double(
            self.reflection_table.size(), 0.0
        )
        do_cov = bool(calc_cov and self.var_cov_matrix.non_zeroes > 0)

        scales_list = []
        derivs_list = []
        for component in self.components.values():
            component.update_reflection_data()
            s_gpu, d_gpu = evaluate_component_gpu(
                component, block_id=0, compute_derivatives=do_cov, device=dev
            )
            scales_list.append(s_gpu)
            if do_cov:
                derivs_list.append(d_gpu)

        if scales_list:
            all_scales_gpu = scales_list[0].clone()
            for s_gpu in scales_list[1:]:
                all_scales_gpu = all_scales_gpu * s_gpu
            all_scales_np = np.ascontiguousarray(
                all_scales_gpu.cpu().numpy(), dtype=np.float64
            )
            all_scales = flumpy.from_numpy(all_scales_np)
        else:
            all_scales = flex.double(self.n_suitable_refl, 1.0)
        self._last_all_scales = all_scales

        scaled_isel = self.suitable_refl_for_scaling_sel.iselection()
        self.reflection_table["inverse_scale_factor"].set_selected(
            scaled_isel, all_scales
        )

        if do_cov and scales_list:
            n_refl = self.n_suitable_refl
            comp_list = list(self.components.values())
            scaled_blocks = []
            for j, d_j in enumerate(derivs_list):
                n_param = comp_list[j].n_params
                if n_param == 0:
                    continue
                mult = torch.ones(n_refl, dtype=torch.float64, device=dev)
                for k, s_k in enumerate(scales_list):
                    if j != k:
                        mult = mult * s_k
                sb = d_j * mult[:, None]
                if sb.shape[1] < n_param:
                    pad = torch.zeros(
                        (n_refl, n_param - sb.shape[1]),
                        dtype=torch.float64,
                        device=dev,
                    )
                    sb = torch.cat([sb, pad], dim=1)
                scaled_blocks.append(sb)
            if scaled_blocks:
                J_gpu = torch.cat(scaled_blocks, dim=1)
                v_np = np.asarray(
                    flumpy.to_numpy(self._var_cov_matrix.as_dense_matrix()),
                    dtype=np.float64,
                ).reshape(self._var_cov_matrix.n_rows, self._var_cov_matrix.n_cols)
                V_gpu = torch.from_numpy(v_np).to(device=dev, dtype=torch.float64)
                invsfvars_gpu = torch.sum(torch.mm(J_gpu, V_gpu) * J_gpu, dim=1)
                invsfvars_np = np.ascontiguousarray(
                    invsfvars_gpu.cpu().numpy(), dtype=np.float64
                )
                self.reflection_table["inverse_scale_factor_variance"].set_selected(
                    scaled_isel, flumpy.from_numpy(invsfvars_np)
                )

        if caller is None:
            self.global_Ih_table.update_data_in_blocks(
                all_scales,
                dataset_id=0,
                column="inverse_scale_factor",
            )
            for block in self.global_Ih_table.blocked_data_list:
                calc_Ih_gpu(block, sync_to_cpu=True, device=dev)
            if self._free_Ih_table:
                self._free_Ih_table.update_data_in_blocks(
                    all_scales,
                    dataset_id=0,
                    column="inverse_scale_factor",
                )
                for block in self._free_Ih_table.blocked_data_list:
                    calc_Ih_gpu(block, sync_to_cpu=True, device=dev)
            logger.info(
                "Scale factors determined during minimisation have now been\n"
                "applied to all reflections for dataset %s.\n",
                self.reflection_table["id"][0],
            )


class _MultiScalerGPUMixin(_GPUScalerMixin):
    """GPU methods for MultiScalerBase subclasses (MultiScalerGPU, TargetScalerGPU)."""

    def expand_scales_to_all_reflections(self, calc_cov=False):
        for i, scaler in enumerate(self.active_scalers):
            scaler.expand_scales_to_all_reflections(caller=self, calc_cov=calc_cov)
            scales = getattr(scaler, "_last_all_scales", None)
            if scales is None:
                scales = scaler.reflection_table["inverse_scale_factor"].select(
                    scaler.suitable_refl_for_scaling_sel
                )
            self.global_Ih_table.update_data_in_blocks(
                scales,
                dataset_id=i,
                column="inverse_scale_factor",
            )
            if self._free_Ih_table:
                self._free_Ih_table.update_data_in_blocks(
                    scales,
                    dataset_id=i,
                    column="inverse_scale_factor",
                )

        for block in self.global_Ih_table.blocked_data_list:
            calc_Ih_gpu(block, sync_to_cpu=True)
        if self._free_Ih_table:
            for block in self._free_Ih_table.blocked_data_list:
                calc_Ih_gpu(block, sync_to_cpu=True)
        logger.info(
            "Scale factors determined during minimisation have now been\n"
            "applied to all datasets.\n"
        )

    def _update_for_minimisation_gpu(
        self,
        apm,
        block_id: int,
        calc_Ih: bool = True,
        calc_derivs: bool = True,
        sync_to_cpu: bool = False,
    ):
        block = self.Ih_table.blocked_data_list[block_id]
        dev = _get_minimization_device(block.size)
        scales_parts = []
        derivs_blocks = []
        start_row = 0
        for j, apm_i in enumerate(apm.apm_list):
            scales_i, derivs_i = (
                RefinerCalculatorGPU.calculate_scales_and_derivatives_gpu(
                    apm_i, block_id, device=dev, compute_derivatives=calc_derivs
                )
            )
            scales_parts.append(scales_i)
            if calc_derivs and derivs_i is not None:
                n_rows_i = scales_i.shape[0]
                c0 = int(apm.apm_data[j]["start_idx"])
                c1 = int(apm.apm_data[j]["end_idx"])
                derivs_blocks.append((start_row, start_row + n_rows_i, c0, c1, derivs_i))
                start_row += n_rows_i

        scales_gpu = (
            torch.cat(scales_parts, dim=0)
            if scales_parts
            else torch.empty(0, dtype=torch.float64, device=dev)
        )
        block._gpu_derivs_blocks = derivs_blocks if calc_derivs else None
        block._gpu_n_params = int(apm.n_active_params)

        if calc_Ih:
            calc_Ih_gpu(
                block, scales_gpu=scales_gpu, sync_to_cpu=sync_to_cpu, device=dev
            )
        else:
            block._gpu_scales = scales_gpu
            if sync_to_cpu:
                block.Ih_table.loc[:, "inverse_scale_factor"] = (
                    scales_gpu.cpu().numpy()
                )

        if sync_to_cpu and calc_derivs:
            deriv_matrix = sparse.matrix(scales_gpu.shape[0], apm.n_active_params)
            for r0, r1, c0, c1, d_gpu in derivs_blocks:
                if r1 > r0 and c1 > c0:
                    d_np = np.ascontiguousarray(d_gpu.cpu().numpy(), dtype=np.float64)
                    b = flumpy.from_numpy(d_np)
                    b.reshape(flex.grid(r1 - r0, c1 - c0))
                    deriv_matrix.assign_block(b, r0, c0)
            block.derivatives = deriv_matrix

    def update_for_minimisation_gpu(
        self, apm, block_id: int, calc_derivs: bool = True, sync_to_cpu: bool = False
    ):
        self._update_for_minimisation_gpu(
            apm,
            block_id,
            calc_Ih=True,
            calc_derivs=calc_derivs,
            sync_to_cpu=sync_to_cpu,
        )

    def _update_for_minimisation(self, apm, block_id, calc_Ih=True):
        self._update_for_minimisation_gpu(
            apm, block_id, calc_Ih=calc_Ih, calc_derivs=True, sync_to_cpu=True
        )

    def _create_Ih_table(self):
        """Create a new IhTableGPU from the reflection tables."""
        tables = [
            s._get_valid_ih_table(
                sub_selection=s.scaling_selection, anomalous=self.params.anomalous
            )
            if hasattr(s, "_get_valid_ih_table")
            else s.get_reflections_for_model_minimisation()
            for s in self.active_scalers
        ]
        indices_lists = [s.scaling_selection.iselection() for s in self.active_scalers]
        self._Ih_table = IhTableGPU(
            tables,
            self.active_scalers[0].experiment.crystal.get_space_group(),
            indices_lists=indices_lists,
            nblocks=self.params.scaling_options.nproc,
            anomalous=self.params.anomalous,
        )
        for i, scaler in enumerate(self.active_scalers):
            error_model = scaler._experiment.scaling_model.error_model
            if error_model:
                self._Ih_table.update_weights(error_model, dataset_id=i)

    def _create_global_Ih_table(self, anomalous=False, remove_outliers=False):
        tables = []
        for s in self.active_scalers:
            if hasattr(s, "_get_valid_ih_table"):
                tables.append(
                    s._get_valid_ih_table(
                        sub_selection=(~s.outliers if remove_outliers else None),
                        anomalous=anomalous,
                        include_partiality=True,
                    )
                )
            elif remove_outliers:
                tables.append(s.get_valid_reflections().select(~s.outliers))
            else:
                tables.append(s.get_valid_reflections())
        free_set_percentage = 0.0
        free_Ih_table = None
        if self.params.scaling_options.use_free_set:
            free_set_percentage = self.params.scaling_options.free_set_percentage
        space_group = self.active_scalers[0].experiment.crystal.get_space_group()
        global_Ih_table = IhTableGPU(
            tables,
            space_group,
            nblocks=1,
            additional_cols=["partiality"],
            free_set_percentage=free_set_percentage,
            free_set_offset=self.params.scaling_options.free_set_offset,
            anomalous=anomalous,
        )
        if free_set_percentage:
            tables = []
            free_tables = []
            indices_list = []
            free_indices_list = []
            for i, scaler in enumerate(self.active_scalers):
                sel = (
                    global_Ih_table.Ih_table_blocks[-1]
                    .Ih_table["dataset_id"]
                    .to_numpy()
                    == i
                )
                indiv_Ih_block = global_Ih_table.Ih_table_blocks[-1].select(sel)
                loc_indices = indiv_Ih_block.Ih_table["loc_indices"].to_numpy()
                scaler.free_set_selection = flex.bool(scaler.n_suitable_refl, False)
                scaler.free_set_selection.set_selected(
                    flumpy.from_numpy(loc_indices), True
                )
                tables.append(scaler.get_work_set_reflections())
                free_tables.append(scaler.get_free_set_reflections())
                free_indices_list.append(scaler.free_set_selection.iselection())
                indices_list.append((~scaler.free_set_selection).iselection())
            global_Ih_table = IhTableGPU(
                tables,
                space_group,
                indices_list,
                nblocks=1,
                anomalous=anomalous,
            )
            free_Ih_table = IhTableGPU(
                free_tables,
                space_group,
                free_indices_list,
                nblocks=1,
                anomalous=anomalous,
            )
        return global_Ih_table, free_Ih_table

    def _round_of_outlier_rejection(self, target=None):
        assert self.active_scalers is not None
        if not self.global_Ih_table:
            self._global_Ih_table, self._free_Ih_table = self._create_global_Ih_table(
                anomalous=self.params.anomalous
            )
        if self.params.scaling_options.outlier_rejection:
            method = (
                "target"
                if target
                else self.params.scaling_options.outlier_rejection
            )
            outlier_index_arrays = determine_outlier_index_arrays_gpu(
                self.global_Ih_table,
                method,
                self.params.scaling_options.outlier_zmax,
                target=target,
            )
            for outlier_indices, scaler in zip(
                outlier_index_arrays, self.active_scalers
            ):
                scaler.outliers = flex.bool(scaler.n_suitable_refl, False)
                scaler.outliers.set_selected(flumpy.from_numpy(outlier_indices), True)
            if self.params.scaling_options.emax:
                Esq_outlier_indices = determine_Esq_outlier_index_arrays_gpu(
                    self.global_Ih_table,
                    self.active_scalers[0].experiment,
                    self.params.scaling_options.emax,
                )
                for Esq_outliers, scaler in zip(
                    Esq_outlier_indices, self.active_scalers
                ):
                    scaler.outliers.set_selected(flumpy.from_numpy(Esq_outliers), True)
            if self._free_Ih_table:
                free_outlier_index_arrays = determine_outlier_index_arrays_gpu(
                    self._free_Ih_table,
                    method,
                    self.params.scaling_options.outlier_zmax,
                    target=target,
                )
                for outlier_indices, scaler in zip(
                    free_outlier_index_arrays,
                    self.active_scalers,
                ):
                    scaler.outliers.set_selected(
                        flumpy.from_numpy(outlier_indices), True
                    )
                if self.params.scaling_options.emax:
                    free_Esq_outlier_indices = determine_Esq_outlier_index_arrays_gpu(
                        self._free_Ih_table,
                        self.active_scalers[0].experiment,
                        self.params.scaling_options.emax,
                    )
                    for Esq_outlier_indices, scaler in zip(
                        free_Esq_outlier_indices,
                        self.active_scalers,
                    ):
                        scaler.outliers.set_selected(
                            flumpy.from_numpy(Esq_outlier_indices), True
                        )
        logger.debug("Finished outlier rejection.")
        log_memory_usage()

    @Subject.notify_event(event="performed_error_analysis")
    def perform_error_optimisation(self, update_Ih=True):
        minimisation_groups = extract_error_model_groups(
            self.params.weighting.error_model, len(self.active_scalers)
        )
        for g in minimisation_groups:
            scalers = [self.active_scalers[i] for i in g]
            error_model = scalers[0]._experiment.scaling_model.error_model
            if not error_model.params.minimisation:
                continue
            tables = [
                s._get_valid_ih_table(
                    sub_selection=~s.outliers, anomalous=True, include_partiality=True
                )
                if hasattr(s, "_get_valid_ih_table")
                else s.get_valid_reflections().select(~s.outliers)
                for s in scalers
            ]
            space_group = scalers[0].experiment.crystal.get_space_group()
            Ih_table = IhTableGPU(
                tables,
                space_group,
                anomalous=True,
                additional_cols=["partiality"],
            )
            if len(minimisation_groups) == 1:
                logger.info("Determining a combined error model for all datasets")
            else:
                logger.info(
                    f"Error model determination for sweep(s) {','.join(str(i + 1) for i in g)}"
                )
            try:
                model = run_error_model_refinement_gpu(
                    scalers[0]._experiment.scaling_model.error_model,
                    Ih_table,
                    min_partiality=self.params.reflection_selection.min_partiality,
                    use_stills_filtering=scalers[0].is_still,
                )
            except (ValueError, RuntimeError) as e:
                logger.info(e)
                logger.debug(e, exc_info=True)
            else:
                for s in scalers:
                    s._experiment.scaling_model.set_error_model(model)
        if update_Ih:
            if len(minimisation_groups) == 1:
                error_model = self.active_scalers[
                    0
                ]._experiment.scaling_model.error_model
                if not error_model.params.minimisation:
                    return
                self.global_Ih_table.update_weights(error_model)
                if self._free_Ih_table:
                    self._free_Ih_table.update_weights(error_model)
            else:
                for i, scaler in enumerate(self.active_scalers):
                    error_model = scaler._experiment.scaling_model.error_model
                    if not error_model.params.minimisation:
                        continue
                    self.global_Ih_table.update_weights(error_model, dataset_id=i)
                    if self._free_Ih_table:
                        self._free_Ih_table.update_weights(error_model, dataset_id=i)


class MultiScalerGPU(_MultiScalerGPUMixin, MultiScaler):
    """GPU-accelerated MultiScaler."""

    def combine_intensities(self):
        if self.params.reflection_selection.combine.joint_analysis:
            logger.info(
                "Performing multi-dataset profile/summation intensity optimisation."
            )
            try:
                combiner = MultiDatasetIntensityCombinerGPU(self)
            except DialsMergingStatisticsError as e:
                logger.info("Intensity combination failed with the error %s", e)
            else:
                for i, scaler in enumerate(self.active_scalers):
                    (
                        intensity,
                        variance,
                    ) = combiner.calculate_suitable_combined_intensities(i)
                    isel = scaler.suitable_refl_for_scaling_sel.iselection()
                    scaler.reflection_table["intensity"].set_selected(isel, intensity)
                    scaler.reflection_table["variance"].set_selected(isel, variance)
                    self.global_Ih_table.update_data_in_blocks(
                        intensity, i, column="intensity"
                    )
                    self.global_Ih_table.update_data_in_blocks(
                        variance, i, column="variance"
                    )
                    self.global_Ih_table.update_weights(
                        error_model=scaler._experiment.scaling_model.error_model,
                        dataset_id=i,
                    )
                    scaler.experiment.scaling_model.record_intensity_combination_Imid(
                        combiner.max_key
                    )
                    if self._free_Ih_table:
                        self._free_Ih_table.update_data_in_blocks(
                            intensity, i, column="intensity"
                        )
                        self._free_Ih_table.update_data_in_blocks(
                            variance, i, column="variance"
                        )
                        self._free_Ih_table.update_weights(
                            error_model=scaler._experiment.scaling_model.error_model,
                            dataset_id=i,
                        )
                for block in self.global_Ih_table.blocked_data_list:
                    calc_Ih_gpu(block, sync_to_cpu=True)
                if self._free_Ih_table:
                    for block in self._free_Ih_table.blocked_data_list:
                        calc_Ih_gpu(block, sync_to_cpu=True)
        else:
            super().combine_intensities()


class TargetScalerGPU(_MultiScalerGPUMixin, TargetScaler):
    """GPU-accelerated TargetScaler."""

    def update_for_minimisation_gpu(
        self, apm, block_id: int, calc_derivs: bool = True, sync_to_cpu: bool = False
    ):
        self._update_for_minimisation_gpu(
            apm,
            block_id,
            calc_Ih=False,
            calc_derivs=calc_derivs,
            sync_to_cpu=sync_to_cpu,
        )

    @Subject.notify_event(event="performed_scaling")
    def perform_scaling(self, engine=None, max_iterations=None, tolerance=None):
        for scaler in self._target_active_scalers:
            self._active_scalers = [scaler]
            self._create_Ih_table()
            self._update_model_data()
            self._perform_scaling(
                target_type=ScalingTargetFixedIHGPU,
                engine=engine,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
        self._active_scalers = self._target_active_scalers


def _choose_initial_scaling_intensities_fast(
    reflection_table, intensity_choice="profile"
):
    """Vectorized 1D NumPy equivalent of dials.algorithms.scaling.scaling_library.choose_initial_scaling_intensities
    that avoids full-table .select() clones."""
    n = reflection_table.size()
    choice = "prf" if intensity_choice == "profile" else intensity_choice
    flags_np = flumpy.to_numpy(reflection_table["flags"])
    prf_flag = (flags_np & int(reflection_table.flags.integrated_prf)) != 0
    if "intensity.prf.value" not in reflection_table:
        choice = "sum"
    elif choice == "prf" and not prf_flag.any():
        logger.warning(
            "No profile fitted reflections in this dataset, using summation intensities"
        )
        choice = "sum"

    conv_np = np.ones(n, dtype=np.float64)
    if "lp" in reflection_table:
        conv_np = conv_np * flumpy.to_numpy(reflection_table["lp"])
    qe_np = None
    if "qe" in reflection_table:
        qe_np = flumpy.to_numpy(reflection_table["qe"])
    elif "dqe" in reflection_table:
        qe_np = flumpy.to_numpy(reflection_table["dqe"])
    if qe_np is not None:
        nz_qe = qe_np > 0.0
        inv_qe = np.ones(n, dtype=np.float64)
        inv_qe[nz_qe] = 1.0 / qe_np[nz_qe]
        conv_np = conv_np * inv_qe
    reflection_table["prescaling_correction"] = flumpy.from_numpy(conv_np)

    if choice == "prf":
        i_np = flumpy.to_numpy(reflection_table["intensity.prf.value"]) * conv_np
        v_np = (
            flumpy.to_numpy(reflection_table["intensity.prf.variance"])
            * conv_np
            * conv_np
        )
    else:
        if "partiality" in reflection_table:
            part_np = flumpy.to_numpy(reflection_table["partiality"])
            nz_p = part_np > 0.0
            inv_p = np.ones(n, dtype=np.float64)
            inv_p[nz_p] = 1.0 / part_np[nz_p]
            ci = conv_np * inv_p
            i_np = (
                flumpy.to_numpy(reflection_table["intensity.sum.value"])
                * conv_np
                * inv_p
            )
            v_np = flumpy.to_numpy(reflection_table["intensity.sum.variance"]) * (
                ci * ci
            )
            if "partiality.inv.variance" in reflection_table:
                sc = (
                    flumpy.to_numpy(reflection_table["intensity.sum.value"]) * conv_np
                )
                v_np = v_np + (sc * sc) * flumpy.to_numpy(
                    reflection_table["partiality.inv.variance"]
                )
        else:
            i_np = flumpy.to_numpy(reflection_table["intensity.sum.value"]) * conv_np
            v_np = (
                flumpy.to_numpy(reflection_table["intensity.sum.variance"])
                * conv_np
                * conv_np
            )
        if choice == "combine":
            i_prf = flumpy.to_numpy(reflection_table["intensity.prf.value"]) * conv_np
            v_prf = (
                flumpy.to_numpy(reflection_table["intensity.prf.variance"])
                * conv_np
                * conv_np
            )
            i_np = np.where(prf_flag, i_prf, i_np)
            v_np = np.where(prf_flag, v_prf, v_np)

    reflection_table["intensity"] = flumpy.from_numpy(
        np.ascontiguousarray(i_np, dtype=np.float64)
    )
    reflection_table["variance"] = flumpy.from_numpy(
        np.ascontiguousarray(v_np, dtype=np.float64)
    )
    var_mask = flumpy.from_numpy(np.ascontiguousarray(v_np <= 0.0))
    reflection_table.set_flags(var_mask, reflection_table.flags.excluded_for_scaling)
    return reflection_table


class SingleScalerFactoryGPU(SingleScalerFactory):
    """GPU-accelerated SingleScalerFactory."""

    @staticmethod
    def filter_bad_reflections(
        reflections, partiality_cutoff=0.4, min_isigi=-5.0, intensity_choice="combine"
    ):
        """Vectorized 1D NumPy implementation of ScalerFactory.filter_bad_reflections
        that avoids 6 full-table .select()/.del_selected() copies."""
        logger.info(
            "Applying filter of min_isigi > %s, partiality > %s",
            min_isigi,
            partiality_cutoff,
        )
        n = reflections.size()
        if n == 0:
            raise ValueError("All data has been filtered from the reflection table")
        if intensity_choice not in ("combine", "profile", "sum"):
            return ScalerFactory.filter_bad_reflections(
                reflections,
                partiality_cutoff=partiality_cutoff,
                min_isigi=min_isigi,
                intensity_choice=intensity_choice,
            )

        if intensity_choice == "combine":
            if "intensity.sum.value" not in reflections:
                if "intensity.prf.value" not in reflections:
                    return reflections
                mode = "prf"
            elif "intensity.prf.value" not in reflections:
                mode = "sum"
            else:
                mode = "sum_or_prf"
        elif intensity_choice == "profile":
            if "intensity.prf.value" not in reflections:
                raise ValueError("Cannot filter intensity kind 'profile'; missing column")
            mode = "prf"
        else:
            if "intensity.sum.value" not in reflections:
                raise ValueError("Cannot filter intensity kind 'sum'; missing column")
            mode = "sum"

        reflections["original_index"] = flumpy.from_numpy(
            np.arange(n, dtype=np.uint64)
        )
        keep = flumpy.to_numpy(reflections["id"]) >= 0
        if not keep.any():
            raise ValueError("All data has been filtered from the reflection table")

        flags_np = flumpy.to_numpy(reflections["flags"])
        prf_flag = (flags_np & int(reflections.flags.integrated_prf)) != 0
        sum_flag = (flags_np & int(reflections.flags.integrated_sum)) != 0

        if mode == "sum_or_prf":
            if not (keep & prf_flag).any():
                mode = "sum"
            else:
                keep = keep & (prf_flag | sum_flag)
        if mode == "prf":
            keep = keep & prf_flag
        elif mode == "sum":
            keep = keep & sum_flag
        if not keep.any():
            raise ValueError("All data has been filtered from the reflection table")

        v_sum = None
        v_prf = None
        if mode in ("sum", "sum_or_prf"):
            v_sum = flumpy.to_numpy(reflections["intensity.sum.variance"])
            keep = keep & ~((v_sum <= 0.0) & sum_flag)
            if not keep.any():
                raise ValueError("All data has been filtered from the reflection table")
        if mode in ("prf", "sum_or_prf"):
            v_prf = flumpy.to_numpy(reflections["intensity.prf.variance"])
            keep = keep & ~((v_prf <= 0.0) & prf_flag)
            if not keep.any():
                raise ValueError("All data has been filtered from the reflection table")

        qe_np = None
        if "qe" in reflections:
            qe_np = flumpy.to_numpy(reflections["qe"])
            keep = keep & (qe_np > 0.0)
        elif "dqe" in reflections:
            qe_np = flumpy.to_numpy(reflections["dqe"])
            keep = keep & (qe_np > 0.0)
        if not keep.any():
            raise ValueError("All data has been filtered from the reflection table")

        part_np = None
        if "partiality" in reflections:
            part_np = flumpy.to_numpy(reflections["partiality"])
            keep = keep & (part_np > 0.0) & (part_np >= partiality_cutoff)
            if not keep.any():
                raise ValueError("All data has been filtered from the reflection table")
            reflections["fractioncalc"] = reflections["partiality"]

        if min_isigi:
            idx = np.flatnonzero(keep)
            conv = np.ones(idx.size, dtype=np.float64)
            if qe_np is not None:
                conv /= qe_np[idx]
            if "lp" in reflections:
                conv *= flumpy.to_numpy(reflections["lp"])[idx]
            with np.errstate(divide="ignore", invalid="ignore"):
                if mode in ("sum", "sum_or_prf"):
                    sum_conv = (conv / part_np[idx]) if part_np is not None else conv
                    i_s = (
                        flumpy.to_numpy(reflections["intensity.sum.value"])[idx]
                        * sum_conv
                    )
                    v_s = v_sum[idx] * (sum_conv * sum_conv)
                    bad_s = (i_s / np.sqrt(v_s)) < min_isigi
                    if bad_s.any():
                        keep[idx[bad_s]] = False
                        if not keep.any():
                            raise ValueError(
                                "All data has been filtered from the reflection table"
                            )
                        idx = idx[~bad_s]
                        if mode == "sum_or_prf":
                            conv = conv[~bad_s]
                if mode in ("prf", "sum_or_prf"):
                    i_p = (
                        flumpy.to_numpy(reflections["intensity.prf.value"])[idx]
                        * conv
                    )
                    v_p = v_prf[idx] * (conv * conv)
                    bad_p = (i_p / np.sqrt(v_p)) < min_isigi
                    if bad_p.any():
                        keep[idx[bad_p]] = False
                        if not keep.any():
                            raise ValueError(
                                "All data has been filtered from the reflection table"
                            )

        mask = flumpy.from_numpy(np.ascontiguousarray(~keep))
        reflections.set_flags(mask, reflections.flags.excluded_for_scaling)
        return reflections

    @classmethod
    def create(cls, params, experiment, reflection_table, for_multi=False):
        cls.ensure_experiment_identifier(experiment, reflection_table)
        logger.info(
            "The scaling model type being applied is %s. \n",
            experiment.scaling_model.id_,
        )
        try:
            reflection_table = cls.filter_bad_reflections(
                reflection_table,
                partiality_cutoff=params.cut_data.partiality_cutoff,
                min_isigi=params.cut_data.min_isigi,
                intensity_choice=params.reflection_selection.intensity_choice,
            )
        except ValueError:
            raise BadDatasetForScalingException

        reflection_table = sum_partial_reflections(reflection_table)

        if "inverse_scale_factor" not in reflection_table:
            reflection_table["inverse_scale_factor"] = flex.double(
                reflection_table.size(), 1.0
            )
        elif (
            reflection_table["inverse_scale_factor"].count(0.0)
            == reflection_table.size()
        ):
            reflection_table["inverse_scale_factor"] = flex.double(
                reflection_table.size(), 1.0
            )
        reflection_table = _choose_initial_scaling_intensities_fast(
            reflection_table, params.reflection_selection.intensity_choice
        )

        excluded_for_scaling = reflection_table.get_flags(
            reflection_table.flags.excluded_for_scaling
        )
        user_excluded = reflection_table.get_flags(
            reflection_table.flags.user_excluded_in_scaling
        )
        reasons = Reasons()
        reasons.add_reason("user excluded", user_excluded.count(True))
        reasons.add_reason("excluded for scaling", excluded_for_scaling.count(True))
        n_excluded = (excluded_for_scaling | user_excluded).count(True)
        if n_excluded == reflection_table.size():
            logger.info("All reflections were determined to be unsuitable for scaling.")
            logger.info(reasons)
            raise BadDatasetForScalingException(
                "Unable to use this dataset for scaling"
            )
        else:
            logger.info(
                "Excluding %s/%s reflections\n%s",
                n_excluded,
                reflection_table.size(),
                reasons,
            )

        if params.reflection_selection.method == "intensity_ranges":
            try:
                reflection_table = quasi_normalisation(reflection_table, experiment)
            except AssertionError:
                raise BadDatasetForScalingException(
                    "Unable to use this dataset for scaling with the option reflection_selection.method=intensity_ranges"
                )
        if (
            params.reflection_selection.method in (None, Auto, "auto", "quasi_random")
        ) or (
            experiment.scaling_model.id_ == "physical"
            and "absorption" in experiment.scaling_model.components
        ):
            if experiment.scan and (experiment.scan.get_oscillation()[1] != 0.0):
                reflection_table = calc_crystal_frame_vectors_gpu(
                    reflection_table,
                    experiment,
                    align_axis=(1.0, 0.0, 0.0),
                )
        try:
            scaler = SingleScalerGPU(params, experiment, reflection_table, for_multi)
        except BadDatasetForScalingException as e:
            raise ValueError(e)
        else:
            return scaler


class MultiScalerFactoryGPU(MultiScalerFactory):
    """GPU-accelerated MultiScalerFactory."""

    @staticmethod
    def create(params, experiments, reflections):
        single_scalers = []
        idx_to_remove = []
        for i, (expt, refl) in enumerate(zip(experiments, reflections)):
            try:
                scaler = SingleScalerFactoryGPU.create(
                    params, expt, refl, for_multi=True
                )
            except BadDatasetForScalingException as e:
                logger.info(e)
                idx_to_remove.append(i)
            else:
                single_scalers.append(scaler)
        ms = MultiScalerGPU(single_scalers)
        if idx_to_remove:
            for i in idx_to_remove:
                ms.removed_datasets.append(experiments[i].identifier)
            logger.info(
                "Removed experiments %s", " ".join(str(i) for i in idx_to_remove)
            )
        return ms

    @staticmethod
    def create_from_targetscaler(targetscaler):
        single_scalers = []
        for scaler in targetscaler.unscaled_scalers:
            single_scalers.append(scaler)
        single_scalers.extend(targetscaler.single_scalers)
        multiscaler = MultiScalerGPU(single_scalers)
        multiscaler.observers = targetscaler.observers
        return multiscaler


class TargetScalerFactoryGPU(TargetScalerFactory):
    """GPU-accelerated TargetScalerFactory."""

    @staticmethod
    def create_for_target_against_reference(params, experiments, reflections):
        unscaled_scalers = []
        idx_to_remove = []
        for i, (expt, refl) in enumerate(zip(experiments[:-1], reflections[:-1])):
            try:
                scaler = SingleScalerFactoryGPU.create(
                    params, expt, refl, for_multi=True
                )
            except BadDatasetForScalingException as e:
                logger.info(e)
                idx_to_remove.append(i)
            else:
                unscaled_scalers.append(scaler)
        scaled_scalers = [
            NullScalerFactory.create(params, experiments[-1], reflections[-1])
        ]
        ts = TargetScalerGPU(scaled_scalers, unscaled_scalers)
        if idx_to_remove:
            for i in idx_to_remove:
                ts.removed_datasets.append(experiments[i].identifier)
            logger.info(
                "Removed experiments %s", " ".join(str(i) for i in idx_to_remove)
            )
        return ts

    @staticmethod
    def create(params, experiments, reflections):
        scaled_experiments = []
        scaled_scalers = []
        unscaled_scalers = []
        idx_to_remove = []

        for i, (expt, refl) in enumerate(zip(experiments, reflections)):
            try:
                scaler = SingleScalerFactoryGPU.create(
                    params, expt, refl, for_multi=True
                )
            except BadDatasetForScalingException as e:
                logger.info(e)
                idx_to_remove.append(i)
            else:
                if expt.scaling_model.is_scaled:
                    scaled_scalers.append(scaler)
                    scaled_experiments.append(expt)
                else:
                    unscaled_scalers.append(scaler)
        if idx_to_remove:
            for j in idx_to_remove[::-1]:
                del experiments[j]
                del reflections[j]
            logger.info(
                "Removed experiments %s", " ".join(str(i) for i in idx_to_remove)
            )

        return TargetScalerGPU(scaled_scalers, unscaled_scalers)


def create_scaler_gpu(params, experiments, reflections):
    """GPU-accelerated drop-in replacement for create_scaler."""
    if not reflections:
        raise ValueError("No reflection tables provided as input")
    if len(reflections) == 1:
        return SingleScalerFactoryGPU.create(params, experiments[0], reflections[0])
    is_scaled_list = [expt.scaling_model.is_scaled for expt in experiments]
    if params.scaling_options.reference:
        return TargetScalerFactoryGPU.create_for_target_against_reference(
            params, experiments, reflections
        )
    elif params.scaling_options.target_cycle and len(set(is_scaled_list)) == 2:
        return TargetScalerFactoryGPU.create(params, experiments, reflections)
    return MultiScalerFactoryGPU.create(params, experiments, reflections)


def targeted_scaling_algorithm_gpu(scaler):
    """GPU-accelerated targeted_scaling_algorithm."""
    scaler.perform_scaling()
    if scaler.params.scaling_options.only_target:
        scaler.expand_scales_to_all_reflections(calc_cov=True)
        return scaler

    scaler.expand_scales_to_all_reflections()
    scaler.round_of_outlier_rejection()

    if scaler.params.scaling_options.use_free_set:
        scaler.params.scaling_options.use_free_set = False
        logger.warning("Cannot use free set if targeted scaling - disabling option.")
    scaler = MultiScalerFactoryGPU.create_from_targetscaler(scaler)
    return scaling_algorithm(scaler)


_SCALING_MODEL_CLASS_CACHE: dict[str, type] = {}


def create_scaling_model_gpu(params, experiments, reflections):
    """Fast cached version of create_scaling_model avoiding repeated importlib.metadata scans."""
    import importlib.metadata
    from dials.algorithms.scaling.model.model import (
        ArrayScalingModel,
        DoseDecay,
        KBScalingModel,
        PhysicalScalingModel,
    )

    if not _SCALING_MODEL_CLASS_CACHE:
        _SCALING_MODEL_CLASS_CACHE.update(
            {
                "physical": PhysicalScalingModel,
                "KB": KBScalingModel,
                "array": ArrayScalingModel,
                "dose_decay": DoseDecay,
            }
        )

    autos = (None, Auto, "auto", "Auto")
    use_auto_model = params.model in autos
    model_class = None
    if not use_auto_model:
        model_class = _SCALING_MODEL_CLASS_CACHE.get(params.model)
        if model_class is None:
            for entry_point in importlib.metadata.entry_points(
                group="dxtbx.scaling_model_ext"
            ):
                if entry_point.name == params.model:
                    model_class = entry_point.load()
                    _SCALING_MODEL_CLASS_CACHE[params.model] = model_class
                    break
        if not model_class:
            raise ValueError(f"Unable to create scaling model of type {params.model}")

    for expt, refl in zip(experiments, reflections):
        if not expt.scaling_model or params.overwrite_existing_models:
            if use_auto_model:
                if not expt.scan or (expt.scan.get_oscillation()[1] == 0.0):
                    model = KBScalingModel
                else:
                    osc_range = expt.scan.get_oscillation_range()
                    abs_osc_range = abs(osc_range[1] - osc_range[0])
                    if abs_osc_range < 1.0:
                        model = KBScalingModel
                    else:
                        model = PhysicalScalingModel
            else:
                model = model_class
            expt.scaling_model = model.from_data(params, expt, refl)
        else:
            expt.scaling_model.update(params)
    return experiments


class ScalingAlgorithmGPU(ScalingAlgorithm):
    """GPU-accelerated ScalingAlgorithm."""

    def _create_model_and_scaler(self):
        """Create the scaling models and GPU scaler."""
        self.experiments = create_scaling_model_gpu(
            self.params, self.experiments, self.reflections
        )
        logger.info("\nScaling models have been initialised for all experiments.")
        logger.info("%s%s%s", "\n", "=" * 80, "\n")
        if len(self.experiments) == 1 and self.experiments[0].scaling_model.id_ == "KB":
            raise RuntimeError("Invalid model option (KB) for scaling a single dataset")

        self.experiments = set_image_ranges_in_scaling_models(self.experiments)

        self.scaler = create_scaler_gpu(self.params, self.experiments, self.reflections)

    @property
    def scaled_miller_array(self):
        if getattr(self, "_needs_lazy_scaled_miller_array", False):
            self._needs_lazy_scaled_miller_array = False
            self._scaled_miller_array = scaled_data_as_miller_array(
                self.reflections,
                self.experiments,
                anomalous_flag=False,
                best_unit_cell=self.params.reflection_selection.best_unit_cell,
            )
        return getattr(self, "_scaled_miller_array", None)

    @scaled_miller_array.setter
    def scaled_miller_array(self, value):
        self._needs_lazy_scaled_miller_array = False
        self._scaled_miller_array = value

    @property
    def merging_statistics_result(self):
        if getattr(self, "_needs_lazy_merging_stats", False):
            self._needs_lazy_merging_stats = False
            try:
                self.calculate_merging_stats()
            except DialsMergingStatisticsError as e:
                logger.info(e)
        return getattr(self, "_merging_statistics_result", None)

    @merging_statistics_result.setter
    def merging_statistics_result(self, value):
        self._needs_lazy_merging_stats = False
        self._merging_statistics_result = value

    @property
    def anom_merging_statistics_result(self):
        if getattr(self, "_needs_lazy_merging_stats", False):
            self._needs_lazy_merging_stats = False
            try:
                self.calculate_merging_stats()
            except DialsMergingStatisticsError as e:
                logger.info(e)
        return getattr(self, "_anom_merging_statistics_result", None)

    @anom_merging_statistics_result.setter
    def anom_merging_statistics_result(self, value):
        self._needs_lazy_merging_stats = False
        self._anom_merging_statistics_result = value

    def run(self):
        """Run the scaling script."""
        if (
            self.params.output.html
            or self.params.output.json
            or getattr(self.params.output, "additional_stats", False)
        ):
            return super().run()

        start_time = time.time()
        self.scale()
        self.remove_bad_data()
        if not self.experiments:
            raise ValueError("All data sets have been rejected as bad.")
        for table in self.reflections:
            bad = table.get_flags(table.flags.bad_for_scaling, all=False)
            table.unset_flags(flex.bool(table.size(), True), table.flags.scaled)
            table.set_flags(~bad, table.flags.scaled)
        self._needs_lazy_scaled_miller_array = True
        self._needs_lazy_merging_stats = True
        logger.debug("\nTotal time taken: %.4fs ", time.time() - start_time)

    def scale(self):
        """The main scaling algorithm on GPU."""
        if self.scaler.id_ == "target":
            self.scaler.perform_scaling()
            if (
                self.params.scaling_options.only_target
                or self.params.scaling_options.reference
            ):
                self.scaler = targeted_scaling_algorithm(self.scaler)
                return
            self.scaler.expand_scales_to_all_reflections()
            self.scaler = MultiScalerFactoryGPU.create_from_targetscaler(self.scaler)

        self.scaler = scaling_algorithm(self.scaler)


class ScaleAndFilterAlgorithmGPU(ScaleAndFilterAlgorithm, ScalingAlgorithmGPU):
    """GPU-accelerated ScaleAndFilterAlgorithm."""

    pass
