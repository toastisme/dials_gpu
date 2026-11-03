"""
GPU-accelerated symmetry group reductions and IhTableBlock operations using PyTorch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from cctbx import crystal, miller, sgtbx, uctbx
from dxtbx import flumpy
from scipy.sparse import csc_matrix
from scitbx import sparse

from dials.algorithms.scaling.Ih_table import (
    IhTable,
    IhTableBlock,
    get_sorted_asu_indices,
    map_indices_to_asu,
)
from dials.array_family import flex

from .components import _get_device, _get_minimization_device


def _np_to_tensor(
    arr: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Convert a NumPy array (even if read-only from pandas) to a PyTorch tensor without warnings."""
    a = np.ascontiguousarray(arr)
    if not a.flags.writeable:
        a = a.copy()
    return torch.from_numpy(a).to(device=device, dtype=dtype)


def _pack_hkl(hkl_np: np.ndarray) -> np.ndarray:
    """Pack an (N, 3) int32 Miller index array into a 1D int64 key array."""
    h = hkl_np[:, 0].astype(np.int64) + 32768
    k = hkl_np[:, 1].astype(np.int64) + 32768
    l = hkl_np[:, 2].astype(np.int64) + 32768
    return (h << 32) | (k << 16) | l


class IhTableBlockGPU(IhTableBlock):
    """Fast IhTableBlock that stores 1D int64 `_group_ids` directly and avoids
    constructing or slicing `scitbx.sparse.matrix` and `scipy.sparse.csc_matrix`
    unless explicitly accessed."""

    def __init__(
        self, n_groups: int, n_refl: int, n_datasets: int = 1, _fast_init: bool = False
    ):
        self.Ih_table = pd.DataFrame()
        self.block_selections = [None] * n_datasets
        self._n_groups = int(n_groups)
        self._n_refl_expected = int(n_refl)
        self._group_ids = np.empty(0, dtype=np.int64)
        self._setup_info = {"next_row": 0, "next_dataset": 0, "setup_complete": False}
        self.dataset_info = {}
        self.n_datasets = n_datasets
        self.derivatives = None
        self.binner = None
        self._h_index_matrix = None
        self._h_expand_matrix = None
        self._csc_h_index_matrix_cache = None
        self._csc_h_expand_matrix_cache = None
        self._hkl_data = flex.miller_index([]) if not _fast_init else None
        self._hkl_lazy = None

    def _invalidate_sparse_caches(self) -> None:
        self._h_index_matrix = None
        self._h_expand_matrix = None
        self._csc_h_index_matrix_cache = None
        self._csc_h_expand_matrix_cache = None
        self._gpu_group_ids_cache = None
        self._gpu_fast_static = None

    @property
    def _hkl(self) -> flex.miller_index:
        if self._hkl_lazy is not None:
            parent_hkl, sel_bool = self._hkl_lazy
            self._hkl_data = parent_hkl.select(
                flumpy.from_numpy(np.ascontiguousarray(sel_bool))
            )
            self._hkl_lazy = None
        elif self._hkl_data is None:
            self._hkl_data = flex.miller_index([])
        return self._hkl_data

    @_hkl.setter
    def _hkl(self, val: flex.miller_index) -> None:
        self._hkl_lazy = None
        self._hkl_data = val

    @property
    def asu_miller_index(self) -> flex.miller_index:
        return self._hkl

    @property
    def n_groups(self) -> int:
        if self._csc_h_index_matrix_cache is not None:
            return self._csc_h_index_matrix_cache.shape[1]
        return self._n_groups

    @property
    def _csc_h_index_matrix(self) -> csc_matrix:
        if self._csc_h_index_matrix_cache is None:
            n_r = int(self._group_ids.size)
            n_g = int(self._n_groups)
            if n_r == 0 or n_g == 0:
                self._csc_h_index_matrix_cache = csc_matrix(
                    (n_r, n_g), dtype=np.float64
                )
            else:
                rows = np.arange(n_r, dtype=np.int64)
                data = np.ones(n_r, dtype=np.float64)
                self._csc_h_index_matrix_cache = csc_matrix(
                    (data, (rows, self._group_ids)), shape=(n_r, n_g)
                )
        return self._csc_h_index_matrix_cache

    @_csc_h_index_matrix.setter
    def _csc_h_index_matrix(self, val) -> None:
        self._csc_h_index_matrix_cache = val
        if val is not None:
            self._n_groups = int(val.shape[1])

    @property
    def _csc_h_expand_matrix(self) -> csc_matrix:
        if self._csc_h_expand_matrix_cache is None:
            self._csc_h_expand_matrix_cache = self._csc_h_index_matrix.transpose()
        return self._csc_h_expand_matrix_cache

    @_csc_h_expand_matrix.setter
    def _csc_h_expand_matrix(self, val) -> None:
        self._csc_h_expand_matrix_cache = val

    @property
    def h_index_matrix(self) -> sparse.matrix:
        if self._h_index_matrix is None:
            n_r = int(self._group_ids.size)
            n_g = int(self._n_groups)
            mat = sparse.matrix(n_r, n_g)
            for rowidx, gid in enumerate(self._group_ids):
                mat[rowidx, int(gid)] = 1.0
            mat.compact()
            self._h_index_matrix = mat
        return self._h_index_matrix

    @h_index_matrix.setter
    def h_index_matrix(self, val) -> None:
        self._h_index_matrix = val

    @property
    def h_expand_matrix(self) -> sparse.matrix:
        if self._h_expand_matrix is None:
            self._h_expand_matrix = self.h_index_matrix.transpose()
        return self._h_expand_matrix

    @h_expand_matrix.setter
    def h_expand_matrix(self, val) -> None:
        self._h_expand_matrix = val

    def add_data(
        self,
        dataset_id: int,
        group_ids: np.ndarray,
        reflections: pd.DataFrame,
        hkl: flex.miller_index,
    ) -> None:
        assert not self._setup_info["setup_complete"]
        assert self._setup_info["next_row"] + len(group_ids) <= self._n_refl_expected
        assert dataset_id == self._setup_info["next_dataset"]

        g_i64 = np.ascontiguousarray(group_ids, dtype=np.int64)
        if self._group_ids.size == 0:
            self._group_ids = g_i64
        else:
            self._group_ids = np.concatenate([self._group_ids, g_i64])

        if self._setup_info["next_dataset"] == 0 and self.n_datasets == 1:
            self._hkl = hkl
            if isinstance(reflections.index, pd.RangeIndex) and reflections.index.start == 0 and reflections.index.step == 1:
                self.Ih_table = reflections
            else:
                self.Ih_table = reflections.reset_index(drop=True)
        else:
            self._hkl.extend(hkl)
            self.Ih_table = pd.concat([self.Ih_table, reflections], ignore_index=True)

        self.dataset_info[dataset_id] = {"start_index": self._setup_info["next_row"]}
        self._setup_info["next_row"] += len(group_ids)
        self._setup_info["next_dataset"] += 1
        self.dataset_info[dataset_id]["end_index"] = self._setup_info["next_row"]

        if "loc_indices" in reflections:
            self.block_selections[dataset_id] = reflections["loc_indices"].to_numpy()
        else:
            self.block_selections[dataset_id] = np.arange(
                reflections.shape[0], dtype=np.uint64
            )
        if self._setup_info["next_dataset"] == len(self.block_selections):
            self._complete_setup()

    def _complete_setup(self) -> None:
        assert self._setup_info["next_row"] == self._n_refl_expected
        self._invalidate_sparse_caches()
        self.Ih_table["weights"] = 1.0 / self.Ih_table["variance"].to_numpy()
        self._setup_info["setup_complete"] = True

    def sum_in_groups(
        self, array: csc_matrix | np.ndarray, output: str = "per_group"
    ) -> np.ndarray:
        if isinstance(array, np.ndarray) and array.ndim == 1:
            if self._n_groups == 0:
                per_group = np.empty(0, dtype=np.float64)
            else:
                per_group = np.bincount(
                    self._group_ids, weights=array, minlength=self._n_groups
                ).astype(np.float64, copy=False)
            if output == "per_group":
                return per_group
            elif output == "per_refl":
                return per_group[self._group_ids]
            else:
                raise ValueError(
                    f"Bad value for output= parameter (value={output}, allowed values: per_group, per_refl)"
                )
        return super().sum_in_groups(array, output=output)

    def group_multiplicities(self, output: str = "per_group") -> np.ndarray:
        if self._n_groups == 0:
            counts = np.empty(0, dtype=np.float64)
        else:
            counts = np.bincount(self._group_ids, minlength=self._n_groups).astype(
                np.float64
            )
        if output == "per_group":
            return counts
        elif output == "per_refl":
            return counts[self._group_ids]
        raise ValueError(
            f"Bad value for output= parameter (value={output}, allowed values: per_group, per_refl)"
        )

    def calc_nh(self) -> np.ndarray:
        return self.group_multiplicities(output="per_refl")

    def calc_Ih(self) -> None:
        if self.size == 0:
            self.Ih_table["Ih_values"] = np.empty(0, dtype=np.float64)
            return
        if self.size < 20000:
            g = self.inverse_scale_factors
            w = self.weights
            I = self.intensities
            sumgsq = np.bincount(
                self._group_ids, weights=(g * g) * w, minlength=self._n_groups
            )
            sumgI = np.bincount(
                self._group_ids, weights=(g * I) * w, minlength=self._n_groups
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                Ih = sumgI / sumgsq
            self.Ih_table["Ih_values"] = Ih[self._group_ids]
            self._gpu_scales = None
            self._gpu_Ih_values = None
            self._gpu_sumgsq = None
        else:
            calc_Ih_gpu(self, sync_to_cpu=True)

    def select(self, sel: np.ndarray) -> IhTableBlockGPU:
        sel_arr = flumpy.to_numpy(sel) if not isinstance(sel, np.ndarray) else sel
        if sel_arr.dtype != bool:
            mask = np.zeros(self.size, dtype=bool)
            mask[sel_arr] = True
            sel_bool = mask
        else:
            sel_bool = sel_arr

        Ih_table = self.Ih_table[sel_bool].reset_index(drop=True)
        sel_group_ids = self._group_ids[sel_bool]
        if sel_group_ids.size > 0 and self._n_groups > 0:
            present = np.zeros(self._n_groups, dtype=bool)
            present[sel_group_ids] = True
            new_n_groups = int(np.count_nonzero(present))
            remap = np.cumsum(present, dtype=np.int64) - 1
            new_group_ids = remap[sel_group_ids]
        else:
            new_n_groups = 0
            new_group_ids = np.empty(0, dtype=np.int64)

        newtable = IhTableBlockGPU(
            n_groups=new_n_groups,
            n_refl=int(Ih_table.shape[0]),
            n_datasets=self.n_datasets,
            _fast_init=True,
        )
        newtable.Ih_table = Ih_table
        newtable._group_ids = new_group_ids
        newtable._hkl_lazy = (self.asu_miller_index, sel_bool)
        newtable._setup_info["setup_complete"] = True
        newtable.block_selections = []
        offset = 0
        for i in range(newtable.n_datasets):
            newtable.dataset_info[i] = {"start_index": offset}
            block_sel_i = self.block_selections[i]
            n_in_dataset_i = len(block_sel_i)
            newtable.block_selections.append(
                block_sel_i[sel_bool[offset : offset + n_in_dataset_i]]
            )
            offset += n_in_dataset_i
            newtable.dataset_info[i]["end_index"] = offset
        return newtable

    def select_on_groups(self, sel: np.ndarray) -> IhTableBlockGPU:
        sel_arr = flumpy.to_numpy(sel) if not isinstance(sel, np.ndarray) else sel
        if sel_arr.dtype == bool:
            group_mask = sel_arr
        else:
            group_mask = np.zeros(self._n_groups, dtype=bool)
            if sel_arr.size > 0:
                group_mask[sel_arr] = True
        nz_row_sel = group_mask[self._group_ids]
        return self.select(nz_row_sel)

    def match_Ih_values_to_target(self, target_Ih_table: IhTable) -> None:
        super().match_Ih_values_to_target(target_Ih_table)
        if self._csc_h_index_matrix_cache is not None:
            csr = self._csc_h_index_matrix_cache.tocsr()
            self._group_ids = csr.indices.astype(np.int64, copy=True)
            self._n_groups = int(self._csc_h_index_matrix_cache.shape[1])

    def setup_binner(
        self,
        unit_cell: uctbx.unit_cell,
        space_group: sgtbx.space_group,
        n_resolution_bins: int,
    ) -> None:
        miller_set = miller.set(
            crystal_symmetry=crystal.symmetry(
                unit_cell=unit_cell,
                space_group=space_group,
                assert_is_compatible_unit_cell=False,
            ),
            indices=self.asu_miller_index,
            anomalous_flag=False,
        )
        g = self.inverse_scale_factors
        data_np = np.ascontiguousarray(self.intensities / g, dtype=np.float64)
        sigmas_np = np.ascontiguousarray(np.sqrt(self.variances) / g, dtype=np.float64)
        ma = miller.array(miller_set, data=flumpy.from_numpy(data_np))
        ma.set_observation_type_xray_intensity()
        ma.set_sigmas(flumpy.from_numpy(sigmas_np))

        d_star_sq = ma.d_star_sq().data()
        d_star_sq_min = flex.min(d_star_sq)
        d_star_sq_max = flex.max(d_star_sq)
        span = d_star_sq_max - d_star_sq_min
        relative_tolerance = 1e-6
        d_star_sq_max += span * relative_tolerance
        d_star_sq_min -= span * relative_tolerance
        step = max((d_star_sq_max - d_star_sq_min) / n_resolution_bins, 0.004)

        self.binner = ma.setup_binner_d_star_sq_step(
            auto_binning=False,
            d_max=uctbx.d_star_sq_as_d(d_star_sq_max),
            d_min=uctbx.d_star_sq_as_d(d_star_sq_min),
            d_star_sq_step=step,
        )


class IhTableGPU(IhTable):
    """Accelerated IhTable that vectorizes ASU Miller index deduplication, block
    splitting, and group_id assignment using NumPy/PyTorch and IhTableBlockGPU."""

    def _create_empty_Ih_table_blocks(self) -> None:
        for n in range(self.n_work_blocks):
            n_refl_in_block = self.properties_dict["n_reflections_in_each_block"][n]
            n_groups_in_block = self.properties_dict["n_unique_in_each_block"][n]
            self.Ih_table_blocks.append(
                IhTableBlockGPU(
                    n_groups=n_groups_in_block,
                    n_refl=n_refl_in_block,
                    n_datasets=self.n_datasets,
                )
            )

    def _determine_required_block_structures(
        self,
        reflection_tables: list[flex.reflection_table],
        free_set_percentage: float = 0,
        free_set_offset: int = 0,
    ) -> None:
        self._dataset_sort_cache = {}
        if free_set_percentage:
            self._sorted_unique_keys = None
            return super()._determine_required_block_structures(
                reflection_tables, free_set_percentage, free_set_offset
            )

        dataset_keys_list = []
        dataset_hkl_list = []
        total_rows = 0
        for ds_id, table in enumerate(reflection_tables):
            if "asu_miller_index" not in table:
                table["asu_miller_index"] = map_indices_to_asu(
                    table["miller_index"], self.space_group, self.anomalous
                )
            asu_mi = table["asu_miller_index"]
            n_ds = asu_mi.size()
            total_rows += n_ds
            if n_ds > 0:
                hkl_np = flumpy.to_numpy(asu_mi)
                ds_keys = _pack_hkl(hkl_np)
                perm_np = np.argsort(ds_keys, kind="stable")
                sorted_ds_keys = ds_keys[perm_np]
                sorted_ds_hkl = hkl_np[perm_np]
            else:
                perm_np = np.empty(0, dtype=np.int64)
                sorted_ds_keys = np.empty(0, dtype=np.int64)
                sorted_ds_hkl = np.empty((0, 3), dtype=np.int32)
            self._dataset_sort_cache[ds_id] = (perm_np, sorted_ds_keys)
            dataset_keys_list.append(sorted_ds_keys)
            dataset_hkl_list.append(sorted_ds_hkl)

        if total_rows == 0:
            raise ValueError("No data found in input file(s)")

        if len(reflection_tables) == 1:
            keys = dataset_keys_list[0]
            hkl_all = dataset_hkl_list[0]
        else:
            all_keys = np.concatenate(dataset_keys_list)
            all_hkl = np.concatenate(dataset_hkl_list, axis=0)
            joint_order = np.argsort(all_keys, kind="stable")
            keys = all_keys[joint_order]
            hkl_all = all_hkl[joint_order]

        is_new = np.empty(len(keys), dtype=bool)
        is_new[0] = True
        is_new[1:] = keys[1:] != keys[:-1]
        first_idx = np.flatnonzero(is_new)
        unique_keys = keys[first_idx]
        n_unique_groups = int(first_idx.size)
        self.n_work_blocks = min(self.n_work_blocks, n_unique_groups)

        if len(reflection_tables) == 1 and self.n_work_blocks == 1:
            self.properties_dict["n_unique_in_each_block"].append(n_unique_groups)
            self.properties_dict["n_reflections_in_each_block"][0] = len(keys)
            self.properties_dict["miller_index_boundaries"].append((10000, 10000, 10000))
            self.properties_dict["n_unique_in_each_block"].append(0)
            self.properties_dict["miller_index_boundaries"].append((10000, 10000, 10000))
            self.properties_dict["n_reflections_in_each_block"][1] = 0
            self._single_ds_group_ids = np.cumsum(is_new, dtype=np.int64) - 1
            self._sorted_unique_keys = unique_keys
            return

        self._single_ds_group_ids = None
        group_boundaries = [
            int(i * n_unique_groups / self.n_work_blocks)
            for i in range(self.n_work_blocks)
        ]
        group_boundaries.append(n_unique_groups)

        group_id_for_unique = np.empty(n_unique_groups, dtype=np.uint64)
        block_id_for_unique = np.empty(n_unique_groups, dtype=np.int64)
        for b in range(self.n_work_blocks):
            g0 = group_boundaries[b]
            g1 = group_boundaries[b + 1]
            n_grp = g1 - g0
            self.properties_dict["n_unique_in_each_block"].append(n_grp)
            group_id_for_unique[g0:g1] = np.arange(n_grp, dtype=np.uint64)
            block_id_for_unique[g0:g1] = b
            r0 = int(first_idx[g0])
            r1 = int(first_idx[g1]) if g1 < n_unique_groups else len(keys)
            self.properties_dict["n_reflections_in_each_block"][b] = r1 - r0
            if b + 1 < self.n_work_blocks:
                b_hkl = hkl_all[first_idx[g1]]
                self.properties_dict["miller_index_boundaries"].append(
                    (int(b_hkl[0]), int(b_hkl[1]), int(b_hkl[2]))
                )
            else:
                self.properties_dict["miller_index_boundaries"].append(
                    (10000, 10000, 10000)
                )

        self.properties_dict["n_unique_in_each_block"].append(0)
        self.properties_dict["miller_index_boundaries"].append((10000, 10000, 10000))
        self.properties_dict["n_reflections_in_each_block"][self.n_work_blocks] = 0

        # unique_keys is already sorted in ascending order because `keys` is sorted!
        self._sorted_unique_keys = unique_keys
        self._sorted_group_ids = group_id_for_unique
        self._sorted_block_ids = block_id_for_unique

    def _add_dataset_to_blocks(
        self,
        dataset_id: int,
        reflections: flex.reflection_table,
        indices_array: flex.size_t | None = None,
        additional_cols: list[str] | None = None,
    ) -> None:
        if getattr(self, "_sorted_unique_keys", None) is None:
            return super()._add_dataset_to_blocks(
                dataset_id, reflections, indices_array, additional_cols
            )

        hkl = reflections["asu_miller_index"]
        n_rows = reflections.size()
        cached = self._dataset_sort_cache.get(dataset_id)
        if cached is not None:
            perm_np, ds_keys = cached
            perm = flumpy.from_numpy(np.ascontiguousarray(perm_np, dtype=np.uint64))
        else:
            sorted_asu_indices, perm = get_sorted_asu_indices(
                hkl, self.space_group, self.anomalous
            )
            perm_np = flumpy.to_numpy(perm)
            ds_keys = (
                _pack_hkl(flumpy.to_numpy(sorted_asu_indices))
                if n_rows > 0
                else np.empty(0, dtype=np.int64)
            )

        data_dict = {
            "intensity": flumpy.to_numpy(reflections["intensity"])[perm_np],
            "variance": flumpy.to_numpy(reflections["variance"])[perm_np],
            "inverse_scale_factor": flumpy.to_numpy(
                reflections["inverse_scale_factor"]
            )[perm_np],
        }
        if isinstance(additional_cols, list):
            for col in additional_cols:
                if col in reflections:
                    data_dict[col] = flumpy.to_numpy(reflections[col])[perm_np]
        if indices_array:
            data_dict["loc_indices"] = flumpy.to_numpy(indices_array)[perm_np]
        else:
            data_dict["loc_indices"] = perm_np.astype(np.uint64, copy=True)
        data_dict["dataset_id"] = np.full(n_rows, dataset_id, dtype=np.uint64)
        df = pd.DataFrame(data_dict, copy=False)
        hkl = hkl.select(perm)

        if n_rows == 0:
            group_ids = np.zeros(0, dtype=np.int64)
            boundaries_for_this_datset = [0] * (self.n_work_blocks + 1)
        elif getattr(self, "_single_ds_group_ids", None) is not None:
            group_ids = self._single_ds_group_ids
            boundaries_for_this_datset = [0, n_rows]
        else:
            pos = np.searchsorted(self._sorted_unique_keys, ds_keys)
            group_ids = self._sorted_group_ids[pos]
            if self.n_work_blocks == 1:
                boundaries_for_this_datset = [0, n_rows]
            else:
                block_ids = self._sorted_block_ids[pos]
                splits = np.searchsorted(
                    block_ids, np.arange(1, self.n_work_blocks, dtype=np.int64)
                )
                boundaries_for_this_datset = (
                    [0] + [int(x) for x in splits] + [n_rows]
                )

        if self.n_work_blocks == 1:
            self.Ih_table_blocks[0].add_data(dataset_id, group_ids, df, hkl)
        else:
            for i, val in enumerate(boundaries_for_this_datset[:-1]):
                start = val
                end = boundaries_for_this_datset[i + 1]
                self.Ih_table_blocks[i].add_data(
                    dataset_id, group_ids[start:end], df[start:end], hkl[start:end]
                )

    def as_miller_array(
        self, unit_cell: uctbx.unit_cell, return_free_set_data: bool = False
    ) -> miller.array:
        blocked_data_list = self.blocked_data_list
        if self.free_Ih_table:
            if return_free_set_data:
                blocked_data_list = [blocked_data_list[-1]]
            else:
                blocked_data_list = blocked_data_list[:-1]
        if len(blocked_data_list) == 1:
            block = blocked_data_list[0]
            miller_set = miller.set(
                crystal_symmetry=crystal.symmetry(
                    unit_cell=unit_cell,
                    space_group=self.space_group,
                    assert_is_compatible_unit_cell=False,
                ),
                indices=block.asu_miller_index,
                anomalous_flag=False,
            )
            g = block.inverse_scale_factors
            data_np = np.ascontiguousarray(block.intensities / g, dtype=np.float64)
            sigmas_np = np.ascontiguousarray(
                np.sqrt(block.variances) / g, dtype=np.float64
            )
            i_obs = miller.array(miller_set, data=flumpy.from_numpy(data_np))
            i_obs.set_observation_type_xray_intensity()
            i_obs.set_sigmas(flumpy.from_numpy(sigmas_np))
            i_obs.set_info(
                miller.array_info(source="DIALS", source_type="reflection_tables")
            )
            return i_obs
        return super().as_miller_array(
            unit_cell, return_free_set_data=return_free_set_data
        )

    def update_data_in_blocks(self, data, dataset_id, column):
        for block in self.Ih_table_blocks:
            block._gpu_I_cache = None
            block._gpu_w_cache = None
        super().update_data_in_blocks(data, dataset_id, column)

    def update_weights(self, error_model=None, dataset_id=None):
        for block in self.Ih_table_blocks:
            block._gpu_w_cache = None
        super().update_weights(error_model=error_model, dataset_id=dataset_id)


def get_block_group_ids_gpu(
    block: IhTableBlock, device: torch.device | str | None = None
) -> torch.Tensor:
    """Return a cached 1D int64 tensor of length block.size mapping each reflection
    to its symmetry group index in [0, block.n_groups)."""
    dev = _get_device(device)
    gids_np = getattr(block, "_group_ids", None)
    if gids_np is not None:
        cache = getattr(block, "_gpu_group_ids_cache", None)
        if (
            cache is None
            or cache[0] is not gids_np
            or cache[1] != block.size
            or cache[2].device != dev
        ):
            if block.size == 0:
                group_ids = torch.empty(0, dtype=torch.int64, device=dev)
            else:
                group_ids = _np_to_tensor(gids_np, dev, dtype=torch.int64)
            block._gpu_group_ids_cache = (gids_np, block.size, group_ids)
        return block._gpu_group_ids_cache[2]

    csc_mat = block._csc_h_index_matrix
    cache = getattr(block, "_gpu_group_ids_cache", None)
    if (
        cache is None
        or cache[0] is not csc_mat
        or cache[1] != block.size
        or cache[2].device != dev
    ):
        if block.size == 0:
            group_ids = torch.empty(0, dtype=torch.int64, device=dev)
        else:
            csr = csc_mat.tocsr()
            group_ids = _np_to_tensor(csr.indices, dev, dtype=torch.int64)
        block._gpu_group_ids_cache = (csc_mat, block.size, group_ids)
    return block._gpu_group_ids_cache[2]


def get_block_Static_tensors_gpu(
    block: IhTableBlock, device: torch.device | str | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (group_ids_gpu, intensities_gpu, weights_gpu) for an IhTableBlock on GPU."""
    dev = _get_device(device)
    if getattr(block, "_gpu_in_minimisation", False):
        fast_cache = getattr(block, "_gpu_fast_static", None)
        if fast_cache is not None and fast_cache[0] == dev and fast_cache[1] == block.size:
            return fast_cache[2], fast_cache[3], fast_cache[4]

    group_ids_gpu = get_block_group_ids_gpu(block, dev)

    I_np = block.intensities
    w_np = block.weights
    # Check if cached tensors match current numpy buffers
    I_cache = getattr(block, "_gpu_I_cache", None)
    if (
        I_cache is None
        or I_cache[0] != block.size
        or I_cache[1] != (float(I_np[0]) if block.size > 0 else 0.0)
        or I_cache[2] != (float(I_np[-1]) if block.size > 0 else 0.0)
        or I_cache[3].device != dev
    ):
        I_t = _np_to_tensor(I_np, dev, dtype=torch.float64)
        block._gpu_I_cache = (
            block.size,
            float(I_np[0]) if block.size > 0 else 0.0,
            float(I_np[-1]) if block.size > 0 else 0.0,
            I_t,
        )
    else:
        I_t = I_cache[3]

    w_cache = getattr(block, "_gpu_w_cache", None)
    if (
        w_cache is None
        or w_cache[0] != block.size
        or w_cache[1] != (float(w_np[0]) if block.size > 0 else 0.0)
        or w_cache[2] != (float(w_np[-1]) if block.size > 0 else 0.0)
        or w_cache[3].device != dev
    ):
        w_t = _np_to_tensor(w_np, dev, dtype=torch.float64)
        block._gpu_w_cache = (
            block.size,
            float(w_np[0]) if block.size > 0 else 0.0,
            float(w_np[-1]) if block.size > 0 else 0.0,
            w_t,
        )
    else:
        w_t = w_cache[3]

    if getattr(block, "_gpu_in_minimisation", False):
        block._gpu_fast_static = (dev, block.size, group_ids_gpu, I_t, w_t)

    return group_ids_gpu, I_t, w_t


def sum_in_groups_gpu(
    values_gpu: torch.Tensor,
    n_groups: int,
    group_ids_gpu: torch.Tensor,
    output: str = "per_group",
) -> torch.Tensor:
    """Sum a 1D or 2D tensor over symmetry equivalent groups on the GPU."""
    if values_gpu.ndim == 1:
        group_sums = torch.zeros(
            n_groups, dtype=values_gpu.dtype, device=values_gpu.device
        )
        group_sums.index_add_(0, group_ids_gpu, values_gpu)
    else:
        group_sums = torch.zeros(
            (n_groups, values_gpu.shape[1]),
            dtype=values_gpu.dtype,
            device=values_gpu.device,
        )
        group_sums.index_add_(0, group_ids_gpu, values_gpu)

    if output == "per_group":
        return group_sums
    elif output == "per_refl":
        return group_sums[group_ids_gpu]
    else:
        raise ValueError(
            f"Bad value for output= parameter (value={output}, allowed values: per_group, per_refl)"
        )


def calc_Ih_gpu(
    block: IhTableBlock,
    scales_gpu: torch.Tensor | None = None,
    sync_to_cpu: bool = True,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate best estimate Ih for each reflection group on the GPU.

    Returns (Ih_per_refl_gpu, sumgsq_per_group_gpu, scales_gpu) and stores them
    on `block` for zero-copy reuse by `ScalingTargetGPU`.
    """
    if device is None and scales_gpu is not None:
        dev = scales_gpu.device
    else:
        dev = _get_minimization_device(block.size, device)
    group_ids_gpu, I_gpu, w_gpu = get_block_Static_tensors_gpu(block, dev)
    n_groups = block.n_groups

    if scales_gpu is None:
        scales_gpu = _np_to_tensor(block.inverse_scale_factors, dev, dtype=torch.float64)
    else:
        if scales_gpu.device != dev or scales_gpu.dtype != torch.float64:
            scales_gpu = scales_gpu.to(device=dev, dtype=torch.float64)
        if sync_to_cpu:
            block.Ih_table["inverse_scale_factor"] = scales_gpu.cpu().numpy()

    gsq = (scales_gpu * scales_gpu) * w_gpu
    sumgsq = sum_in_groups_gpu(gsq, n_groups, group_ids_gpu, "per_group")
    gI = (scales_gpu * I_gpu) * w_gpu
    sumgI = sum_in_groups_gpu(gI, n_groups, group_ids_gpu, "per_group")
    Ih_group = sumgI / sumgsq
    Ih_refl = Ih_group[group_ids_gpu]

    block._gpu_scales = scales_gpu
    block._gpu_Ih_values = Ih_refl
    block._gpu_sumgsq = sumgsq

    if sync_to_cpu:
        block.Ih_table["Ih_values"] = Ih_refl.cpu().numpy()

    return Ih_refl, sumgsq, scales_gpu
