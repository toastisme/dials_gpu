"""
GPU-accelerated symmetry group reductions and IhTableBlock operations using PyTorch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from dxtbx import flumpy

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


class IhTableGPU(IhTable):
    """Accelerated IhTable that vectorizes ASU Miller index deduplication, block
    splitting, and group_id assignment using NumPy/PyTorch instead of Python loops
    and OrderedSet."""

    def _determine_required_block_structures(
        self,
        reflection_tables: list[flex.reflection_table],
        free_set_percentage: float = 0,
        free_set_offset: int = 0,
    ) -> None:
        if free_set_percentage:
            self._sorted_unique_keys = None
            return super()._determine_required_block_structures(
                reflection_tables, free_set_percentage, free_set_offset
            )

        joint_asu_indices = flex.miller_index()
        for table in reflection_tables:
            if "asu_miller_index" not in table:
                table["asu_miller_index"] = map_indices_to_asu(
                    table["miller_index"], self.space_group, self.anomalous
                )
            joint_asu_indices.extend(table["asu_miller_index"])
        sorted_joint_asu_indices, _ = get_sorted_asu_indices(
            joint_asu_indices, self.space_group, self.anomalous
        )
        if not sorted_joint_asu_indices:
            raise ValueError("No data found in input file(s)")

        hkl_all = np.asarray(
            flumpy.to_numpy(sorted_joint_asu_indices.as_vec3_double()), dtype=np.int32
        ).reshape(-1, 3)
        keys = _pack_hkl(hkl_all)

        is_new = np.empty(len(keys), dtype=bool)
        is_new[0] = True
        is_new[1:] = keys[1:] != keys[:-1]
        first_idx = np.flatnonzero(is_new)
        unique_keys = keys[first_idx]
        n_unique_groups = int(first_idx.size)
        self.n_work_blocks = min(self.n_work_blocks, n_unique_groups)

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

        # Record the number in the free block
        self.properties_dict["n_unique_in_each_block"].append(0)
        self.properties_dict["miller_index_boundaries"].append((10000, 10000, 10000))
        self.properties_dict["n_reflections_in_each_block"][self.n_work_blocks] = 0

        order = np.argsort(unique_keys)
        self._sorted_unique_keys = unique_keys[order]
        self._sorted_group_ids = group_id_for_unique[order]
        self._sorted_block_ids = block_id_for_unique[order]

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

        sorted_asu_indices, perm = get_sorted_asu_indices(
            reflections["asu_miller_index"], self.space_group, self.anomalous
        )
        hkl = reflections["asu_miller_index"]
        n_rows = reflections.size()
        perm_np = flumpy.to_numpy(perm)

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
        df = pd.DataFrame(data_dict, index=perm_np)
        hkl = hkl.select(perm)

        if n_rows == 0:
            group_ids = np.zeros(0, dtype=np.uint64)
            boundaries_for_this_datset = [0] * (self.n_work_blocks + 1)
        else:
            ds_hkl = np.asarray(
                flumpy.to_numpy(sorted_asu_indices.as_vec3_double()), dtype=np.int32
            ).reshape(-1, 3)
            ds_keys = _pack_hkl(ds_hkl)
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
    elif (
        values_gpu.device.type == "cpu"
        and torch.cuda.is_available()
        and values_gpu.numel() >= 8192
    ):
        dev_cuda = torch.device("cuda", torch.cuda.current_device())
        gs_cuda = torch.zeros(
            (n_groups, values_gpu.shape[1]),
            dtype=values_gpu.dtype,
            device=dev_cuda,
        )
        gs_cuda.index_add_(
            0, group_ids_gpu.to(device=dev_cuda), values_gpu.to(device=dev_cuda)
        )
        group_sums = gs_cuda.cpu()
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
            block.Ih_table.loc[:, "inverse_scale_factor"] = scales_gpu.cpu().numpy()

    wg = w_gpu * scales_gpu
    sumgsq = sum_in_groups_gpu(wg * scales_gpu, n_groups, group_ids_gpu, "per_group")
    sumgI = sum_in_groups_gpu(wg * I_gpu, n_groups, group_ids_gpu, "per_group")
    Ih_group = sumgI / sumgsq
    Ih_refl = Ih_group[group_ids_gpu]

    block._gpu_scales = scales_gpu
    block._gpu_Ih_values = Ih_refl
    block._gpu_sumgsq = sumgsq

    if sync_to_cpu:
        block.Ih_table.loc[:, "Ih_values"] = Ih_refl.cpu().numpy()

    return Ih_refl, sumgsq, scales_gpu
