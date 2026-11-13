from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl
from dxtbx import flumpy

import dials_array_family_flex_ext
from dials.algorithms.shoebox import MaskCode
from dials.array_family import flex
from dials.model.data import PixelList
from dials_gpu.spot_finding.threshold import flex_to_tensor, get_default_device


@triton.jit
def _triton_hook_edges_kernel(
    u_ptr, v_ptr, labels_ptr, changed_ptr, E, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < E
    u = tl.load(u_ptr + offs, mask=m, other=0)
    v = tl.load(v_ptr + offs, mask=m, other=0)
    lu = tl.load(labels_ptr + u, mask=m, other=0)
    lv = tl.load(labels_ptr + v, mask=m, other=0)
    diff = m & (lu != lv)
    if tl.max(diff.to(tl.int32), axis=0) != 0:
        hi = tl.maximum(lu, lv)
        lo = tl.minimum(lu, lv)
        tl.atomic_min(labels_ptr + hi, lo, mask=diff)
        tl.atomic_min(labels_ptr + u, lo, mask=diff & (u > lo))
        tl.atomic_min(labels_ptr + v, lo, mask=diff & (v > lo))
        tl.store(changed_ptr, 1)


@triton.jit
def _triton_pointer_jump_kernel(labels_ptr, M, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < M
    p = tl.load(labels_ptr + offs, mask=m, other=0)
    pp = tl.load(labels_ptr + p, mask=m, other=0)
    ppp = tl.load(labels_ptr + pp, mask=m, other=0)
    tl.store(labels_ptr + offs, ppp, mask=m)


class PixelListLabellerGPU:
    """
    GPU-accelerated replacement for dials.model.data.PixelListLabeller.
    Accumulates strong pixels across frames for a single detector panel and
    performs 2D or 3D connected component labelling and spot extraction on the GPU.
    """

    def __init__(self, device: torch.device | None = None):
        self.device = device or get_default_device()
        self._size: tuple[int, int] = (0, 0)
        self._first_frame: int = 0
        self._last_frame: int = 0
        self._coords_list: list[torch.Tensor] = []
        self._values_list: list[torch.Tensor] = []

    def add_tensor(self, frame: int, image_t: torch.Tensor, mask_t: torch.Tensor):
        """
        Extract strong pixels directly from GPU tensors (H, W).
        """
        H, W = image_t.shape
        if self._last_frame == self._first_frame:
            self._first_frame = frame
            self._size = (H, W)
        else:
            assert frame == self._last_frame, f"Expected frame {self._last_frame}, got {frame}"
            assert self._size == (H, W), f"Inconsistent image size {self._size} vs {(H, W)}"
        self._last_frame = frame + 1

        yx = torch.nonzero(mask_t, as_tuple=False)  # (N, 2) in row-major order
        if yx.numel() > 0:
            vals = image_t[mask_t].to(torch.float64)
            z = torch.full((yx.shape[0], 1), frame, dtype=torch.int64, device=self.device)
            coords = torch.cat([z, yx.to(torch.int64)], dim=1)  # (N, 3): (z, y, x)
            self._coords_list.append(coords)
            self._values_list.append(vals)

    def add_volume(
        self,
        first_frame: int,
        last_frame: int,
        size: tuple[int, int],
        coords: torch.Tensor,
        values: torch.Tensor,
    ):
        """
        Add strong pixels for a contiguous 3D volume [first_frame, last_frame) at once.
        """
        H, W = size
        if self._last_frame == self._first_frame:
            self._first_frame = first_frame
            self._size = (H, W)
        else:
            assert first_frame == self._last_frame
            assert self._size == (H, W)
        self._last_frame = last_frame
        if coords.shape[0] > 0:
            self._coords_list.append(coords.to(self.device))
            self._values_list.append(values.to(self.device))

    def add(self, pixel_list):
        """
        Add a dials.model.data.PixelList or GPU PixelListResult.
        """
        if isinstance(pixel_list, GPUFramePixelList):
            frame = pixel_list.frame
            H, W = pixel_list.size
            if self._last_frame == self._first_frame:
                self._first_frame = frame
                self._size = (H, W)
            else:
                assert frame == self._last_frame
                assert self._size == (H, W)
            self._last_frame = frame + 1
            if pixel_list.coords.shape[0] > 0:
                self._coords_list.append(pixel_list.coords.to(self.device))
                self._values_list.append(pixel_list.values.to(self.device))
            return

        # Standard dials.model.data.PixelList
        frame = pixel_list.frame()
        size = tuple(pixel_list.size())
        H, W = size[0], size[1]
        if self._last_frame == self._first_frame:
            self._first_frame = frame
            self._size = (H, W)
        else:
            assert frame == self._last_frame
            assert self._size == (H, W)
        self._last_frame = frame + 1

        if len(pixel_list) > 0:
            idx_np = flumpy.to_numpy(pixel_list.index())
            val_np = flumpy.to_numpy(pixel_list.value())
            idx_t = torch.from_numpy(idx_np.astype(np.int64)).to(self.device)
            val_t = torch.from_numpy(val_np.astype(np.float64)).to(self.device)
            y = torch.div(idx_t, W, rounding_mode="floor")
            x = idx_t - y * W
            z = torch.full_like(y, frame)
            coords = torch.stack([z, y, x], dim=1)
            self._coords_list.append(coords)
            self._values_list.append(val_t)

    def size(self) -> tuple[int, int]:
        return self._size

    def num_pixels(self) -> int:
        return sum(c.shape[0] for c in self._coords_list)

    def first_frame(self) -> int:
        return self._first_frame

    def last_frame(self) -> int:
        return self._last_frame

    def frame_range(self) -> tuple[int, int]:
        return (self._first_frame, self._last_frame)

    def num_frames(self) -> int:
        return self._last_frame - self._first_frame

    def get_coords_and_values_gpu(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._coords_list:
            return (
                torch.empty((0, 3), dtype=torch.int64, device=self.device),
                torch.empty((0,), dtype=torch.float64, device=self.device),
            )
        if len(self._coords_list) == 1:
            return self._coords_list[0], self._values_list[0]
        coords = torch.cat(self._coords_list, dim=0)
        values = torch.cat(self._values_list, dim=0)
        self._coords_list = [coords]
        self._values_list = [values]
        return coords, values

    def coords(self):
        coords_t, _ = self.get_coords_and_values_gpu()
        if coords_t.shape[0] == 0:
            return flex.vec3_int()
        c_np = np.ascontiguousarray(coords_t.detach().cpu().numpy(), dtype=np.int32)
        return flumpy.vec_from_numpy(c_np)

    def values(self) -> flex.double:
        _, values_t = self.get_coords_and_values_gpu()
        if values_t.shape[0] == 0:
            return flex.double()
        v_np = np.ascontiguousarray(values_t.detach().cpu().numpy(), dtype=np.float64)
        return flumpy.from_numpy(v_np)

    def labels_3d(self) -> flex.int:
        _, _, labels_t, _ = self.labels_gpu(twod=False)
        if labels_t.shape[0] == 0:
            return flex.int()
        l_np = np.ascontiguousarray(labels_t.detach().cpu().numpy(), dtype=np.int32)
        return flumpy.from_numpy(l_np)

    def labels_2d(self) -> flex.int:
        _, _, labels_t, _ = self.labels_gpu(twod=True)
        if labels_t.shape[0] == 0:
            return flex.int()
        l_np = np.ascontiguousarray(labels_t.detach().cpu().numpy(), dtype=np.int32)
        return flumpy.from_numpy(l_np)

    def labels_gpu(self, twod: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Perform 2D (4-connected) or 3D (6-connected) connected components labelling on GPU.
        Returns (coords, values, labels, num_components) where labels are 0..num_components-1
        ordered by the first pixel appearance in lexicographical (z, y, x) order, matching
        boost::connected_components in dials/model/data/pixel_list.h.
        """
        coords, values = self.get_coords_and_values_gpu()
        M = coords.shape[0]
        if M == 0:
            return coords, values, torch.empty((0,), dtype=torch.int64, device=self.device), 0

        H, W = self._size
        z = coords[:, 0]
        y = coords[:, 1]
        x = coords[:, 2]

        # Linear index for fast neighbor search: ((z - first_frame) * H + y) * W + x
        lin_idx = ((z - self._first_frame) * H + y) * W + x

        # Ensure lexicographical sorting if needed
        if M > 1 and not bool(torch.all(lin_idx[:-1] < lin_idx[1:]).item()):
            perm = torch.argsort(lin_idx)
            coords = coords[perm]
            values = values[perm]
            lin_idx = lin_idx[perm]
            z = coords[:, 0]
            y = coords[:, 1]
            x = coords[:, 2]

        use_triton_cc = coords.is_cuda and M < (1 << 31)
        idx_dtype = torch.int32 if use_triton_cc else torch.int64
        edge_src = []
        edge_dst = []

        # 1. Right neighbor (z, y, x + 1): if present, must be at k + 1
        if M > 1:
            has_right = (x[:-1] < W - 1) & (lin_idx[1:] == lin_idx[:-1] + 1)
            u_x = torch.nonzero(has_right, as_tuple=False).squeeze(1).to(idx_dtype)
            del has_right
            if u_x.numel() > 0:
                edge_src.append(u_x)
                edge_dst.append(u_x + 1)
            else:
                del u_x

        # 2. Down neighbor (z, y + 1, x): target linear index is lin_idx + W
        valid_y = y < H - 1
        if bool(valid_y.any().item()):
            k_y = torch.nonzero(valid_y, as_tuple=False).squeeze(1)
            del valid_y
            target_y = lin_idx[k_y] + W
            pos_y = torch.searchsorted(lin_idx, target_y)
            pos_y_clamped = torch.clamp(pos_y, max=M - 1)
            found_y = (pos_y < M) & (lin_idx[pos_y_clamped] == target_y)
            del target_y, pos_y
            if bool(found_y.any().item()):
                edge_src.append(k_y[found_y].to(idx_dtype))
                edge_dst.append(pos_y_clamped[found_y].to(idx_dtype))
            del k_y, pos_y_clamped, found_y
        else:
            del valid_y

        # 3. Next frame neighbor (z + 1, y, x) if 3D
        if not twod:
            valid_z = z < self._last_frame - 1
            if bool(valid_z.any().item()):
                k_z = torch.nonzero(valid_z, as_tuple=False).squeeze(1)
                del valid_z
                target_z = lin_idx[k_z] + (H * W)
                pos_z = torch.searchsorted(lin_idx, target_z)
                pos_z_clamped = torch.clamp(pos_z, max=M - 1)
                found_z = (pos_z < M) & (lin_idx[pos_z_clamped] == target_z)
                del target_z, pos_z
                if bool(found_z.any().item()):
                    edge_src.append(k_z[found_z].to(idx_dtype))
                    edge_dst.append(pos_z_clamped[found_z].to(idx_dtype))
                del k_z, pos_z_clamped, found_z
            else:
                del valid_z

        del lin_idx
        if use_triton_cc:
            labels_i32 = torch.arange(M, dtype=torch.int32, device=self.device)
            if edge_src:
                u = torch.cat(edge_src, dim=0)
                v = torch.cat(edge_dst, dim=0)
                del edge_src, edge_dst
                E = int(u.numel())
                if E > 0:
                    changed = torch.zeros(1, dtype=torch.int32, device=self.device)
                    grid_e = ((E + 1023) // 1024,)
                    grid_m = ((M + 1023) // 1024,)
                    while True:
                        changed.zero_()
                        _triton_hook_edges_kernel[grid_e](
                            u, v, labels_i32, changed, E, BLOCK=1024
                        )
                        _triton_pointer_jump_kernel[grid_m](labels_i32, M, BLOCK=1024)
                        if int(changed.item()) == 0:
                            break
                    del changed
                del u, v
            unique_roots, comp_labels = torch.unique(
                labels_i32, sorted=True, return_inverse=True
            )
            num_components = int(unique_roots.numel())
            del labels_i32, unique_roots
            return coords, values, comp_labels, num_components

        labels = torch.arange(M, dtype=torch.int64, device=self.device)
        if edge_src:
            u = torch.cat(edge_src, dim=0)
            v = torch.cat(edge_dst, dim=0)
            del edge_src, edge_dst
            # Undirected edges
            u_all = torch.cat([u, v], dim=0)
            v_all = torch.cat([v, u], dim=0)
            del u, v

            while True:
                next_labels = labels.clone()
                lu = labels[u_all]
                lv = labels[v_all]
                next_labels.scatter_reduce_(
                    0, lu, lv, reduce="amin", include_self=True
                )
                next_labels.scatter_reduce_(
                    0, u_all, lv, reduce="amin", include_self=True
                )
                # Pointer jumping
                next_labels = next_labels[next_labels]
                next_labels = next_labels[next_labels]
                if torch.equal(next_labels, labels):
                    break
                labels = next_labels
            del u_all, v_all

        unique_roots, comp_labels = torch.unique(labels, sorted=True, return_inverse=True)
        num_components = int(unique_roots.numel())
        del labels, unique_roots
        return coords, values, comp_labels, num_components


class GPUFramePixelList:
    """
    Lightweight container holding strong pixel coordinates and values on GPU for one frame & panel.
    """

    def __init__(self, frame: int, size: tuple[int, int], coords: torch.Tensor, values: torch.Tensor):
        self.frame = frame
        self.size = size
        self.coords = coords
        self.values = values

    def __len__(self) -> int:
        return int(self.coords.shape[0])

    def to_cpu_pixel_list(self) -> PixelList:
        H, W = self.size
        if len(self) == 0:
            return PixelList(self.frame, self.size, flex.double(), flex.size_t())
        y = self.coords[:, 1]
        x = self.coords[:, 2]
        flat_idx = (y * W + x).detach().cpu().numpy().astype(np.uint64)
        vals = self.values.detach().cpu().numpy().astype(np.float64)
        return PixelList(
            self.frame,
            self.size,
            flumpy.from_numpy(np.ascontiguousarray(vals)),
            flumpy.from_numpy(np.ascontiguousarray(flat_idx)),
        )


def extract_spots_from_labeller_gpu(
    labeller: PixelListLabellerGPU,
    panel_id: int,
    zstart: int,
    twod: bool,
    min_pixels: int,
    max_pixels: int,
    find_hot_pixels: bool,
    compute_shoeboxes: bool = True,
) -> dict:
    """
    GPU replacement for PixelListShoeboxCreator + centroid_valid + summed_intensity.
    Computes bounding boxes, spot sizes, centroids, variances, summed intensities,
    peak coordinates, hot pixels, and optionally shoeboxes.
    """
    device = labeller.device
    coords, values, labels, num_spots = labeller.labels_gpu(twod=twod)
    H, W = labeller.size()
    first_frame, last_frame = labeller.frame_range()

    if num_spots == 0:
        return {
            "num_total_spots": 0,
            "num_below_min": 0,
            "num_above_max": 0,
            "num_valid_spots": 0,
            "panel": torch.empty((0,), dtype=torch.int64, device=device),
            "bbox": torch.empty((0, 6), dtype=torch.int64, device=device),
            "xyzobs_px_value": torch.empty((0, 3), dtype=torch.float64, device=device),
            "xyzobs_px_variance": torch.empty((0, 3), dtype=torch.float64, device=device),
            "intensity_sum_value": torch.empty((0,), dtype=torch.float64, device=device),
            "intensity_sum_variance": torch.empty((0,), dtype=torch.float64, device=device),
            "n_signal": torch.empty((0,), dtype=torch.int64, device=device),
            "peak_coords": torch.empty((0, 3), dtype=torch.float64, device=device),
            "hot_pixels": flex.size_t(),
            "shoeboxes": flex.shoebox() if compute_shoeboxes else None,
            "pixel_coords": coords,
            "pixel_values": values,
            "pixel_spot_idx": torch.empty((0,), dtype=torch.int64, device=device),
        }

    # Spot sizes
    spot_sizes = torch.bincount(labels, minlength=num_spots)
    below_min = spot_sizes < min_pixels
    above_max = spot_sizes > max_pixels
    valid_spot_mask = (~below_min) & (~above_max)

    num_below_min = int(below_min.sum().item())
    num_above_max = int(above_max.sum().item())
    num_valid_spots = int(valid_spot_mask.sum().item())

    # Hot pixel detection on GPU matching PixelListShoeboxCreator
    hot_pixels_flex = flex.size_t()
    if find_hot_pixels:
        num_frames = last_frame - first_frame
        if num_frames > 0 and coords.shape[0] > 0:
            flat_2d = coords[:, 1] * W + coords[:, 2]
            # Since coords are unique per (frame, y, x), a pixel is present in every
            # frame [first_frame, last_frame) iff its count across frames == num_frames
            counts_2d = torch.bincount(flat_2d, minlength=H * W)
            hot_idx = torch.nonzero(counts_2d == num_frames, as_tuple=False).squeeze(1)
            if hot_idx.numel() > 0:
                hot_np = np.ascontiguousarray(hot_idx.detach().cpu().numpy(), dtype=np.uint64)
                hot_pixels_flex = flumpy.from_numpy(hot_np)

    if num_valid_spots == 0:
        return {
            "num_total_spots": num_spots,
            "num_below_min": num_below_min,
            "num_above_max": num_above_max,
            "num_valid_spots": 0,
            "panel": torch.empty((0,), dtype=torch.int64, device=device),
            "bbox": torch.empty((0, 6), dtype=torch.int64, device=device),
            "xyzobs_px_value": torch.empty((0, 3), dtype=torch.float64, device=device),
            "xyzobs_px_variance": torch.empty((0, 3), dtype=torch.float64, device=device),
            "intensity_sum_value": torch.empty((0,), dtype=torch.float64, device=device),
            "intensity_sum_variance": torch.empty((0,), dtype=torch.float64, device=device),
            "n_signal": torch.empty((0,), dtype=torch.int64, device=device),
            "peak_coords": torch.empty((0, 3), dtype=torch.float64, device=device),
            "hot_pixels": hot_pixels_flex,
            "shoeboxes": flex.shoebox() if compute_shoeboxes else None,
            "pixel_coords": torch.empty((0, 3), dtype=torch.int64, device=device),
            "pixel_values": torch.empty((0,), dtype=torch.float64, device=device),
            "pixel_spot_idx": torch.empty((0,), dtype=torch.int64, device=device),
        }

    # Map valid spots to 0 .. K-1
    K = num_valid_spots
    new_spot_id = torch.full((num_spots,), -1, dtype=torch.int64, device=device)
    new_spot_id[valid_spot_mask] = torch.arange(K, dtype=torch.int64, device=device)

    pixel_spot_id = new_spot_id[labels]
    del labels, new_spot_id
    valid_pix_mask = pixel_spot_id >= 0

    v_coords = coords[valid_pix_mask]
    v_values = values[valid_pix_mask]
    v_labels = pixel_spot_id[valid_pix_mask]
    del coords, values, pixel_spot_id, valid_pix_mask
    labeller._coords_list.clear()
    labeller._values_list.clear()

    vz = v_coords[:, 0]
    vy = v_coords[:, 1]
    vx = v_coords[:, 2]

    # Bounding boxes [x0, x1, y0, y1, z0, z1]
    x0 = torch.full((K,), W, dtype=torch.int64, device=device).scatter_reduce_(
        0, v_labels, vx, reduce="amin", include_self=False
    )
    x1 = torch.zeros((K,), dtype=torch.int64, device=device).scatter_reduce_(
        0, v_labels, vx + 1, reduce="amax", include_self=False
    )
    y0 = torch.full((K,), H, dtype=torch.int64, device=device).scatter_reduce_(
        0, v_labels, vy, reduce="amin", include_self=False
    )
    y1 = torch.zeros((K,), dtype=torch.int64, device=device).scatter_reduce_(
        0, v_labels, vy + 1, reduce="amax", include_self=False
    )
    z0_rel = torch.full((K,), last_frame, dtype=torch.int64, device=device).scatter_reduce_(
        0, v_labels, vz, reduce="amin", include_self=False
    )
    z1_rel = torch.full((K,), first_frame, dtype=torch.int64, device=device).scatter_reduce_(
        0, v_labels, vz + 1, reduce="amax", include_self=False
    )
    z0 = z0_rel + zstart
    z1 = z1_rel + zstart
    bbox = torch.stack([x0, x1, y0, y1, z0, z1], dim=1)

    n_signal = spot_sizes[valid_spot_mask]

    # Match DIALS Shoebox<float> and CentroidPoints<float, vec3<double>> precision:
    # 1. sum_pixels_ and sum_pixels_sq_ are accumulated in float32 via scitbx::af::sum / sum_sq,
    #    then cast to double (value_type).
    # 2. Coordinates passed to CentroidPoints are relative to bbox (x - x0 + 0.5, y - y0 + 0.5, z - z0 + 0.5),
    #    and bbox_offset (x0, y0, z0) is added to cent.px.value afterwards.
    w_f32 = v_values.to(torch.float32)
    w = w_f32.to(torch.float64)

    rx = (vx - x0[v_labels]).to(torch.float64) + 0.5
    ry = (vy - y0[v_labels]).to(torch.float64) + 0.5
    rz = ((vz + zstart) - z0[v_labels]).to(torch.float64) + 0.5

    sum_w_f64 = torch.zeros((K,), dtype=torch.float64, device=device).scatter_add_(0, v_labels, w)
    sum_w_sq_f64 = torch.zeros((K,), dtype=torch.float64, device=device).scatter_add_(
        0, v_labels, (w_f32 * w_f32).to(torch.float64)
    )
    sum_w = sum_w_f64.to(torch.float32).to(torch.float64)
    sum_w_sq = sum_w_sq_f64.to(torch.float32).to(torch.float64)

    # For spots where float32 sequential accumulation can round differently from float64
    # (sum_sq >= 2^23 or non-integer pixel weights), match scitbx::af::sum / sum_sq sequential float32 order:
    has_non_int = not bool(torch.all(w_f32 == torch.floor(w_f32)).item())
    needs_seq = (sum_w_sq_f64 >= 8388608.0) | (sum_w_f64 >= 8388608.0)
    if K <= 50000 and (has_non_int or bool(torch.any(needs_seq).item())):
        if has_non_int:
            sel_pix = torch.ones_like(v_labels, dtype=torch.bool)
            spot_ids = torch.arange(K, dtype=torch.int64, device=device)
        else:
            sel_pix = needs_seq[v_labels]
            spot_ids = torch.nonzero(needs_seq, as_tuple=False).squeeze(1)
        sub_labels = v_labels[sel_pix].detach().cpu().numpy()
        sub_w = w_f32[sel_pix].detach().cpu().numpy()
        sub_w2 = (w_f32[sel_pix] * w_f32[sel_pix]).detach().cpu().numpy()
        order = np.argsort(sub_labels, kind="stable")
        sub_labels = sub_labels[order]
        sub_w = sub_w[order]
        sub_w2 = sub_w2[order]
        uniq_k, idx_start, counts_k = np.unique(sub_labels, return_index=True, return_counts=True)
        sw_fix = np.empty(len(uniq_k), dtype=np.float64)
        sw2_fix = np.empty(len(uniq_k), dtype=np.float64)
        for idx_u, (st, cnt) in enumerate(zip(idx_start, counts_k)):
            acc_w = np.float32(0.0)
            acc_w2 = np.float32(0.0)
            for j in range(st, st + cnt):
                acc_w = np.float32(acc_w + sub_w[j])
                acc_w2 = np.float32(acc_w2 + sub_w2[j])
            sw_fix[idx_u] = float(acc_w)
            sw2_fix[idx_u] = float(acc_w2)
        uniq_k_t = torch.from_numpy(uniq_k).to(device=device, dtype=torch.int64)
        sum_w[uniq_k_t] = torch.from_numpy(sw_fix).to(device=device, dtype=torch.float64)
        sum_w_sq[uniq_k_t] = torch.from_numpy(sw2_fix).to(device=device, dtype=torch.float64)

    sum_wx = torch.zeros((K,), dtype=torch.float64, device=device).scatter_add_(0, v_labels, w * rx)
    sum_wy = torch.zeros((K,), dtype=torch.float64, device=device).scatter_add_(0, v_labels, w * ry)
    sum_wz = torch.zeros((K,), dtype=torch.float64, device=device).scatter_add_(0, v_labels, w * rz)

    pos_w = sum_w > 0
    safe_sum_w = torch.where(pos_w, sum_w, torch.ones_like(sum_w))
    cx_rel = torch.where(pos_w, sum_wx / safe_sum_w, (x1 - x0).to(torch.float64) * 0.5)
    cy_rel = torch.where(pos_w, sum_wy / safe_sum_w, (y1 - y0).to(torch.float64) * 0.5)
    cz_rel = torch.where(pos_w, sum_wz / safe_sum_w, (z1 - z0).to(torch.float64) * 0.5)

    # Variance (unbiased_standard_error_sq + 1/12 in centroid_points.h):
    # unbiased_variance = sum_pixels_delta_sq * sum_pixels / (sum_pixels^2 - sum_pixels_sq)
    # unbiased_standard_error_sq = unbiased_variance / sum_pixels
    dx = rx - cx_rel[v_labels]
    dy = ry - cy_rel[v_labels]
    dz = rz - cz_rel[v_labels]
    sum_wdx2 = torch.zeros((K,), dtype=torch.float64, device=device).scatter_add_(0, v_labels, w * dx * dx)
    sum_wdy2 = torch.zeros((K,), dtype=torch.float64, device=device).scatter_add_(0, v_labels, w * dy * dy)
    sum_wdz2 = torch.zeros((K,), dtype=torch.float64, device=device).scatter_add_(0, v_labels, w * dz * dz)

    denom = sum_w * sum_w - sum_w_sq
    valid_var = pos_w & (denom > 0)
    safe_denom = torch.where(valid_var, denom, torch.ones_like(denom))
    var_x = torch.where(valid_var, (sum_wdx2 * sum_w / safe_denom) / safe_sum_w + 1.0 / 12.0, torch.full_like(sum_w, 1.0 / 12.0))
    var_y = torch.where(valid_var, (sum_wdy2 * sum_w / safe_denom) / safe_sum_w + 1.0 / 12.0, torch.full_like(sum_w, 1.0 / 12.0))
    var_z = torch.where(valid_var, (sum_wdz2 * sum_w / safe_denom) / safe_sum_w + 1.0 / 12.0, torch.full_like(sum_w, 1.0 / 12.0))
    xyzobs_px_variance = torch.stack([var_x, var_y, var_z], dim=1)

    # Add bbox_offset to relative centroid
    cx = cx_rel + x0.to(torch.float64)
    cy = cy_rel + y0.to(torch.float64)
    cz = cz_rel + z0.to(torch.float64)
    # Match Shoebox::centroid_valid: if zsize() == 1, cent.px.value[2] = bbox[4] + 0.5
    cz = torch.where(z1 - z0 == 1, z0.to(torch.float64) + 0.5, cz)
    xyzobs_px_value = torch.stack([cx, cy, cz], dim=1)

    px = vx.to(torch.float64) + 0.5
    py = vy.to(torch.float64) + 0.5
    pz = (vz + zstart).to(torch.float64) + 0.5

    # Summed intensity: Summation<float> accumulates in float32 in Shoebox::summed_intensity()
    # We compute in float64 from float32 values (which for integer photon counts <= 2^24 is exact)
    intensity_sum_value = sum_w
    intensity_sum_variance = sum_w

    # Peak coordinates for PeakCentroidDistanceFilter
    # Matches dials_array_family_flex_ext.so Shoebox::peak_coordinates()
    # which selects the maximum-valued pixel nearest the centroid (breaking
    # distance ties by earliest pixel in (z, y, x) order).
    max_w = torch.full((K,), -float("inf"), dtype=torch.float64, device=device).scatter_reduce_(
        0, v_labels, w, reduce="amax", include_self=False
    )
    is_max = w == max_w[v_labels]
    dx_c = px - cx[v_labels]
    dy_c = py - cy[v_labels]
    dz_c = pz - cz[v_labels]
    d2_c = dx_c * dx_c + dy_c * dy_c + dz_c * dz_c
    cand_d2 = torch.where(is_max, d2_c, torch.full_like(d2_c, float("inf")))
    min_d2 = torch.full((K,), float("inf"), dtype=torch.float64, device=device).scatter_reduce_(
        0, v_labels, cand_d2, reduce="amin", include_self=False
    )
    is_best = is_max & (d2_c == min_d2[v_labels])
    pix_idx = torch.arange(v_coords.shape[0], dtype=torch.int64, device=device)
    cand_pix_idx = torch.where(is_best, pix_idx, torch.full_like(pix_idx, v_coords.shape[0] + 1))
    best_pix_idx = torch.full((K,), v_coords.shape[0] + 1, dtype=torch.int64, device=device).scatter_reduce_(
        0, v_labels, cand_pix_idx, reduce="amin", include_self=False
    )
    peak_coords = torch.stack([px[best_pix_idx], py[best_pix_idx], pz[best_pix_idx]], dim=1)
    del px, py, pz, max_w, is_max, dx_c, dy_c, dz_c, d2_c, cand_d2, min_d2, is_best, pix_idx, cand_pix_idx, best_pix_idx

    panel_tensor = torch.full((K,), panel_id, dtype=torch.int64, device=device)

    return {
        "num_total_spots": num_spots,
        "num_below_min": num_below_min,
        "num_above_max": num_above_max,
        "num_valid_spots": K,
        "panel": panel_tensor,
        "bbox": bbox,
        "xyzobs_px_value": xyzobs_px_value,
        "xyzobs_px_variance": xyzobs_px_variance,
        "intensity_sum_value": intensity_sum_value,
        "intensity_sum_variance": intensity_sum_variance,
        "n_signal": n_signal,
        "peak_coords": peak_coords,
        "hot_pixels": hot_pixels_flex,
        "pixel_coords": v_coords,
        "pixel_values": w_f32,
        "pixel_spot_idx": v_labels,
        "zstart": zstart,
    }


def build_shoeboxes_from_gpu_spots(
    panel_t: torch.Tensor,
    bbox_t: torch.Tensor,
    pixel_coords: torch.Tensor,
    pixel_values_f32: torch.Tensor,
    pixel_spot_idx: torch.Tensor,
    zstart: int,
    selected_spot_mask: torch.Tensor | None = None,
) -> flex.shoebox:
    """
    Construct dials.array_family.flex.shoebox for the (optionally filtered) spots
    and populate their data, background, and mask arrays using ShoeboxExtractFromData.
    """
    device = bbox_t.device
    if selected_spot_mask is not None:
        keep_idx = torch.nonzero(selected_spot_mask, as_tuple=False).squeeze(1)
        K_out = int(keep_idx.numel())
        if K_out == 0:
            return flex.shoebox()
        remap = torch.full((bbox_t.shape[0],), -1, dtype=torch.int64, device=device)
        remap[keep_idx] = torch.arange(K_out, dtype=torch.int64, device=device)
        panel_t = panel_t[keep_idx]
        bbox_t = bbox_t[keep_idx]
        new_pix_spot = remap[pixel_spot_idx]
        valid_p = new_pix_spot >= 0
        pixel_coords = pixel_coords[valid_p]
        pixel_values_f32 = pixel_values_f32[valid_p]
        pixel_spot_idx = new_pix_spot[valid_p]
    else:
        K_out = int(bbox_t.shape[0])
        if K_out == 0:
            return flex.shoebox()

    # Create flex.shoebox(panel, bbox, allocate=True)
    panel_np = np.ascontiguousarray(panel_t.detach().cpu().numpy(), dtype=np.uint64)
    bbox_np = np.ascontiguousarray(bbox_t.detach().cpu().numpy(), dtype=np.int32)
    panel_flex = flumpy.from_numpy(panel_np)
    # flex.int6 can be constructed from flat int32 numpy array of shape (K_out, 6)
    bbox_flex = flex.int6(
        flumpy.from_numpy(np.ascontiguousarray(bbox_np[:, 0])),
        flumpy.from_numpy(np.ascontiguousarray(bbox_np[:, 1])),
        flumpy.from_numpy(np.ascontiguousarray(bbox_np[:, 2])),
        flumpy.from_numpy(np.ascontiguousarray(bbox_np[:, 3])),
        flumpy.from_numpy(np.ascontiguousarray(bbox_np[:, 4])),
        flumpy.from_numpy(np.ascontiguousarray(bbox_np[:, 5])),
    )
    shoeboxes = flex.shoebox(panel_flex, bbox_flex, allocate=True)

    # Compute sparse foreground offsets on GPU and populate flat buffers on CPU
    # (avoids allocating 9 * total_elems bytes of dense cuboid zeros in GPU VRAM,
    # which can exceed 4-8 GB when a near-zero RL threshold produces >1M noise spots).
    xsize = bbox_t[:, 1] - bbox_t[:, 0]
    ysize = bbox_t[:, 3] - bbox_t[:, 2]
    zsize = bbox_t[:, 5] - bbox_t[:, 4]
    sbox_sizes = xsize * ysize * zsize
    sbox_ends = torch.cumsum(sbox_sizes, dim=0)
    sbox_starts = sbox_ends - sbox_sizes
    total_elems = int(sbox_ends[-1].item())

    flat_data_np = np.zeros(total_elems, dtype=np.float32)
    flat_mask_np = np.zeros(total_elems, dtype=np.uint8)

    if pixel_coords.shape[0] > 0:
        kk = (pixel_coords[:, 0] + zstart) - bbox_t[pixel_spot_idx, 4]
        jj = pixel_coords[:, 1] - bbox_t[pixel_spot_idx, 2]
        ii = pixel_coords[:, 2] - bbox_t[pixel_spot_idx, 0]
        ys = ysize[pixel_spot_idx]
        xs = xsize[pixel_spot_idx]
        offsets_np = (
            (sbox_starts[pixel_spot_idx] + (kk * ys + jj) * xs + ii)
            .detach()
            .cpu()
            .numpy()
        )
        vals_np = pixel_values_f32.detach().cpu().numpy()
        flat_data_np[offsets_np] = vals_np
        flat_mask_np[offsets_np] = int(MaskCode.Valid | MaskCode.Foreground)

    data_flex = flumpy.from_numpy(flat_data_np)
    bg_flex = flex.float(total_elems, 0.0)
    mask_flex = flumpy.from_numpy(flat_mask_np)

    dials_array_family_flex_ext.ShoeboxExtractFromData(shoeboxes, data_flex, bg_flex, mask_flex)
    return shoeboxes
