from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from dxtbx import flumpy

from dials.array_family import flex


_TRITON_SCRATCH_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}
_TRITON_BATCH_SCRATCH_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}


def _get_triton_scratch(device: torch.device, H: int, W: int) -> dict[str, torch.Tensor]:
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    key = (device.type, dev_idx, H, W)
    bufs = _TRITON_SCRATCH_CACHE.get(key)
    if bufs is None:
        bufs = {
            "mh": torch.empty((H, W), dtype=torch.int32, device=device),
            "xh": torch.empty((H, W), dtype=torch.float64, device=device),
            "yh": torch.empty((H, W), dtype=torch.float64, device=device),
            "dst": torch.empty((H, W), dtype=torch.bool, device=device),
            "ev": torch.empty((H, W), dtype=torch.bool, device=device),
            "bg_mask": torch.empty((H, W), dtype=torch.bool, device=device),
        }
        _TRITON_SCRATCH_CACHE[key] = bufs
    return bufs


def _get_triton_batch_scratch(
    device: torch.device, B: int, H: int, W: int
) -> dict[str, torch.Tensor]:
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    key = (device.type, dev_idx, B, H, W)
    bufs = _TRITON_BATCH_SCRATCH_CACHE.get(key)
    if bufs is None:
        _TRITON_BATCH_SCRATCH_CACHE.clear()
        bufs = {
            "mh": torch.empty((B, H, W), dtype=torch.uint8, device=device),
            "xh": torch.empty((B, H, W), dtype=torch.float32, device=device),
            "yh": torch.empty((B, H, W), dtype=torch.float32, device=device),
            "dst": torch.empty((B, H, W), dtype=torch.bool, device=device),
            "ev": torch.empty((B, H, W), dtype=torch.bool, device=device),
            "bg_mask": torch.empty((B, H, W), dtype=torch.bool, device=device),
        }
        _TRITON_BATCH_SCRATCH_CACHE[key] = bufs
    return bufs


@triton.jit
def _batch_pass1_horiz(
    src_ptr, mask_ptr, mh_ptr, xh_ptr, yh_ptr, H, W, kx, BLOCK_W: tl.constexpr
):
    br = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    in_b = c < W
    row_off = br * W
    m = tl.zeros((BLOCK_W,), dtype=tl.int32)
    x = tl.zeros((BLOCK_W,), dtype=tl.float32)
    y = tl.zeros((BLOCK_W,), dtype=tl.float32)
    for dx in range(-kx, kx + 1):
        nc = c + dx
        v_c = (nc >= 0) & (nc < W)
        idx = row_off + nc
        mk = tl.load(mask_ptr + idx, mask=v_c, other=False).to(tl.int1)
        val = tl.load(src_ptr + idx, mask=v_c & mk, other=0).to(tl.float32)
        v = v_c & mk & (val < 16777216.0)
        vx = tl.where(v, val, 0.0)
        m += tl.where(v, 1, 0)
        x += vx
        y += vx * vx
    out_idx = row_off + c
    tl.store(mh_ptr + out_idx, m.to(tl.uint8), mask=in_b)
    tl.store(xh_ptr + out_idx, x, mask=in_b)
    tl.store(yh_ptr + out_idx, y, mask=in_b)


@triton.jit
def _batch_pass1_vert(
    src_ptr,
    mask_ptr,
    mh_ptr,
    xh_ptr,
    yh_ptr,
    dst_ptr,
    H,
    W,
    ky,
    min_count,
    nsig_b,
    nsig_s,
    global_threshold,
    gain_ptr,
    gain_val,
    mode: tl.constexpr,
    is_extended: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    br = tl.program_id(0)
    b = br // H
    r = br - b * H
    c = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    in_b = c < W
    base_b = b * H * W
    m_i = tl.zeros((BLOCK_W,), dtype=tl.int32)
    x = tl.zeros((BLOCK_W,), dtype=tl.float32)
    y = tl.zeros((BLOCK_W,), dtype=tl.float32)
    for dy in range(-ky, ky + 1):
        nr = r + dy
        v_r = (nr >= 0) & (nr < H) & in_b
        idx = base_b + nr * W + c
        m_i += tl.load(mh_ptr + idx, mask=v_r, other=0).to(tl.int32)
        x += tl.load(xh_ptr + idx, mask=v_r, other=0.0)
        y += tl.load(yh_ptr + idx, mask=v_r, other=0.0)
    out_idx = br * W + c
    mk = tl.load(mask_ptr + out_idx, mask=in_b, other=False).to(tl.int1)
    m = m_i.to(tl.float32)
    m1 = m - 1.0
    sqrt_2m1 = tl.sqrt(tl.maximum(2.0 * m1, 0.0))
    if mode == 0:
        a = m * y - x * x - x * m1
        c_val = x * nsig_b * sqrt_2m1
    elif mode == 1:
        a = m * y - x * x
        c_val = gain_val * x * (m1 + nsig_b * sqrt_2m1)
    else:
        g = tl.load(gain_ptr + r * W + c, mask=in_b, other=1.0).to(tl.float32)
        a = m * y - x * x
        c_val = g * x * (m1 + nsig_b * sqrt_2m1)
    if is_extended:
        disp_pass = (m_i >= min_count) & (x >= 0.0) & (a > c_val)
        out = tl.where(mk, disp_pass, True)
    else:
        src_val = tl.load(src_ptr + out_idx, mask=in_b, other=0).to(tl.float32)
        b_val = m * src_val - x
        if mode == 0:
            d = nsig_s * tl.sqrt(tl.maximum(x * m, 0.0))
        elif mode == 1:
            d = nsig_s * tl.sqrt(tl.maximum(gain_val * x * m, 0.0))
        else:
            d = nsig_s * tl.sqrt(tl.maximum(g * x * m, 0.0))
        out = (
            mk
            & (m_i >= min_count)
            & (x >= 0.0)
            & (src_val > global_threshold)
            & (a > c_val)
            & (b_val > d)
        )
    tl.store(dst_ptr + out_idx, out, mask=in_b)


@triton.jit
def _batch_erode_vert(dst_ptr, ev_ptr, H, W, r_erode, BLOCK_W: tl.constexpr):
    br = tl.program_id(0)
    b = br // H
    r = br - b * H
    c = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    in_b = c < W
    base_b = b * H * W
    acc = tl.full((BLOCK_W,), True, dtype=tl.int1)
    for dy in range(-r_erode, r_erode + 1):
        nr = r + dy
        v_r = (nr >= 0) & (nr < H) & in_b
        val = tl.load(dst_ptr + base_b + nr * W + c, mask=v_r, other=True).to(tl.int1)
        acc = acc & val
    tl.store(ev_ptr + br * W + c, acc, mask=in_b)


@triton.jit
def _batch_erode_horiz_bg(
    mask_ptr, ev_ptr, bg_mask_ptr, H, W, r_erode, BLOCK_W: tl.constexpr
):
    br = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    in_b = c < W
    row_off = br * W
    eroded = tl.full((BLOCK_W,), True, dtype=tl.int1)
    for dx in range(-r_erode, r_erode + 1):
        nc = c + dx
        v_c = (nc >= 0) & (nc < W)
        val = tl.load(ev_ptr + row_off + nc, mask=v_c, other=True).to(tl.int1)
        eroded = eroded & val
    mk = tl.load(mask_ptr + row_off + c, mask=in_b, other=False).to(tl.int1)
    bg = mk & (~eroded)
    tl.store(bg_mask_ptr + row_off + c, bg, mask=in_b)


@triton.jit
def _batch_pass2_sparse(
    src_ptr,
    mask_ptr,
    bg_mask_ptr,
    out_ptr,
    H,
    W,
    ky2,
    kx2,
    nsig_s,
    global_threshold,
    gain_ptr,
    gain_val,
    mode: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    br = tl.program_id(0)
    b = br // H
    r = br - b * H
    c = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    in_b = c < W
    out_idx = br * W + c
    mk = tl.load(mask_ptr + out_idx, mask=in_b, other=False).to(tl.int1)
    bg = tl.load(bg_mask_ptr + out_idx, mask=in_b, other=False).to(tl.int1)
    src_val = tl.load(src_ptr + out_idx, mask=in_b, other=0).to(tl.float32)
    cand = mk & (~bg) & (src_val > global_threshold)
    if tl.max(cand.to(tl.int32), axis=0) == 0:
        tl.store(out_ptr + out_idx, False, mask=in_b)
        return
    base_b = b * H * W
    m_i = tl.zeros((BLOCK_W,), dtype=tl.int32)
    x = tl.zeros((BLOCK_W,), dtype=tl.float32)
    for dy in range(-ky2, ky2 + 1):
        nr = r + dy
        v_r = (nr >= 0) & (nr < H)
        row_off = base_b + nr * W
        for dx in range(-kx2, kx2 + 1):
            nc = c + dx
            v_c = v_r & (nc >= 0) & (nc < W) & cand
            idx = row_off + nc
            bg_nb = tl.load(bg_mask_ptr + idx, mask=v_c, other=False).to(tl.int1)
            val_nb = tl.load(src_ptr + idx, mask=v_c & bg_nb, other=0).to(tl.float32)
            v = v_c & bg_nb & (val_nb < 16777216.0)
            m_i += tl.where(v, 1, 0)
            x += tl.where(v, val_nb, 0.0)
    m = m_i.to(tl.float32)
    mean = tl.where(m_i >= 2, x / tl.maximum(m, 1.0), 0.0)
    if mode == 0:
        local_thresh = mean + nsig_s * tl.sqrt(tl.maximum(mean, 0.0))
    elif mode == 1:
        local_thresh = mean + nsig_s * tl.sqrt(tl.maximum(gain_val * mean, 0.0))
    else:
        g = tl.load(gain_ptr + r * W + c, mask=in_b, other=1.0).to(tl.float32)
        local_thresh = mean + nsig_s * tl.sqrt(tl.maximum(g * mean, 0.0))
    final = cand & (m_i >= 0) & (x >= 0.0) & (src_val >= local_thresh)
    tl.store(out_ptr + out_idx, final, mask=in_b)


@triton.jit
def _triton_pass1_horiz(
    src_ptr,
    mask_ptr,
    mh_ptr,
    xh_ptr,
    yh_ptr,
    H,
    W,
    kx: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    r = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    in_b = c < W
    m = tl.zeros((BLOCK_W,), dtype=tl.int32)
    x = tl.zeros((BLOCK_W,), dtype=tl.float64)
    y = tl.zeros((BLOCK_W,), dtype=tl.float64)
    row_off = r * W
    for dx in tl.static_range(-kx, kx + 1):
        nc = c + dx
        v_c = (nc >= 0) & (nc < W)
        idx = row_off + nc
        mk = tl.load(mask_ptr + idx, mask=v_c, other=False).to(tl.int1)
        val = tl.load(src_ptr + idx, mask=v_c & mk, other=0).to(tl.float64)
        v = v_c & mk & (val < 16777216.0)
        vx = tl.where(v, val, 0.0)
        m += tl.where(v, 1, 0)
        x += vx
        y += vx * vx
    out_idx = row_off + c
    tl.store(mh_ptr + out_idx, m, mask=in_b)
    tl.store(xh_ptr + out_idx, x, mask=in_b)
    tl.store(yh_ptr + out_idx, y, mask=in_b)


@triton.jit
def _triton_pass1_vert(
    src_ptr,
    mask_ptr,
    mh_ptr,
    xh_ptr,
    yh_ptr,
    dst_ptr,
    H,
    W,
    ky: tl.constexpr,
    min_count,
    nsig_b,
    nsig_s,
    global_threshold,
    gain_ptr,
    gain_val,
    mode: tl.constexpr,
    is_extended: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    r = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    in_b = c < W
    m_i = tl.zeros((BLOCK_W,), dtype=tl.int32)
    x = tl.zeros((BLOCK_W,), dtype=tl.float64)
    y = tl.zeros((BLOCK_W,), dtype=tl.float64)
    for dy in tl.static_range(-ky, ky + 1):
        nr = r + dy
        v_r = (nr >= 0) & (nr < H) & in_b
        idx = nr * W + c
        m_i += tl.load(mh_ptr + idx, mask=v_r, other=0)
        x += tl.load(xh_ptr + idx, mask=v_r, other=0.0)
        y += tl.load(yh_ptr + idx, mask=v_r, other=0.0)
    out_idx = r * W + c
    mk = tl.load(mask_ptr + out_idx, mask=in_b, other=False).to(tl.int1)
    m = m_i.to(tl.float64)
    m1 = m - 1.0
    sqrt_2m1 = tl.sqrt(tl.maximum(2.0 * m1, 0.0))
    if mode == 0:
        a = m * y - x * x - x * m1
        c_val = x * nsig_b * sqrt_2m1
    elif mode == 1:
        a = m * y - x * x
        c_val = gain_val * x * (m1 + nsig_b * sqrt_2m1)
    else:
        g = tl.load(gain_ptr + out_idx, mask=in_b, other=1.0).to(tl.float64)
        a = m * y - x * x
        c_val = g * x * (m1 + nsig_b * sqrt_2m1)
    if is_extended:
        disp_pass = (m_i >= min_count) & (x >= 0.0) & (a > c_val)
        out = tl.where(mk, disp_pass, True)
    else:
        src_val = tl.load(src_ptr + out_idx, mask=in_b, other=0).to(tl.float64)
        b = m * src_val - x
        if mode == 0:
            d = nsig_s * tl.sqrt(tl.maximum(x * m, 0.0))
        elif mode == 1:
            d = nsig_s * tl.sqrt(tl.maximum(gain_val * x * m, 0.0))
        else:
            d = nsig_s * tl.sqrt(tl.maximum(g * x * m, 0.0))
        out = (
            mk
            & (m_i >= min_count)
            & (x >= 0.0)
            & (src_val > global_threshold)
            & (a > c_val)
            & (b > d)
        )
    tl.store(dst_ptr + out_idx, out, mask=in_b)


@triton.jit
def _triton_erode_vert(dst_ptr, ev_ptr, H, W, r_erode: tl.constexpr, BLOCK_W: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    in_b = c < W
    acc = tl.full((BLOCK_W,), True, dtype=tl.int1)
    for dy in tl.static_range(-r_erode, r_erode + 1):
        nr = r + dy
        v_r = (nr >= 0) & (nr < H) & in_b
        val = tl.load(dst_ptr + nr * W + c, mask=v_r, other=True).to(tl.int1)
        acc = acc & val
    tl.store(ev_ptr + r * W + c, acc, mask=in_b)


@triton.jit
def _triton_erode_horiz_bg(
    mask_ptr, ev_ptr, bg_mask_ptr, H, W, r_erode: tl.constexpr, BLOCK_W: tl.constexpr
):
    r = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    in_b = c < W
    row_off = r * W
    eroded = tl.full((BLOCK_W,), True, dtype=tl.int1)
    for dx in tl.static_range(-r_erode, r_erode + 1):
        nc = c + dx
        v_c = (nc >= 0) & (nc < W)
        val = tl.load(ev_ptr + row_off + nc, mask=v_c, other=True).to(tl.int1)
        eroded = eroded & val
    mk = tl.load(mask_ptr + row_off + c, mask=in_b, other=False).to(tl.int1)
    bg = mk & (~eroded)
    tl.store(bg_mask_ptr + row_off + c, bg, mask=in_b)


@triton.jit
def _triton_pass2_horiz(
    src_ptr, bg_mask_ptr, m2h_ptr, x2h_ptr, H, W, kx2: tl.constexpr, BLOCK_W: tl.constexpr
):
    r = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    in_b = c < W
    m = tl.zeros((BLOCK_W,), dtype=tl.int32)
    x = tl.zeros((BLOCK_W,), dtype=tl.float64)
    row_off = r * W
    for dx in tl.static_range(-kx2, kx2 + 1):
        nc = c + dx
        v_c = (nc >= 0) & (nc < W)
        idx = row_off + nc
        bg = tl.load(bg_mask_ptr + idx, mask=v_c, other=False).to(tl.int1)
        val = tl.load(src_ptr + idx, mask=v_c & bg, other=0).to(tl.float64)
        v = v_c & bg & (val < 16777216.0)
        vx = tl.where(v, val, 0.0)
        m += tl.where(v, 1, 0)
        x += vx
    out_idx = row_off + c
    tl.store(m2h_ptr + out_idx, m, mask=in_b)
    tl.store(x2h_ptr + out_idx, x, mask=in_b)


@triton.jit
def _triton_pass2_vert(
    src_ptr,
    mask_ptr,
    bg_mask_ptr,
    m2h_ptr,
    x2h_ptr,
    out_ptr,
    H,
    W,
    ky2: tl.constexpr,
    nsig_s,
    global_threshold,
    gain_ptr,
    gain_val,
    mode: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    r = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    in_b = c < W
    m_i = tl.zeros((BLOCK_W,), dtype=tl.int32)
    x = tl.zeros((BLOCK_W,), dtype=tl.float64)
    for dy in tl.static_range(-ky2, ky2 + 1):
        nr = r + dy
        v_r = (nr >= 0) & (nr < H) & in_b
        idx = nr * W + c
        m_i += tl.load(m2h_ptr + idx, mask=v_r, other=0)
        x += tl.load(x2h_ptr + idx, mask=v_r, other=0.0)
    out_idx = r * W + c
    mk = tl.load(mask_ptr + out_idx, mask=in_b, other=False).to(tl.int1)
    bg = tl.load(bg_mask_ptr + out_idx, mask=in_b, other=False).to(tl.int1)
    src_val = tl.load(src_ptr + out_idx, mask=in_b, other=0).to(tl.float64)
    m = m_i.to(tl.float64)
    mean = tl.where(m_i >= 2, x / tl.maximum(m, 1.0), 0.0)
    if mode == 0:
        local_thresh = mean + nsig_s * tl.sqrt(tl.maximum(mean, 0.0))
    elif mode == 1:
        local_thresh = mean + nsig_s * tl.sqrt(tl.maximum(gain_val * mean, 0.0))
    else:
        g = tl.load(gain_ptr + out_idx, mask=in_b, other=1.0).to(tl.float64)
        local_thresh = mean + nsig_s * tl.sqrt(tl.maximum(g * mean, 0.0))
    final = (
        mk
        & (m_i >= 0)
        & (x >= 0.0)
        & (~bg)
        & (src_val > global_threshold)
        & (src_val >= local_thresh)
    )
    tl.store(out_ptr + out_idx, final, mask=in_b)


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def flex_to_tensor(arr, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    if device is None:
        device = get_default_device()
    if isinstance(arr, torch.Tensor):
        t = arr.to(device=device)
        if dtype is not None:
            t = t.to(dtype=dtype)
        return t
    np_arr = flumpy.to_numpy(arr)
    t = torch.from_numpy(np_arr).to(device=device)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t


def tensor_to_flex_bool(t: torch.Tensor) -> flex.bool:
    np_arr = np.ascontiguousarray(t.detach().cpu().numpy(), dtype=bool)
    return flumpy.from_numpy(np_arr)


def estimate_global_threshold(
    image,
    mask=None,
    plot: bool = False,
    device: torch.device | None = None,
) -> float:
    """
    GPU implementation of dials.extensions.dispersion_spotfinder_threshold_ext.estimate_global_threshold.
    """
    if device is None:
        device = get_default_device()

    img_t = flex_to_tensor(image, device=device, dtype=torch.float64)
    thresholds = [float(int(1.5**i)) for i in range(1, 20)]
    thresh_t = torch.tensor(thresholds, dtype=torch.float64, device=device)

    # Vectorized count of pixels > g for all 19 thresholds on GPU
    n_above_t = (img_t.reshape(1, -1) > thresh_t.unsqueeze(1)).sum(dim=1).to(torch.float64)
    x = np.array(thresholds, dtype=np.float64)
    y = n_above_t.cpu().numpy()

    slopes = (y[-1] - y[:-1]) / (x[-1] - x[:-1])
    p_m = int(np.argmin(slopes))

    x1 = np.array([x[p_m], y[p_m]], dtype=np.float64)
    x2 = np.array([x[-1], y[-1]], dtype=np.float64)
    v = np.array([x2[1] - x1[1], -(x2[0] - x1[0])], dtype=np.float64)
    v = v / np.linalg.norm(v)

    gaps = []
    for i in range(p_m, len(x)):
        x0 = np.array([x[i], y[i]], dtype=np.float64)
        r = x1 - x0
        gaps.append(abs(float(np.dot(v, r))))

    p_g = int(np.argmax(gaps))
    x_g = float(x[p_g + p_m - 2])
    return x_g


def _box_sum_2d(arr: torch.Tensor, ky: int, kx: int) -> torch.Tensor:
    H, W = arr.shape
    cs = arr.cumsum(dim=0)
    v = torch.empty_like(cs)
    v[: ky + 1, :] = cs[ky : 2 * ky + 1, :]
    v[ky + 1 : H - ky, :] = cs[2 * ky + 1 :, :] - cs[: H - 2 * ky - 1, :]
    if ky > 0:
        v[H - ky :, :] = cs[-1:, :] - cs[H - 2 * ky - 1 : H - ky - 1, :]

    cs = v.cumsum(dim=1)
    out = v
    out[:, : kx + 1] = cs[:, kx : 2 * kx + 1]
    out[:, kx + 1 : W - kx] = cs[:, 2 * kx + 1 :] - cs[:, : W - 2 * kx - 1]
    if kx > 0:
        out[:, W - kx :] = cs[:, -1:] - cs[:, W - 2 * kx - 1 : W - kx - 1]
    return out


def _compute_sat_stats(
    src: torch.Tensor,
    mask: torch.Tensor,
    kernel_size: tuple[int, int],
    compute_y: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Compute 2D Summed-Area Table (SAT) statistics over window (2*ky+1, 2*kx+1)
    truncated at image boundaries, matching dials/algorithms/image/threshold/local.h.

    Args:
        src: (H, W) float64 tensor of pixel values
        mask: (H, W) bool tensor of valid pixels
        kernel_size: (ky, kx) where ky is slow/row radius and kx is fast/col radius
        compute_y: Whether to also compute sum of squares y

    Returns:
        (m, x, y) where m is pixel count, x is sum(src), y is sum(src^2)
    """
    ky, kx = int(kernel_size[0]), int(kernel_size[1])

    # Match local.h: m = (mask[i] && (src[i] < (1 << 24))) ? 1 : 0
    valid = mask & (src < float(1 << 24))
    m_in_f32 = valid.to(torch.float32)
    x_in = torch.where(valid, src, torch.zeros_like(src))

    m = _box_sum_2d(m_in_f32, ky, kx).to(src.dtype)
    x = _box_sum_2d(x_in, ky, kx)
    y = _box_sum_2d(x_in * x_in, ky, kx) if compute_y else None
    return m, x, y


def dispersion_threshold_gpu(
    src: torch.Tensor,
    mask: torch.Tensor,
    gain: torch.Tensor | float | None = None,
    kernel_size: tuple[int, int] = (3, 3),
    nsig_b: float = 6.0,
    nsig_s: float = 3.0,
    global_threshold: float = 0.0,
    min_local: int = 2,
) -> torch.Tensor:
    """
    PyTorch GPU implementation of DispersionThreshold (Kabsch 2010).
    """
    assert nsig_b >= 0 and nsig_s >= 0
    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    min_count = (2 * kx + 1) * (2 * ky + 1) if min_local <= 0 else min_local

    if src.is_cuda and src.is_contiguous() and mask.is_contiguous():
        H, W = src.shape
        bufs = _get_triton_scratch(src.device, H, W)
        out = torch.empty((H, W), dtype=torch.bool, device=src.device)
        grid = (H, triton.cdiv(W, 256))
        if gain is None:
            mode = 0
            gain_ptr = src
            gain_val = 1.0
        elif isinstance(gain, torch.Tensor):
            mode = 2
            gain_ptr = gain.contiguous()
            gain_val = 1.0
        else:
            mode = 1
            gain_ptr = src
            gain_val = float(gain)

        _triton_pass1_horiz[grid](
            src, mask, bufs["mh"], bufs["xh"], bufs["yh"], H, W, kx, 256
        )
        _triton_pass1_vert[grid](
            src,
            mask,
            bufs["mh"],
            bufs["xh"],
            bufs["yh"],
            out,
            H,
            W,
            ky,
            int(min_count),
            float(nsig_b),
            float(nsig_s),
            float(global_threshold),
            gain_ptr,
            gain_val,
            mode,
            False,
            256,
        )
        return out

    src_d = src.to(torch.float64)
    m, x, y = _compute_sat_stats(src_d, mask, (ky, kx), compute_y=True)
    assert y is not None

    b = m * src_d - x
    sqrt_2m1 = torch.sqrt(torch.clamp(2.0 * (m - 1.0), min=0.0))

    if gain is None:
        a = m * y - x * x - x * (m - 1.0)
        c = x * nsig_b * sqrt_2m1
        d = nsig_s * torch.sqrt(torch.clamp(x * m, min=0.0))
    else:
        g = gain.to(torch.float64) if isinstance(gain, torch.Tensor) else float(gain)
        a = m * y - x * x
        c = g * x * (m - 1.0 + nsig_b * sqrt_2m1)
        d = nsig_s * torch.sqrt(torch.clamp(g * x * m, min=0.0))

    dst = (
        mask
        & (m >= min_count)
        & (x >= 0)
        & (src_d > global_threshold)
        & (a > c)
        & (b > d)
    )
    return dst


def dispersion_extended_threshold_gpu(
    src: torch.Tensor,
    mask: torch.Tensor,
    gain: torch.Tensor | float | None = None,
    kernel_size: tuple[int, int] = (3, 3),
    nsig_b: float = 6.0,
    nsig_s: float = 3.0,
    global_threshold: float = 0.0,
    min_local: int = 2,
) -> torch.Tensor:
    """
    PyTorch GPU implementation of DispersionExtendedThreshold.
    Matches dials/algorithms/image/threshold/local.h DispersionExtendedThreshold.
    """
    assert nsig_b >= 0 and nsig_s >= 0
    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    min_count = (2 * kx + 1) * (2 * ky + 1) if min_local <= 0 else min_local
    erosion_distance = min(ky, kx)

    if (
        src.is_cuda
        and src.is_contiguous()
        and mask.is_contiguous()
        and erosion_distance >= 1
    ):
        H, W = src.shape
        bufs = _get_triton_scratch(src.device, H, W)
        out = torch.empty((H, W), dtype=torch.bool, device=src.device)
        grid = (H, triton.cdiv(W, 256))
        if gain is None:
            mode = 0
            gain_ptr = src
            gain_val = 1.0
        elif isinstance(gain, torch.Tensor):
            mode = 2
            gain_ptr = gain.contiguous()
            gain_val = 1.0
        else:
            mode = 1
            gain_ptr = src
            gain_val = float(gain)

        _triton_pass1_horiz[grid](
            src, mask, bufs["mh"], bufs["xh"], bufs["yh"], H, W, kx, 256
        )
        _triton_pass1_vert[grid](
            src,
            mask,
            bufs["mh"],
            bufs["xh"],
            bufs["yh"],
            bufs["dst"],
            H,
            W,
            ky,
            int(min_count),
            float(nsig_b),
            float(nsig_s),
            float(global_threshold),
            gain_ptr,
            gain_val,
            mode,
            True,
            256,
        )
        r_erode = erosion_distance - 1
        _triton_erode_vert[grid](bufs["dst"], bufs["ev"], H, W, r_erode, 256)
        _triton_erode_horiz_bg[grid](
            mask, bufs["ev"], bufs["bg_mask"], H, W, r_erode, 256
        )
        _triton_pass2_horiz[grid](
            src, bufs["bg_mask"], bufs["mh"], bufs["xh"], H, W, kx + 2, 256
        )
        _triton_pass2_vert[grid](
            src,
            mask,
            bufs["bg_mask"],
            bufs["mh"],
            bufs["xh"],
            out,
            H,
            W,
            ky + 2,
            float(nsig_s),
            float(global_threshold),
            gain_ptr,
            gain_val,
            mode,
            256,
        )
        return out

    src_d = src.to(torch.float64)
    m, x, y = _compute_sat_stats(src_d, mask, (ky, kx), compute_y=True)
    assert y is not None

    sqrt_2m1 = torch.sqrt(torch.clamp(2.0 * (m - 1.0), min=0.0))
    if gain is None:
        a = m * y - x * x - x * (m - 1.0)
        c = x * nsig_b * sqrt_2m1
    else:
        g = gain.to(torch.float64) if isinstance(gain, torch.Tensor) else float(gain)
        a = m * y - x * x
        c = g * x * (m - 1.0 + nsig_b * sqrt_2m1)

    # Pass 1: Dispersion test. Invalid (masked-out) pixels are set to True
    # so they do not act as background seeds during Chebyshev erosion.
    disp_pass = (m >= min_count) & (x >= 0) & (a > c)
    dst = torch.where(mask, disp_pass, torch.ones_like(mask))

    # Stage 1b: Erode dispersion mask using Chebyshev distance transform
    erosion_distance = min(ky, kx)
    if erosion_distance <= 0:
        eroded_dst = torch.ones_like(dst)
    else:
        r = erosion_distance - 1
        if r == 0:
            has_bg = ~dst
        else:
            not_dst_f = (~dst).to(torch.float32).unsqueeze(0).unsqueeze(0)
            pooled = F.max_pool2d(not_dst_f, kernel_size=2 * r + 1, stride=1, padding=r)
            has_bg = pooled.squeeze(0).squeeze(0) > 0.5
        eroded_dst = dst & (~has_bg)

    # Background mask for pass 2: valid pixels that were not in the eroded strong mask
    bg_mask = mask & (~eroded_dst)

    # Stage 2: Recompute SAT with widened kernel (ky + 2, kx + 2) over background pixels
    m2, x2, _ = _compute_sat_stats(src_d, bg_mask, (ky + 2, kx + 2), compute_y=False)
    mean = torch.where(m2 >= 2, x2 / torch.clamp(m2, min=1.0), torch.zeros_like(x2))

    if gain is None:
        local_thresh = mean + nsig_s * torch.sqrt(torch.clamp(mean, min=0.0))
    else:
        g = gain.to(torch.float64) if isinstance(gain, torch.Tensor) else float(gain)
        local_thresh = mean + nsig_s * torch.sqrt(torch.clamp(g * mean, min=0.0))

    final_mask = (
        mask
        & (m2 >= 0)
        & (x2 >= 0)
        & (~bg_mask)
        & (src_d > global_threshold)
        & (src_d >= local_thresh)
    )
    return final_mask


def dispersion_threshold_batch_gpu(
    src: torch.Tensor,
    mask: torch.Tensor,
    gain: torch.Tensor | float | None = None,
    kernel_size: tuple[int, int] = (3, 3),
    nsig_b: float = 6.0,
    nsig_s: float = 3.0,
    global_threshold: float = 0.0,
    min_local: int = 2,
) -> torch.Tensor:
    """
    Batched 3D [B, H, W] GPU implementation of DispersionThreshold using float32 Triton kernels.
    """
    assert nsig_b >= 0 and nsig_s >= 0
    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    min_count = (2 * kx + 1) * (2 * ky + 1) if min_local <= 0 else min_local
    B, H, W = src.shape
    bufs = _get_triton_batch_scratch(src.device, B, H, W)
    out = bufs["dst"]
    grid = (B * H, triton.cdiv(W, 256))
    if gain is None:
        mode = 0
        gain_ptr = src
        gain_val = 1.0
    elif isinstance(gain, torch.Tensor):
        mode = 2
        gain_ptr = gain.contiguous()
        gain_val = 1.0
    else:
        mode = 1
        gain_ptr = src
        gain_val = float(gain)

    _batch_pass1_horiz[grid](src, mask, bufs["mh"], bufs["xh"], bufs["yh"], H, W, kx, 256)
    _batch_pass1_vert[grid](
        src,
        mask,
        bufs["mh"],
        bufs["xh"],
        bufs["yh"],
        out,
        H,
        W,
        ky,
        int(min_count),
        float(nsig_b),
        float(nsig_s),
        float(global_threshold),
        gain_ptr,
        gain_val,
        mode,
        False,
        256,
    )
    return out


def dispersion_extended_threshold_batch_gpu(
    src: torch.Tensor,
    mask: torch.Tensor,
    gain: torch.Tensor | float | None = None,
    kernel_size: tuple[int, int] = (3, 3),
    nsig_b: float = 6.0,
    nsig_s: float = 3.0,
    global_threshold: float = 0.0,
    min_local: int = 2,
) -> torch.Tensor:
    """
    Batched 3D [B, H, W] GPU implementation of DispersionExtendedThreshold using float32 Triton kernels
    with sparse Pass 2 evaluation.
    """
    assert nsig_b >= 0 and nsig_s >= 0
    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    min_count = (2 * kx + 1) * (2 * ky + 1) if min_local <= 0 else min_local
    erosion_distance = min(ky, kx)
    if erosion_distance < 1:
        return torch.stack(
            [
                dispersion_extended_threshold_gpu(
                    src[b],
                    mask[b],
                    gain=gain,
                    kernel_size=kernel_size,
                    nsig_b=nsig_b,
                    nsig_s=nsig_s,
                    global_threshold=global_threshold,
                    min_local=min_local,
                )
                for b in range(src.shape[0])
            ],
            dim=0,
        )

    B, H, W = src.shape
    bufs = _get_triton_batch_scratch(src.device, B, H, W)
    out = bufs["dst"]
    grid = (B * H, triton.cdiv(W, 256))
    if gain is None:
        mode = 0
        gain_ptr = src
        gain_val = 1.0
    elif isinstance(gain, torch.Tensor):
        mode = 2
        gain_ptr = gain.contiguous()
        gain_val = 1.0
    else:
        mode = 1
        gain_ptr = src
        gain_val = float(gain)

    _batch_pass1_horiz[grid](src, mask, bufs["mh"], bufs["xh"], bufs["yh"], H, W, kx, 256)
    _batch_pass1_vert[grid](
        src,
        mask,
        bufs["mh"],
        bufs["xh"],
        bufs["yh"],
        bufs["dst"],
        H,
        W,
        ky,
        int(min_count),
        float(nsig_b),
        float(nsig_s),
        float(global_threshold),
        gain_ptr,
        gain_val,
        mode,
        True,
        256,
    )
    r_erode = erosion_distance - 1
    _batch_erode_vert[grid](bufs["dst"], bufs["ev"], H, W, r_erode, 256)
    _batch_erode_horiz_bg[grid](mask, bufs["ev"], bufs["bg_mask"], H, W, r_erode, 256)
    _batch_pass2_sparse[grid](
        src,
        mask,
        bufs["bg_mask"],
        out,
        H,
        W,
        ky + 2,
        kx + 2,
        float(nsig_s),
        float(global_threshold),
        gain_ptr,
        gain_val,
        mode,
        256,
    )
    return out


class DispersionSpotFinderThresholdGPU:
    """
    Drop-in GPU replacement for DispersionSpotFinderThreshold / ThresholdDebugWrapper.
    """

    def __init__(
        self,
        kernel_size: tuple[int, int] = (3, 3),
        gain: float | None = None,
        mask=None,
        n_sigma_b: float = 6.0,
        n_sigma_s: float = 3.0,
        min_count: int = 2,
        global_threshold: float | None = 0.0,
        device: torch.device | None = None,
    ):
        self.kernel_size = kernel_size
        self.gain = gain
        self.mask = mask
        self.n_sigma_b = n_sigma_b
        self.n_sigma_s = n_sigma_s
        self.min_count = min_count
        self.global_threshold = global_threshold
        self.device = device or get_default_device()

    def compute_threshold(self, image, mask, gain=None):
        img_t = flex_to_tensor(image, device=self.device, dtype=torch.float64)
        mask_t = flex_to_tensor(mask, device=self.device, dtype=torch.bool)
        dst_t = self.compute_threshold_tensor(img_t, mask_t, gain=gain)
        return tensor_to_flex_bool(dst_t)

    def compute_threshold_tensor(
        self,
        img_t: torch.Tensor,
        mask_t: torch.Tensor,
        gain=None,
    ) -> torch.Tensor:
        if self.mask is not None and not isinstance(self.mask, (str, tuple, list)):
            ext_mask_t = flex_to_tensor(self.mask, device=self.device, dtype=torch.bool)
            mask_t = mask_t & ext_mask_t

        if self.global_threshold is None:
            global_threshold = estimate_global_threshold(img_t, mask_t, device=self.device)
        else:
            global_threshold = float(self.global_threshold)

        gain_val = None
        if gain is not None and self.gain is not None:
            gain_t = flex_to_tensor(gain, device=self.device, dtype=torch.float64)
            gain_val = gain_t * float(self.gain)
        elif gain is not None:
            gain_val = flex_to_tensor(gain, device=self.device, dtype=torch.float64)
        elif self.gain is not None:
            gain_val = float(self.gain)

        return dispersion_threshold_gpu(
            src=img_t,
            mask=mask_t,
            gain=gain_val,
            kernel_size=self.kernel_size,
            nsig_b=self.n_sigma_b,
            nsig_s=self.n_sigma_s,
            global_threshold=global_threshold,
            min_local=self.min_count,
        )

    def compute_threshold_batch_tensor(
        self,
        img_vol: torch.Tensor,
        mask_vol: torch.Tensor,
        gain=None,
    ) -> torch.Tensor:
        if self.mask is not None and not isinstance(self.mask, (str, tuple, list)):
            ext_mask_t = flex_to_tensor(self.mask, device=self.device, dtype=torch.bool)
            mask_vol = mask_vol & ext_mask_t.unsqueeze(0)

        if self.global_threshold is None:
            return torch.stack(
                [
                    self.compute_threshold_tensor(img_vol[b], mask_vol[b], gain=gain)
                    for b in range(img_vol.shape[0])
                ],
                dim=0,
            )
        global_threshold = float(self.global_threshold)

        gain_val = None
        if gain is not None and self.gain is not None:
            gain_t = flex_to_tensor(gain, device=self.device, dtype=torch.float32)
            gain_val = gain_t * float(self.gain)
        elif gain is not None:
            gain_val = flex_to_tensor(gain, device=self.device, dtype=torch.float32)
        elif self.gain is not None:
            gain_val = float(self.gain)

        return dispersion_threshold_batch_gpu(
            src=img_vol,
            mask=mask_vol,
            gain=gain_val,
            kernel_size=self.kernel_size,
            nsig_b=self.n_sigma_b,
            nsig_s=self.n_sigma_s,
            global_threshold=global_threshold,
            min_local=self.min_count,
        )


class DispersionExtendedSpotFinderThresholdGPU(DispersionSpotFinderThresholdGPU):
    """
    Drop-in GPU replacement for DispersionExtendedSpotFinderThreshold.
    """

    def compute_threshold_tensor(
        self,
        img_t: torch.Tensor,
        mask_t: torch.Tensor,
        gain=None,
    ) -> torch.Tensor:
        if self.mask is not None and not isinstance(self.mask, (str, tuple, list)):
            ext_mask_t = flex_to_tensor(self.mask, device=self.device, dtype=torch.bool)
            mask_t = mask_t & ext_mask_t

        if self.global_threshold is None:
            global_threshold = estimate_global_threshold(img_t, mask_t, device=self.device)
        else:
            global_threshold = float(self.global_threshold)

        gain_val = None
        if gain is not None and self.gain is not None:
            gain_t = flex_to_tensor(gain, device=self.device, dtype=torch.float64)
            gain_val = gain_t * float(self.gain)
        elif gain is not None:
            gain_val = flex_to_tensor(gain, device=self.device, dtype=torch.float64)
        elif self.gain is not None:
            gain_val = float(self.gain)

        return dispersion_extended_threshold_gpu(
            src=img_t,
            mask=mask_t,
            gain=gain_val,
            kernel_size=self.kernel_size,
            nsig_b=self.n_sigma_b,
            nsig_s=self.n_sigma_s,
            global_threshold=global_threshold,
            min_local=self.min_count,
        )

    def compute_threshold_batch_tensor(
        self,
        img_vol: torch.Tensor,
        mask_vol: torch.Tensor,
        gain=None,
    ) -> torch.Tensor:
        if self.mask is not None and not isinstance(self.mask, (str, tuple, list)):
            ext_mask_t = flex_to_tensor(self.mask, device=self.device, dtype=torch.bool)
            mask_vol = mask_vol & ext_mask_t.unsqueeze(0)

        if self.global_threshold is None:
            return torch.stack(
                [
                    self.compute_threshold_tensor(img_vol[b], mask_vol[b], gain=gain)
                    for b in range(img_vol.shape[0])
                ],
                dim=0,
            )
        global_threshold = float(self.global_threshold)

        gain_val = None
        if gain is not None and self.gain is not None:
            gain_t = flex_to_tensor(gain, device=self.device, dtype=torch.float32)
            gain_val = gain_t * float(self.gain)
        elif gain is not None:
            gain_val = flex_to_tensor(gain, device=self.device, dtype=torch.float32)
        elif self.gain is not None:
            gain_val = float(self.gain)

        return dispersion_extended_threshold_batch_gpu(
            src=img_vol,
            mask=mask_vol,
            gain=gain_val,
            kernel_size=self.kernel_size,
            nsig_b=self.n_sigma_b,
            nsig_s=self.n_sigma_s,
            global_threshold=global_threshold,
            min_local=self.min_count,
        )


class RadialProfileSpotFinderThresholdGPU:
    """
    PyTorch GPU implementation of RadialProfileSpotFinderThresholdExt.
    """

    _NARROW_KERNEL = (
        0.0625, 0.125, 0.0625,
        0.125,  0.25,  0.125,
        0.0625, 0.125, 0.0625,
    )
    _WIDE_KERNEL = tuple(
        x / 273.0
        for x in (
            1,  4,  7,  4,  1,
            4, 16, 26, 16,  4,
            7, 26, 41, 26,  7,
            4, 16, 26, 16,  4,
            1,  4,  7,  4,  1,
        )
    )

    def __init__(
        self,
        params=None,
        *,
        imageset=None,
        n_iqr: float = 6,
        blur: str | None = None,
        n_bins: int = 100,
        device: torch.device | None = None,
    ):
        self.params = params
        self.imageset = imageset
        if params is not None:
            rp_params = params.spotfinder.threshold.radial_profile
            self.n_iqr = rp_params.n_iqr
            self.blur = rp_params.blur
            self.n_bins = rp_params.n_bins
        else:
            self.n_iqr = n_iqr
            self.blur = blur
            self.n_bins = n_bins

        self.device = device or get_default_device()

        if self.blur == "narrow":
            self._kernel = torch.tensor(self._NARROW_KERNEL, dtype=torch.float64, device=self.device).reshape(1, 1, 3, 3)
            self._pad = 1
        elif self.blur == "wide":
            self._kernel = torch.tensor(self._WIDE_KERNEL, dtype=torch.float64, device=self.device).reshape(1, 1, 5, 5)
            self._pad = 2
        else:
            assert self.blur is None
            self._kernel = None
            self._pad = 0

        self._panel_cache: dict[tuple, tuple[torch.Tensor, float]] = {}

    def _get_panel_binning(self, imageset, i_panel: int, region_of_interest=None):
        key = (id(imageset), i_panel, tuple(region_of_interest) if region_of_interest else None)
        if key not in self._panel_cache:
            beam = imageset.get_beam()
            panel = imageset.get_detector()[i_panel]
            s0 = beam.get_unit_s0()
            two_theta = panel.get_two_theta_array(s0)
            if region_of_interest:
                x0, x1, y0, y1 = region_of_interest
                two_theta = two_theta[y0:y1, x0:x1]
            tt_t = flex_to_tensor(two_theta, device=self.device, dtype=torch.float64)
            lookup = tt_t - tt_t.min()
            multiplier = self.n_bins / float((lookup + 1e-10).max().item())
            lookup = lookup * multiplier
            bins = torch.floor(lookup).round().to(torch.int64)
            bins = torch.clamp(bins, min=0, max=self.n_bins - 1)
            adu = 1.0 / panel.get_gain()
            self._panel_cache[key] = (bins.reshape(-1), adu)
        return self._panel_cache[key]

    def compute_threshold(
        self,
        image,
        mask,
        *,
        imageset=None,
        i_panel: int = 0,
        region_of_interest=None,
        **kwargs,
    ):
        img_t = flex_to_tensor(image, device=self.device, dtype=torch.float64)
        mask_t = flex_to_tensor(mask, device=self.device, dtype=torch.bool)
        dst_t = self.compute_threshold_tensor(
            img_t,
            mask_t,
            imageset=imageset or self.imageset,
            i_panel=i_panel,
            region_of_interest=region_of_interest,
        )
        return tensor_to_flex_bool(dst_t)

    def compute_threshold_tensor(
        self,
        img_t: torch.Tensor,
        mask_t: torch.Tensor,
        *,
        imageset=None,
        i_panel: int = 0,
        region_of_interest=None,
    ) -> torch.Tensor:
        imageset = imageset or self.imageset
        H, W = img_t.shape
        if self._kernel is not None:
            padded = F.pad(img_t.unsqueeze(0).unsqueeze(0), (self._pad, self._pad, self._pad, self._pad), mode="replicate")
            work_img = F.conv2d(padded, self._kernel).squeeze(0).squeeze(0)
        else:
            work_img = img_t

        full_bins, adu = self._get_panel_binning(imageset, i_panel, region_of_interest=region_of_interest)
        mask_flat = mask_t.reshape(-1)
        val_valid = work_img.reshape(-1)[mask_flat]
        bin_valid = full_bins[mask_flat]

        # Sort by (bin, value) on GPU
        val_sorted, perm1 = torch.sort(val_valid)
        bin_after_val = bin_valid[perm1]
        bin_sorted, perm2 = torch.sort(bin_after_val, stable=True)
        val_sorted = val_sorted[perm2]

        counts = torch.bincount(bin_sorted, minlength=self.n_bins)
        ends = torch.cumsum(counts, dim=0)
        starts = ends - counts

        valid_bins = counts > 2
        n_f = counts.to(torch.float64)

        def _interp_quantile(p: float) -> torch.Tensor:
            q_idx = (n_f - 1.0) * p
            idx_lo = torch.clamp(torch.floor(q_idx).to(torch.int64), min=0)
            idx_hi = torch.clamp(idx_lo + 1, max=torch.clamp(counts - 1, min=0))
            frac = q_idx - idx_lo.to(torch.float64)
            safe_starts = torch.where(valid_bins, starts, torch.zeros_like(starts))
            safe_lo = torch.where(valid_bins, safe_starts + idx_lo, torch.zeros_like(starts))
            safe_hi = torch.where(valid_bins, safe_starts + idx_hi, torch.zeros_like(starts))
            if val_sorted.numel() == 0:
                return torch.zeros(self.n_bins, dtype=torch.float64, device=self.device)
            v_lo = val_sorted[safe_lo]
            v_hi = val_sorted[safe_hi]
            res = v_lo * (1.0 - frac) + v_hi * frac
            return torch.where(valid_bins, res, torch.zeros_like(res))

        med_I = _interp_quantile(0.5)
        q1 = _interp_quantile(0.25)
        q3 = _interp_quantile(0.75)
        iqr = q3 - q1

        add_level = self.n_iqr * iqr
        add_level = torch.where(add_level <= adu, torch.full_like(add_level, 2.0 * adu), add_level)
        thresh_bins = med_I + add_level

        thresh_img = thresh_bins[full_bins].reshape(H, W)
        result = (work_img > thresh_img) & mask_t
        return result
