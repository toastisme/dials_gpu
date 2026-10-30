from __future__ import annotations

import math

import numpy as np
import torch
from scitbx import lbfgs
from scitbx.array_family import flex


def optimise_basis_vectors_gpu(reciprocal_lattice_points, vectors, device=None):
    """
    GPU-accelerated replacement for dials.algorithms.indexing.basis_vector_search.optimise.optimise_basis_vectors.
    Uses GPU tensor evaluation for large reflection sets and L1-cached vectorized
    evaluation during sequential L-BFGS steps when N < 16384 to avoid PCIe sync latency.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(vectors) == 0:
        return flex.vec3_double()

    target = BasisVectorTargetGPU(reciprocal_lattice_points, device=device)
    optimised = flex.vec3_double()
    for vector in vectors:
        minimised = BasisVectorMinimiserGPU(
            reciprocal_lattice_points, vector, target=target
        )
        optimised.append(tuple(minimised.x))

    # Compute functionals for all input vectors in a single batched matrix multiply
    vecs_np = np.array([tuple(v) for v in vectors], dtype=np.float64)
    if target.use_gpu:
        vecs_t = torch.from_numpy(vecs_np).to(device=device, dtype=torch.float64)
        dots = vecs_t @ target.two_pi_rlp_t.T
        funcs_np = np.ascontiguousarray(
            (-torch.cos(dots).sum(dim=1)).cpu().numpy(), dtype=np.float64
        )
    else:
        dots = vecs_np @ target.two_pi_rlp_np.T
        funcs_np = np.ascontiguousarray(-np.cos(dots).sum(axis=1), dtype=np.float64)
    functionals = flex.double(funcs_np)
    perm = flex.sort_permutation(functionals)
    return optimised.select(perm)


class BasisVectorTargetGPU:
    def __init__(
        self, reciprocal_lattice_points, device=None, gpu_threshold: int = 16384
    ):
        self.reciprocal_lattice_points = reciprocal_lattice_points
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if isinstance(reciprocal_lattice_points, torch.Tensor):
            rlp_np = reciprocal_lattice_points.detach().cpu().numpy()
            self.n_points = rlp_np.shape[0]
            self.use_gpu = (
                self.device.type == "cuda" and self.n_points >= gpu_threshold
            )
            if self.use_gpu:
                self.rlp_t = reciprocal_lattice_points.to(
                    device=self.device, dtype=torch.float64
                )
                self.two_pi_rlp_t = (2.0 * math.pi) * self.rlp_t
            self.two_pi_rlp_np = (2.0 * math.pi) * np.ascontiguousarray(
                rlp_np, dtype=np.float64
            )
        else:
            rlp_np = reciprocal_lattice_points.as_numpy_array()
            self.n_points = rlp_np.shape[0]
            self.use_gpu = (
                self.device.type == "cuda" and self.n_points >= gpu_threshold
            )
            if self.use_gpu:
                self.rlp_t = torch.from_numpy(rlp_np).to(
                    device=self.device, dtype=torch.float64
                )
                self.two_pi_rlp_t = (2.0 * math.pi) * self.rlp_t
            self.two_pi_rlp_np = (2.0 * math.pi) * np.ascontiguousarray(
                rlp_np, dtype=np.float64
            )

    def compute_functional(self, vector):
        if self.use_gpu:
            v_t = torch.tensor(tuple(vector), dtype=torch.float64, device=self.device)
            two_pi_s_dot_v = self.two_pi_rlp_t @ v_t
            return float(-torch.cos(two_pi_s_dot_v).sum().item())
        v_np = np.array(tuple(vector), dtype=np.float64)
        two_pi_s_dot_v = self.two_pi_rlp_np @ v_np
        return float(-np.cos(two_pi_s_dot_v).sum())

    def compute_functional_and_gradients(self, vector):
        assert len(vector) == 3
        if self.use_gpu:
            v_t = torch.tensor(tuple(vector), dtype=torch.float64, device=self.device)
            two_pi_s_dot_v = self.two_pi_rlp_t @ v_t
            f = float(-torch.cos(two_pi_s_dot_v).sum().item())
            sin_part = torch.sin(two_pi_s_dot_v)
            g_t = sin_part @ self.two_pi_rlp_t
            return f, flex.double(g_t.tolist())
        v_np = np.array(tuple(vector), dtype=np.float64)
        two_pi_s_dot_v = self.two_pi_rlp_np @ v_np
        f = float(-np.cos(two_pi_s_dot_v).sum())
        sin_part = np.sin(two_pi_s_dot_v)
        g_np = sin_part @ self.two_pi_rlp_np
        return f, flex.double(
            (float(g_np[0]), float(g_np[1]), float(g_np[2]))
        )


class BasisVectorMinimiserGPU:
    def __init__(
        self,
        reciprocal_lattice_points,
        vector,
        lbfgs_termination_params=None,
        lbfgs_core_params=lbfgs.core_parameters(m=20),
        target=None,
    ):
        self.reciprocal_lattice_points = reciprocal_lattice_points
        if not isinstance(vector, flex.double):
            self.x = flex.double(vector)
        else:
            self.x = vector.deep_copy()
        self.n = len(self.x)
        assert self.n == 3
        self.target = target or BasisVectorTargetGPU(self.reciprocal_lattice_points)
        self.minimizer = lbfgs.run(
            target_evaluator=self,
            termination_params=lbfgs_termination_params,
            core_params=lbfgs_core_params,
        )

    def compute_functional_and_gradients(self):
        return self.target.compute_functional_and_gradients(tuple(self.x))
